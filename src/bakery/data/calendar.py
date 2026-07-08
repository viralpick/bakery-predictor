"""Calendar / holiday frame.

Source-of-truth for date-level external calendar features. Because future
holiday dates are known in advance, every column here is leakage-safe by
construction — it depends only on `date`.

Real-data swap point: replace `build_calendar_daily` body with a loader that
hits the 천문연 특일정보 API. Schema must match `CALENDAR_DAILY_COLUMNS`.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import pandas as pd

CALENDAR_DAILY_COLUMNS: dict[str, str] = {
    "date": "datetime64[ns]",
    "holiday_name": "string",
    "is_public_holiday": "int8",
    "is_substitute_holiday": "int8",
    "is_weekend": "int8",
    "is_off_day": "int8",
    "is_day_before_off": "int8",
    "is_day_after_off": "int8",
    "off_streak_length": "int8",
    "off_position_in_streak": "int8",
    "is_xmas_eve": "int8",
    "is_xmas": "int8",
    "is_valentine": "int8",
    "is_white_day": "int8",
    "is_children_day": "int8",
    "is_pepero": "int8",
}

# `is_off_day` = public holiday OR weekend. Streak length is computed over
# consecutive off days. We expose both the public-holiday flag and the
# combined off flag because Korean weekend baseline already captures dow.

# 음력 명절 (양력 변동) → 연도별 양력 날짜 lookup. 광교 효과 검증:
# 추석 +11.4% (p=0.029), 설날 +11.0% (p=0.005) — the largest bakery demand
# events, so lead-up (days_to_*) matters. Single source of truth: both
# calendar_features (v0~v3) and category_aggregate (v4) import this so the
# dates never drift apart. Extend the tables as new years are announced.
LUNAR_EVENT_DATES: dict[str, dict[int, str]] = {
    "days_to_chuseok": {
        2021: "2021-09-21", 2022: "2022-09-10", 2023: "2023-09-29",
        2024: "2024-09-17", 2025: "2025-10-06", 2026: "2026-09-25",
        2027: "2027-09-15", 2028: "2028-10-03",
    },
    "days_to_seollal": {
        2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
        2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17",
        2027: "2027-02-06", 2028: "2028-01-26",
    },
}


def build_calendar_daily(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Return a daily calendar frame for [start, end] inclusive."""
    dates = pd.date_range(start, end, freq="D")
    if len(dates) == 0:
        raise ValueError(f"empty date range: {start} → {end}")
    kr = holidays.KR(years=range(dates.min().year, dates.max().year + 1))

    out = pd.DataFrame({"date": dates})
    holiday_name = out["date"].map(lambda d: kr.get(d.date(), "") or "").astype("string")
    out["holiday_name"] = holiday_name
    out["is_public_holiday"] = (holiday_name != "").astype("int8")
    out["is_substitute_holiday"] = (
        holiday_name.str.contains("대체", regex=False, na=False)
        | holiday_name.str.contains("Alternative", case=False, na=False)
        | holiday_name.str.contains("observed", case=False, na=False)
    ).astype("int8")
    out["is_weekend"] = (out["date"].dt.dayofweek >= 5).astype("int8")
    out["is_off_day"] = ((out["is_public_holiday"] == 1) | (out["is_weekend"] == 1)).astype("int8")

    _attach_streak(out)
    _attach_adjacency(out)
    _attach_event_flags(out)

    out = out[list(CALENDAR_DAILY_COLUMNS.keys())]
    return _coerce_dtypes(out)


def _attach_streak(out: pd.DataFrame) -> None:
    is_off = out["is_off_day"].astype(bool)
    group_id = (is_off != is_off.shift()).cumsum()
    streak_len = is_off.groupby(group_id).transform("sum").astype("int16")
    position = is_off.groupby(group_id).cumcount().add(1).astype("int16")
    out["off_streak_length"] = streak_len.where(is_off, 0).astype("int8")
    out["off_position_in_streak"] = position.where(is_off, 0).astype("int8")


def _attach_adjacency(out: pd.DataFrame) -> None:
    is_off = out["is_off_day"].astype(bool)
    next_off = is_off.shift(-1, fill_value=False)
    prev_off = is_off.shift(1, fill_value=False)
    out["is_day_before_off"] = (next_off & ~is_off).astype("int8")
    out["is_day_after_off"] = (prev_off & ~is_off).astype("int8")


def _attach_event_flags(out: pd.DataFrame) -> None:
    month = out["date"].dt.month
    day = out["date"].dt.day
    out["is_xmas_eve"] = ((month == 12) & (day == 24)).astype("int8")
    out["is_xmas"] = ((month == 12) & (day == 25)).astype("int8")
    out["is_valentine"] = ((month == 2) & (day == 14)).astype("int8")
    out["is_white_day"] = ((month == 3) & (day == 14)).astype("int8")
    out["is_children_day"] = ((month == 5) & (day == 5)).astype("int8")
    out["is_pepero"] = ((month == 11) & (day == 11)).astype("int8")


def _coerce_dtypes(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["holiday_name"] = out["holiday_name"].astype("string")
    for col, dtype in CALENDAR_DAILY_COLUMNS.items():
        if col in {"date", "holiday_name"}:
            continue
        out[col] = out[col].astype(dtype)
    return out


def load_calendar_from_local(
    parquet_path: Path | str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build a calendar daily frame from the raw 천문연 parquet.

    raw schema (long form): [date, name, is_holiday, kind, source]
      source ∈ {"holiday", "division"}. We currently only consume "holiday"
      rows; 24절기 ("division") rows are kept in the raw parquet for future
      feature work but not surfaced in CALENDAR_DAILY_COLUMNS yet.
    """
    raw = pd.read_parquet(parquet_path)
    holidays = raw[raw["source"] == "holiday"].copy()
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.normalize()
    return _build_calendar_from_holiday_rows(holidays, start=start, end=end)


def _build_calendar_from_holiday_rows(
    holidays: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    if len(dates) == 0:
        raise ValueError(f"empty date range: {start} → {end}")
    public_holiday_dates = set(holidays.loc[holidays["is_holiday"], "date"])
    name_lookup = (
        holidays.drop_duplicates(subset=["date"], keep="first")
        .set_index("date")["name"]
        .to_dict()
    )
    # KMA's 대체공휴일 row has kind == "02"; we also tolerate the name string.
    substitute_dates = set(
        holidays.loc[
            (holidays["kind"].astype(str) == "02")
            | holidays["name"].fillna("").str.contains("대체")
            | holidays["name"].fillna("").str.contains("Alternative", case=False),
            "date",
        ]
    )
    out = pd.DataFrame({"date": dates})
    out["holiday_name"] = out["date"].map(lambda d: name_lookup.get(d, "")).astype("string")
    out["is_public_holiday"] = out["date"].isin(public_holiday_dates).astype("int8")
    out["is_substitute_holiday"] = out["date"].isin(substitute_dates).astype("int8")
    out["is_weekend"] = (out["date"].dt.dayofweek >= 5).astype("int8")
    out["is_off_day"] = ((out["is_public_holiday"] == 1) | (out["is_weekend"] == 1)).astype("int8")
    _attach_streak(out)
    _attach_adjacency(out)
    _attach_event_flags(out)
    out = out[list(CALENDAR_DAILY_COLUMNS.keys())]
    return _coerce_dtypes(out)


def validate_calendar(df: pd.DataFrame) -> None:
    missing = set(CALENDAR_DAILY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"calendar frame missing columns: {sorted(missing)}")
    if df["date"].dt.normalize().ne(df["date"]).any():
        raise ValueError("calendar 'date' must be normalized to midnight")
    if df["date"].duplicated().any():
        raise ValueError("calendar has duplicate dates")

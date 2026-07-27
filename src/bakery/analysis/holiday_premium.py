"""공휴일 프리미엄 분해 — 요일·주말·연휴·대체공휴일 축.

lift = actual / 로컬 동일요일 baseline(±HALFWIN주, 공휴일 제외) — 추세·요일 동시 통제.
출처: scripts/holiday_premium_decompose.py(2026-07-18). 스크립트는 print만 담당하고
계산은 이 모듈로 옮겼다(순수함수 = 회귀 대조 가능).

주의: 캘린더는 `bakery.data.calendar.build_calendar_daily`로 만든 것을 주입해야 한다.
`calendar_raw` parquet 직독은 2021-23 공휴일이 누락돼 프리미엄이 −18.5% 과소평가된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALFWIN = 6                      # 동일요일 baseline ±주
MIN_BASELINE_SAMPLES = 3         # 이보다 적으면 baseline NaN
SERIES_VALUE_COLUMN = "adjusted_demand_unit"
WEEKEND_START_DOW = 5            # 토=5
STREAK_BUCKET_EDGES = [0, 1, 2, 10]
STREAK_BUCKET_LABELS = ["1(고립)", "2", "3+(연휴)"]
_WEEKDAY = "평일"
_WEEKEND = "주말"


def local_dow_baseline(series: pd.DataFrame, calendar: pd.DataFrame, *,
                       halfwin: int = HALFWIN) -> pd.Series:
    """각 날짜의 '평상 동일요일' median(±halfwin주, 공휴일 제외, 자기 자신 제외)."""
    merged = series.merge(calendar, on="date", how="left")
    normal = merged[merged["is_public_holiday"] == 0].set_index("date")[SERIES_VALUE_COLUMN]
    index = normal.index
    out: dict[pd.Timestamp, float] = {}
    for date in merged["date"]:
        low, high = date - pd.Timedelta(weeks=halfwin), date + pd.Timedelta(weeks=halfwin)
        same = normal[(index >= low) & (index <= high)
                      & (index.dayofweek == date.dayofweek) & (index != date)]
        out[date] = float(same.median()) if len(same) >= MIN_BASELINE_SAMPLES else np.nan
    return pd.Series(out, name="dow_base")


def normalize_holiday_name(name: object) -> str:
    """대체공휴일을 원 명절로 통합, 영문 표기 정리."""
    if not isinstance(name, str):
        return ""
    return name.replace("Alternative holiday for ", "").replace(" (observed)", "")


def _build_full(series: pd.DataFrame, calendar: pd.DataFrame, halfwin: int) -> pd.DataFrame:
    full = series.merge(calendar, on="date", how="left")
    full["dow_base"] = full["date"].map(local_dow_baseline(series, calendar, halfwin=halfwin))
    full["lift"] = full[SERIES_VALUE_COLUMN] / full["dow_base"]
    full["dow"] = full["date"].dt.dayofweek
    full["is_weekend"] = full["dow"] >= WEEKEND_START_DOW
    full["base_name"] = full["holiday_name"].map(normalize_holiday_name)
    return full


def _by_holiday(holidays: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "base_name", "dow", "is_weekend", "lift",
               "off_streak_length", "off_position_in_streak", "is_substitute_holiday"]
    return holidays[columns].sort_values(["base_name", "date"]).reset_index(drop=True)


def _dow_class(holidays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, subset in ((_WEEKDAY, holidays[~holidays["is_weekend"]]),
                          (_WEEKEND, holidays[holidays["is_weekend"]])):
        lifts = subset["lift"].dropna()
        rows.append({"dow_class": label, "n": int(len(lifts)),
                     "median_lift": float(lifts.median()),
                     "q25": float(lifts.quantile(0.25)),
                     "q75": float(lifts.quantile(0.75))})
    return pd.DataFrame(rows)


def _event_ranking(holidays: pd.DataFrame) -> pd.DataFrame:
    weekday = holidays[~holidays["is_weekend"]]
    rows = []
    for name, group in weekday.groupby("base_name"):
        lifts = group["lift"].dropna()
        if len(lifts) == 0:
            continue
        rows.append({"base_name": name, "median_lift": float(lifts.median()),
                     "n_weekday": int(len(lifts))})
    return (pd.DataFrame(rows).sort_values("median_lift", ascending=False)
            .reset_index(drop=True))


def _streak_buckets(holidays: pd.DataFrame) -> pd.DataFrame:
    weekday = holidays[~holidays["is_weekend"]].copy()
    weekday["streak_bucket"] = pd.cut(weekday["off_streak_length"],
                                      STREAK_BUCKET_EDGES, labels=STREAK_BUCKET_LABELS)
    rows = []
    for bucket in STREAK_BUCKET_LABELS:
        lifts = weekday[weekday["streak_bucket"] == bucket]["lift"].dropna()
        rows.append({"streak_bucket": bucket, "n": int(len(lifts)),
                     "median_lift": float(lifts.median()) if len(lifts) else np.nan})
    return pd.DataFrame(rows)


def decompose_holiday_premium(series: pd.DataFrame, calendar: pd.DataFrame, *,
                              halfwin: int = HALFWIN) -> dict[str, pd.DataFrame]:
    """공휴일 프리미엄 4축 분해. series는 (date, adjusted_demand_unit) 일별 시리즈."""
    full = _build_full(series, calendar, halfwin)
    holidays = full[full["is_public_holiday"] == 1].copy()
    return {
        "full": full,
        "by_holiday": _by_holiday(holidays),
        "dow_class": _dow_class(holidays),
        "event_ranking": _event_ranking(holidays),
        "streak_buckets": _streak_buckets(holidays),
    }

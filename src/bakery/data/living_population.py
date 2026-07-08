"""Seoul living-population (생활인구) data — admin-dong × date × hour.

The Seoul Open Data Plaza `LOCAL_PEOPLE_DONG` dataset estimates how many
people are physically present in each admin dong on a given day/hour from
KT mobile signal + satellite data. It's the closest public proxy to
"foot traffic in this neighborhood" and was chosen to replace the subway
ridership signal we dropped (CardSubwayTime turned out to be month-grain).

Schema is long-form keyed by `(admin_dong_code, date, hour)`. For PoC we
only need a slim total-population column; downstream features aggregate
it into static per-store baselines (daily avg / peak-hour share / weekend
ratio).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.store_mapping import StationMapping

LIVING_POP_COLUMNS: dict[str, str] = {
    "admin_dong_code": "string",
    "date": "datetime64[ns]",
    "hour": "int8",
    "total_pop": "float32",
}

# Per-dong shape parameters used by the synthetic generator. Calibrated
# loosely against published Seoul living-population magnitudes (tens of
# thousands at peak) so backtest values are realistic.
_SYNTH_PROFILE: dict[str, dict[str, float]] = {
    "11680565": {  # 청담동 — residential
        "base": 14_000, "peak_lunch_lift": 0.10, "peak_evening_lift": 0.20,
        "weekend_ratio": 1.05,
    },
    "11440660": {  # 서교동 — transit hub
        "base": 32_000, "peak_lunch_lift": 0.40, "peak_evening_lift": 0.55,
        "weekend_ratio": 1.30,
    },
    "11560540": {  # 여의동 — office
        "base": 28_000, "peak_lunch_lift": 0.65, "peak_evening_lift": 0.10,
        "weekend_ratio": 0.45,
    },
}
_DEFAULT_PROFILE = {
    "base": 18_000, "peak_lunch_lift": 0.20, "peak_evening_lift": 0.20, "weekend_ratio": 0.90,
}


def build_synthetic_living_population(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    mapping: dict[str, StationMapping],
    seed: int = 42,
) -> pd.DataFrame:
    """Hand-built living-population frame, one row per (dong, date, hour)."""
    dates = pd.date_range(start, end, freq="D")
    if len(dates) == 0:
        raise ValueError(f"empty date range: {start} → {end}")
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    dong_codes = sorted({s["admin_dong_code"] for s in mapping.values()})

    frames: list[pd.DataFrame] = []
    for dong in dong_codes:
        profile = _SYNTH_PROFILE.get(dong, _DEFAULT_PROFILE)
        for date in dates:
            hour_pop = _hourly_population(date, hours, profile, rng)
            frames.append(
                pd.DataFrame(
                    {
                        "admin_dong_code": dong,
                        "date": date,
                        "hour": hours.astype("int8"),
                        "total_pop": hour_pop.astype("float32"),
                    }
                )
            )
    return _coerce_dtypes(pd.concat(frames, ignore_index=True))


def _hourly_population(
    date: pd.Timestamp, hours: np.ndarray, profile: dict, rng: np.random.Generator
) -> np.ndarray:
    """Daily population curve — overnight low, lunch + evening peaks."""
    base = profile["base"]
    weekend = date.weekday() >= 5
    weekend_mult = profile["weekend_ratio"] if weekend else 1.0
    diurnal = 1.0 + 0.6 * np.sin((hours - 6) * np.pi / 18).clip(min=0)  # crude bell 6~24
    lunch_bump = profile["peak_lunch_lift"] * np.exp(-((hours - 12.5) ** 2) / 4)
    evening_bump = profile["peak_evening_lift"] * np.exp(-((hours - 18.5) ** 2) / 4)
    curve = diurnal + lunch_bump + evening_bump
    noise = rng.normal(1.0, 0.05, len(hours))
    return (base * weekend_mult * curve * noise).clip(min=base * 0.2)


def load_living_population_from_local(
    parquet_path: Path | str,
    *,
    admin_dong_codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Filter ingested raw parquet to our store dongs and date window."""
    raw = pd.read_parquet(parquet_path)
    raw["admin_dong_code"] = raw["admin_dong_code"].astype("string")
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    mask = (
        raw["admin_dong_code"].isin(admin_dong_codes)
        & (raw["date"] >= pd.Timestamp(start))
        & (raw["date"] <= pd.Timestamp(end))
    )
    return _coerce_dtypes(raw[mask].copy())


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["admin_dong_code"] = df["admin_dong_code"].astype("string")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["hour"] = df["hour"].astype("int8")
    df["total_pop"] = df["total_pop"].astype("float32")
    return df[list(LIVING_POP_COLUMNS.keys())]


def validate_living_population(df: pd.DataFrame) -> None:
    missing = set(LIVING_POP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"living population frame missing columns: {sorted(missing)}")
    dup = df.duplicated(subset=["admin_dong_code", "date", "hour"])
    if dup.any():
        raise ValueError(f"living population has {int(dup.sum())} duplicate (dong, date, hour) rows")

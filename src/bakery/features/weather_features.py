"""Merge weather daily into a sales daily frame and derive threshold flags.

PoC assumption: one weather observation per date, shared across all stores.
When a real per-station weather frame arrives, this function will need a
`station_id` join key — keep that change isolated here.
"""

from __future__ import annotations

import pandas as pd

# Raw numeric columns that come straight from the weather frame.
WEATHER_RAW_COLUMNS: list[str] = [
    "avg_temp",
    "max_temp",
    "min_temp",
    "diurnal_range",
    "humidity",
    "precipitation_mm",
    "is_rain",
    "snow_depth_cm",
    "is_snow",
    "sunshine_hours",
]

# Threshold-derived flags. Thresholds match Korean meteorological conventions.
HEATWAVE_TEMP_C = 28.0  # avg_temp above this — note KMA uses max_temp ≥ 33; we use avg as a softer signal
COLDSNAP_TEMP_C = -5.0
HEAVY_RAIN_MM = 10.0
HEAVY_SNOW_CM = 5.0

WEATHER_DERIVED_COLUMNS: list[str] = [
    "is_heavy_rain",
    "is_heavy_snow",
    "is_heatwave",
    "is_coldsnap",
]

WEATHER_FEATURE_COLUMNS: list[str] = WEATHER_RAW_COLUMNS + WEATHER_DERIVED_COLUMNS


def add_weather_features(
    df: pd.DataFrame, weather_df: pd.DataFrame, *, date_col: str = "date"
) -> pd.DataFrame:
    missing = set(WEATHER_RAW_COLUMNS) - set(weather_df.columns)
    if missing:
        raise ValueError(f"weather_df missing columns: {sorted(missing)}")
    out = df.merge(weather_df[[date_col, *WEATHER_RAW_COLUMNS]], on=date_col, how="left")
    for col in WEATHER_RAW_COLUMNS:
        out[col] = out[col].fillna(0)
    out["is_heavy_rain"] = (out["precipitation_mm"] >= HEAVY_RAIN_MM).astype("int8")
    out["is_heavy_snow"] = (out["snow_depth_cm"] >= HEAVY_SNOW_CM).astype("int8")
    out["is_heatwave"] = (out["avg_temp"] >= HEATWAVE_TEMP_C).astype("int8")
    out["is_coldsnap"] = (out["avg_temp"] <= COLDSNAP_TEMP_C).astype("int8")
    return out

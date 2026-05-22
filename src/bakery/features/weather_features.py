"""Merge weather daily into a sales daily frame.

Weather frames are long-form keyed by `(store_id, date)`. PoC review noted
that threshold flags (`is_heavy_rain` / `is_heavy_snow` / `is_heatwave` /
`is_coldsnap`) are fully redundant with the raw numeric columns — LightGBM
splits learn the threshold cutoffs natively — so they were dropped.
`is_rain` / `is_snow` are kept because the "any precipitation / any snow"
binary is a cleaner 0-vs-positive split than `precipitation_mm > 0`
across noisy near-zero values.
"""

from __future__ import annotations

import pandas as pd

WEATHER_FEATURE_COLUMNS: list[str] = [
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

# Back-compat alias for callers that still import the older raw-vs-derived split.
WEATHER_RAW_COLUMNS = WEATHER_FEATURE_COLUMNS


def add_weather_features(
    df: pd.DataFrame, weather_df: pd.DataFrame, *, date_col: str = "date"
) -> pd.DataFrame:
    missing = set(WEATHER_FEATURE_COLUMNS) - set(weather_df.columns)
    if missing:
        raise ValueError(f"weather_df missing columns: {sorted(missing)}")
    if "store_id" not in weather_df.columns:
        raise ValueError("weather_df missing 'store_id' — expected long-form (store_id, date) keys")
    if "store_id" not in df.columns:
        raise ValueError("df missing 'store_id' — cannot merge per-store weather")
    join_cols = ["store_id", date_col]
    out = df.merge(weather_df[[*join_cols, *WEATHER_FEATURE_COLUMNS]], on=join_cols, how="left")
    for col in WEATHER_FEATURE_COLUMNS:
        out[col] = out[col].fillna(0)
    return out

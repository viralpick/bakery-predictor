"""Weather frame shape, threshold flag correctness, merge leakage safety."""

from __future__ import annotations

import pandas as pd

from bakery.data.weather import (
    WEATHER_DAILY_COLUMNS,
    build_synthetic_weather,
    validate_weather,
)
from bakery.features.weather_features import (
    HEATWAVE_TEMP_C,
    HEAVY_RAIN_MM,
    HEAVY_SNOW_CM,
    WEATHER_FEATURE_COLUMNS,
    add_weather_features,
)


def test_weather_frame_validates_and_has_all_columns():
    df = build_synthetic_weather("2024-01-01", "2024-12-31", seed=42)
    validate_weather(df)
    for col in WEATHER_DAILY_COLUMNS:
        assert col in df.columns
    assert len(df) == 366  # 2024 leap year


def test_weather_seasonality_plausible():
    """Summer (Jun-Aug) should average warmer than winter (Dec-Feb)."""
    df = build_synthetic_weather("2024-01-01", "2024-12-31", seed=42)
    summer = df[(df["date"].dt.month >= 6) & (df["date"].dt.month <= 8)]["avg_temp"].mean()
    winter = df[df["date"].dt.month.isin([12, 1, 2])]["avg_temp"].mean()
    assert summer > 20
    assert winter < 5
    assert summer > winter + 15


def test_threshold_flags_match_raw_columns():
    weather = pd.DataFrame(
        {
            "date": pd.date_range("2024-06-01", periods=5, freq="D"),
            "avg_temp": [30.0, -7.0, 15.0, 15.0, 15.0],
            "max_temp": [35.0, -2.0, 20.0, 20.0, 20.0],
            "min_temp": [25.0, -12.0, 10.0, 10.0, 10.0],
            "diurnal_range": [10.0, 10.0, 10.0, 10.0, 10.0],
            "humidity": [70.0, 30.0, 50.0, 50.0, 50.0],
            "precipitation_mm": [0.0, 0.0, 15.0, 2.0, 0.0],
            "is_rain": [0, 0, 1, 1, 0],
            "snow_depth_cm": [0.0, 8.0, 0.0, 0.0, 0.0],
            "is_snow": [0, 1, 0, 0, 0],
            "sunshine_hours": [10.0, 3.0, 1.0, 4.0, 8.0],
        }
    )
    sales = pd.DataFrame(
        {
            "store_id": ["s1"] * 5,
            "item_id": ["i1"] * 5,
            "date": weather["date"],
            "sold_units": [10] * 5,
        }
    )
    merged = add_weather_features(sales, weather)
    for col in WEATHER_FEATURE_COLUMNS:
        assert col in merged.columns
    # Row 0: heatwave (avg_temp=30 >= HEATWAVE_TEMP_C)
    assert merged.iloc[0]["is_heatwave"] == 1
    assert merged.iloc[0]["is_coldsnap"] == 0
    # Row 1: coldsnap + heavy snow
    assert merged.iloc[1]["is_coldsnap"] == 1
    assert merged.iloc[1]["is_heavy_snow"] == 1
    # Row 2: heavy rain
    assert merged.iloc[2]["is_heavy_rain"] == 1
    # Row 3: light rain — not heavy
    assert merged.iloc[3]["is_heavy_rain"] == 0
    # Row 4: nothing flagged
    assert merged.iloc[4][["is_heatwave", "is_coldsnap", "is_heavy_rain", "is_heavy_snow"]].sum() == 0


def test_thresholds_are_documented_constants():
    # Hands-off pin so threshold drift gets caught in PR review.
    assert HEATWAVE_TEMP_C == 28.0
    assert HEAVY_RAIN_MM == 10.0
    assert HEAVY_SNOW_CM == 5.0


def test_weather_merge_no_future_leakage():
    """Per-row merge: mutating a future weather row cannot change past merged rows."""
    weather = build_synthetic_weather("2024-01-01", "2024-12-31", seed=42)
    sales = pd.DataFrame(
        {
            "store_id": ["s1"] * 30,
            "item_id": ["i1"] * 30,
            "date": pd.date_range("2024-02-01", periods=30, freq="D"),
            "sold_units": [10] * 30,
        }
    )
    merged_a = add_weather_features(sales, weather)
    pivot = pd.Timestamp("2024-02-15")
    mut = weather.copy()
    mut.loc[mut["date"] >= pivot, "avg_temp"] = 999.0
    mut.loc[mut["date"] >= pivot, "precipitation_mm"] = 999.0
    merged_b = add_weather_features(sales, mut)
    past_a = merged_a[merged_a["date"] < pivot][WEATHER_FEATURE_COLUMNS].reset_index(drop=True)
    past_b = merged_b[merged_b["date"] < pivot][WEATHER_FEATURE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_a, past_b)

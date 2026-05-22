"""Weather frame shape, threshold flag correctness, merge leakage safety."""

from __future__ import annotations

import pandas as pd

from bakery.data.weather import (
    WEATHER_DAILY_COLUMNS,
    build_synthetic_weather,
    validate_weather,
)
from bakery.features.weather_features import (
    WEATHER_FEATURE_COLUMNS,
    add_weather_features,
)


def test_weather_frame_validates_and_has_all_columns():
    df = build_synthetic_weather(
        "2024-01-01", "2024-12-31", store_ids=["s1", "s2", "s3"], seed=42
    )
    validate_weather(df)
    for col in WEATHER_DAILY_COLUMNS:
        assert col in df.columns
    # long form: 366 days (2024 leap) × 3 stores
    assert len(df) == 366 * 3
    assert set(df["store_id"].unique()) == {"s1", "s2", "s3"}
    # per-store count equal
    assert (df.groupby("store_id").size() == 366).all()


def test_weather_seasonality_plausible():
    """Summer (Jun-Aug) should average warmer than winter (Dec-Feb) per store."""
    df = build_synthetic_weather("2024-01-01", "2024-12-31", store_ids=["s1"], seed=42)
    summer = df[(df["date"].dt.month >= 6) & (df["date"].dt.month <= 8)]["avg_temp"].mean()
    winter = df[df["date"].dt.month.isin([12, 1, 2])]["avg_temp"].mean()
    assert summer > 20
    assert winter < 5
    assert summer > winter + 15


def test_per_store_weather_differs_but_correlates():
    """Stores share seasonal base but get small distinct noise."""
    df = build_synthetic_weather(
        "2024-01-01", "2024-03-31", store_ids=["s1", "s2"], seed=42
    )
    a = df[df["store_id"] == "s1"].sort_values("date")["avg_temp"].to_numpy()
    b = df[df["store_id"] == "s2"].sort_values("date")["avg_temp"].to_numpy()
    # Different (perturbations applied), but highly correlated (same base).
    assert (a != b).any()
    corr = pd.Series(a).corr(pd.Series(b))
    assert corr > 0.95


def test_weather_merge_preserves_raw_columns():
    weather = pd.DataFrame(
        {
            "store_id": ["s1"] * 5,
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
    # raw values survive the merge
    assert merged.iloc[0]["avg_temp"] == 30.0
    assert merged.iloc[2]["precipitation_mm"] == 15.0
    assert merged.iloc[1]["snow_depth_cm"] == 8.0


def test_weather_merge_no_future_leakage():
    """Per-row merge: mutating a future weather row cannot change past merged rows."""
    weather = build_synthetic_weather(
        "2024-01-01", "2024-12-31", store_ids=["s1"], seed=42
    )
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


def test_weather_merge_picks_per_store_rows():
    """Two stores with intentionally different weather → merged rows differ."""
    weather = pd.DataFrame(
        {
            "store_id": ["s1", "s1", "s2", "s2"],
            "date": pd.to_datetime(["2024-06-01", "2024-06-02"] * 2),
            "avg_temp": [30.0, 30.0, 10.0, 10.0],
            "max_temp": [35.0, 35.0, 15.0, 15.0],
            "min_temp": [25.0, 25.0, 5.0, 5.0],
            "diurnal_range": [10.0] * 4,
            "humidity": [70.0] * 4,
            "precipitation_mm": [0.0] * 4,
            "is_rain": [0] * 4,
            "snow_depth_cm": [0.0] * 4,
            "is_snow": [0] * 4,
            "sunshine_hours": [10.0] * 4,
        }
    )
    sales = pd.DataFrame(
        {
            "store_id": ["s1", "s1", "s2", "s2"],
            "item_id": ["i1"] * 4,
            "date": pd.to_datetime(["2024-06-01", "2024-06-02"] * 2),
            "sold_units": [10] * 4,
        }
    )
    merged = add_weather_features(sales, weather)
    s1_rows = merged[merged["store_id"] == "s1"]
    s2_rows = merged[merged["store_id"] == "s2"]
    assert (s1_rows["avg_temp"] == 30.0).all()
    assert (s2_rows["avg_temp"] == 10.0).all()

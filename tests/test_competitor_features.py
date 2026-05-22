"""Competitor radius + temporal-integrity tests.

Pins:
- A business open on date d is counted; one with license_date > d is not.
- A business closed before d (close_date ≤ d) is excluded; close_date strictly
  after d still counts as active.
- 90-day new/closed windows are strictly past — mutating future license/close
  events cannot move past-date counts.
- Radius cutoffs (500m / 1km) are honored exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.data.competitor import (
    COMPETITOR_RAW_COLUMNS,
    build_synthetic_competitor,
    validate_competitor,
)
from bakery.features.competitor_features import (
    COMPETITOR_FEATURE_COLUMNS,
    add_competitor_features,
    compute_competitor_features,
)
from bakery.ingest.store_mapping import DEFAULT_STATIONS


def _two_store_mapping() -> dict:
    """Compact mapping: only store_A + store_B for tighter assertions."""
    return {k: DEFAULT_STATIONS[k] for k in ("store_A", "store_B")}


def _businesses_around(store_lat: float, store_lon: float, count: int, *, distance_offsets_m: list[int]) -> list[dict]:
    """Place `count` businesses at exact distances east of the store along a parallel."""
    rows = []
    for i, d in enumerate(distance_offsets_m):
        # 1 degree longitude at this latitude ≈ 111_320 * cos(lat) meters
        meters_per_deg = 111_320 * np.cos(np.radians(store_lat))
        d_lon = d / meters_per_deg
        rows.append(
            {
                "business_id": f"b-{store_lat:.4f}-{i}",
                "category": "bakery",
                "license_date": pd.Timestamp("2024-01-01"),
                "close_date": pd.NaT,
                "lat": store_lat,
                "lon": store_lon + d_lon,
                "business_status": "active",
            }
        )
    return rows


def test_synthetic_competitor_validates_and_has_all_columns():
    df = build_synthetic_competitor(mapping=DEFAULT_STATIONS, seed=3)
    validate_competitor(df)
    for col in COMPETITOR_RAW_COLUMNS:
        assert col in df.columns
    # Sanity — at least ~150 businesses overall, mix of bakery + cafe
    assert len(df) > 150
    assert set(df["category"].unique()) == {"bakery", "cafe"}


def test_radius_cutoff_is_strict():
    """Distance-only test: 200m / 800m / 1200m businesses → only first two within 1km."""
    mapping = {k: DEFAULT_STATIONS[k] for k in ("store_A",)}
    entry = mapping["store_A"]
    rows = _businesses_around(entry["lat"], entry["lon"], 3, distance_offsets_m=[200, 800, 1200])
    competitor = pd.DataFrame(rows)
    dates = pd.date_range("2024-06-01", periods=1, freq="D")
    feats = compute_competitor_features(competitor, mapping, dates)
    row = feats.iloc[0]
    assert row["competitor_bakery_500m"] == 1   # only 200m
    assert row["competitor_bakery_1km"] == 2    # 200m + 800m
    # 1200m is over 1km, must be excluded


def test_temporal_filtering_active_vs_closed():
    """Future-licensed business should not count on past dates; closed must drop after close_date."""
    mapping = {k: DEFAULT_STATIONS[k] for k in ("store_A",)}
    entry = mapping["store_A"]
    rows = [
        # licensed 2024-01-01, never closed
        {"business_id": "b1", "category": "bakery",
         "license_date": pd.Timestamp("2024-01-01"), "close_date": pd.NaT,
         "lat": entry["lat"], "lon": entry["lon"] + 0.001,
         "business_status": "active"},
        # not yet licensed on 2024-03-01
        {"business_id": "b2", "category": "bakery",
         "license_date": pd.Timestamp("2024-06-01"), "close_date": pd.NaT,
         "lat": entry["lat"], "lon": entry["lon"] + 0.001,
         "business_status": "active"},
        # closed 2024-04-01 — gone from 2024-04-01 onward (strictly after)
        {"business_id": "b3", "category": "bakery",
         "license_date": pd.Timestamp("2024-01-01"), "close_date": pd.Timestamp("2024-04-01"),
         "lat": entry["lat"], "lon": entry["lon"] + 0.001,
         "business_status": "closed"},
    ]
    competitor = pd.DataFrame(rows)
    dates = pd.to_datetime(["2024-02-01", "2024-05-01", "2024-07-01"])
    feats = compute_competitor_features(competitor, mapping, dates)
    by_date = feats.set_index("date")["competitor_bakery_500m"]
    assert by_date.loc["2024-02-01"] == 2   # b1 + b3
    assert by_date.loc["2024-05-01"] == 1   # b1 only (b3 closed, b2 not yet)
    assert by_date.loc["2024-07-01"] == 2   # b1 + b2


def test_future_events_dont_leak_into_past_counts():
    """Mutate license/close events after a pivot date → counts on past dates unchanged."""
    mapping = _two_store_mapping()
    competitor = build_synthetic_competitor(mapping=mapping, seed=11)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    feats_a = compute_competitor_features(competitor, mapping, dates)

    pivot = pd.Timestamp("2024-06-15")
    mut = competitor.copy()
    # Move every "future" event later — must not affect past counts
    mut.loc[mut["license_date"] >= pivot, "license_date"] += pd.Timedelta(days=120)
    mut.loc[mut["close_date"].notna() & (mut["close_date"] >= pivot), "close_date"] += pd.Timedelta(days=120)
    feats_b = compute_competitor_features(mut, mapping, dates)

    past_a = feats_a[feats_a["date"] < pivot].sort_values(["store_id", "date"]).reset_index(drop=True)
    past_b = feats_b[feats_b["date"] < pivot].sort_values(["store_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(past_a[COMPETITOR_FEATURE_COLUMNS], past_b[COMPETITOR_FEATURE_COLUMNS])


def test_window_counts_90d_new_closed():
    """`new_bakery_90d_1km` = bakery licenses in [d-90, d) within 1km."""
    mapping = {k: DEFAULT_STATIONS[k] for k in ("store_A",)}
    entry = mapping["store_A"]
    rows = [
        # licensed 60 days before pivot — counted
        {"business_id": "b1", "category": "bakery",
         "license_date": pd.Timestamp("2024-04-02"), "close_date": pd.NaT,
         "lat": entry["lat"], "lon": entry["lon"] + 0.001,
         "business_status": "active"},
        # licensed 100 days before pivot — outside window
        {"business_id": "b2", "category": "bakery",
         "license_date": pd.Timestamp("2024-02-21"), "close_date": pd.NaT,
         "lat": entry["lat"], "lon": entry["lon"] + 0.001,
         "business_status": "active"},
        # closed 30 days before pivot — counted in closed window
        {"business_id": "b3", "category": "bakery",
         "license_date": pd.Timestamp("2023-01-01"), "close_date": pd.Timestamp("2024-05-02"),
         "lat": entry["lat"], "lon": entry["lon"] + 0.001,
         "business_status": "closed"},
    ]
    competitor = pd.DataFrame(rows)
    pivot = pd.Timestamp("2024-06-01")
    feats = compute_competitor_features(competitor, mapping, pd.DatetimeIndex([pivot]))
    row = feats.iloc[0]
    assert row["competitor_new_bakery_90d_1km"] == 1     # only b1 within last 90 days
    assert row["competitor_closed_bakery_90d_1km"] == 1  # only b3


def test_add_competitor_features_merge_per_store_date():
    mapping = _two_store_mapping()
    competitor = build_synthetic_competitor(mapping=mapping, seed=7)
    dates = pd.date_range("2024-06-01", periods=10, freq="D")
    feats = compute_competitor_features(competitor, mapping, dates)
    sales = pd.DataFrame(
        {
            "store_id": ["store_A", "store_B"] * 10,
            "item_id": ["i1"] * 20,
            "date": dates.repeat(2),
            "sold_units": [10] * 20,
        }
    )
    merged = add_competitor_features(sales, feats)
    for col in COMPETITOR_FEATURE_COLUMNS:
        assert col in merged.columns
    # Both stores get values; transit hub (B) has more competitors than residential (A)
    a_mean = merged[merged["store_id"] == "store_A"]["competitor_bakery_1km"].mean()
    b_mean = merged[merged["store_id"] == "store_B"]["competitor_bakery_1km"].mean()
    assert b_mean > a_mean

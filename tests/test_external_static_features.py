"""Schema + static-feature sanity for living-pop, population, consumption.

PoC-grade: validate schema, verify per-store features are deterministic
given the synthetic generators, and confirm store_id-only merges leave
every row constant within a store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.data.consumption import (
    CONSUMPTION_COLUMNS,
    build_synthetic_consumption,
    validate_consumption,
)
from bakery.data.living_population import (
    LIVING_POP_COLUMNS,
    build_synthetic_living_population,
    validate_living_population,
)
from bakery.data.population import (
    AGE_BINS,
    POPULATION_COLUMNS,
    build_synthetic_population,
    validate_population,
)
from bakery.features.consumption_features import (
    CONSUMPTION_FEATURE_COLUMNS,
    add_consumption_features,
    compute_store_consumption_features,
)
from bakery.features.living_population_features import (
    LIVING_POP_FEATURE_COLUMNS,
    add_living_pop_features,
    compute_store_living_features,
)
from bakery.features.population_features import (
    POPULATION_FEATURE_COLUMNS,
    add_population_features,
    compute_store_population_features,
)
from bakery.ingest.store_mapping import DEFAULT_STATIONS


# ---------------- living population ----------------

def test_living_population_schema_and_synthetic_density():
    df = build_synthetic_living_population(
        "2024-01-01", "2024-01-31", mapping=DEFAULT_STATIONS, seed=3
    )
    validate_living_population(df)
    for col in LIVING_POP_COLUMNS:
        assert col in df.columns
    # 31 days × 24 hours × N dongs (N = stores in DEFAULT_STATIONS)
    assert len(df) == 31 * 24 * len(DEFAULT_STATIONS)
    # Transit hub (서교동) should average highest
    by_dong = df.groupby("admin_dong_code")["total_pop"].mean()
    assert by_dong["11440660"] > by_dong["11680565"]


def test_living_pop_static_features_per_store():
    df = build_synthetic_living_population(
        "2024-01-01", "2024-03-31", mapping=DEFAULT_STATIONS, seed=7
    )
    static = compute_store_living_features(df, DEFAULT_STATIONS)
    assert set(static["store_id"]) == set(DEFAULT_STATIONS.keys())
    for col in LIVING_POP_FEATURE_COLUMNS:
        assert col in static.columns
        assert static[col].notna().all()
    # Office hub (여의동) should show < 1 weekend ratio; transit > 1
    scores = static.set_index("store_id")["living_pop_weekend_ratio"]
    assert scores["store_C"] < 1.0
    assert scores["store_B"] > 1.0


def test_living_pop_merge_constant_within_store():
    df = build_synthetic_living_population(
        "2024-01-01", "2024-02-29", mapping=DEFAULT_STATIONS, seed=5
    )
    static = compute_store_living_features(df, DEFAULT_STATIONS)
    sales = pd.DataFrame(
        {
            "store_id": list(DEFAULT_STATIONS) * 5,
            "date": pd.date_range("2024-02-01", periods=5, freq="D").repeat(len(DEFAULT_STATIONS)),
        }
    )
    merged = add_living_pop_features(sales, static)
    for store_id in DEFAULT_STATIONS:
        rows = merged[merged["store_id"] == store_id]
        for col in LIVING_POP_FEATURE_COLUMNS:
            assert rows[col].nunique() == 1, f"{store_id}/{col} not constant"


# ---------------- population (age/sex) ----------------

def test_population_schema_and_age_bins():
    df = build_synthetic_population(mapping=DEFAULT_STATIONS)
    validate_population(df)
    for col in POPULATION_COLUMNS:
        assert col in df.columns
    assert set(df["age_bin"].unique()) == set(AGE_BINS)
    assert set(df["sex"].unique()) == {"M", "F"}


def test_population_features_shapes_make_sense():
    df = build_synthetic_population(mapping=DEFAULT_STATIONS)
    static = compute_store_population_features(df, DEFAULT_STATIONS)
    for col in POPULATION_FEATURE_COLUMNS:
        assert col in static.columns
        assert (static[col] >= 0).all()
        assert (static[col] <= 1).all()
    # 청담동(store_A) older skew → 60+ share highest among stores
    shares = static.set_index("store_id")["pop_share_60_plus"]
    assert shares["store_A"] > shares["store_B"]
    # 서교동(store_B) young skew → 20-39 highest
    young = static.set_index("store_id")["pop_share_20_39"]
    assert young["store_B"] > young["store_A"]


def test_population_merge_constant_within_store():
    df = build_synthetic_population(mapping=DEFAULT_STATIONS)
    static = compute_store_population_features(df, DEFAULT_STATIONS)
    sales = pd.DataFrame(
        {
            "store_id": list(DEFAULT_STATIONS) * 3,
            "date": pd.date_range("2024-06-01", periods=3, freq="D").repeat(len(DEFAULT_STATIONS)),
        }
    )
    merged = add_population_features(sales, static)
    for store_id in DEFAULT_STATIONS:
        rows = merged[merged["store_id"] == store_id]
        for col in POPULATION_FEATURE_COLUMNS:
            assert rows[col].nunique() == 1


# ---------------- consumption ----------------

def test_consumption_schema():
    df = build_synthetic_consumption(mapping=DEFAULT_STATIONS)
    validate_consumption(df)
    for col in CONSUMPTION_COLUMNS:
        assert col in df.columns
    # food_retail_spend ≤ total_spend always
    assert (df["food_retail_spend"] <= df["total_spend"]).all()


def test_consumption_features_log_scaled_and_finite():
    df = build_synthetic_consumption(mapping=DEFAULT_STATIONS)
    static = compute_store_consumption_features(df, DEFAULT_STATIONS)
    for col in CONSUMPTION_FEATURE_COLUMNS:
        assert col in static.columns
        assert np.isfinite(static[col]).all()
    # Transit hub spend is largest in our synthetic profile
    totals = static.set_index("store_id")["consumption_total_log"]
    assert totals["store_B"] > totals["store_A"]


def test_consumption_merge_constant_within_store():
    df = build_synthetic_consumption(mapping=DEFAULT_STATIONS)
    static = compute_store_consumption_features(df, DEFAULT_STATIONS)
    sales = pd.DataFrame(
        {
            "store_id": list(DEFAULT_STATIONS) * 2,
            "date": pd.date_range("2024-09-01", periods=2, freq="D").repeat(len(DEFAULT_STATIONS)),
        }
    )
    merged = add_consumption_features(sales, static)
    for store_id in DEFAULT_STATIONS:
        rows = merged[merged["store_id"] == store_id]
        for col in CONSUMPTION_FEATURE_COLUMNS:
            assert rows[col].nunique() == 1

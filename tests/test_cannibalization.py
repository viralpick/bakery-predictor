"""Cannibalization features: leakage discipline + aggregate correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.features.cannibalization import (
    CANNIBALIZATION_FEATURE_COLUMNS,
    add_cannibalization_features,
)


def _toy(n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store in ("s1", "s2"):
        for item, cat in [("i1", "A"), ("i2", "A"), ("i3", "B"), ("i4", "B")]:
            for _i, d in enumerate(dates):
                rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "category_id": cat,
                        "date": d,
                        "sold_units": int(20 + rng.integers(0, 10)),
                        "is_stockout": bool(rng.random() < 0.3),
                    }
                )
    return pd.DataFrame(rows)


def test_columns_present():
    df = _toy()
    out = add_cannibalization_features(df)
    for col in CANNIBALIZATION_FEATURE_COLUMNS:
        assert col in out.columns


def test_no_future_leakage():
    df = _toy(n_days=60)
    feat_a = add_cannibalization_features(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    pivot = pd.Timestamp("2024-02-01")
    mut = df.copy()
    mut.loc[mut["date"] >= pivot, "is_stockout"] = True
    mut.loc[mut["date"] >= pivot, "sold_units"] = 9999
    feat_b = add_cannibalization_features(mut).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    past_a = feat_a[feat_a["date"] < pivot][CANNIBALIZATION_FEATURE_COLUMNS].reset_index(drop=True)
    past_b = feat_b[feat_b["date"] < pivot][CANNIBALIZATION_FEATURE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_a, past_b)


def test_store_aggregate_matches_yesterday_average():
    df = _toy(n_days=10)
    out = add_cannibalization_features(df)
    row = out[(out["store_id"] == "s1") & (out["date"] == pd.Timestamp("2024-01-05"))].iloc[0]
    yesterday = df[(df["store_id"] == "s1") & (df["date"] == pd.Timestamp("2024-01-04"))]
    assert row["store_stockout_rate_lag1"] == yesterday["is_stockout"].mean()
    assert row["store_total_sold_lag1"] == yesterday["sold_units"].sum()


def test_category_aggregate_isolated_per_store():
    """Same category, different stores should have independent aggregates."""
    df = _toy(n_days=10)
    out = add_cannibalization_features(df)
    s1 = out[(out["store_id"] == "s1") & (out["category_id"] == "A") & (out["date"] == pd.Timestamp("2024-01-05"))].iloc[0]
    s2 = out[(out["store_id"] == "s2") & (out["category_id"] == "A") & (out["date"] == pd.Timestamp("2024-01-05"))].iloc[0]
    # different stores → different aggregates (vanishingly unlikely to coincide on random seeded toy data)
    assert s1["cat_total_sold_lag1"] != s2["cat_total_sold_lag1"]

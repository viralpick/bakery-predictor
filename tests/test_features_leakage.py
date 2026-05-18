"""Pin the no-leakage property of lag/rolling features.

The check: take a daily frame, compute features. Mutate any future row's
sold_units to wildly different values, recompute features, and confirm that
past rows' feature values are byte-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.features.date_features import add_date_features
from bakery.features.lag_features import LAG_FEATURE_COLUMNS, add_lag_features
from bakery.features.rolling_features import ROLLING_FEATURE_COLUMNS, add_rolling_features


def _toy_daily(n_days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store in ("s1", "s2"):
        for item in ("i1", "i2"):
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "date": d,
                        "sold_units": 10 + (i % 7) + (5 if store == "s2" else 0),
                    }
                )
    return pd.DataFrame(rows)


def test_lag_features_do_not_leak_future():
    df = _toy_daily()
    feat_a = add_lag_features(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    pivot_date = pd.Timestamp("2024-03-01")
    mutated = df.copy()
    mutated.loc[mutated["date"] >= pivot_date, "sold_units"] = 99999
    feat_b = (
        add_lag_features(mutated).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    )
    past_a = feat_a[feat_a["date"] < pivot_date][LAG_FEATURE_COLUMNS]
    past_b = feat_b[feat_b["date"] < pivot_date][LAG_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(past_a, past_b)


def test_rolling_features_do_not_leak_future():
    df = _toy_daily()
    feat_a = (
        add_rolling_features(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    )
    pivot_date = pd.Timestamp("2024-03-01")
    mutated = df.copy()
    mutated.loc[mutated["date"] >= pivot_date, "sold_units"] = -1
    feat_b = (
        add_rolling_features(mutated)
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )
    past_a = feat_a[feat_a["date"] < pivot_date][ROLLING_FEATURE_COLUMNS]
    past_b = feat_b[feat_b["date"] < pivot_date][ROLLING_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(past_a, past_b)


def test_lag_does_not_cross_group_boundary():
    df = _toy_daily(n_days=10)
    out = add_lag_features(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    # First row per (store, item) must have NaN lag_1 (no prior row in its group).
    firsts = out.groupby(["store_id", "item_id"]).head(1)
    assert firsts["sales_lag_1"].isna().all()


def test_date_features_present_and_expected_dtypes():
    df = _toy_daily(n_days=14)
    out = add_date_features(df)
    for col in ("dow", "is_weekend", "month", "quarter"):
        assert col in out.columns
    assert out["dow"].between(0, 6).all()
    assert out["is_weekend"].isin([0, 1]).all()


def test_lag_7_equals_one_week_prior_value():
    df = _toy_daily(n_days=30)
    out = add_lag_features(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    row = out[(out["store_id"] == "s1") & (out["item_id"] == "i1") & (out["date"] == pd.Timestamp("2024-01-15"))].iloc[0]
    expected_prior = df[
        (df["store_id"] == "s1") & (df["item_id"] == "i1") & (df["date"] == pd.Timestamp("2024-01-08"))
    ]["sold_units"].iloc[0]
    assert row["sales_lag_7"] == expected_prior


def test_rolling_uses_only_past_values():
    df = _toy_daily(n_days=30)
    out = add_rolling_features(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    row = out[(out["store_id"] == "s1") & (out["item_id"] == "i1") & (out["date"] == pd.Timestamp("2024-01-15"))].iloc[0]
    history = df[
        (df["store_id"] == "s1")
        & (df["item_id"] == "i1")
        & (df["date"] < pd.Timestamp("2024-01-15"))
        & (df["date"] >= pd.Timestamp("2024-01-08"))
    ]["sold_units"]
    np.testing.assert_allclose(row["roll_mean_7"], history.mean())

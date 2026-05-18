"""v2 (unified demand model) regression tests.

Pin: v2 LightGBM trains on potential_demand by default, accepts/needs
cannibalization features, and supports quantile objective for the
recommended-production model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.data.calendar import build_calendar_daily
from bakery.data.weather import build_synthetic_weather
from bakery.features.calendar_features import add_calendar_features
from bakery.features.cannibalization import CANNIBALIZATION_FEATURE_COLUMNS
from bakery.features.potential_demand import StoreHours, attach_potential_demand
from bakery.features.weather_features import add_weather_features
from bakery.models.lightgbm_regressor import GlobalLGBM, LGBMParams


def _v2_daily(n_days: int = 180, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store in ("s1", "s2"):
        for item, cat in [("i1", "bread"), ("i2", "bread"), ("i3", "cake"), ("i4", "cake")]:
            for i, d in enumerate(dates):
                sold = int(40 + (i % 7) * 5 + rng.integers(0, 8))
                is_so = bool(rng.random() < 0.25)
                rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "category_id": cat,
                        "date": d,
                        "sold_units": sold,
                        "is_stockout": is_so,
                        "stockout_time": d + pd.Timedelta(hours=15) if is_so else pd.NaT,
                    }
                )
    df = pd.DataFrame(rows)
    cal = build_calendar_daily(dates.min(), dates.max())
    w = build_synthetic_weather(dates.min(), dates.max(), seed=seed)
    df = add_calendar_features(df, cal)
    df = add_weather_features(df, w)
    stores = [StoreHours("s1", 9, 22), StoreHours("s2", 8, 21)]
    df = attach_potential_demand(df, stores)
    return df


def test_v2_default_target_is_potential_demand():
    m = GlobalLGBM(feature_set="v2")
    assert m.y_col == "potential_demand"
    assert m.name == "lightgbm_v2"


def test_v2_includes_cannibalization_features():
    m = GlobalLGBM(feature_set="v2")
    for col in CANNIBALIZATION_FEATURE_COLUMNS:
        assert col in m.numeric_columns


def test_v2_fit_predict_roundtrip():
    df = _v2_daily(n_days=180)
    cutoff = pd.Timestamp("2024-05-01")
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    m = GlobalLGBM(feature_set="v2").fit(train)
    yhat = m.predict(val)
    assert len(yhat) == len(val)
    assert (yhat >= 0).all()
    assert np.isfinite(yhat.to_numpy()).all()


def test_v2_target_frame_can_skip_sold_units_and_is_stockout():
    """predict-next-week passes a horizon frame without observed sold_units."""
    df = _v2_daily(n_days=120)
    cutoff = pd.Timestamp("2024-04-01")
    train = df[df["date"] < cutoff].copy()
    horizon = (
        train[["store_id", "item_id", "category_id"]].drop_duplicates()
        .merge(pd.DataFrame({"date": pd.date_range(cutoff, periods=7, freq="D")}), how="cross")
    )
    # Carry calendar/weather to satisfy v2 check (predict-next-week does this).
    horizon_full = horizon.merge(df.drop_duplicates("date")[["date"] + [c for c in df.columns if c not in {"store_id","item_id","category_id","date","sold_units","is_stockout","stockout_time","potential_demand"}]], on="date", how="left")
    m = GlobalLGBM(feature_set="v2").fit(train)
    yhat = m.predict(horizon_full)
    assert len(yhat) == len(horizon_full)
    assert (yhat >= 0).all()


def test_v2_quantile_objective_runs_and_renames_model():
    df = _v2_daily(n_days=150)
    cutoff = pd.Timestamp("2024-04-15")
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    params = LGBMParams(objective="quantile", alpha=0.85)
    m = GlobalLGBM(feature_set="v2", params=params).fit(train)
    assert m.name == "lightgbm_v2_q85"
    yhat = m.predict(val)
    assert len(yhat) == len(val)


def test_v2_quantile_85_predicts_higher_than_quantile_50():
    """0.85 quantile should be ≥ 0.5 quantile on most rows (newsvendor safety)."""
    df = _v2_daily(n_days=200, seed=11)
    cutoff = pd.Timestamp("2024-05-01")
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    median = GlobalLGBM(feature_set="v2", params=LGBMParams(objective="quantile", alpha=0.5)).fit(train).predict(val)
    high = GlobalLGBM(feature_set="v2", params=LGBMParams(objective="quantile", alpha=0.85)).fit(train).predict(val)
    # Allow a small minority to invert (quantile fits aren't strictly nested), but
    # the bulk should respect ordering.
    assert (high.to_numpy() >= median.to_numpy()).mean() > 0.75


def test_v2_rejects_train_without_potential_demand_target():
    df = _v2_daily(n_days=120).drop(columns=["potential_demand"])
    with pytest.raises(KeyError):
        GlobalLGBM(feature_set="v2").fit(df)

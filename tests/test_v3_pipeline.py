"""v3 pipeline regression tests.

Pin:
- GlobalLGBM(feature_set="v3") refuses unenriched frames missing competitor columns.
- v3 trains on potential_demand.
- A horizon target frame with competitor features (and no observed sales)
  predicts cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.data.calendar import build_calendar_daily
from bakery.data.competitor import build_synthetic_competitor
from bakery.data.consumption import build_synthetic_consumption
from bakery.data.living_population import build_synthetic_living_population
from bakery.data.population import build_synthetic_population
from bakery.data.weather import build_synthetic_weather
from bakery.features.calendar_features import add_calendar_features
from bakery.features.competitor_features import (
    COMPETITOR_FEATURE_COLUMNS,
    add_competitor_features,
    compute_competitor_features,
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
from bakery.features.potential_demand import StoreHours, attach_potential_demand
from bakery.features.weather_features import add_weather_features
from bakery.ingest.store_mapping import DEFAULT_STATIONS
from bakery.models.lightgbm_regressor import GlobalLGBM


def _v3_daily(n_days: int = 180, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    store_ids = list(DEFAULT_STATIONS.keys())
    rows = []
    for store in store_ids:
        for item, cat in [("i1", "bread"), ("i2", "sandwich"), ("i3", "cake")]:
            for i, d in enumerate(dates):
                sold = int(40 + (i % 7) * 5 + rng.integers(0, 8))
                is_so = bool(rng.random() < 0.20)
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
    w = build_synthetic_weather(
        dates.min(), dates.max(), store_ids=store_ids, seed=seed
    )
    competitor = build_synthetic_competitor(mapping=DEFAULT_STATIONS, seed=seed)
    competitor_feats = compute_competitor_features(competitor, DEFAULT_STATIONS, dates)
    living_pop = build_synthetic_living_population(
        dates.min(), dates.max(), mapping=DEFAULT_STATIONS, seed=seed
    )
    living_static = compute_store_living_features(living_pop, DEFAULT_STATIONS)
    population = build_synthetic_population(mapping=DEFAULT_STATIONS)
    pop_static = compute_store_population_features(population, DEFAULT_STATIONS)
    consumption = build_synthetic_consumption(mapping=DEFAULT_STATIONS)
    cons_static = compute_store_consumption_features(consumption, DEFAULT_STATIONS)
    df = add_calendar_features(df, cal)
    df = add_weather_features(df, w)
    df = add_competitor_features(df, competitor_feats)
    df = add_living_pop_features(df, living_static)
    df = add_population_features(df, pop_static)
    df = add_consumption_features(df, cons_static)
    df = attach_potential_demand(
        df, [StoreHours(s, 9, 22) for s in store_ids]
    )
    return df


def test_v3_default_target_is_potential_demand():
    m = GlobalLGBM(feature_set="v3")
    assert m.y_col == "potential_demand"
    assert m.name == "lightgbm_v3"


def test_v3_includes_all_external_features():
    m = GlobalLGBM(feature_set="v3")
    for group in (
        COMPETITOR_FEATURE_COLUMNS,
        LIVING_POP_FEATURE_COLUMNS,
        POPULATION_FEATURE_COLUMNS,
        CONSUMPTION_FEATURE_COLUMNS,
    ):
        for col in group:
            assert col in m.numeric_columns


def test_v3_rejects_train_without_competitor_columns():
    df = _v3_daily(n_days=90).drop(columns=COMPETITOR_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="missing required columns"):
        GlobalLGBM(feature_set="v3").fit(df)


def test_v3_rejects_train_without_living_pop_columns():
    df = _v3_daily(n_days=90).drop(columns=LIVING_POP_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="missing required columns"):
        GlobalLGBM(feature_set="v3").fit(df)


def test_v3_rejects_train_without_population_columns():
    df = _v3_daily(n_days=90).drop(columns=POPULATION_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="missing required columns"):
        GlobalLGBM(feature_set="v3").fit(df)


def test_v3_rejects_train_without_consumption_columns():
    df = _v3_daily(n_days=90).drop(columns=CONSUMPTION_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="missing required columns"):
        GlobalLGBM(feature_set="v3").fit(df)


def test_v3_fit_predict_roundtrip():
    df = _v3_daily(n_days=180)
    cutoff = pd.Timestamp("2024-05-01")
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    m = GlobalLGBM(feature_set="v3").fit(train)
    yhat = m.predict(val)
    assert len(yhat) == len(val)
    assert (yhat >= 0).all()
    assert np.isfinite(yhat.to_numpy()).all()


def test_v3_horizon_target_without_observed_sold_units():
    """Horizon frame needs calendar/weather/competitor merged — no per-day sales lag."""
    df = _v3_daily(n_days=150)
    cutoff = pd.Timestamp("2024-05-01")
    train = df[df["date"] < cutoff].copy()
    horizon_dates = pd.date_range(cutoff, periods=7, freq="D")
    horizon = (
        train[["store_id", "item_id", "category_id"]].drop_duplicates()
        .merge(pd.DataFrame({"date": horizon_dates}), how="cross")
    )
    cal = build_calendar_daily(horizon_dates.min(), horizon_dates.max())
    w = build_synthetic_weather(
        horizon_dates.min(), horizon_dates.max(), store_ids=list(DEFAULT_STATIONS.keys()), seed=99
    )
    competitor = build_synthetic_competitor(mapping=DEFAULT_STATIONS, seed=99)
    competitor_feats = compute_competitor_features(competitor, DEFAULT_STATIONS, horizon_dates)
    living_pop = build_synthetic_living_population(
        horizon_dates.min(), horizon_dates.max(), mapping=DEFAULT_STATIONS, seed=99
    )
    living_static = compute_store_living_features(living_pop, DEFAULT_STATIONS)
    population = build_synthetic_population(mapping=DEFAULT_STATIONS)
    pop_static = compute_store_population_features(population, DEFAULT_STATIONS)
    consumption = build_synthetic_consumption(mapping=DEFAULT_STATIONS)
    cons_static = compute_store_consumption_features(consumption, DEFAULT_STATIONS)
    horizon = add_calendar_features(horizon, cal)
    horizon = add_weather_features(horizon, w)
    horizon = add_competitor_features(horizon, competitor_feats)
    horizon = add_living_pop_features(horizon, living_static)
    horizon = add_population_features(horizon, pop_static)
    horizon = add_consumption_features(horizon, cons_static)

    m = GlobalLGBM(feature_set="v3").fit(train)
    yhat = m.predict(horizon)
    assert len(yhat) == len(horizon)
    assert (yhat >= 0).all()

"""v1 wiring regression tests.

These pin two properties that the v1 pipeline must hold:

1. Enriching daily with calendar + weather does not introduce leakage
   into lag/rolling features (lag/rolling depend only on sold_units, but
   we test the property explicitly).
2. `GlobalLGBM(feature_set="v1")` refuses to fit on a non-enriched frame
   and round-trips fit→predict on an enriched one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.data.calendar import build_calendar_daily
from bakery.data.synthetic import generate_synthetic_bundle
from bakery.data.weather import build_synthetic_weather
from bakery.features.calendar_features import (
    CALENDAR_FEATURE_COLUMNS,
    add_calendar_features,
)
from bakery.features.lag_features import LAG_FEATURE_COLUMNS, add_lag_features
from bakery.features.rolling_features import ROLLING_FEATURE_COLUMNS, add_rolling_features
from bakery.features.weather_features import WEATHER_FEATURE_COLUMNS, add_weather_features
from bakery.models.lightgbm_regressor import GlobalLGBM


def _enriched_toy(n_days: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store in ("s1", "s2"):
        for item, cat in [("i1", "bread"), ("i2", "cake")]:
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "category_id": cat,
                        "date": d,
                        "sold_units": int(20 + (i % 7) * 3 + rng.integers(0, 5)),
                    }
                )
    df = pd.DataFrame(rows)
    cal = build_calendar_daily(dates.min(), dates.max())
    weather = build_synthetic_weather(dates.min(), dates.max(), seed=seed)
    df = add_calendar_features(df, cal)
    df = add_weather_features(df, weather)
    return df


def test_enriched_daily_preserves_lag_no_leakage():
    """Enrichment shouldn't change lag/rolling computation."""
    df = _enriched_toy()
    feat_a = (
        add_rolling_features(add_lag_features(df))
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )
    # mutate future weather/calendar columns; lag/rolling are sold_units-based
    # so past lag values must not move.
    mut = df.copy()
    pivot = pd.Timestamp("2024-02-15")
    for col in (*CALENDAR_FEATURE_COLUMNS, *WEATHER_FEATURE_COLUMNS):
        mut.loc[mut["date"] >= pivot, col] = 0
    feat_b = (
        add_rolling_features(add_lag_features(mut))
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )
    past_cols = [*LAG_FEATURE_COLUMNS, *ROLLING_FEATURE_COLUMNS]
    past_a = feat_a[feat_a["date"] < pivot][past_cols].reset_index(drop=True)
    past_b = feat_b[feat_b["date"] < pivot][past_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_a, past_b)


def test_lightgbm_v1_rejects_unenriched_frame():
    df = _enriched_toy()
    bare = df.drop(columns=[*CALENDAR_FEATURE_COLUMNS, *WEATHER_FEATURE_COLUMNS])
    model = GlobalLGBM(feature_set="v1")
    with pytest.raises(ValueError, match="missing required columns"):
        model.fit(bare)


def test_lightgbm_v1_fit_predict_roundtrip():
    df = _enriched_toy(n_days=120)
    cutoff = pd.Timestamp("2024-04-01")
    train = df[df["date"] < cutoff].copy()
    target = df[df["date"] >= cutoff].copy()
    model = GlobalLGBM(feature_set="v1")
    model.fit(train)
    yhat = model.predict(target)
    assert len(yhat) == len(target)
    assert (yhat >= 0).all()
    assert np.isfinite(yhat.to_numpy()).all()
    # feature_importance includes v1 columns
    imp = model.feature_importance()
    v1_only = set(CALENDAR_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS)
    assert v1_only.issubset(set(imp["feature"]))


def test_lightgbm_v0_still_works_on_enriched_frame():
    """Adding columns shouldn't break the v0 model — it just ignores them."""
    df = _enriched_toy(n_days=120)
    cutoff = pd.Timestamp("2024-04-01")
    train = df[df["date"] < cutoff].copy()
    target = df[df["date"] >= cutoff].copy()
    model = GlobalLGBM(feature_set="v0")
    model.fit(train)
    yhat = model.predict(target)
    assert len(yhat) == len(target)
    assert (yhat >= 0).all()


def test_synthetic_bundle_dgp_signals_align():
    """The bundle's daily reflects calendar/weather effects baked into the DGP.

    Crude sanity check: Christmas-period cake demand should exceed mid-November.
    """
    bundle = generate_synthetic_bundle(start="2024-01-01", end="2024-12-31", seed=0)
    daily = bundle.daily
    cake = daily[daily["category_id"] == "cake"]
    xmas = cake[(cake["date"] >= "2024-12-23") & (cake["date"] <= "2024-12-25")]["sold_units"].mean()
    midnov = cake[(cake["date"] >= "2024-11-13") & (cake["date"] <= "2024-11-15")]["sold_units"].mean()
    assert xmas > midnov * 1.5, f"xmas={xmas:.1f} not materially > midnov={midnov:.1f}"

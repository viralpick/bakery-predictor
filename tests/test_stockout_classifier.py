"""StockoutClassifier + classifier metrics regression tests.

Pins: fit/predict roundtrip, AUC/precision/recall sanity, leakage discipline
of stockout-history features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.data.calendar import build_calendar_daily
from bakery.data.weather import build_synthetic_weather
from bakery.evaluation.classifier_metrics import (
    base_rate,
    precision_at_k,
    recall_at_k,
    roc_auc,
)
from bakery.features.calendar_features import add_calendar_features
from bakery.features.stockout_history import (
    STOCKOUT_HISTORY_COLUMNS,
    add_stockout_history,
)
from bakery.features.weather_features import add_weather_features
from bakery.models.stockout_classifier import StockoutClassifier


def _enriched(n_days: int = 150, seed: int = 11) -> pd.DataFrame:
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
                        "sold_units": int(30 + (i % 7) * 4 + rng.integers(0, 5)),
                        # stockouts cluster on weekends to make signal learnable
                        "is_stockout": bool(d.dayofweek >= 5 and rng.random() < 0.55),
                    }
                )
    df = pd.DataFrame(rows)
    cal = build_calendar_daily(dates.min(), dates.max())
    weather = build_synthetic_weather(dates.min(), dates.max(), seed=seed)
    df = add_calendar_features(df, cal)
    df = add_weather_features(df, weather)
    return df


def test_stockout_history_no_future_leakage():
    df = _enriched(n_days=80)
    feat_a = add_stockout_history(df).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    pivot = pd.Timestamp("2024-02-15")
    mut = df.copy()
    mut.loc[mut["date"] >= pivot, "is_stockout"] = True
    feat_b = add_stockout_history(mut).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    past_a = feat_a[feat_a["date"] < pivot][STOCKOUT_HISTORY_COLUMNS]
    past_b = feat_b[feat_b["date"] < pivot][STOCKOUT_HISTORY_COLUMNS]
    pd.testing.assert_frame_equal(past_a, past_b)


def test_classifier_fit_predict_roundtrip():
    df = _enriched(n_days=180)
    cutoff = pd.Timestamp("2024-05-01")
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    clf = StockoutClassifier(feature_set="v1")
    clf.fit(train)
    proba = clf.predict_proba(val)
    assert len(proba) == len(val)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_classifier_beats_random_on_weekend_signal():
    """We engineered weekend-clustered stockouts — AUC should be materially > 0.5."""
    df = _enriched(n_days=180, seed=7)
    cutoff = pd.Timestamp("2024-05-01")
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    clf = StockoutClassifier(feature_set="v1").fit(train)
    proba = clf.predict_proba(val)
    auc = roc_auc(val["is_stockout"].astype(int).to_numpy(), proba.to_numpy())
    assert auc > 0.6, f"expected AUC > 0.6 on engineered weekend stockouts, got {auc:.3f}"


def test_classifier_rejects_frame_without_is_stockout():
    df = _enriched(n_days=60).drop(columns=["is_stockout"])
    with pytest.raises(ValueError, match="is_stockout"):
        StockoutClassifier(feature_set="v1").fit(df)


def test_roc_auc_basic_cases():
    perfect = roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect == 1.0
    inverse = roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert inverse == 0.0
    flat = roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert flat == 0.5


def test_precision_recall_at_k():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6])
    assert precision_at_k(y, s, 4) == 1.0  # top-4 scores all correspond to positives
    assert recall_at_k(y, s, 4) == 1.0  # all 4 positives captured


def test_base_rate_matches_mean():
    y = np.array([0, 1, 1, 1, 0])
    assert base_rate(y) == pytest.approx(0.6)

import numpy as np
import pandas as pd
import pytest

from bakery.models.category_total import (
    expanding_calibration_folds,
    fit_category_total,
)


def _toy_cat_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    trend = np.arange(n) / n
    noise = rng.normal(0, 5, n)
    base = 100 + 20 * np.sin(2 * np.pi * dow / 7) + 30 * trend
    return pd.DataFrame(
        {
            "date": dates,
            "dow": dow,
            "trend": trend,
            "feat": rng.normal(0, 1, n),
            "adjusted_demand_unit": base + noise,
        }
    )


def test_fit_with_q_lo_trains_lower_model():
    df = _toy_cat_df()
    m = fit_category_total(df, q_lo=0.10)
    assert m.production_lo is not None
    lo = m.predict_production_lo(df)
    hi = m.predict_production(df)
    # q0.10 lower model must sit below the q0.90 anchor on average
    assert lo.mean() < hi.mean()


def test_fit_without_q_lo_has_no_lower_model():
    df = _toy_cat_df()
    m = fit_category_total(df)
    assert m.production_lo is None


def test_predict_production_lo_without_model_raises():
    df = _toy_cat_df()
    m = fit_category_total(df)
    with pytest.raises(ValueError):
        m.predict_production_lo(df)


def test_calibration_folds_are_time_ordered():
    df = _toy_cat_df(n=600)
    folds = expanding_calibration_folds(
        df, n_folds=3, min_train_days=300, calibration_days=60, horizon_days=30
    )
    assert len(folds) == 3
    for f in folds:
        assert f.train["date"].max() < f.calibration["date"].min()
        assert f.calibration["date"].max() < f.test["date"].min()


def test_calibration_folds_have_disjoint_dates():
    df = _toy_cat_df(n=600)
    folds = expanding_calibration_folds(
        df, n_folds=2, min_train_days=300, calibration_days=60, horizon_days=30
    )
    for f in folds:
        tr = set(f.train["date"])
        ca = set(f.calibration["date"])
        te = set(f.test["date"])
        assert tr.isdisjoint(ca)
        assert ca.isdisjoint(te)
        assert tr.isdisjoint(te)


def test_calibration_window_sizes_match_params():
    df = _toy_cat_df(n=600)
    folds = expanding_calibration_folds(
        df, n_folds=2, min_train_days=300, calibration_days=60, horizon_days=30
    )
    for f in folds:
        assert f.calibration["date"].nunique() == 60
        assert f.test["date"].nunique() == 30


def test_calibration_folds_insufficient_history_raises():
    df = _toy_cat_df(n=200)
    with pytest.raises(ValueError):
        expanding_calibration_folds(
            df, n_folds=2, min_train_days=300, calibration_days=60, horizon_days=30
        )

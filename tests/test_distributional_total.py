import numpy as np
import pandas as pd
import pytest

from bakery.models.distributional_total import (
    DistributionalTotalModel,
    fit_distributional_total,
)

TARGET = "adjusted_demand_unit"


def _synth(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """양수 target + feature 2개 + LEAK_COL 1개(sold_total_unit)."""
    rng = np.random.RandomState(seed)
    x1, x2 = rng.rand(n), rng.rand(n)
    y = np.exp(5.0 + 0.5 * x1 + 0.3 * rng.randn(n))  # 양수, lognormal-ish
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "f1": x1, "f2": x2,
        "sold_total_unit": y * 1.1,   # LEAK_COL — feature에서 제외돼야
        TARGET: y,
    })


def _fit(df: pd.DataFrame | None = None) -> DistributionalTotalModel:
    df = _synth() if df is None else df
    return fit_distributional_total(df, target_col=TARGET, n_estimators=50)


def test_predict_quantile_shape():
    m, df = _fit(), _synth()
    assert len(m.predict_quantile(df, 0.85)) == len(df)


def test_quantile_monotonic():
    m, df = _fit(), _synth()
    q50 = m.predict_quantile(df, 0.5)
    q85 = m.predict_quantile(df, 0.85)
    q95 = m.predict_quantile(df, 0.95)
    assert np.all(q50 <= q85)
    assert np.all(q85 <= q95)


def test_median_matches_dist():
    m, df = _fit(), _synth()
    assert np.allclose(m.predict_median(df), np.ravel(m.predict_dist(df).ppf(0.5)))


def test_predictions_positive():
    m, df = _fit(), _synth()
    assert np.all(m.predict_quantile(df, 0.85) > 0)


def test_sigma_shape_and_positive():
    m, df = _fit(), _synth()
    sigma = m.predict_sigma(df)
    assert len(sigma) == len(df)
    assert np.all(sigma > 0)


def test_feature_cols_exclude_leak_and_target():
    m = _fit()
    assert TARGET not in m.feature_cols
    assert "sold_total_unit" not in m.feature_cols
    assert "date" not in m.feature_cols
    assert "f1" in m.feature_cols
    assert "f2" in m.feature_cols


def test_predict_without_target_column():
    m = _fit()
    df_no_target = _synth().drop(columns=[TARGET])
    assert len(m.predict_quantile(df_no_target, 0.85)) == len(df_no_target)


def test_nonpositive_target_raises():
    df = _synth()
    df.loc[0, TARGET] = 0.0
    with pytest.raises(ValueError):
        fit_distributional_total(df, target_col=TARGET, n_estimators=50)


def test_deterministic():
    df = _synth()
    m1 = fit_distributional_total(df, target_col=TARGET, n_estimators=50)
    m2 = fit_distributional_total(df, target_col=TARGET, n_estimators=50)
    assert np.allclose(m1.predict_quantile(df, 0.85), m2.predict_quantile(df, 0.85))


def test_alias_expected_is_median():
    m, df = _fit(), _synth()
    assert np.array_equal(m.predict_expected(df), m.predict_median(df))


def test_alias_production_is_quantile():
    m, df = _fit(), _synth()
    assert np.array_equal(m.predict_production(df, 0.85), m.predict_quantile(df, 0.85))
    # 기본 production_q=0.85 확인
    assert np.array_equal(m.predict_production(df), m.predict_quantile(df, 0.85))

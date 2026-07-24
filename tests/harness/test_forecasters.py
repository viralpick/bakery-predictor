import numpy as np
import pytest

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.forecasters import CategoryTotalForecaster, DistributionalTotalForecaster
from bakery.models.category_total import fit_category_total

TARGET = "adjusted_demand_unit"


def _feat():
    cd = build_category_daily(alpha=0.8)
    return build_features(cd, target_col=TARGET).dropna().reset_index(drop=True)


def test_category_adapter_matches_direct_fit():
    feat = _feat()
    train, test = feat.iloc[:400], feat.iloc[400:407]
    adapted = CategoryTotalForecaster().fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    direct = fit_category_total(train, target_col=TARGET, alpha_demand=0.8, production_q=0.85)
    np.testing.assert_array_equal(adapted.predict_expected(test), direct.predict_expected(test))
    np.testing.assert_array_equal(adapted.predict_production(test), direct.predict_production(test))


def test_distributional_adapter_deterministic():
    feat = _feat()
    train, test = feat.iloc[:400], feat.iloc[400:407]
    fc = DistributionalTotalForecaster()
    m1 = fc.fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    m2 = fc.fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    np.testing.assert_array_equal(m1.predict_expected(test), m2.predict_expected(test))
    np.testing.assert_array_equal(m1.predict_production(test), m2.predict_production(test))


def test_distributional_adapter_no_rng_leak():
    """hermetic seed: fit이 전역 numpy RNG 스트림을 소비/변경하지 않아야."""
    feat = _feat()
    train = feat.iloc[:400]
    np.random.seed(7)
    a = np.random.rand()
    np.random.seed(7)
    DistributionalTotalForecaster().fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    b = np.random.rand()
    assert a == b


def test_distributional_production_ge_expected():
    feat = _feat()
    train, test = feat.iloc[:400], feat.iloc[400:407]
    m = DistributionalTotalForecaster().fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    assert bool((m.predict_production(test) >= m.predict_expected(test)).all())

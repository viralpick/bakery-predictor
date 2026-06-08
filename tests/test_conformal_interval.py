import numpy as np
import pytest

from bakery.models.conformal_interval import ConformalInterval


def test_symmetric_interval_is_centered_on_anchor():
    rng = np.random.default_rng(0)
    n = 2000
    actual = rng.normal(0, 10, n)
    center = np.zeros(n)
    dow = rng.integers(0, 7, n)
    ci = ConformalInterval(mode="symmetric", coverage=0.80).calibrate(
        actual=actual, center_pred=center, dow=dow
    )
    anchor = 100.0
    lower, upper = ci.predict_interval(center_pred=np.array([anchor]), dow=np.array([2]))
    # symmetric: equal distance from the anchor on both sides
    assert (anchor - lower[0]) == pytest.approx(upper[0] - anchor)


def test_symmetric_coverage_matches_nominal():
    rng = np.random.default_rng(1)
    n = 6000
    actual = rng.normal(0, 10, n)
    center = np.zeros(n)
    dow = rng.integers(0, 7, n)
    ci = ConformalInterval(mode="symmetric", coverage=0.80).calibrate(
        actual=actual, center_pred=center, dow=dow
    )
    # fresh test sample from the same distribution
    actual_t = rng.normal(0, 10, n)
    center_t = np.zeros(n)
    dow_t = rng.integers(0, 7, n)
    lower, upper = ci.predict_interval(center_pred=center_t, dow=dow_t)
    cov = np.mean((actual_t >= lower) & (actual_t <= upper))
    assert abs(cov - 0.80) < 0.03


def test_coverage_monotonic_wider_for_higher_level():
    rng = np.random.default_rng(2)
    n = 4000
    actual = rng.normal(0, 10, n)
    center = np.zeros(n)
    dow = rng.integers(0, 7, n)
    ci80 = ConformalInterval(mode="symmetric", coverage=0.80).calibrate(
        actual=actual, center_pred=center, dow=dow
    )
    ci95 = ConformalInterval(mode="symmetric", coverage=0.95).calibrate(
        actual=actual, center_pred=center, dow=dow
    )
    c = np.zeros(n)
    d = rng.integers(0, 7, n)
    lo80, up80 = ci80.predict_interval(center_pred=c, dow=d)
    lo95, up95 = ci95.predict_interval(center_pred=c, dow=d)
    assert np.mean(up95 - lo95) >= np.mean(up80 - lo80)


def test_asymmetric_margins_independent():
    # Upper residual r_hi = actual-center spans [0,10]; lower residual
    # r_lo = lo_pred-actual spans [0,20]. Their quantiles must differ,
    # proving the two sides are computed independently.
    actual = np.full(100, 50.0)
    r_hi = np.linspace(0, 10, 100)
    r_lo = np.linspace(0, 20, 100)
    center = actual - r_hi
    lo_pred = actual + r_lo
    dow = np.zeros(100, dtype=int)
    ci = ConformalInterval(mode="asymmetric", coverage=0.80).calibrate(
        actual=actual, center_pred=center, dow=dow, lo_pred=lo_pred
    )
    m_lo, m_hi = ci.margins_by_dow[0]
    assert m_hi == pytest.approx(8.0, abs=0.3)
    assert m_lo == pytest.approx(16.0, abs=0.3)
    assert m_lo != m_hi


def test_asymmetric_lower_uses_lo_pred_base():
    rng = np.random.default_rng(3)
    n = 1000
    actual = rng.normal(50, 5, n)
    center = np.full(n, 70.0)
    lo_pred = np.full(n, 30.0)
    dow = np.zeros(n, dtype=int)
    ci = ConformalInterval(mode="asymmetric", coverage=0.80).calibrate(
        actual=actual, center_pred=center, dow=dow, lo_pred=lo_pred
    )
    lower, upper = ci.predict_interval(
        center_pred=np.array([70.0]), dow=np.array([0]), lo_pred=np.array([30.0])
    )
    m_lo, m_hi = ci.margins_by_dow[0]
    # lower bound anchored on lo_pred, upper bound anchored on center
    assert lower[0] == pytest.approx(30.0 - m_lo)
    assert upper[0] == pytest.approx(70.0 + m_hi)


def test_asymmetric_requires_lo_pred():
    ci = ConformalInterval(mode="asymmetric", coverage=0.80)
    with pytest.raises(ValueError):
        ci.calibrate(
            actual=np.array([1.0]), center_pred=np.array([1.0]), dow=np.array([0])
        )


def test_undersized_dow_uses_pooled_margin():
    rng = np.random.default_rng(4)
    # 6 well-populated dows + 1 sparse dow (5 samples < min_group_n=30)
    big = np.repeat(np.arange(6), 100)
    small = np.full(5, 6)
    dow = np.concatenate([big, small])
    actual = rng.normal(0, 10, len(dow))
    center = np.zeros(len(dow))
    ci = ConformalInterval(mode="symmetric", coverage=0.80, min_group_n=30).calibrate(
        actual=actual, center_pred=center, dow=dow
    )
    assert ci.margins_by_dow[6] == ci.pooled_margin


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        ConformalInterval(mode="bogus", coverage=0.80).calibrate(
            actual=np.array([1.0]), center_pred=np.array([1.0]), dow=np.array([0])
        )

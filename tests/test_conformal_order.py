import numpy as np
import pytest

from bakery.models.conformal_order import ConformalOrderCalibrator, DEFAULT_SERVICE_LEVEL


def test_default_service_level_is_cost_optimal():
    assert DEFAULT_SERVICE_LEVEL == 0.74


def test_q_s_is_higher_quantile_of_scores():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    calib = ConformalOrderCalibrator().fit(scores, service_level=0.8)
    # method="higher" 0.8-quantile of 5 sorted values → index ceil(0.8*4)=4 → 4.0... use np ref
    assert calib.q_s == pytest.approx(np.quantile(scores, 0.8, method="higher"))


def test_apply_adds_margin_scaled_and_clips_negative():
    calib = ConformalOrderCalibrator().fit(np.array([-5.0, -5.0]), service_level=0.5)  # q_s=-5
    base = np.array([10.0, 3.0])
    scales = np.array([2.0, 1.0])
    # order = base + q_s*scale = [10-10, 3-5] = [0, -2] → clip → [0, 0]
    got = calib.apply(base, scales)
    assert got.tolist() == [0.0, 0.0]


def test_apply_positive_margin_exact():
    calib = ConformalOrderCalibrator().fit(np.array([1.0, 1.0]), service_level=0.5)  # q_s=1.0
    got = calib.apply(np.array([10.0]), np.array([3.0]))
    assert got.tolist() == [13.0]  # 10 + 1.0*3


def test_apply_before_fit_raises_runtime_error():
    calib = ConformalOrderCalibrator()  # fresh, unfitted — q_s never set
    with pytest.raises(RuntimeError, match="before fit"):
        calib.apply(np.array([10.0]), np.array([1.0]))


def test_coverage_contract_on_exchangeable_synthetic():
    # 이유: conformal coverage는 통계적 보장이라 정확값 불가 → 허용오차 단언.
    rng = np.random.default_rng(42)
    n = 4000
    scale = np.full(n, 5.0)
    resid = rng.normal(0, 5.0, size=n)      # y - base, exchangeable
    scores = resid / scale
    half = n // 2
    calib = ConformalOrderCalibrator().fit(scores[:half], service_level=0.8)
    base = np.zeros(half)
    order = calib.apply(base, scale[half:])
    y = resid[half:]                         # base=0 → y == resid
    exceed = float((y > order).mean())
    assert abs(exceed - 0.2) < 0.03          # nominal miss = 1 - 0.8

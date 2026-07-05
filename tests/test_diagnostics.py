import numpy as np
import pytest
from bakery.evaluation.diagnostics import decoupling_score

def test_uncorrected_sales_negative_score():
    # 미복원(원판매): 품절률 높을수록 관측수요 낮음 → 강한 음의 상관
    stockout_rate = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    observed = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    assert decoupling_score(observed, stockout_rate) == pytest.approx(-1.0, abs=1e-9)

def test_perfect_recovery_zero_score():
    # 완전복원: 품절률과 무관하게 실수요 100 → 상관 정의 안 됨(분산0) → 0 반환
    stockout_rate = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    recovered = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    assert decoupling_score(recovered, stockout_rate) == 0.0

def test_weighted_matches_manual():
    demand = np.array([10.0, 20.0, 30.0])
    sr = np.array([0.1, 0.2, 0.3])
    w = np.array([1.0, 1.0, 2.0])
    # 양의 완전 선형 → +1
    assert decoupling_score(demand, sr, weights=w) == pytest.approx(1.0, abs=1e-9)

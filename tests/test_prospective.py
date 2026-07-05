import numpy as np
import pytest
from bakery.features.potential_demand import bakery_hour_profile
from bakery.evaluation.prospective import simulate_soldout

OPEN, CLOSE = 8, 22


def _uniform_profile():
    # 균등 도착: alpha=0 → 개점~폐점 균등
    return bakery_hour_profile(OPEN, CLOSE, alpha=0.0)


def test_order_ge_demand_never_sells_out():
    prof = _uniform_profile()
    t, is_so = simulate_soldout(100.0, 80.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert t is None
    assert is_so is False


def test_uniform_half_demand_sells_out_midday():
    # 균등 14시간(08~22) 영업, 수요 100, 발주 50 → 정확히 중간 시각 15:00
    prof = _uniform_profile()
    t, is_so = simulate_soldout(50.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert is_so is True
    assert t == pytest.approx(15.0, abs=1e-6)   # 08 + 14*0.5


def test_zero_order_sells_out_at_open():
    prof = _uniform_profile()
    t, is_so = simulate_soldout(0.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert is_so is True
    assert t == pytest.approx(float(OPEN), abs=1e-6)


def test_monotone_higher_order_later_soldout():
    prof = _uniform_profile()
    t_low, _ = simulate_soldout(30.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    t_high, _ = simulate_soldout(70.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert t_high > t_low

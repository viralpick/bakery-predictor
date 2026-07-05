import numpy as np
import pandas as pd
import pytest
from bakery.features.potential_demand import bakery_hour_profile
from bakery.evaluation.prospective import simulate_soldout, build_arrival_profile

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


def test_build_arrival_profile_sums_by_hour():
    receipts = pd.DataFrame({
        "item_id": ["a", "a", "a", "b"],
        "hour":    [9,   9,   14,  10],
        "qty":     [2,   3,   5,   1],
    })
    prof = build_arrival_profile(receipts, group_cols=["item_id"])
    assert prof[("a",)][9] == 5.0
    assert prof[("a",)][14] == 5.0
    assert prof[("a",)].sum() == 10.0
    assert prof[("b",)][10] == 1.0
    assert prof[("a",)].shape == (24,)


def test_build_arrival_profile_excludes_keys():
    receipts = pd.DataFrame({
        "item_id": ["a", "a"],
        "date":    ["2025-01-01", "2025-01-02"],
        "hour":    [9, 9],
        "qty":     [2, 7],
    })
    prof = build_arrival_profile(
        receipts, group_cols=["item_id"], exclude_keys={("a", "2025-01-02")},
        exclude_cols=["item_id", "date"],
    )
    assert prof[("a",)][9] == 2.0   # 01-02 제외


def test_reconstruct_baseline_order_identity():
    from bakery.evaluation.prospective import reconstruct_baseline_order
    df = pd.DataFrame({
        "normal_units":  [10.0, 5.0],
        "closing_units": [3.0,  0.0],
        "waste_units":   [2.0,  1.0],
    })
    got = reconstruct_baseline_order(df)
    assert list(got) == [15.0, 6.0]


def test_reconstruct_baseline_order_nan_as_zero():
    from bakery.evaluation.prospective import reconstruct_baseline_order
    df = pd.DataFrame({
        "normal_units":  [10.0, np.nan],
        "closing_units": [np.nan, 4.0],
        "waste_units":   [2.0, 1.0],
    })
    got = reconstruct_baseline_order(df)
    assert list(got) == [12.0, 5.0]

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


def test_build_arrival_profile_string_key_contract_with_int_group_col():
    # group_cols에 정수 dtype 컬럼(store_id)이 와도 build_arrival_profile의
    # dict key가 str-cast되어, simulate_item_day_kpis 쪽
    # tuple(str(r[c]) for c in group_cols) 조회와 어긋나지 않아야 한다.
    # 어긋나면 lookup이 조용히 miss → 기본 arrival curve로 fallback되어
    # 에러 없이 잘못된 결과를 낸다 (silent failure).
    receipts = pd.DataFrame({
        "store_id": [1, 1, 2],
        "hour":     [9, 14, 10],
        "qty":      [3, 7, 4],
    })
    prof = build_arrival_profile(receipts, group_cols=["store_id"])
    # int로 조회하면 실패해야 정상(키가 str로 저장됨을 증명)
    assert (1,) not in prof
    lookup_key = tuple(str(v) for v in (1,))
    assert lookup_key in prof
    assert prof[lookup_key][9] == 3.0
    assert prof[lookup_key][14] == 7.0
    assert prof[lookup_key].sum() == 10.0
    lookup_key_2 = tuple(str(v) for v in (2,))
    assert prof[lookup_key_2][10] == 4.0


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


from bakery.features.potential_demand import StoreHours, bakery_hour_profile
from bakery.evaluation.business_metrics import CostParams
from bakery.evaluation.prospective import simulate_item_day_kpis


def test_item_day_kpis_waste_and_soldout():
    rows = pd.DataFrame({
        "item_id": ["a", "a"],
        "date":    ["2025-01-01", "2025-01-02"],
        "potential_demand": [100.0, 100.0],
        "order_qty": [120.0, 50.0],   # 1일차 과발주(미매진, 폐기), 2일차 부족(매진)
    })
    prof = {("a",): bakery_hour_profile(8, 22, alpha=0.0)}
    out = simulate_item_day_kpis(
        rows, prof, order_col="order_qty",
        store_hours=StoreHours("gwangyo", 8, 22),
        group_cols=["item_id"],
        params=CostParams(), unit_prices={"a": 1000.0},
    )
    r0, r1 = out.iloc[0], out.iloc[1]
    assert r0["is_stockout"] == False
    assert r0["waste_units"] == 20.0            # 120-100
    assert pd.isna(r0["soldout_hour"])
    assert r1["is_stockout"] == True
    assert r1["soldout_hour"] == pytest.approx(15.0, abs=1e-6)  # 발주50/수요100 균등
    assert r1["lost_sale_units"] == 50.0        # 100-50


def _kpi_frame(waste, lost, stockouts, soldout):
    return pd.DataFrame({
        "waste_cost_krw": waste, "lost_margin_krw": lost,
        "is_stockout": stockouts, "soldout_hour": soldout,
    })


def test_compare_policies_delta():
    from bakery.evaluation.prospective import compare_policies
    our = _kpi_frame([100.0, 0.0], [0.0, 50.0], [False, True], [np.nan, 16.0])
    base = _kpi_frame([200.0, 0.0], [0.0, 80.0], [False, True], [np.nan, 14.0])
    out = compare_policies(our, base).set_index("policy")
    assert out.loc["our", "waste_cost_krw"] == 100.0
    assert out.loc["baseline", "waste_cost_krw"] == 200.0
    assert out.loc["delta", "waste_cost_krw"] == -100.0     # 우리가 폐기 100 적음
    assert out.loc["our", "stockout_rate"] == 0.5
    assert out.loc["our", "soldout_median_h"] == 16.0       # 매진일만 median
    assert out.loc["delta", "soldout_median_h"] == 2.0      # 16 - 14


def test_simulate_uses_named_demand_col():
    rows = pd.DataFrame({
        "item_id": ["A"],
        "date": pd.to_datetime(["2021-01-01"]),
        "adjusted_demand": [10.0],
        "potential_demand": [999.0],  # 잘못된 잣대 — 선택되면 결과가 달라짐
        "our_order": [10.0],
    })
    sh = StoreHours("store_gw01", 8, 22)
    out = simulate_item_day_kpis(
        rows, profiles={}, order_col="our_order", store_hours=sh,
        group_cols=["item_id"], demand_col="adjusted_demand",
    )
    # order == adjusted_demand == 10 → 폐기 0, lost 0
    assert float(out["waste_units"].iloc[0]) == 0.0
    assert float(out["lost_sale_units"].iloc[0]) == 0.0

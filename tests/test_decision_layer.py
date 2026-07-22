"""Unit tests for the v6 decision layer (policy / risk / lineage / pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.decision import (
    PolicyParams,
    RiskParams,
    apply_policy,
    build_recommendation,
    lineage_to_frame,
    simulate_item_risk,
)


# --- policy + lineage -------------------------------------------------------

def test_policy_adds_safety_margin_and_rounds_up():
    order, lineage = apply_policy("A", 20.0, PolicyParams(safety_margin=0.15, round_unit=1))
    # 20 * 1.15 = 23.0 → round = 23
    assert order == 23.0
    names = [s.name for s in lineage.steps]
    assert names == ["safety_margin", "rounding"]


def test_lineage_conservation_holds():
    """base + Σ contributions must equal the final order qty (보존 법칙)."""
    for point in (0.0, 3.7, 20.0, 137.4):
        order, lineage = apply_policy("X", point, PolicyParams(display_floor=5, round_unit=4))
        assert lineage.is_conserved(order)
        assert lineage.order_qty == pytest.approx(order)


def test_display_floor_applies_only_when_below():
    _, low = apply_policy("A", 1.0, PolicyParams(safety_margin=0.0, display_floor=5, round_unit=1))
    assert any(s.name == "display_floor" for s in low.steps)
    _, high = apply_policy("B", 50.0, PolicyParams(safety_margin=0.0, display_floor=5, round_unit=1))
    assert not any(s.name == "display_floor" for s in high.steps)


def test_round_unit_batches_up():
    order, _ = apply_policy("A", 21.0, PolicyParams(safety_margin=0.0, round_unit=6))
    assert order == 24.0  # ceil(21/6)*6


# --- risk -------------------------------------------------------------------

def test_risk_is_reproducible_with_seed():
    p = RiskParams(n_samples=2000, seed=7)
    a = simulate_item_risk(20.0, 23.0, p)
    b = simulate_item_risk(20.0, 23.0, p)
    assert a == b


def test_higher_order_lowers_stockout_raises_waste():
    p = RiskParams(n_samples=20000, demand_cv=0.3, seed=1)
    low = simulate_item_risk(20.0, 18.0, p)
    high = simulate_item_risk(20.0, 28.0, p)
    assert high.p_stockout < low.p_stockout
    assert high.p_waste > low.p_waste


def test_far_excess_order_drives_stockout_to_zero():
    p = RiskParams(n_samples=20000, demand_cv=0.2, seed=3)
    res = simulate_item_risk(20.0, 100.0, p)
    assert res.p_stockout == pytest.approx(0.0, abs=1e-3)
    assert res.expected_short == pytest.approx(0.0, abs=1e-3)


def test_expected_cost_combines_margin_and_waste():
    p = RiskParams(n_samples=20000, demand_cv=0.3, unit_margin=3.0, unit_cost=1.0, seed=5)
    res = simulate_item_risk(20.0, 20.0, p)
    assert res.expected_cost == pytest.approx(3.0 * res.expected_short + 1.0 * res.expected_waste, rel=1e-9)


def test_lognormal_median_gives_half_stockout_at_order_equals_point():
    # LogNormal: P(X > median) = 0.5 exactly. order=point=median → p_stockout≈0.5.
    p = RiskParams(n_samples=60000, seed=4)
    res = simulate_item_risk(20.0, 20.0, p, demand_sigma_log=0.4)
    assert abs(res.p_stockout - 0.5) < 0.02


def test_higher_sigma_raises_stockout_when_order_above_point():
    # order(25) > point(20): 분산 클수록 상단 tail이 order를 더 자주 넘음.
    p = RiskParams(n_samples=60000, seed=8)
    lo = simulate_item_risk(20.0, 25.0, p, demand_sigma_log=0.15)
    hi = simulate_item_risk(20.0, 25.0, p, demand_sigma_log=0.55)
    assert hi.p_stockout > lo.p_stockout


def test_sigma_none_matches_normal_default_exactly():
    # backward-compat: None 전달 == 미전달 (기존 Normal cv 경로, 동일 seed).
    p = RiskParams(n_samples=5000, seed=7)
    a = simulate_item_risk(20.0, 23.0, p)
    b = simulate_item_risk(20.0, 23.0, p, demand_sigma_log=None)
    assert a == b


def test_nonpositive_point_with_sigma_falls_back_to_normal():
    # demand_point<=0 → LogNormal 불가 → Normal fallback(예외 없이 all-waste).
    res = simulate_item_risk(0.0, 5.0, RiskParams(n_samples=5000, seed=10), demand_sigma_log=0.3)
    assert res.p_stockout == 0.0
    assert res.p_waste == 1.0


# --- pipeline ---------------------------------------------------------------

def _items():
    return pd.DataFrame({
        "store_id": ["gwangyo", "gwangyo"],
        "category_id": ["bread", "pastry"],
        "item_id": ["통팥빵", "치즈롤"],
        "demand_point": [20.0, 8.0],
    })


def test_pipeline_outputs_v6_record_per_item():
    rec = build_recommendation(_items(), PolicyParams(), RiskParams(n_samples=3000, seed=11))
    assert len(rec.table) == 2
    for col in ("demand_point", "order_qty", "p_stockout", "p_waste", "expected_cost"):
        assert col in rec.table.columns
    # carried-through id columns preserved
    assert set(rec.table["item_id"]) == {"통팥빵", "치즈롤"}
    assert "store_id" in rec.table.columns


def test_pipeline_lineage_aligns_and_flattens():
    rec = build_recommendation(_items(), PolicyParams(), RiskParams(n_samples=1000, seed=1))
    assert len(rec.lineages) == len(rec.table)
    for order, lin in zip(rec.table["order_qty"], rec.lineages):
        assert lin.is_conserved(order)
    flat = lineage_to_frame(rec.lineages)
    assert set(flat["step"]) >= {"base", "safety_margin", "rounding"}


def test_pipeline_rejects_missing_columns():
    with pytest.raises(ValueError, match="demand_point"):
        build_recommendation(pd.DataFrame({"item_id": ["a"]}))


def test_pipeline_seed_makes_table_deterministic():
    risk = RiskParams(n_samples=2000, seed=9)
    a = build_recommendation(_items(), PolicyParams(), risk)
    b = build_recommendation(_items(), PolicyParams(), risk)
    np.testing.assert_array_equal(a.table["p_stockout"].to_numpy(), b.table["p_stockout"].to_numpy())


def test_zero_demand_with_positive_order_is_all_waste():
    res = simulate_item_risk(0.0, 5.0, RiskParams(n_samples=5000, seed=10))
    assert res.p_stockout == pytest.approx(0.0, abs=1e-3)
    assert res.p_waste == pytest.approx(1.0, abs=1e-3)
    assert res.expected_waste == pytest.approx(5.0, abs=1e-2)


def test_pipeline_rejects_negative_demand():
    df = pd.DataFrame({"item_id": ["a"], "demand_point": [-5.0]})
    with pytest.raises(ValueError, match="demand_point must be"):
        build_recommendation(df)


def test_pipeline_rejects_nan_demand():
    df = pd.DataFrame({"item_id": ["a"], "demand_point": [float("nan")]})
    with pytest.raises(ValueError, match="NaN"):
        build_recommendation(df)


def test_pipeline_threads_demand_sigma_log():
    # 정책상 order>demand_point(안전마진) → σ 클수록 p_stockout 커야 함.
    base = _items()
    risk = RiskParams(n_samples=30000, seed=3)
    r_lo = build_recommendation(base.assign(demand_sigma_log=[0.1, 0.1]), PolicyParams(), risk)
    r_hi = build_recommendation(base.assign(demand_sigma_log=[0.6, 0.6]), PolicyParams(), risk)
    assert (r_hi.table["p_stockout"].to_numpy() > r_lo.table["p_stockout"].to_numpy()).all()


def test_pipeline_nan_sigma_falls_back_to_cv():
    # demand_sigma_log 컬럼이 NaN이면 None 취급 → 기존 cv 경로와 동일 결과.
    base = _items()
    risk = RiskParams(n_samples=4000, seed=9)
    with_nan = base.assign(demand_sigma_log=[float("nan"), float("nan")])
    a = build_recommendation(base, PolicyParams(), risk)
    b = build_recommendation(with_nan, PolicyParams(), risk)
    assert a.table["p_stockout"].tolist() == b.table["p_stockout"].tolist()

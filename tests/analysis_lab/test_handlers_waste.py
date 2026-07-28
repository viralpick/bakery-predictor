import pandas as pd
import pytest

from bakery.analysis.lab.handlers.waste import (
    MIN_PRODUCTION_FOR_ITEM_RATE,
    identity_residual,
    overproduction_breakdown,
    overproduction_by_category,
    waste_alpha_identity,
    waste_rate,
    waste_rate_by_item,
    waste_rate_by_store,
)
from bakery.analysis.lab.result import KIND_DATA


def _waste():
    """광교 2품목 2일. b1: 생산 100 폐기 10, p1: 생산 50 폐기 20."""
    return pd.DataFrame({
        "cd": ["1000000047"] * 4,
        "store": ["광교"] * 4,
        "item_id": ["b1", "b1", "p1", "p1"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
        "production_qty": [60, 40, 25, 25],
        "waste_qty": [6, 4, 10, 10],
        "normal_qty": [50.0, 34.0, 15.0, 13.0],
        "closing_qty": [4.0, 2.0, 0.0, 2.0],
        "unit_price": [3000, 3000, 5000, 5000],
        "waste_cost": [18000, 12000, 50000, 50000],
        "identity_diff": [0.0, 0.0, 0.0, 0.0],
        "sold_total": [54.0, 36.0, 15.0, 15.0],
    })


def _item_to_category():
    return pd.Series({"b1": "bread", "p1": "pastry"})


def test_min_production_constant():
    assert MIN_PRODUCTION_FOR_ITEM_RATE == 30


def test_waste_rate_by_store_exact():
    row = waste_rate_by_store(_waste()).iloc[0]
    assert row["production_qty"] == 150          # 60+40+25+25
    assert row["waste_qty"] == 30                # 6+4+10+10
    assert row["waste_rate"] == pytest.approx(0.2)
    assert row["waste_cost"] == 130000
    assert row["n_carry_in"] == 0                 # 이 fixture엔 음수 폐기 없음
    assert row["carry_in_units"] == pytest.approx(0.0)


def test_waste_rate_by_store_surfaces_carry_in_negatives():
    """음수 waste_qty(전일 재고 이월)는 clip하지 않고 순합 + 건수/합계로 노출한다."""
    frame = _waste().copy()
    frame.loc[0, "waste_qty"] = -4               # 판매가 당일 생산 초과 → carry-in
    row = waste_rate_by_store(frame).iloc[0]
    assert row["waste_qty"] == 20                 # -4+4+10+10 (clip 안 함)
    assert row["n_carry_in"] == 1
    assert row["carry_in_units"] == pytest.approx(-4.0)
    assert row["waste_rate"] == pytest.approx(20 / 150)


def test_waste_rate_by_item_exact_and_filters_low_production():
    frame = waste_rate_by_item(_waste(), _item_to_category()).set_index("item_id")
    assert frame.loc["b1", "waste_rate"] == pytest.approx(10 / 100)
    assert frame.loc["p1", "waste_rate"] == pytest.approx(20 / 50)
    assert frame.loc["b1", "category_id"] == "bread"
    # 생산 30 미만 품목은 비율이 불안정해 제외
    small = _waste().assign(production_qty=[5, 5, 25, 25])
    assert waste_rate_by_item(small, _item_to_category())["item_id"].tolist() == ["p1"]


def test_identity_residual_reports_zero_when_consistent():
    row = identity_residual(_waste()).iloc[0]
    assert row["n_rows"] == 4
    assert row["n_nonzero"] == 0
    assert row["max_abs_diff"] == pytest.approx(0.0)
    assert row["zero_frac"] == pytest.approx(1.0)


def test_identity_residual_detects_mismatch():
    broken = _waste().copy()
    broken.loc[0, "waste_qty"] = 10             # 60 − 54 − 10 = −4 잔차
    row = identity_residual(broken).iloc[0]
    assert row["n_nonzero"] == 1
    assert row["max_abs_diff"] == pytest.approx(4.0)
    assert row["zero_frac"] == pytest.approx(0.75)


def test_overproduction_by_category_cost_share_sums_to_one():
    frame = overproduction_by_category(_waste(), _item_to_category())
    assert frame["cost_share"].sum() == pytest.approx(1.0)
    bread = frame[frame["category_id"] == "bread"].iloc[0]
    assert bread["waste_cost"] == 30000
    assert bread["cost_share"] == pytest.approx(30000 / 130000)


def test_three_handlers_are_data_kind_without_verdict(stub_inputs):
    inputs = stub_inputs(waste=_waste(), item_to_category=_item_to_category())
    for handler, tables in ((waste_rate, ["by_store", "by_item"]),
                            (waste_alpha_identity, ["residual"]),
                            (overproduction_breakdown, ["by_category"])):
        result = handler(inputs)
        assert result.kind == KIND_DATA
        assert result.verdict is None
        assert [label for label, _ in result.tables] == tables

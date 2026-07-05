import pandas as pd
from bakery.features.potential_demand import StoreHours
from bakery.evaluation.prospective import (
    build_arrival_profile, simulate_item_day_kpis, compare_policies,
)
from bakery.evaluation.business_metrics import CostParams
from bakery.cli import _assemble_real_rows


def test_end_to_end_our_beats_worse_baseline():
    # 합성: item a, 2일. 복원수요 100/100. 우리 발주=수요근접, baseline=과발주.
    receipts = pd.DataFrame({
        "item_id": ["a"]*4, "date": ["2025-01-01"]*2 + ["2025-01-02"]*2,
        "hour": [9, 14, 9, 14], "qty": [50, 50, 50, 50],
    })
    prof = build_arrival_profile(receipts, group_cols=["item_id"])
    rows = pd.DataFrame({
        "item_id": ["a", "a"], "date": ["2025-01-01", "2025-01-02"],
        "potential_demand": [100.0, 100.0],
        "our_order": [105.0, 105.0], "base_order": [140.0, 140.0],
    })
    sh = StoreHours("gwangyo", 8, 22)
    our = simulate_item_day_kpis(rows, prof, order_col="our_order",
                                 store_hours=sh, group_cols=["item_id"],
                                 params=CostParams(), unit_prices={"a": 1000.0})
    base = simulate_item_day_kpis(rows, prof, order_col="base_order",
                                  store_hours=sh, group_cols=["item_id"],
                                  params=CostParams(), unit_prices={"a": 1000.0})
    cmp = compare_policies(our, base).set_index("policy")
    # 우리 발주가 baseline보다 과발주 덜 함 → 폐기비용 Δ < 0
    assert cmp.loc["delta", "waste_cost_krw"] < 0.0


def _real_shaped_daily() -> pd.DataFrame:
    """bonavi_daily.parquet 서브셋 형태 (store/category 필터 이후 상태)."""
    return pd.DataFrame({
        "item_id": ["101", "101", "102", "999"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-01"]),
        "category_id": ["bread", "bread", "pastry", "bread"],
        "sold_units": [10, 12, 5, 3],
        "is_stockout": [False, True, False, False],
        "potential_demand": [12.0, 15.0, 5.0, 3.0],
    })


def _real_shaped_inventory() -> pd.DataFrame:
    """load_inventory 반환 형태 — date는 YYYYMMDD 문자열(daily의 datetime과 다른 표현)."""
    return pd.DataFrame({
        "date": ["20210101", "20210102", "20210101"],
        "item_id": ["101", "101", "102"],
        "production_qty": [15, 18, 6],
        "waste_qty": [3, 2, 1],
    })


def test_assemble_real_rows_base_order_matches_production_qty():
    result = _assemble_real_rows(_real_shaped_daily(), _real_shaped_inventory())

    assert list(result.columns) == [
        "item_id", "date", "category_id", "potential_demand",
        "sold_units", "is_stockout", "base_order", "waste_qty",
    ]

    row = result[
        (result["item_id"] == "101") & (result["date"] == pd.Timestamp("2021-01-02"))
    ].iloc[0]
    assert row["base_order"] == 18
    assert row["waste_qty"] == 2
    assert row["potential_demand"] == 15.0
    assert row["category_id"] == "bread"
    assert bool(row["is_stockout"]) is True

    row2 = result[result["item_id"] == "102"].iloc[0]
    assert row2["base_order"] == 6
    assert row2["waste_qty"] == 1
    assert row2["category_id"] == "pastry"


def test_assemble_real_rows_drops_item_days_without_inventory_match():
    result = _assemble_real_rows(_real_shaped_daily(), _real_shaped_inventory())

    # item "999" (재고정보 매칭 없음) 은 base_order 미정이라 평가셋에서 제외된다
    assert set(result["item_id"]) == {"101", "102"}
    assert len(result) == 3

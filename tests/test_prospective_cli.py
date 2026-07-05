import numpy as np
import pandas as pd
import pytest
from bakery.features.potential_demand import StoreHours, bakery_hour_profile
from bakery.evaluation.prospective import (
    build_arrival_profile, simulate_item_day_kpis, compare_policies,
)
from bakery.evaluation.business_metrics import CostParams


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

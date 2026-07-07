import pandas as pd
from bakery.evaluation.prospective import characterize_baseline_proxy


def test_characterize_baseline_proxy_shares():
    rows = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "date": ["2024-01-01"] * 4,
        "is_stockout": [True, False, True, False],  # 2/4 = 0.5
    })
    waste_report = {"policy": "clip", "n_negative": 1, "n_total": 8, "min_value": -3.0}
    result = characterize_baseline_proxy(rows, waste_report)
    assert result["n_item_days"] == 4
    assert result["stockout_share"] == 0.5
    assert result["negative_waste_share"] == 0.125  # 1/8
    assert isinstance(result["carryover_note"], str)

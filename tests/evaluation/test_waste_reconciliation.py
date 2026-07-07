import numpy as np
import pandas as pd
from bakery.evaluation.prospective import compare_actual_vs_simulated_waste


def test_actual_vs_simulated_waste_totals_and_ratio():
    rows = pd.DataFrame({
        "item_id": ["a", "b"], "date": ["2024-01-01", "2024-01-01"],
        "waste_qty": [4.0, 6.0],  # actual total = 10
    })
    base_kpis = pd.DataFrame({
        "item_id": ["a", "b"], "date": ["2024-01-01", "2024-01-01"],
        "waste_units": [3.0, 2.0],  # simulated total = 5
    })
    result = compare_actual_vs_simulated_waste(rows, base_kpis)
    assert result == {"actual_total": 10.0, "simulated_total": 5.0,
                      "ratio": 0.5, "n_rows": 2}


def test_actual_zero_gives_nan_ratio():
    rows = pd.DataFrame({"item_id": ["a"], "date": ["2024-01-01"], "waste_qty": [0.0]})
    base_kpis = pd.DataFrame({"item_id": ["a"], "date": ["2024-01-01"], "waste_units": [2.0]})
    result = compare_actual_vs_simulated_waste(rows, base_kpis)
    assert np.isnan(result["ratio"])
    assert result["actual_total"] == 0.0

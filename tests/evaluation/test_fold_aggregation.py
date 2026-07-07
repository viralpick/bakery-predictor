import numpy as np
import pandas as pd
from bakery.evaluation.prospective import compare_policies_by_fold, aggregate_fold_kpis


def test_compare_policies_by_fold_computes_delta_per_fold():
    our = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "waste_cost_krw": [10.0, 10.0, 20.0, 20.0],
        "lost_margin_krw": [1.0, 1.0, 2.0, 2.0],
        "is_stockout": [True, False, True, True],
        "soldout_hour": [15.0, np.nan, 16.0, 14.0],
    })
    base = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "waste_cost_krw": [4.0, 4.0, 5.0, 5.0],
        "lost_margin_krw": [0.5, 0.5, 1.0, 1.0],
        "is_stockout": [True, True, True, True],
        "soldout_hour": [16.0, 17.0, 18.0, 18.0],
    })
    out = compare_policies_by_fold(our, base)
    row0 = out[out["fold"] == 0].iloc[0]
    # fold0: waste Δ = (10+10) - (4+4) = 12 ; stockout_rate Δ = 0.5 - 1.0 = -0.5
    assert row0["waste_cost_krw"] == 12.0
    assert row0["stockout_rate"] == -0.5


def test_aggregate_fold_kpis_mean_and_ci():
    per_fold = pd.DataFrame({"fold": [0, 1, 2], "waste_cost_krw": [10.0, 20.0, 30.0]})
    agg = aggregate_fold_kpis(per_fold, ["waste_cost_krw"])
    r = agg.iloc[0]
    assert r["metric"] == "waste_cost_krw"
    assert r["mean"] == 20.0
    assert r["n"] == 3
    # std(ddof=1)=10, sem=10/sqrt(3)
    assert abs(r["sem"] - (10.0 / np.sqrt(3))) < 1e-9
    assert abs(r["ci95_low"] - (20.0 - 1.96 * 10.0 / np.sqrt(3))) < 1e-9

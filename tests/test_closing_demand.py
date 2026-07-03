import math

import numpy as np
import pandas as pd
import pytest
from bakery.analysis.closing_demand import (
    build_closing_panel,
    build_intraday_curve,
    depth_time_overlap,
    fit_depth_elasticity,
    fit_kink,
    fit_surplus_counterfactual,
)


def _rows():
    # 1 day, 1 category(bread) via item->cat map; 2 normal + 3 closing(30%) + 2 closing(20%)
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05"] * 7),
        "hour": [10, 11, 20, 20, 21, 20, 21],
        "minute": [0, 0, 5, 10, 0, 15, 30],
        "item_id": ["A"] * 7,
        "qty": [2, 3, 1, 1, 1, 1, 1],   # normal:5, closing30:3, closing20:2
        "label": ["none", "none", "closing", "closing", "closing", "closing", "closing"],
        "discount_code": ["", "", "0077", "0077", "0077", "0069", "0069"],
    })


def test_panel_decomposition_identity():
    rows = _rows()
    waste = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "item_id": ["A"], "waste_qty": [4]})
    itc = pd.Series({"A": "bread"})
    panel = build_closing_panel(rows, waste, itc)
    r = panel.iloc[0]
    assert r["normal_qty"] == 5
    assert r["closing_qty"] == 5           # 3 + 2
    assert r["closing_qty_30"] == 3
    assert r["closing_qty_20"] == 2
    assert r["waste_qty"] == 4
    assert r["surplus"] == 9               # closing 5 + waste 4


def test_surplus_equals_closing_plus_waste_all_rows():
    rows = _rows()
    waste = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "item_id": ["A"], "waste_qty": [4]})
    panel = build_closing_panel(rows, waste, pd.Series({"A": "bread"}))
    assert (panel["surplus"] == panel["closing_qty"] + panel["waste_qty"]).all()


def _depth_panel(slope, base_per_depth, n=200):
    """Generate synthetic depth panel: closing_qty_d = base + slope*depth."""
    rng = np.arange(n)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng, "D")
    c30 = base_per_depth + slope * 0.30
    c20 = base_per_depth + slope * 0.20
    return pd.DataFrame({
        "category_id": ["bread"] * n, "date": dates,
        "closing_qty_30": [c30] * n, "closing_qty_20": [c20] * n,
        "closing_qty": [c30 + c20] * n, "surplus": [100.0] * n,
        "dow": (rng % 7), "month": 1, "trend": rng,
    })


def test_depth_elasticity_recovers_slope():
    panel = _depth_panel(slope=50.0, base_per_depth=10.0)
    res = fit_depth_elasticity(panel)
    assert res.slope == pytest.approx(50.0, abs=1.0)
    # base at depth=0 == 10 per depth-observation; α = base / mean(observed depth qty)
    assert res.base == pytest.approx(10.0, abs=0.5)
    assert 0.0 <= res.alpha <= 1.0


def test_depth_time_confound_flag():
    rows = pd.DataFrame({
        "discount_code": ["0069","0069","0077","0077"],
        "label": ["closing"]*4, "hour": [20,20,21,21], "minute":[0,5,0,5],
    })
    ov = depth_time_overlap(rows)   # returns dict: median hour per depth + separated flag
    assert ov["median_hour_20"] == 20.0
    assert ov["median_hour_30"] == 21.0
    assert ov["time_separated"] is True   # medians differ ≥ 1h


def _surplus_panel(slope, n=200):
    rng = np.arange(n)
    surplus = 10.0 + rng % 40          # varies 10..49
    closing = slope * surplus          # supply-driven if slope~1
    return pd.DataFrame({
        "category_id": ["bread"]*n,
        "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng, "D"),
        "closing_qty": closing, "surplus": surplus,
        "normal_qty": 100.0, "dow": rng % 7, "trend": rng,
    })


def test_surplus_slope_supply_driven():
    res = fit_surplus_counterfactual(_surplus_panel(slope=0.9))
    assert res.slope == pytest.approx(0.9, abs=0.05)   # tracks surplus → supply-driven
    assert res.clearance_high == pytest.approx(0.9, abs=0.05)


def test_surplus_slope_saturated():
    # closing fixed regardless of surplus → demand-driven base
    p = _surplus_panel(slope=0.9); p["closing_qty"] = 8.0
    res = fit_surplus_counterfactual(p)
    assert res.slope == pytest.approx(0.0, abs=0.05)


def test_surplus_ill_posed_regression():
    """Test that ill-posed regression (singular design matrix) returns note='ill-posed' not 'demand-limited'."""
    # Create a panel where surplus is perfectly collinear with trend,
    # making the design matrix singular and _ols_hc3 return None.
    rng = np.arange(200)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng, "D")
    # surplus = trend (perfect collinearity)
    surplus_vals = rng % 40 + 10
    trend_vals = rng % 40 + 10
    return_panel = pd.DataFrame({
        "category_id": ["bread"] * 200,
        "date": dates,
        "closing_qty": 5.0,  # constant, arbitrary
        "surplus": surplus_vals,
        "normal_qty": 100.0,
        "dow": rng % 7,
        "trend": trend_vals,
    })
    res = fit_surplus_counterfactual(return_panel)
    assert res.note == "ill-posed", f"Expected note='ill-posed', got '{res.note}'"
    assert math.isnan(res.slope), f"Expected slope to be NaN, got {res.slope}"


def _intraday_rows(days=30):
    # pre-onset(17-19h) flat rate 2/bin; closing window(20-21h) observed 5/bin.
    # counterfactual base in closing window = 2/bin. induced=3/bin. α = base/closing = 2/5=0.4
    recs = []
    for d in range(days):
        date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=d)
        for h in [17, 18, 19]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 2,
                         "label": "none", "item_id": "A"})
        for h in [20, 21]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 5,
                         "label": "closing", "item_id": "A"})
    return pd.DataFrame(recs)


def test_kink_recovers_alpha():
    rows = _intraday_rows()
    curve = build_intraday_curve(rows, pd.Series({"A": "bread"}), "bread", bin_min=60)
    res = fit_kink(curve)
    assert res.alpha == pytest.approx(0.4, abs=0.05)

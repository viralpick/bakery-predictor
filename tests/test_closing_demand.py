import math

import numpy as np
import pandas as pd
import pytest
from bakery.analysis.closing_demand import (
    aggregate_alpha,
    build_closing_panel,
    build_intraday_curve,
    depth_time_overlap,
    evening_traffic_check,
    fit_depth_elasticity,
    fit_kink,
    fit_surplus_counterfactual,
    run_closing_demand,
    DepthResult,
    KinkResult,
    SurplusResult,
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


def test_depth_elasticity_degenerate_negative_extrapolation():
    """Steep positive slope (30%-depth qty >> 20%-depth qty) can extrapolate to a
    negative base at depth=0 -- a degenerate linear extrapolation (depth
    endogeneity), not a valid '0% real demand' estimate. Must return NaN alpha
    with a 'degenerate' note, not silently clip to 0.0.
    """
    panel = _depth_panel(slope=100.0, base_per_depth=-10.0)
    res = fit_depth_elasticity(panel)
    assert math.isnan(res.alpha)
    assert res.note.startswith("degenerate")
    # Sanity check against the pre-fix behavior: clipping the raw negative base
    # to 0 first would have produced alpha=0.0 here, which misreads as "method
    # finds 0% real demand" -- the opposite of the true conclusion.
    pre_fix_base = max(res.base, 0.0)
    pre_fix_alpha = pre_fix_base / 15.0  # y.mean() for this synthetic panel
    assert pre_fix_alpha == pytest.approx(0.0)


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


def test_surplus_clearance_over_one_marker():
    """clearance_high > 1.0 signals a data-quality issue (negative waste can make
    surplus < closing_qty). The note must carry a visible marker, not pass silently.
    """
    n = 200
    rng = np.arange(n)
    surplus = 10.0 + rng % 40
    closing = 1.1 * surplus  # closing exceeds surplus -> clearance > 1
    panel = pd.DataFrame({
        "category_id": ["bread"] * n,
        "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng, "D"),
        "closing_qty": closing, "surplus": surplus,
        "normal_qty": 100.0, "dow": rng % 7, "trend": rng,
    })
    res = fit_surplus_counterfactual(panel)
    assert res.clearance_high > 1.0
    assert "[clearance>1: check waste data]" in res.note


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


def test_kink_scope_consistent_base_partial_coverage():
    """Verify α uses scope-consistent base (pre_rate × num_closing_bins), not all-days bias.

    If some days have pre-onset bins but NO closing bins, the old formula
    base = pre_rate * bins_per_day * days would inflate base (wrong denominator).
    Scope-consistent base = pre_rate * len(win) avoids this.

    Setup: 20 days with pre+closing, 10 days pre-only.
    Expected α = (2 × 40 bins) / 200 = 0.4 (scope-consistent).
    Wrong α = (2 × 2 × 30) / 200 = 0.6 (days-based, would fail this test).
    """
    recs = []
    # Days 0-19: both pre-onset (17-19) and closing (20-21)
    for d in range(20):
        date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=d)
        for h in [17, 18, 19]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 2,
                         "label": "none", "item_id": "A"})
        for h in [20, 21]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 5,
                         "label": "closing", "item_id": "A"})
    # Days 20-29: pre-onset only, NO closing
    for d in range(20, 30):
        date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=d)
        for h in [17, 18, 19]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 2,
                         "label": "none", "item_id": "A"})
    rows = pd.DataFrame(recs)
    curve = build_intraday_curve(rows, pd.Series({"A": "bread"}), "bread", bin_min=60)
    res = fit_kink(curve)
    # α should be 0.4 (scope-consistent), not 0.6 (days-biased)
    assert res.alpha == pytest.approx(0.4, abs=0.05)


def test_kink_degenerate_base_exceeds_closing():
    """When the afternoon-rate extrapolated base exceeds the observed closing
    total, the discount produced no measurable lift -- a degenerate signal, not
    a confident lower-bound estimate. Alpha still clips to 1.0, but the note
    must say so honestly instead of claiming 'lower-bound (evening commute uplift)'.
    """
    recs = []
    for d in range(10):
        date = pd.Timestamp("2025-05-01") + pd.Timedelta(days=d)
        for h in (17, 18, 19):
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 10,
                         "label": "none", "item_id": "A"})
        for h in (20, 21):
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 1,
                         "label": "closing", "item_id": "A"})
    rows = pd.DataFrame(recs)
    curve = build_intraday_curve(rows, pd.Series({"A": "bread"}), "bread", bin_min=60)
    res = fit_kink(curve)
    assert res.base > res.closing_total
    assert res.alpha == 1.0
    assert res.note.startswith("degenerate")


def test_evening_traffic_check_declining_evening():
    """Afternoon rate 4/h, evening rate 2/h -> ratio 0.5, a1_floor_valid False.

    This is the case where fit_kink's A1 lower-bound interpretation breaks:
    evening traffic is naturally lower than afternoon, so A1 overstates alpha.
    """
    recs = []
    for d in range(5):
        date = pd.Timestamp("2025-04-01") + pd.Timedelta(days=d)
        for h in (17, 18, 19):
            recs.append({"date": date, "hour": h, "minute": 0, "item_id": "A",
                         "qty": 4, "label": "none"})
        for h in (20, 21):
            recs.append({"date": date, "hour": h, "minute": 0, "item_id": "A",
                         "qty": 2, "label": "none"})
    rows = pd.DataFrame(recs)
    result = evening_traffic_check(rows, pd.Series({"A": "bread"}), "bread")
    assert result["afternoon_rate"] == pytest.approx(4.0)
    assert result["evening_rate"] == pytest.approx(2.0)
    assert result["evening_to_afternoon_ratio"] == pytest.approx(0.5)
    assert result["a1_floor_valid"] is False


def test_evening_traffic_check_degenerate_no_evening_rows():
    """No non-closing rows in the evening window -> NaN ratio, a1_floor_valid=None."""
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-01"] * 3),
        "hour": [17, 18, 19], "minute": [0, 0, 0], "item_id": ["A"] * 3,
        "qty": [4, 4, 4], "label": ["none"] * 3,
    })
    result = evening_traffic_check(rows, pd.Series({"A": "bread"}), "bread")
    assert math.isnan(result["evening_rate"])
    assert math.isnan(result["evening_to_afternoon_ratio"])
    assert result["a1_floor_valid"] is None


def test_aggregate_alpha_interval():
    kink = KinkResult(30, 2.0, 5.0, 0.40, "")
    depth = DepthResult(200, 50.0, 2.0, 10.0, 0.45, "")
    surplus = SurplusResult(200, 0.9, 0.05, 0.9, "supply-driven (low α)")
    est = aggregate_alpha(kink, depth, surplus)
    # lower bound = max of the two lower-bound methods (A1, A2)
    assert est.alpha_low == pytest.approx(0.45, abs=1e-6)
    assert est.a1 == pytest.approx(0.40, abs=1e-6)
    assert est.a2 == pytest.approx(0.45, abs=1e-6)
    assert 0.0 <= est.alpha_low <= est.alpha_high <= 1.0


def test_aggregate_alpha_both_nan_lower_bounds():
    """Regression test: both lower-bound methods (kink.alpha, depth.alpha) fail (NaN).

    The 'both NaN' case exercised the nan > x bug class. Verify that:
    1. When BOTH lowers are NaN, alpha_low correctly becomes NaN (not 0).
    2. Supply-driven case: alpha_high also becomes NaN (due to clipping with NaN bound).
    3. Demand-limited case: alpha_high remains 1.0, even when alpha_low is NaN.
    """
    # Case 1: Supply-driven surplus (slope > 0.5)
    kink_nan = KinkResult(30, 2.0, 5.0, float("nan"), "kink failed")
    depth_nan = DepthResult(200, 50.0, 2.0, 10.0, float("nan"), "depth failed")
    surplus_supply = SurplusResult(200, 0.8, 0.05, 0.3, "supply-driven")
    est_supply = aggregate_alpha(kink_nan, depth_nan, surplus_supply)
    assert math.isnan(est_supply.alpha_low), "Expected both-NaN lowers to produce NaN alpha_low"
    # In supply-driven case, alpha_high is clipped with NaN lower bound → also NaN
    assert math.isnan(est_supply.alpha_high), "Supply-driven with NaN alpha_low should produce NaN alpha_high"

    # Case 2: Demand-limited surplus (slope ≈ 0)
    surplus_demand = SurplusResult(200, 0.0, 0.05, 0.5, "demand-limited")
    est_demand = aggregate_alpha(kink_nan, depth_nan, surplus_demand)
    assert math.isnan(est_demand.alpha_low), "Expected both-NaN lowers to produce NaN alpha_low"
    assert est_demand.alpha_high == 1.0, "Demand-limited should allow alpha_high = 1.0 even when alpha_low is NaN"


def _orchestrator_rows(days=30):
    """Synthetic bread-category rows: pre-onset flat rate + closing window split
    across both discount depths (30%/20%) with day-varying qty for depth
    identification, sized to clear MIN_DEPTH_ROWS/MIN_SURPLUS_ROWS."""
    recs = []
    for d in range(days):
        date = pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)
        for h in (17, 18, 19):
            recs.append({"date": date, "hour": h, "minute": 0, "item_id": "A",
                         "qty": 2, "label": "none", "discount_code": ""})
        recs.append({"date": date, "hour": 20, "minute": 0, "item_id": "A",
                     "qty": 3 + (d % 5), "label": "closing", "discount_code": "0077"})
        recs.append({"date": date, "hour": 21, "minute": 0, "item_id": "A",
                     "qty": 2 + (d % 3), "label": "closing", "discount_code": "0069"})
    return pd.DataFrame(recs)


def test_run_closing_demand_smoke():
    rows = _orchestrator_rows()
    dates = rows["date"].drop_duplicates()
    waste = pd.DataFrame({"date": dates, "item_id": "A", "waste_qty": 1})
    itc = pd.Series({"A": "bread"})
    result = run_closing_demand(rows, waste, itc, category="bread")
    assert set(result.keys()) == {"alpha", "depth", "surplus", "kink", "panel"}
    assert 0.0 <= result["alpha"].alpha_low <= 1.0

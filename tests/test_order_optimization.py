import numpy as np
import pandas as pd
import pytest
from bakery.analysis.order_optimization import (
    load_category_daily,
    conditional_demand_samples,
    demand_quantile,
    demand_cdf,
    implied_cost_rate,
    newsvendor_order,
    backtest_savings,
    run_phaseb,
    _backtest_one_c,
    _trailing_closing_frac,
    CLOSING_DELTA,
    MIN_HISTORY_DAYS,
)


def _rows():
    # 2 items in bread, 2 days. day2 item B has out<0 (clip) ; day1 clean.
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01","2025-01-01","2025-01-02","2025-01-02"]),
        "item_id": ["A","B","A","B"],
        "made":[10,20,10,20], "out":[2,0,0,-3],
        "normal_qty":[6,15,7,18], "closing_qty":[2,5,3,5], "sold_total":[8,20,10,23],
        "unit_price":[1000,2000,1000,2000], "identity_diff":[0.0,0.0,0.0,0.0],
    })

def test_category_daily_aggregation():
    itc = pd.Series({"A":"bread","B":"bread"})
    cd = load_category_daily(_rows(), itc, "bread")
    d1 = cd[cd["date"]=="2025-01-01"].iloc[0]
    assert d1["demand"] == 28          # 8+20
    assert d1["made"] == 30            # 10+20
    assert d1["out"] == 2              # 2+0
    assert d1["normal"] == 21 and d1["closing"] == 7
    # price = (8*1000 + 20*2000)/28
    assert d1["price"] == pytest.approx((8*1000+20*2000)/28)

def test_out_negative_clipped():
    itc = pd.Series({"A":"bread","B":"bread"})
    cd = load_category_daily(_rows(), itc, "bread")
    d2 = cd[cd["date"]=="2025-01-02"].iloc[0]
    assert d2["out"] == 0              # item B out=-3 clipped to 0, item A out=0


def _hist():
    dates = pd.date_range("2025-01-06", periods=140, freq="D")  # Mondays start
    # demand = 100 on Mondays(dow=0), 50 otherwise, deterministic
    dow = dates.dayofweek
    demand = np.where(dow==0, 100.0, 50.0)
    return pd.DataFrame({"date":dates,"demand":demand,"dow":dow})


def test_conditional_samples_dow_and_leakage():
    hist = _hist()
    target = pd.Timestamp("2025-04-07")  # a Monday
    s = conditional_demand_samples(hist, target, dow=0, window_weeks=8)
    assert (s==100.0).all()                      # only Monday demand
    assert len(s)==8                             # last 8 Mondays before target
    # leakage: only strictly-before target
    assert hist[hist["date"]>=target].shape[0]>0 # future rows exist...
    s2 = conditional_demand_samples(hist[hist["date"]<target], target, 0, 8)
    assert np.array_equal(s, s2)                 # ...but they don't change the estimate


def test_quantile_and_cdf():
    s = np.array([10.,20.,30.,40.,50.])
    assert demand_quantile(s, 0.5) == pytest.approx(30.0)
    assert demand_cdf(s, 30.0) == pytest.approx(0.6)   # P(<=30)=3/5


def test_level1_is_1minus_c_quantile():
    s = np.arange(1, 101, dtype=float)  # 1..100 uniform
    res = newsvendor_order(s, c=0.35, closing_frac=0.0, delta=0.30)
    # closing_frac=0 → no salvage band → Level2==Level1==(1-0.35) quantile
    assert res.q_l1 == pytest.approx(np.quantile(s, 0.65), abs=1.0)
    assert res.q_l2 == pytest.approx(res.q_l1, abs=2.0)


def test_level2_le_level1_with_closing_band():
    s = np.arange(1, 101, dtype=float)
    res = newsvendor_order(s, c=0.35, closing_frac=0.3, delta=0.30)
    # closing band earns only discount margin → optimal produces no more than L1
    assert res.q_l2 <= res.q_l1 + 1e-6


def test_implied_c_from_service_level():
    s = np.arange(1, 101, dtype=float)
    # made=90 → SL=P(demand<=90)=0.90 → implied c=0.10
    assert implied_cost_rate(s, made=90.0) == pytest.approx(0.10, abs=0.01)
    # made=65 → SL=0.65 → implied c=0.35
    assert implied_cost_rate(s, made=65.0) == pytest.approx(0.35, abs=0.01)


def _cat_daily(n=200):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    dow = dates.dayofweek
    demand = np.where(dow >= 5, 120.0, 60.0)     # weekend higher, deterministic
    return pd.DataFrame({"date": dates, "demand": demand, "made": demand.copy(),
                         "out": np.zeros(n), "normal": demand * 0.7, "closing": demand * 0.3,
                         "price": np.full(n, 1000.0), "dow": dow, "month": dates.month})


def test_backtest_placebo_and_savings_sign():
    cd = _cat_daily()
    res = backtest_savings(cd, c_grid=[0.35])
    row = res.iloc[0]
    # made==demand (perfect) → placebo cost ~0; Q* can't beat it → savings <= 0
    assert row["cost_made"] == pytest.approx(0.0, abs=1e-6)
    assert row["savings_vs_made"] <= 1e-6
    assert row["n_days"] > 0


def test_backtest_is_leakage_safe():
    """Corrupted future rows must not change decisions on days before the corruption.

    The corrupted rows (index >= 150, demand=9999, closing blown up too) are
    NOT truncated away before calling _backtest_one_c -- they genuinely pass
    through the function. Day i's decision depends only on hist =
    rows.iloc[:i] (strictly-past data). For i < 150, that slice never touches
    a corrupted row, so the resulting "impl" AND "q_l2" entries for the early
    evaluated window (days 90..149, MIN_HISTORY_DAYS=90) must be bit-identical
    between the clean and corrupted runs -- if the implementation ever used an
    inclusive slice (rows.iloc[:i+1]) or leaked the full array into sample or
    closing_frac construction, corruption would reach these early days and the
    equality below would fail. This covers both the impl (mean_implied_c) path
    and the qstar/cost path (q_l2, which cost_qstar is a deterministic
    function of via _cost).
    """
    cd = _cat_daily()
    cd2 = cd.copy()
    cd2.loc[cd2.index[150:], "demand"] = 9999  # corrupt all "future" rows from day 150 on
    cd2.loc[cd2.index[150:], "closing"] = 9999
    cd_corrupt = pd.concat([cd.iloc[:150], cd2.iloc[150:]])

    acc_clean = _backtest_one_c(cd, c=0.35, delta=CLOSING_DELTA)
    acc_corrupt = _backtest_one_c(cd_corrupt, c=0.35, delta=CLOSING_DELTA)

    # Days 90..149 -> 60 evaluated entries at the head of "impl"/"q_l2". These
    # only ever see rows.iloc[:i] with i<150, i.e. exclusively clean data.
    n_early = 60
    early_clean = np.asarray(acc_clean["impl"][:n_early], dtype=float)
    early_corrupt = np.asarray(acc_corrupt["impl"][:n_early], dtype=float)
    assert early_clean.shape == early_corrupt.shape == (n_early,)
    finite_mask = np.isfinite(early_clean) & np.isfinite(early_corrupt)
    assert finite_mask.sum() == n_early  # sanity: window isn't degenerate/NaN
    assert np.array_equal(early_clean[finite_mask], early_corrupt[finite_mask])

    q_l2_clean = np.asarray(acc_clean["q_l2"][:n_early], dtype=float)
    q_l2_corrupt = np.asarray(acc_corrupt["q_l2"][:n_early], dtype=float)
    assert np.array_equal(q_l2_clean, q_l2_corrupt)

    # Sanity check: corruption must have actually reached the function and
    # changed later days. Otherwise the early-window equality above would be
    # vacuous (e.g. if corrupted rows were silently dropped instead of used).
    late_clean = np.asarray(acc_clean["impl"][-10:], dtype=float)
    late_corrupt = np.asarray(acc_corrupt["impl"][-10:], dtype=float)
    assert not np.array_equal(late_clean, late_corrupt)
    late_q_l2_clean = np.asarray(acc_clean["q_l2"][-10:], dtype=float)
    late_q_l2_corrupt = np.asarray(acc_corrupt["q_l2"][-10:], dtype=float)
    assert not np.array_equal(late_q_l2_clean, late_q_l2_corrupt)


def test_qstar_closing_frac_is_trailing_not_same_day():
    """Fix-1 regression: closing_frac for day i's order must come from strictly
    -before history, never day i's own realized closing (order-time info only).

    Fixture: closing==0 every day (a single-class world, so the trailing
    closing_frac is always exactly 0 for any window). We corrupt ONLY day
    index 150's own "closing" value to a large nonzero fraction of that day's
    demand -- no other row, past or future, is touched.

    - Fixed behavior (_trailing_closing_frac): day 150's decision is built
      from hist=rows.iloc[:150], which never includes day 150 itself, so
      q_l2 at day 150 must be bit-identical between clean and corrupted runs.
    - Old same-day cf = r["closing"]/r["demand"] reads the corrupted value
      directly for day 150's own decision, so this assertion FAILS under
      that implementation -- confirmed manually: with the old formula,
      q_l2(clean)=61.40 vs q_l2(corrupted)=63.28 at this boundary (the
      corrupted closing pushes cf from 0 to 0.5, switching the newsvendor
      from the single-class to the two-class salvage-band regime for that
      one day's order).
    """
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    dow = dates.dayofweek
    base = np.where(dow >= 5, 120.0, 60.0)
    demand = np.maximum(base + rng.normal(0, 8.0, size=n), 1.0)
    cd = pd.DataFrame({
        "date": dates, "demand": demand, "made": demand.copy(),
        "out": np.zeros(n), "normal": demand.copy(), "closing": np.zeros(n),
        "price": np.full(n, 1000.0), "dow": dow, "month": dates.month,
    })
    corrupt_day = 150
    cd_corrupt = cd.copy()
    cd_corrupt.loc[corrupt_day, "closing"] = cd_corrupt.loc[corrupt_day, "demand"] * 0.5

    acc_clean = _backtest_one_c(cd, c=0.35, delta=CLOSING_DELTA)
    acc_corrupt = _backtest_one_c(cd_corrupt, c=0.35, delta=CLOSING_DELTA)

    boundary = corrupt_day - MIN_HISTORY_DAYS
    assert acc_clean["q_l2"][boundary] == pytest.approx(acc_corrupt["q_l2"][boundary])


def test_trailing_closing_frac_empty_history_returns_zero():
    hist = pd.DataFrame({"date": [], "demand": [], "closing": [], "dow": []})
    assert _trailing_closing_frac(hist, "2024-01-01", 0) == 0.0


def _rows_for_smoke(n=200):
    # item-day rows (pre-aggregation) for 2 bread items, enough history for
    # MIN_HISTORY_DAYS(90) + MIN_SAMPLES(6) same-dow samples.
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    dow = dates.dayofweek
    demand = np.where(dow >= 5, 60.0, 30.0)
    base = pd.DataFrame({
        "date": dates, "made": demand, "out": 0.0,
        "normal_qty": demand * 0.7, "closing_qty": demand * 0.3,
        "sold_total": demand, "unit_price": 1000, "identity_diff": 0.0,
    })
    a = base.copy(); a["item_id"] = "A"
    b = base.copy(); b["item_id"] = "B"
    return pd.concat([a, b], ignore_index=True)


def test_run_phaseb_smoke():
    cd_rows = _rows_for_smoke()   # 충분한 일수의 합성 parquet-형 rows (bread), 헬퍼 상단 정의
    itc = pd.Series({"A": "bread", "B": "bread"})
    out = run_phaseb(cd_rows, itc, "bread", c_grid=[0.35])
    assert set(out) == {"implied_c_current", "savings_table"}
    # made==demand every day in the fixture -> P(demand<=made)=1.0 deterministically
    # -> implied_c_current must be exactly 0.0 (regression guard on the primary deliverable).
    assert out["implied_c_current"] == pytest.approx(0.0, abs=1e-9)

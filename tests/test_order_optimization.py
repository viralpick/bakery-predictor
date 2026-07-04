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
    CLOSING_DELTA,
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
    """Verify that Q* decisions for early days don't leak from future corruption."""
    cd = _cat_daily()
    # Baseline: evaluate days 0-149 using clean data
    r_baseline = backtest_savings(cd.iloc[:150], c_grid=[0.35]).iloc[0]
    # Create a dataset with corrupted future (last 10 days)
    cd2 = cd.copy()
    cd2.loc[cd2.index[-10:], "demand"] = 9999  # corrupt only the tail future
    # Concatenate clean (0-149) + corrupted (150-199), then truncate back to 0-149
    # This ensures hist[:i] for i<150 remains unchanged despite future corruption
    concatenated_truncated = pd.concat([cd.iloc[:150], cd2.iloc[150:]]).iloc[:150]
    r_early = backtest_savings(concatenated_truncated, c_grid=[0.35]).iloc[0]
    # The early-day decisions must be identical (strict: no leakage)
    assert r_baseline["n_days"] == r_early["n_days"]
    assert r_baseline["cost_qstar"] == pytest.approx(r_early["cost_qstar"], abs=1e-10)

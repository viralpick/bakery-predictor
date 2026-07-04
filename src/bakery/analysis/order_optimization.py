"""Phase B — 발주 최적화 + implied c 갭.

카테고리 총수요 two-class salvage-newsvendor. 원가율 c=파라미터.
주 산출물=현행 발주의 implied c. 절감액은 placebo(Q=made) 대비.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

IDENTITY_TOL = 1.0
WINDOW_WEEKS = 13
MIN_SAMPLES = 6
Q_GRID_STEPS = 50
CLOSING_DELTA = 0.28
MIN_HISTORY_DAYS = 90


def load_category_daily(rows, item_to_category, category):
    """Aggregate items to category-day level with identity & out constraints.

    Parameters
    ----------
    rows : pd.DataFrame
        Parquet rows with columns: date, item_id, made, out, normal_qty,
        closing_qty, sold_total, unit_price, identity_diff.
    item_to_category : pd.Series
        Mapping item_id -> category.
    category : str
        Category to filter by.

    Returns
    -------
    pd.DataFrame
        Columns: date, demand, made, out, normal, closing, price, dow, month.
        Rows are sorted by date and indexed from 0.
    """
    df = rows.copy()
    df["category_id"] = df["item_id"].astype(str).map(item_to_category)
    df = df[df["category_id"] == category].copy()
    df["out"] = df["out"].clip(lower=0.0)

    if "identity_diff" in df.columns:
        df = df[df["identity_diff"].abs() <= IDENTITY_TOL]

    g = df.groupby("date", observed=True).agg(
        demand=("sold_total", "sum"),
        made=("made", "sum"),
        out=("out", "sum"),
        normal=("normal_qty", "sum"),
        closing=("closing_qty", "sum"),
        rev=("sold_total", lambda s: float((s * df.loc[s.index, "unit_price"]).sum())),
    ).reset_index()

    g["price"] = g["rev"] / g["demand"].where(g["demand"] > 0, np.nan)
    d = pd.to_datetime(g["date"])
    g["dow"] = d.dt.dayofweek
    g["month"] = d.dt.month

    return g.drop(columns=["rev"]).sort_values("date").reset_index(drop=True)


def conditional_demand_samples(hist, target_date, dow, window_weeks=WINDOW_WEEKS):
    """Extract demand samples for same DOW strictly before target_date.

    Parameters
    ----------
    hist : pd.DataFrame
        Historical data with columns: date, demand, dow. Must be sorted ascending by date.
    target_date : pd.Timestamp or str
        Prediction date (exclusive).
    dow : int
        Day-of-week (0=Monday, 6=Sunday).
    window_weeks : int
        Number of past weeks to include.

    Returns
    -------
    np.ndarray
        Sorted demand samples for same DOW before target_date.
    """
    past = hist[(pd.to_datetime(hist["date"]) < pd.Timestamp(target_date))
                & (hist["dow"] == dow)]
    return past["demand"].to_numpy(float)[-window_weeks:]


def demand_quantile(samples, q):
    """Compute quantile of empirical demand distribution.

    Parameters
    ----------
    samples : np.ndarray
        Demand samples.
    q : float
        Quantile level in [0, 1].

    Returns
    -------
    float
        Quantile value, or NaN if empty.
    """
    if len(samples) == 0:
        return float("nan")
    return float(np.quantile(samples, q, method="linear"))


def demand_cdf(samples, x):
    """Compute empirical CDF P(demand <= x).

    Parameters
    ----------
    samples : np.ndarray
        Demand samples.
    x : float
        Threshold.

    Returns
    -------
    float
        P(demand <= x), or NaN if empty.
    """
    if len(samples) == 0:
        return float("nan")
    return float(np.mean(samples <= x))


def implied_cost_rate(samples, made):
    """Compute implied cost rate from current policy (made quantity).

    Service level SL = P(demand <= made) is calculated via demand_cdf.
    Implied cost rate c = 1 - SL, representing the stockout probability
    under the assumption that made quantity equals the (1-c) quantile.

    Parameters
    ----------
    samples : np.ndarray
        Demand samples.
    made : float
        Made (produced) quantity.

    Returns
    -------
    float
        Implied cost rate (1 - SL), or NaN if insufficient data or invalid made.
    """
    if len(samples) < MIN_SAMPLES or not np.isfinite(made):
        return float("nan")
    sl = demand_cdf(samples, made)
    return float(1.0 - sl)


@dataclass(frozen=True)
class OrderResult:
    """Newsvendor order result (Level 1 & 2)."""
    q_l1: float
    q_l2: float
    c: float


def _expected_profit(q, samples, c, closing_frac, delta):
    """Compute expected profit for order quantity q.

    Parameters
    ----------
    q : float
        Order quantity.
    samples : np.ndarray
        Demand samples.
    c : float
        Cost ratio.
    closing_frac : float
        Fraction of demand in closing band (0 to 1).
    delta : float
        Discount depth (salvage margin = 1 - delta - c).

    Returns
    -------
    float
        Expected profit (normalized to price=1).
    """
    d = samples
    normal = (1.0 - closing_frac) * d
    full = np.minimum(q, normal) * (1.0 - c)
    band = np.clip(np.minimum(q, d) - normal, 0.0, None) * (1.0 - delta - c)
    waste = np.clip(q - d, 0.0, None) * (-c)
    return float(np.mean(full + band + waste))


def newsvendor_order(samples, c, closing_frac, delta=CLOSING_DELTA, q_grid_steps=Q_GRID_STEPS):
    """Compute two-class salvage-newsvendor order quantities.

    Parameters
    ----------
    samples : np.ndarray
        Demand samples.
    c : float
        Cost ratio.
    closing_frac : float
        Fraction of demand in closing band (0 to 1).
    delta : float, optional
        Discount depth (default CLOSING_DELTA).
    q_grid_steps : int, optional
        Grid size for Level 2 search (default Q_GRID_STEPS).

    Returns
    -------
    OrderResult
        q_l1: (1-c) quantile upper bound.
        q_l2: profit-maximizing quantity.
        c: cost ratio.
    """
    if len(samples) < MIN_SAMPLES:
        return OrderResult(float("nan"), float("nan"), c)
    q_l1 = demand_quantile(samples, 1.0 - c)
    grid = np.linspace(samples.min(), samples.max(), q_grid_steps)
    profits = [_expected_profit(q, samples, c, closing_frac, delta) for q in grid]
    q_l2 = float(grid[int(np.argmax(profits))])
    return OrderResult(float(q_l1), q_l2, c)


def _cost(order, d, price, c):
    """Compute newsvendor cost for single day.

    Parameters
    ----------
    order : float
        Ordered quantity.
    d : float
        Realized demand.
    price : float
        Unit price.
    c : float
        Cost rate.

    Returns
    -------
    float
        Cost = c·price·overage + (1−c)·price·underage.
    """
    return c * price * max(order - d, 0.0) + (1.0 - c) * price * max(d - order, 0.0)


def _backtest_one_c(cat_daily, c, delta):
    """Backtest a single cost rate c across all days.

    Parameters
    ----------
    cat_daily : pd.DataFrame
        Rows: date, demand, made, closing, dow, price.
    c : float
        Cost rate.
    delta : float
        Closing discount depth.

    Returns
    -------
    dict
        Keys: qstar, made, l1, impl (list of implied c), n (count).
    """
    rows = cat_daily.reset_index(drop=True)
    acc = {"qstar": 0.0, "made": 0.0, "l1": 0.0, "impl": [], "n": 0}
    for i in range(len(rows)):
        r = rows.iloc[i]
        hist = rows.iloc[:i]
        if len(hist) < MIN_HISTORY_DAYS:
            continue
        samples = conditional_demand_samples(hist, r["date"], int(r["dow"]))
        if len(samples) < MIN_SAMPLES:
            continue
        cf = float(r["closing"] / r["demand"]) if r["demand"] > 0 else 0.0
        res = newsvendor_order(samples, c, cf, delta)
        d, price = float(r["demand"]), float(r["price"])
        acc["qstar"] += _cost(res.q_l2, d, price, c)
        acc["made"] += _cost(float(r["made"]), d, price, c)
        acc["l1"] += _cost(res.q_l1, d, price, c)
        acc["impl"].append(implied_cost_rate(samples, float(r["made"])))
        acc["n"] += 1
    return acc


def backtest_savings(cat_daily, c_grid, delta=CLOSING_DELTA):
    """Rolling backtest across cost rates with placebo arm and savings.

    Parameters
    ----------
    cat_daily : pd.DataFrame
        Category-daily data.
    c_grid : list of float
        Cost rates to evaluate.
    delta : float, optional
        Closing discount depth (default CLOSING_DELTA).

    Returns
    -------
    pd.DataFrame
        Columns: c, mean_implied_c, cost_qstar, cost_made, cost_l1,
                 savings_vs_made, n_days.
    """
    out = []
    for c in c_grid:
        a = _backtest_one_c(cat_daily, c, delta)
        out.append({
            "c": c,
            "mean_implied_c": float(np.nanmean(a["impl"])) if a["impl"] else float("nan"),
            "cost_qstar": a["qstar"],
            "cost_made": a["made"],
            "cost_l1": a["l1"],
            "savings_vs_made": a["made"] - a["qstar"],
            "n_days": a["n"]
        })
    return pd.DataFrame(out)

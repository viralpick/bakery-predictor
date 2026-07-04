"""Phase B — 발주 최적화 + implied c 갭.

카테고리 총수요 two-class salvage-newsvendor. 원가율 c=파라미터.
주 산출물=현행 발주의 implied c. 절감액은 placebo(Q=made) 대비.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IDENTITY_TOL = 1.0
WINDOW_WEEKS = 13
MIN_SAMPLES = 6


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

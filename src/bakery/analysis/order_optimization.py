"""Phase B — 발주 최적화 + implied c 갭.

카테고리 총수요 two-class salvage-newsvendor. 원가율 c=파라미터.
주 산출물=현행 발주의 implied c. 절감액은 placebo(Q=made) 대비.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IDENTITY_TOL = 1.0


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

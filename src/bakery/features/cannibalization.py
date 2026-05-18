"""Cross-item / cross-category signals.

When item A stocks out, traffic that would have bought A often spills over
to substitute items in the same category (or to other categories entirely).
These features expose store- and category-level aggregates as inputs so a
single tree can split on "did my neighbors stock out yesterday?" without
needing one feature per neighbor.

Leakage discipline: every feature is built from `shift(1)`-or-older
aggregates on the date axis. Today's same-store stockouts are never visible
to today's prediction.
"""

from __future__ import annotations

import pandas as pd

CANNIBALIZATION_FEATURE_COLUMNS: list[str] = [
    "store_stockout_rate_lag1",
    "store_stockout_rate_7d",
    "store_total_sold_lag1",
    "cat_stockout_rate_lag1",
    "cat_stockout_rate_7d",
    "cat_total_sold_lag1",
]


def add_cannibalization_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    store_col: str = "store_id",
    cat_col: str = "category_id",
) -> pd.DataFrame:
    out = df.sort_values([store_col, date_col]).reset_index(drop=True).copy()

    store_daily = (
        out.groupby([store_col, date_col], observed=True)
        .agg(
            store_stockout_rate=("is_stockout", "mean"),
            store_total_sold=("sold_units", "sum"),
        )
        .reset_index()
        .sort_values([store_col, date_col])
    )
    store_grp = store_daily.groupby(store_col, observed=True)
    store_daily["store_stockout_rate_lag1"] = store_grp["store_stockout_rate"].shift(1)
    store_daily["store_total_sold_lag1"] = store_grp["store_total_sold"].shift(1)
    store_daily["store_stockout_rate_7d"] = store_grp["store_stockout_rate"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).mean()
    )

    cat_daily = (
        out.groupby([store_col, cat_col, date_col], observed=True)
        .agg(
            cat_stockout_rate=("is_stockout", "mean"),
            cat_total_sold=("sold_units", "sum"),
        )
        .reset_index()
        .sort_values([store_col, cat_col, date_col])
    )
    cat_grp = cat_daily.groupby([store_col, cat_col], observed=True)
    cat_daily["cat_stockout_rate_lag1"] = cat_grp["cat_stockout_rate"].shift(1)
    cat_daily["cat_total_sold_lag1"] = cat_grp["cat_total_sold"].shift(1)
    cat_daily["cat_stockout_rate_7d"] = cat_grp["cat_stockout_rate"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).mean()
    )

    store_cols = [store_col, date_col, "store_stockout_rate_lag1", "store_stockout_rate_7d", "store_total_sold_lag1"]
    cat_cols = [store_col, cat_col, date_col, "cat_stockout_rate_lag1", "cat_stockout_rate_7d", "cat_total_sold_lag1"]
    out = out.merge(store_daily[store_cols], on=[store_col, date_col], how="left")
    out = out.merge(cat_daily[cat_cols], on=[store_col, cat_col, date_col], how="left")
    return out

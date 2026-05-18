"""Lag and same-day-of-week features.

Uses groupby(store_id, item_id).shift(k) on a date-sorted frame, which is
mathematically incapable of pulling future values for past rows. The
test_features_leakage.py regression test pins this property: mutating
future-row targets must not change past-row lag features.
"""

from __future__ import annotations

import pandas as pd

LAG_FEATURE_COLUMNS: list[str] = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_same_dow_1w",
    "sales_same_dow_2w",
    "sales_same_dow_4w",
]


def add_lag_features(
    df: pd.DataFrame,
    *,
    y_col: str = "sold_units",
    group_keys: tuple[str, ...] = ("store_id", "item_id"),
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.sort_values([*group_keys, date_col]).copy()
    grouped = out.groupby(list(group_keys), observed=True)[y_col]
    for k in (1, 7, 14, 28):
        out[f"sales_lag_{k}"] = grouped.shift(k)
    out["sales_same_dow_1w"] = grouped.shift(7)
    out["sales_same_dow_2w"] = _avg_shifts(grouped, [7, 14])
    out["sales_same_dow_4w"] = _avg_shifts(grouped, [7, 14, 21, 28])
    return out


def _avg_shifts(grouped, lags: list[int]) -> pd.Series:
    shifted = [grouped.shift(lag) for lag in lags]
    stacked = pd.concat(shifted, axis=1)
    return stacked.mean(axis=1)

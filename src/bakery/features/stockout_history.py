"""Stockout-history features for the stockout-risk classifier.

All features `shift(1)` first so today's is_stockout never enters the row's
own features — same leakage discipline as lag/rolling_features.
"""

from __future__ import annotations

import pandas as pd

STOCKOUT_HISTORY_COLUMNS: list[str] = [
    "stockout_lag_1",
    "stockout_lag_7",
    "stockout_rate_7d",
    "stockout_rate_28d",
    "same_dow_stockout_4w",
]


def add_stockout_history(
    df: pd.DataFrame,
    *,
    group_keys: tuple[str, ...] = ("store_id", "item_id"),
    date_col: str = "date",
    flag_col: str = "is_stockout",
) -> pd.DataFrame:
    out = df.sort_values([*group_keys, date_col]).copy()
    flag = out[flag_col].astype("int8")
    out["_flag_int"] = flag
    grp = out.groupby(list(group_keys), observed=True)["_flag_int"]
    out["stockout_lag_1"] = grp.shift(1)
    out["stockout_lag_7"] = grp.shift(7)
    shifted = grp.shift(1)
    shifted.index = out.index
    by = [out[k] for k in group_keys]
    out["stockout_rate_7d"] = shifted.groupby(by, observed=True).transform(
        lambda s: s.rolling(7, min_periods=2).mean()
    )
    out["stockout_rate_28d"] = shifted.groupby(by, observed=True).transform(
        lambda s: s.rolling(28, min_periods=2).mean()
    )
    same_dow = [grp.shift(s) for s in (7, 14, 21, 28)]
    out["same_dow_stockout_4w"] = pd.concat(same_dow, axis=1).mean(axis=1)
    return out.drop(columns=["_flag_int"])

"""Rolling statistics that always shift(1) before the window — never include
the target day's own value. Mirrors the leakage discipline of lag_features.
"""

from __future__ import annotations

import pandas as pd

ROLLING_FEATURE_COLUMNS: list[str] = [
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_std_7",
    "roll_std_28",
    "roll_median_28",
    "same_dow_roll_mean_4w",
]


def add_rolling_features(
    df: pd.DataFrame,
    *,
    y_col: str = "sold_units",
    group_keys: tuple[str, ...] = ("store_id", "item_id"),
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.sort_values([*group_keys, date_col]).copy()
    shifted = out.groupby(list(group_keys), observed=True)[y_col].shift(1)
    out["roll_mean_7"] = _roll(shifted, out, group_keys, 7, "mean")
    out["roll_mean_14"] = _roll(shifted, out, group_keys, 14, "mean")
    out["roll_mean_28"] = _roll(shifted, out, group_keys, 28, "mean")
    out["roll_std_7"] = _roll(shifted, out, group_keys, 7, "std")
    out["roll_std_28"] = _roll(shifted, out, group_keys, 28, "std")
    out["roll_median_28"] = _roll(shifted, out, group_keys, 28, "median")
    out["same_dow_roll_mean_4w"] = _same_dow_4w_mean(out, y_col, group_keys)
    return out


def _roll(shifted: pd.Series, frame: pd.DataFrame, group_keys: tuple[str, ...], window: int, agg: str) -> pd.Series:
    series = shifted.copy()
    series.index = frame.index
    grouped = series.groupby([frame[k] for k in group_keys], observed=True)
    result = grouped.transform(lambda s: getattr(s.rolling(window, min_periods=2), agg)())
    return result


def _same_dow_4w_mean(frame: pd.DataFrame, y_col: str, group_keys: tuple[str, ...]) -> pd.Series:
    grouped = frame.groupby(list(group_keys), observed=True)[y_col]
    shifts = [grouped.shift(s) for s in (7, 14, 21, 28)]
    stacked = pd.concat(shifts, axis=1)
    return stacked.mean(axis=1)

"""Time-series splits. Random split is forbidden (CLAUDE.md absolute rule #3).

generate_time_splits yields (train_mask, val_mask) tuples in chronological order
with a fixed validation horizon (default 7 days). The last n_splits windows are
used as validation folds; train can be expanding (default) or rolling.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp  # inclusive
    fold_index: int


def generate_time_splits(
    dates: pd.Series,
    *,
    n_splits: int = 4,
    val_horizon_days: int = 7,
    step_days: int = 7,
    min_train_days: int = 90,
    mode: str = "expanding",
) -> list[SplitWindow]:
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"mode must be expanding|rolling, got {mode!r}")
    unique = pd.to_datetime(pd.Series(dates).dropna().unique())
    unique = pd.DatetimeIndex(sorted(unique))
    if len(unique) < min_train_days + val_horizon_days:
        raise ValueError(
            f"need ≥{min_train_days + val_horizon_days} unique days, got {len(unique)}"
        )
    last_day = unique[-1]
    windows: list[SplitWindow] = []
    for fold in range(n_splits):
        val_end = last_day - pd.Timedelta(days=step_days * fold)
        val_start = val_end - pd.Timedelta(days=val_horizon_days - 1)
        train_end = val_start - pd.Timedelta(days=1)
        if mode == "expanding":
            train_start = unique[0]
        else:
            train_start = train_end - pd.Timedelta(days=min_train_days - 1)
        if train_end < unique[0] + pd.Timedelta(days=min_train_days - 1):
            break
        windows.append(SplitWindow(train_start, train_end, val_start, val_end, fold))
    return list(reversed(windows))


def apply_split(df: pd.DataFrame, window: SplitWindow, *, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[date_col]
    train = df[(d >= window.train_start) & (d <= window.train_end)].copy()
    val = df[(d >= window.val_start) & (d <= window.val_end)].copy()
    assert_no_leakage(train, val, date_col=date_col)
    return train, val


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, *, date_col: str = "date") -> None:
    if train.empty or val.empty:
        return
    train_max = train[date_col].max()
    val_min = val[date_col].min()
    if train_max >= val_min:
        raise AssertionError(
            f"time leakage: train_max ({train_max}) >= val_min ({val_min})"
        )

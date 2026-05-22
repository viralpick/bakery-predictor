"""Pure date-derived features. No leakage risk — depends only on the date column."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_date_features(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """Add cyclic-encoded day-of-month + plain dow/month/quarter/week_of_year.

    `day_of_month` raw (1~31) was replaced with `day_of_month_sin`/`cos` so the
    model sees 28/30/31 as roughly equivalent positions in the month — the
    cyclic encoding wraps day 1 ↔ days_in_month neighbours. `is_month_start`/
    `is_month_end` were dropped: LightGBM can recover them from day_of_month
    splits already, and the cyclic encoding captures the same info.
    """
    out = df.copy()
    d = out[date_col]
    out["dow"] = d.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["month"] = d.dt.month.astype("int8")
    out["quarter"] = d.dt.quarter.astype("int8")
    out["week_of_year"] = d.dt.isocalendar().week.astype("int16")

    day = d.dt.day.to_numpy()
    days_in_month = d.dt.days_in_month.to_numpy()
    # Position within the month, normalized to [0, 2π). Subtract 1 so day 1 → 0,
    # last day → ~2π·(N-1)/N. sin/cos pair lets the model treat month-end as
    # adjacent to month-start.
    angle = 2.0 * np.pi * (day - 1) / days_in_month
    out["day_of_month_sin"] = np.sin(angle).astype("float32")
    out["day_of_month_cos"] = np.cos(angle).astype("float32")
    return out


DATE_FEATURE_COLUMNS: list[str] = [
    "dow",
    "is_weekend",
    "month",
    "quarter",
    "week_of_year",
    "day_of_month_sin",
    "day_of_month_cos",
]

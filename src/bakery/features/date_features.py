"""Pure date-derived features. No leakage risk — depends only on the date column."""

from __future__ import annotations

import pandas as pd


def add_date_features(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = out[date_col]
    out["dow"] = d.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["month"] = d.dt.month.astype("int8")
    out["quarter"] = d.dt.quarter.astype("int8")
    out["day_of_month"] = d.dt.day.astype("int8")
    out["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    out["is_month_start"] = d.dt.is_month_start.astype("int8")
    out["is_month_end"] = d.dt.is_month_end.astype("int8")
    return out


DATE_FEATURE_COLUMNS: list[str] = [
    "dow",
    "is_weekend",
    "month",
    "quarter",
    "day_of_month",
    "week_of_year",
    "is_month_start",
    "is_month_end",
]

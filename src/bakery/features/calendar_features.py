"""Merge calendar daily into a sales daily frame.

Calendar features are functions of `date` alone, so no leakage risk is
introduced by merging — past rows cannot pick up signals from future rows.
"""

from __future__ import annotations

import pandas as pd

from ..data.calendar import CALENDAR_DAILY_COLUMNS

# Columns we feed into the model. `holiday_name` and `is_weekend` are
# excluded: name is a high-cardinality string we don't encode in v1, and
# weekend is already covered by `dow` in date_features.
CALENDAR_FEATURE_COLUMNS: list[str] = [
    "is_public_holiday",
    "is_substitute_holiday",
    "is_off_day",
    "is_day_before_off",
    "is_day_after_off",
    "off_streak_length",
    "off_position_in_streak",
    "is_xmas_eve",
    "is_xmas",
    "is_valentine",
    "is_white_day",
    "is_children_day",
    "is_pepero",
]


def add_calendar_features(
    df: pd.DataFrame, calendar_df: pd.DataFrame, *, date_col: str = "date"
) -> pd.DataFrame:
    """Left-merge calendar columns onto `df` on `date_col`. Missing dates default to 0."""
    missing = set(CALENDAR_FEATURE_COLUMNS) - set(calendar_df.columns)
    if missing:
        raise ValueError(f"calendar_df missing columns: {sorted(missing)}")
    cols = [date_col, *CALENDAR_FEATURE_COLUMNS]
    out = df.merge(calendar_df[cols], on=date_col, how="left")
    # In case the calendar frame doesn't cover every sales date, fill with 0.
    for col in CALENDAR_FEATURE_COLUMNS:
        out[col] = out[col].fillna(0).astype(CALENDAR_DAILY_COLUMNS[col])
    return out

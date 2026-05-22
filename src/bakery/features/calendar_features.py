"""Merge calendar daily into a sales daily frame.

Calendar features are functions of `date` alone, so no leakage risk is
introduced by merging — past rows cannot pick up signals from future rows.

Design (after PoC review):
- `is_public_holiday` + `is_weekend` (from date_features) cover the
  off-day signal; `is_off_day` / `is_substitute_holiday` were dropped.
- `is_day_before_off` / `is_day_after_off` were replaced by signed
  `days_to_*` features per event — those capture the 1-week lead-up that
  cake / sweets sales actually exhibit before special dates.
- Event-day booleans (`is_xmas`, `is_valentine`, ...) collapsed into the
  `days_to_*` features (event day = 0). `is_white_day` kept as boolean
  since we don't model its lead-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.calendar import CALENDAR_DAILY_COLUMNS

# Booleans we still merge straight from calendar_df.
_PASSTHROUGH_COLUMNS: list[str] = [
    "is_public_holiday",
    "off_streak_length",
    "off_position_in_streak",
    "is_white_day",
]

# Signed integer "days to event" features (negative = before, 0 = on event,
# positive = after). Clipped to ±14 so distant dates collapse to the cap.
_EVENT_TO_MONTH_DAY: dict[str, tuple[int, int]] = {
    "days_to_xmas": (12, 25),
    "days_to_valentine": (2, 14),
    "days_to_children_day": (5, 5),
    "days_to_pepero": (11, 11),
}
_EVENT_CLIP = 14

CALENDAR_FEATURE_COLUMNS: list[str] = _PASSTHROUGH_COLUMNS + list(_EVENT_TO_MONTH_DAY.keys())


def add_calendar_features(
    df: pd.DataFrame, calendar_df: pd.DataFrame, *, date_col: str = "date"
) -> pd.DataFrame:
    """Left-merge passthrough calendar columns + compute signed days_to_* features."""
    missing = set(_PASSTHROUGH_COLUMNS) - set(calendar_df.columns)
    if missing:
        raise ValueError(f"calendar_df missing columns: {sorted(missing)}")
    cols = [date_col, *_PASSTHROUGH_COLUMNS]
    out = df.merge(calendar_df[cols], on=date_col, how="left")
    for col in _PASSTHROUGH_COLUMNS:
        out[col] = out[col].fillna(0).astype(CALENDAR_DAILY_COLUMNS[col])

    dates = pd.to_datetime(out[date_col])
    for feat_name, (month, day) in _EVENT_TO_MONTH_DAY.items():
        out[feat_name] = _days_to_event(dates, month=month, day=day).astype("int8")
    return out


def _days_to_event(dates: pd.Series, *, month: int, day: int) -> pd.Series:
    """For each date, signed days to the nearest occurrence of (month, day),
    clipped to ±_EVENT_CLIP. Positive = event is in the future; 0 = on event day."""
    out = np.full(len(dates), _EVENT_CLIP + 1, dtype="int64")
    for offset in range(-1, 2):  # check event in prev / current / next year
        years = dates.dt.year + offset
        try:
            event_dates = pd.to_datetime({"year": years, "month": month, "day": day})
        except ValueError:
            # Handles dates like Feb 29 when year isn't leap — skip those rows
            event_dates = pd.to_datetime(
                {"year": years, "month": month, "day": day}, errors="coerce"
            )
        delta = (event_dates - dates).dt.days.fillna(_EVENT_CLIP + 1).to_numpy()
        # Keep the smallest absolute delta seen so far.
        better = np.abs(delta) < np.abs(out)
        out = np.where(better, delta, out)
    return pd.Series(np.clip(out, -_EVENT_CLIP, _EVENT_CLIP), index=dates.index)

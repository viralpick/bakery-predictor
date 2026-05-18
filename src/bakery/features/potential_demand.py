"""Censored-demand correction.

For non-stockout days, potential_demand == sold_units (no censoring).
For stockout days, we scale sold_units up by the fraction of the day's
expected sales window that was actually available before stockout. The
expected fraction comes from a blended hourly profile:

    combined = α × hour_weight + (1 − α) × uniform

`α=1` uses the bakery's hour curve (morning/lunch/afternoon peaks).
`α=0` is the flat "time-proportional" baseline. `α=0.5` is the default —
respects the curve but softens its peaks so a very-early stockout doesn't
explode the correction. Safety clips (`max_multiplier`, `min_cum_weight`)
backstop both extremes.

Inputs: a daily frame (DAILY_COLUMNS) plus a `store_id → (open_hour,
close_hour)` mapping. Output: same frame with a `potential_demand` column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_ALPHA: float = 0.5
DEFAULT_MAX_MULTIPLIER: float = 3.0
DEFAULT_MIN_CUM_WEIGHT: float = 0.15
PEAKS: list[tuple[float, float, float]] = [
    (9.0, 1.3, 1.0),   # morning commute
    (13.0, 1.0, 0.9),  # lunch
    (16.0, 1.2, 1.1),  # afternoon snack
    (19.0, 1.0, 0.6),  # evening
]


@dataclass(frozen=True)
class StoreHours:
    store_id: str
    open_hour: int
    close_hour: int


def bakery_hour_profile(open_hour: int, close_hour: int, *, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Return a 24-length array, 0 outside open hours, summing to 1 over open hours.

    alpha=1: full bakery curve. alpha=0: uniform over open hours.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if close_hour <= open_hour:
        raise ValueError(f"close_hour ({close_hour}) must be > open_hour ({open_hour})")
    hours = np.arange(24, dtype=float)
    curve = np.zeros(24)
    for peak, width, height in PEAKS:
        curve += height * np.exp(-0.5 * ((hours - peak) / width) ** 2)
    mask = (hours >= open_hour) & (hours < close_hour)
    curve = curve * mask
    if curve.sum() <= 0:
        raise ValueError("bakery curve collapsed to zero over the open window")
    curve = curve / curve.sum()
    uniform = mask.astype(float) / mask.sum()
    return alpha * curve + (1.0 - alpha) * uniform


def cumulative_weight_at(profile: np.ndarray, hour_float: float) -> float:
    """Weight from open through `hour_float`. Linear interpolation within an hour."""
    if hour_float <= 0:
        return 0.0
    if hour_float >= 24:
        return float(profile.sum())
    full = int(np.floor(hour_float))
    frac = hour_float - full
    return float(profile[:full].sum() + profile[full] * frac)


def compute_row_potential(
    sold_units: float,
    stockout_time: pd.Timestamp | None,
    date: pd.Timestamp,
    open_hour: int,
    close_hour: int,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_multiplier: float = DEFAULT_MAX_MULTIPLIER,
    min_cum_weight: float = DEFAULT_MIN_CUM_WEIGHT,
) -> float:
    """Single-row potential demand. NaT stockout_time → no correction."""
    if stockout_time is None or pd.isna(stockout_time):
        return float(sold_units)
    if sold_units <= 0:
        return float(sold_units)
    hour_float = (stockout_time - date).total_seconds() / 3600.0
    if hour_float <= open_hour:
        # Stockout before or right at opening — no reliable signal; clip aggressively.
        return float(sold_units) * max_multiplier
    profile = bakery_hour_profile(open_hour, close_hour, alpha=alpha)
    cum = cumulative_weight_at(profile, hour_float)
    cum = max(cum, min_cum_weight)
    multiplier = min(1.0 / cum, max_multiplier)
    return float(sold_units) * multiplier


def attach_potential_demand(
    daily: pd.DataFrame,
    stores: list[StoreHours],
    *,
    alpha: float = DEFAULT_ALPHA,
    max_multiplier: float = DEFAULT_MAX_MULTIPLIER,
    min_cum_weight: float = DEFAULT_MIN_CUM_WEIGHT,
) -> pd.DataFrame:
    """Return a copy of `daily` with a `potential_demand` column."""
    hours_lookup = {s.store_id: (s.open_hour, s.close_hour) for s in stores}
    profiles = {
        sid: bakery_hour_profile(oh, ch, alpha=alpha) for sid, (oh, ch) in hours_lookup.items()
    }

    sold = daily["sold_units"].astype(float).to_numpy()
    stockout_time = pd.to_datetime(daily["stockout_time"])
    date_norm = pd.to_datetime(daily["date"]).dt.normalize()

    n = len(daily)
    potential = sold.copy()
    has_stockout = ~stockout_time.isna().to_numpy()
    if not has_stockout.any():
        out = daily.copy()
        out["potential_demand"] = potential.astype("float32")
        return out

    store_ids = daily["store_id"].to_numpy()
    hours_from_open = (stockout_time - date_norm).dt.total_seconds().to_numpy() / 3600.0

    for i in range(n):
        if not has_stockout[i] or sold[i] <= 0:
            continue
        store_id = store_ids[i]
        if store_id not in hours_lookup:
            continue  # unknown store → leave sold_units as-is
        open_h, close_h = hours_lookup[store_id]
        h = hours_from_open[i]
        if h <= open_h:
            potential[i] = sold[i] * max_multiplier
            continue
        cum = cumulative_weight_at(profiles[store_id], h)
        cum = max(cum, min_cum_weight)
        mult = min(1.0 / cum, max_multiplier)
        potential[i] = sold[i] * mult

    out = daily.copy()
    out["potential_demand"] = potential.astype("float32")
    return out

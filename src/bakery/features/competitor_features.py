"""Competitor features per spec §2.4.

For each (store_id, date) we count nearby businesses that were **operating
on that date** — `license_date ≤ date AND (close_date is null OR close_date
> date)`. The radii (500m, 1km) and categories (bakery, cafe) follow the
spec's "주요 파생 변수" list. We also expose 90-day new/closed trends for
the bakery category at 1km — a slower but interpretable regime signal.

Forecast-safe: every feature uses only license/close timestamps that fall
strictly before the prediction date, so the same compute path works for
both training and inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.competitor import attach_distance_to_stores
from ..ingest.store_mapping import StationMapping

COMPETITOR_FEATURE_COLUMNS: list[str] = [
    "competitor_bakery_500m",
    "competitor_bakery_1km",
    "competitor_cafe_500m",
    "competitor_cafe_1km",
    "competitor_new_bakery_90d_1km",
    "competitor_closed_bakery_90d_1km",
]

_NEW_CLOSED_WINDOW_DAYS = 90


def compute_competitor_features(
    competitor_df: pd.DataFrame,
    mapping: dict[str, StationMapping],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-form (store_id, date, ...features) over the requested dates.

    Distance is computed once per (store, business) via
    `attach_distance_to_stores`; per-date counts then use sorted-array
    binary search so total work is O(stores × radii × dates × log N).
    """
    distance = attach_distance_to_stores(competitor_df, mapping)
    if distance.empty:
        rows = [
            {"store_id": sid, "date": d, **{c: 0 for c in COMPETITOR_FEATURE_COLUMNS}}
            for sid in mapping for d in dates
        ]
        return _coerce(pd.DataFrame(rows))

    dates_np = pd.DatetimeIndex(dates).to_numpy()
    rows: list[dict] = []
    for store_id in mapping:
        sub = distance[distance["store_id"] == store_id]
        counts = {
            "bakery_500m": _active_counts(sub, "bakery", 500, dates_np),
            "bakery_1km": _active_counts(sub, "bakery", 1000, dates_np),
            "cafe_500m": _active_counts(sub, "cafe", 500, dates_np),
            "cafe_1km": _active_counts(sub, "cafe", 1000, dates_np),
        }
        new_bakery_1km = _window_counts(sub, "bakery", 1000, dates_np, "license_date", _NEW_CLOSED_WINDOW_DAYS)
        closed_bakery_1km = _window_counts(sub, "bakery", 1000, dates_np, "close_date", _NEW_CLOSED_WINDOW_DAYS)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "store_id": store_id,
                    "date": d,
                    "competitor_bakery_500m": int(counts["bakery_500m"][i]),
                    "competitor_bakery_1km": int(counts["bakery_1km"][i]),
                    "competitor_cafe_500m": int(counts["cafe_500m"][i]),
                    "competitor_cafe_1km": int(counts["cafe_1km"][i]),
                    "competitor_new_bakery_90d_1km": int(new_bakery_1km[i]),
                    "competitor_closed_bakery_90d_1km": int(closed_bakery_1km[i]),
                }
            )
    return _coerce(pd.DataFrame(rows))


def _active_counts(
    distance_sub: pd.DataFrame, category: str, radius_m: int, dates: np.ndarray
) -> np.ndarray:
    """For each date, count businesses (category, ≤radius) operating on that date."""
    mask = (distance_sub["category"] == category) & (distance_sub["distance_m"] <= radius_m)
    filtered = distance_sub[mask]
    if filtered.empty:
        return np.zeros(len(dates), dtype="int64")
    license_sorted = np.sort(filtered["license_date"].dropna().to_numpy())
    close_sorted = np.sort(filtered["close_date"].dropna().to_numpy())
    # active = (#licensed by d) - (#closed by d)
    plus = np.searchsorted(license_sorted, dates, side="right")
    minus = np.searchsorted(close_sorted, dates, side="right")
    return plus - minus


def _window_counts(
    distance_sub: pd.DataFrame,
    category: str,
    radius_m: int,
    dates: np.ndarray,
    timestamp_col: str,
    window_days: int,
) -> np.ndarray:
    """Count events with timestamp_col ∈ [date - window_days, date)."""
    mask = (distance_sub["category"] == category) & (distance_sub["distance_m"] <= radius_m)
    filtered = distance_sub[mask].dropna(subset=[timestamp_col])
    if filtered.empty:
        return np.zeros(len(dates), dtype="int64")
    sorted_ts = np.sort(filtered[timestamp_col].to_numpy())
    cutoff = dates - np.timedelta64(window_days, "D")
    upper = np.searchsorted(sorted_ts, dates, side="left")
    lower = np.searchsorted(sorted_ts, cutoff, side="left")
    return upper - lower


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["store_id"] = df["store_id"].astype("string")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in COMPETITOR_FEATURE_COLUMNS:
        df[col] = df[col].astype("int32")
    return df[["store_id", "date", *COMPETITOR_FEATURE_COLUMNS]]


def add_competitor_features(
    df: pd.DataFrame, competitor_features: pd.DataFrame, *, date_col: str = "date"
) -> pd.DataFrame:
    """Merge (store_id, date) → 6 competitor feature columns."""
    missing = set(COMPETITOR_FEATURE_COLUMNS) - set(competitor_features.columns)
    if missing:
        raise ValueError(
            f"competitor_features missing columns: {sorted(missing)}. "
            "Call compute_competitor_features() first."
        )
    if "store_id" not in df.columns:
        raise ValueError("df missing 'store_id' — required for per-store competitor merge")
    join_cols = ["store_id", date_col]
    out = df.merge(
        competitor_features[[*join_cols, *COMPETITOR_FEATURE_COLUMNS]],
        on=join_cols, how="left",
    )
    for col in COMPETITOR_FEATURE_COLUMNS:
        out[col] = out[col].fillna(0).astype("int32")
    return out

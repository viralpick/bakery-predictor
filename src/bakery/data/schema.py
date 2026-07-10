"""Canonical data schemas. Real loader and synthetic generator must both conform."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Required columns for hourly-grain raw sales table.
HOURLY_COLUMNS: dict[str, str] = {
    "store_id": "string",
    "item_id": "string",
    "category_id": "string",
    "ts": "datetime64[ns]",
    "sold_units": "int32",
    "is_open": "bool",
    "is_stockout_hour": "bool",
}

# Required columns for daily-grain table consumed by models.
# `potential_demand` is censoring-corrected sold_units (== sold_units for
# non-stockout days; > sold_units for early-stockout days). See
# features/potential_demand.py for the correction formula. Stored as float32
# because it's a derived continuous quantity, not a count.
# ⚠️ real에서는 stockout_time 버그로 오염 — real 경로는 adjusted_demand 사용(#3 감사).
DAILY_COLUMNS: dict[str, str] = {
    "store_id": "string",
    "item_id": "string",
    "category_id": "string",
    "date": "datetime64[ns]",
    "sold_units": "int32",
    "is_stockout": "bool",
    "stockout_time": "datetime64[ns]",
    "open_hours": "int16",
    "capacity": "int32",
    "potential_demand": "float32",
}


@dataclass(frozen=True)
class StoreSpec:
    store_id: str
    profile: str
    open_hour: int
    close_hour: int
    weekend_multiplier: float


@dataclass(frozen=True)
class ItemSpec:
    item_id: str
    category_id: str
    group: str
    base_demand: float
    weekday_pattern: tuple[float, ...]
    month_pattern: tuple[float, ...]
    sparsity: float
    active_from: str | None = None
    active_until: str | None = None


def validate_daily(df: pd.DataFrame) -> None:
    missing = set(DAILY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")
    if df["date"].dt.normalize().ne(df["date"]).any():
        raise ValueError("daily frame 'date' must be normalized to midnight")
    dup = df.duplicated(subset=["store_id", "item_id", "date"])
    if dup.any():
        raise ValueError(f"daily frame has {dup.sum()} duplicate (store,item,date) rows")


def validate_hourly(df: pd.DataFrame) -> None:
    missing = set(HOURLY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"hourly frame missing columns: {sorted(missing)}")
    dup = df.duplicated(subset=["store_id", "item_id", "ts"])
    if dup.any():
        raise ValueError(f"hourly frame has {dup.sum()} duplicate (store,item,ts) rows")

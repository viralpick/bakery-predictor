"""Upstream Scenario lever (v7 S6) — what_if_driver.

Perturb a driver (weather/calendar), re-run the real LightGBM forecast, and
propagate the demand change through to stockout risk / cost. Palantir Scenario
(model-on-object) homolog. Read-only: this never mutates ontology state.

See docs/superpowers/specs/2026-06-30-whatif-driver-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WEATHER_DRIVERS = frozenset({"is_rain", "is_snow"})
CALENDAR_DRIVERS = frozenset({"is_weekend", "is_off_day", "is_public_holiday"})
VALID_DRIVERS = WEATHER_DRIVERS | CALENDAR_DRIVERS

_WEATHER_LINK = "dailysales_observed_on_weather"
_CALENDAR_LINK = "dailysales_observed_on_calendar"
_ITEM_LINK = "item_sold_as_dailysales"


@dataclass(frozen=True)
class WhatIfDriverResult:
    store_id: str
    item_id: str
    driver_overrides: dict[str, float]
    before_demand: float
    after_demand: float
    demand_delta: float
    before_p_stockout: float
    after_p_stockout: float
    before_expected_cost: float
    after_expected_cost: float
    out_of_support: bool
    propagation_path: tuple[str, ...]


def _validate_drivers(driver_overrides: dict) -> None:
    if not driver_overrides:
        raise ValueError("driver_overrides is empty; provide at least one driver")
    unknown = set(driver_overrides) - VALID_DRIVERS
    if unknown:
        raise ValueError(f"unknown driver(s): {sorted(unknown)}; valid: {sorted(VALID_DRIVERS)}")


def _propagation_path(driver_overrides: dict) -> tuple[str, ...]:
    path = []
    if any(d in WEATHER_DRIVERS for d in driver_overrides):
        path.append(_WEATHER_LINK)
    if any(d in CALENDAR_DRIVERS for d in driver_overrides):
        path.append(_CALENDAR_LINK)
    path.append(_ITEM_LINK)
    return tuple(path)


def _count_support(enriched: pd.DataFrame, store_id: str, driver_overrides: dict) -> int:
    sub = enriched[enriched["store_id"] == store_id]
    mask = pd.Series(True, index=sub.index)
    for col, val in driver_overrides.items():
        mask &= sub[col] == val
    return int(mask.sum())

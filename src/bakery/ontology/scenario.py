"""Upstream Scenario lever (v7 S6) — what_if_driver.

Perturb a driver (weather/calendar), re-run the real LightGBM forecast, and
propagate the demand change through to stockout risk / cost. Palantir Scenario
(model-on-object) homolog. Read-only: this never mutates ontology state.

See docs/superpowers/specs/2026-06-30-whatif-driver-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features.calendar_features import add_calendar_features
from ..features.weather_features import add_weather_features
from ..models.lightgbm_regressor import GlobalLGBM

WEATHER_DRIVERS = frozenset({"is_rain", "is_snow"})
CALENDAR_DRIVERS = frozenset({"is_public_holiday"})
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


def _build_enriched(daily: pd.DataFrame, calendar: pd.DataFrame,
                    weather: pd.DataFrame) -> pd.DataFrame:
    """Merge the separate ontology frames into the single frame GlobalLGBM needs."""
    return add_weather_features(add_calendar_features(daily, calendar), weather)


def _fit_demand_model(enriched: pd.DataFrame, train_cutoff: str,
                      feature_set: str = "v2") -> GlobalLGBM:
    """Fit on rows strictly before train_cutoff (leakage rule). Caller injects cutoff."""
    train = enriched[pd.to_datetime(enriched["date"]) < pd.Timestamp(train_cutoff)]
    if train.empty:
        raise ValueError(f"no training rows before cutoff {train_cutoff}")
    return GlobalLGBM(feature_set=feature_set).fit(train)


def _period_item_rows(enriched: pd.DataFrame, store_id: str, item_id: str,
                      period: tuple[str, str]) -> pd.DataFrame:
    dates = pd.to_datetime(enriched["date"])
    mask = ((enriched["store_id"] == store_id) & (enriched["item_id"] == item_id)
            & (dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1])))
    rows = enriched.loc[mask]
    if rows.empty:
        raise ValueError(f"no rows for store={store_id} item={item_id} in {period}")
    return rows


def _predict_demand(model: GlobalLGBM, target: pd.DataFrame) -> float:
    """Mean predicted demand over the target rows (matches _item_demand_points mean)."""
    return float(model.predict(target).mean())

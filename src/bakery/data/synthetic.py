"""Synthetic bakery sales data with explicit DGP signals.

We hand-craft weekday/seasonal/holiday/store effects so that:
- baselines (seasonal naive, MA) can recover most of the signal,
- a global LightGBM with lag/rolling features should beat them.

Censored demand is simulated explicitly: each (store,item,day) has a finite
capacity; once cumulative hourly demand exceeds capacity, remaining hours are
marked is_stockout_hour=True and sold_units=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.competitor_features import compute_competitor_features
from ..features.consumption_features import compute_store_consumption_features
from ..features.living_population_features import compute_store_living_features
from ..features.population_features import compute_store_population_features
from ..features.potential_demand import StoreHours, attach_potential_demand
from ..ingest.store_mapping import StationMapping, load_store_mapping
from .calendar import build_calendar_daily, validate_calendar
from .competitor import build_synthetic_competitor, validate_competitor
from .consumption import build_synthetic_consumption, validate_consumption
from .living_population import build_synthetic_living_population, validate_living_population
from .population import build_synthetic_population, validate_population
from .schema import DAILY_COLUMNS, ItemSpec, StoreSpec, validate_daily, validate_hourly
from .weather import build_synthetic_weather, validate_weather


@dataclass(frozen=True)
class SyntheticBundle:
    """All synthetic frames produced together so DGP signals stay aligned."""

    hourly: pd.DataFrame
    daily: pd.DataFrame
    weather: pd.DataFrame
    calendar: pd.DataFrame
    competitor: pd.DataFrame
    living_population: pd.DataFrame
    population: pd.DataFrame
    consumption: pd.DataFrame


def default_stores() -> list[StoreSpec]:
    return [
        StoreSpec("store_A", "residential", 7, 22, 1.20),
        StoreSpec("store_B", "transit", 7, 23, 1.00),
        StoreSpec("store_C", "office", 7, 21, 0.55),
    ]


def default_items() -> list[ItemSpec]:
    flat = (1.0,) * 12
    xmas = (0.9, 0.9, 0.9, 0.95, 1.05, 1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.45)
    summer = (0.4, 0.4, 0.4, 0.6, 1.0, 1.5, 1.7, 1.6, 1.1, 0.6, 0.3, 0.2)
    winter = (1.4, 1.2, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 1.2, 1.5)
    spring = (0.8, 0.9, 1.0, 1.1, 1.5, 1.2, 0.9, 0.7, 0.6, 0.6, 0.7, 0.8)
    stable = (1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2)
    weekend = (0.85, 0.85, 0.85, 0.9, 1.1, 1.6, 1.4)
    office = (1.25, 1.25, 1.25, 1.25, 1.1, 0.55, 0.55)
    return [
        # A group: stable everyday bread/pastry
        ItemSpec("item_A01", "bread", "A", 70, stable, flat, 0.0),
        ItemSpec("item_A02", "bread", "A", 55, stable, flat, 0.0),
        ItemSpec("item_A03", "bread", "A", 45, stable, flat, 0.0),
        ItemSpec("item_A04", "pastry", "A", 60, stable, flat, 0.02),
        ItemSpec("item_A05", "pastry", "A", 50, weekend, flat, 0.0),
        ItemSpec("item_A06", "pastry", "A", 40, stable, flat, 0.02),
        ItemSpec("item_A07", "pastry", "A", 35, weekend, flat, 0.0),
        ItemSpec("item_A08", "bread", "A", 30, stable, flat, 0.05),
        ItemSpec("item_A09", "bread", "A", 28, office, flat, 0.05),
        ItemSpec("item_A10", "pastry", "A", 32, weekend, flat, 0.05),
        # B group: more volatile or weekend-led
        ItemSpec("item_B01", "cake", "B", 18, weekend, xmas, 0.10),
        ItemSpec("item_B02", "cake", "B", 14, weekend, xmas, 0.15),
        ItemSpec("item_B03", "sandwich", "B", 22, office, flat, 0.10),
        ItemSpec("item_B04", "sandwich", "B", 16, office, flat, 0.15),
        ItemSpec("item_B05", "sweets", "B", 12, weekend, flat, 0.20),
        # C group: sparse / seasonal / limited
        ItemSpec("item_C01", "seasonal", "C", 14, weekend, spring, 0.10, "2024-01-01", "2024-05-31"),
        ItemSpec("item_C02", "seasonal", "C", 14, weekend, spring, 0.10, "2025-01-01", "2025-05-31"),
        ItemSpec("item_C03", "seasonal", "C", 10, stable, summer, 0.15),
        ItemSpec("item_C04", "seasonal", "C", 12, stable, winter, 0.15),
        ItemSpec("item_C05", "sweets", "C", 6, weekend, flat, 0.45),
    ]


def _store_group_multiplier(store: StoreSpec, item: ItemSpec) -> float:
    # Residential favors A, transit boosts everything mildly, office boosts B sandwiches.
    table = {
        ("residential", "A"): 1.10,
        ("residential", "B"): 0.95,
        ("residential", "C"): 1.00,
        ("transit", "A"): 1.05,
        ("transit", "B"): 1.10,
        ("transit", "C"): 1.05,
        ("office", "A"): 0.90,
        ("office", "B"): 1.25,
        ("office", "C"): 0.70,
    }
    return table[(store.profile, item.group)]


def _hour_weights(open_hour: int, close_hour: int) -> np.ndarray:
    """Bakery-shaped hourly demand profile, normalized to sum=1 over open hours."""
    hours = np.arange(24, dtype=float)
    profile = np.zeros(24)
    for peak, width, height in [(9, 1.3, 1.0), (13, 1.0, 0.9), (16, 1.2, 1.1), (19, 1.0, 0.6)]:
        profile += height * np.exp(-0.5 * ((hours - peak) / width) ** 2)
    mask = (hours >= open_hour) & (hours < close_hour)
    profile = profile * mask
    total = profile.sum()
    if total <= 0:
        raise ValueError("hour profile collapsed to zero — check open/close hours")
    return profile / total


def _is_item_active(item: ItemSpec, date: pd.Timestamp) -> bool:
    if item.active_from and date < pd.Timestamp(item.active_from):
        return False
    return not (item.active_until and date > pd.Timestamp(item.active_until))


def _daily_lambda(
    store: StoreSpec,
    item: ItemSpec,
    date: pd.Timestamp,
    external_boost: float,
    trend: float,
) -> float:
    if not _is_item_active(item, date):
        return 0.0
    base = item.base_demand * _store_group_multiplier(store, item)
    weekday_mult = item.weekday_pattern[date.weekday()]
    month_mult = item.month_pattern[date.month - 1]
    weekend_mult = store.weekend_multiplier if date.weekday() >= 5 else 1.0
    return base * weekday_mult * month_mult * weekend_mult * external_boost * trend


def _simulate_day(
    store: StoreSpec,
    item: ItemSpec,
    date: pd.Timestamp,
    lam: float,
    rng: np.random.Generator,
    hour_weights: np.ndarray,
) -> list[dict]:
    """Simulate one (store,item,date) day at hourly grain. Honors capacity censoring."""
    if lam <= 0 or rng.random() < item.sparsity:
        return _empty_day(store, item, date, hour_weights, capacity=0)
    capacity = int(round(lam * rng.uniform(0.85, 1.25)))
    hourly_demand = rng.poisson(lam * hour_weights).astype(np.int32)
    rows: list[dict] = []
    sold_so_far = 0
    stocked_out = False
    for hour in range(24):
        if hour < store.open_hour or hour >= store.close_hour:
            rows.append(_hour_row(store, item, date, hour, 0, capacity, is_open=False, stockout=False))
            continue
        if stocked_out:
            rows.append(_hour_row(store, item, date, hour, 0, capacity, is_open=True, stockout=True))
            continue
        demand = int(hourly_demand[hour])
        remaining = max(capacity - sold_so_far, 0)
        sold = min(demand, remaining)
        sold_so_far += sold
        stocked_out = sold_so_far >= capacity and demand > sold
        rows.append(
            _hour_row(store, item, date, hour, sold, capacity, is_open=True, stockout=stocked_out)
        )
    return rows


def _empty_day(
    store: StoreSpec,
    item: ItemSpec,
    date: pd.Timestamp,
    hour_weights: np.ndarray,
    *,
    capacity: int,
) -> list[dict]:
    rows = []
    for hour in range(24):
        is_open = store.open_hour <= hour < store.close_hour
        rows.append(_hour_row(store, item, date, hour, 0, capacity, is_open=is_open, stockout=False))
    return rows


def _hour_row(
    store: StoreSpec,
    item: ItemSpec,
    date: pd.Timestamp,
    hour: int,
    sold: int,
    capacity: int,
    *,
    is_open: bool,
    stockout: bool,
) -> dict:
    return {
        "store_id": store.store_id,
        "item_id": item.item_id,
        "category_id": item.category_id,
        "ts": date + pd.Timedelta(hours=hour),
        "sold_units": np.int32(sold),
        "is_open": is_open,
        "is_stockout_hour": stockout,
        "capacity": np.int32(capacity),
    }


def generate_synthetic_bundle(
    *,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    stores: list[StoreSpec] | None = None,
    items: list[ItemSpec] | None = None,
    weather: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
    competitor: pd.DataFrame | None = None,
    living_population: pd.DataFrame | None = None,
    population: pd.DataFrame | None = None,
    consumption: pd.DataFrame | None = None,
    store_mapping: dict[str, StationMapping] | None = None,
) -> SyntheticBundle:
    """Build sales, weather, calendar, and subway together so DGP signals align.

    Sales DGP reads from calendar (holiday/streak/event), weather (heat/cold/
    rain/snow), and subway (foot-traffic boost for transit/office stores).
    Each external frame can be injected to use real data instead — the model
    then learns against the same signal it'll see at inference time.
    """
    stores = stores or default_stores()
    items = items or default_items()
    mapping = store_mapping or load_store_mapping()
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    if weather is None:
        weather = build_synthetic_weather(
            start, end, store_ids=[s.store_id for s in stores], seed=seed
        )
    if calendar is None:
        calendar = build_calendar_daily(start, end)
    if competitor is None:
        competitor = build_synthetic_competitor(mapping=mapping, seed=seed)
    if living_population is None:
        living_population = build_synthetic_living_population(start, end, mapping=mapping, seed=seed)
    if population is None:
        population = build_synthetic_population(mapping=mapping)
    if consumption is None:
        consumption = build_synthetic_consumption(mapping=mapping)
    validate_weather(weather)
    validate_calendar(calendar)
    validate_competitor(competitor)
    validate_living_population(living_population)
    validate_population(population)
    validate_consumption(consumption)
    calendar_lookup = calendar.set_index("date").to_dict(orient="index")
    weather_lookup = weather.set_index(["store_id", "date"]).to_dict(orient="index")
    competitor_features = compute_competitor_features(competitor, mapping, dates)
    competitor_lookup = competitor_features.set_index(["store_id", "date"]).to_dict(orient="index")
    competitor_baselines = _competitor_baselines(competitor_features)
    living_static = compute_store_living_features(living_population, mapping)
    population_static = compute_store_population_features(population, mapping)
    consumption_static = compute_store_consumption_features(consumption, mapping)
    static_lookup = _merge_static_lookups(living_static, population_static, consumption_static)
    weights_cache = {(s.open_hour, s.close_hour): _hour_weights(s.open_hour, s.close_hour) for s in stores}

    rows: list[dict] = []
    for store in stores:
        weights = weights_cache[(store.open_hour, store.close_hour)]
        static_meta = static_lookup.get(store.store_id, {})
        for item in items:
            rows.extend(
                _simulate_item(
                    store, item, dates, rng, calendar_lookup, weather_lookup,
                    competitor_lookup, competitor_baselines,
                    static_meta, weights,
                )
            )
    hourly = pd.DataFrame(rows)
    hourly = _coerce_hourly_dtypes(hourly)
    validate_hourly(hourly)
    daily = aggregate_to_daily(hourly)
    store_hours = [StoreHours(s.store_id, s.open_hour, s.close_hour) for s in stores]
    daily = attach_potential_demand(daily, store_hours)
    daily = _coerce_daily_dtypes(daily)
    validate_daily(daily)
    return SyntheticBundle(
        hourly=hourly, daily=daily, weather=weather, calendar=calendar,
        competitor=competitor, living_population=living_population,
        population=population, consumption=consumption,
    )


def _merge_static_lookups(*frames: pd.DataFrame) -> dict[str, dict]:
    """Combine per-store static feature frames into a single store_id → dict lookup."""
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="store_id", how="outer")
    out: dict[str, dict] = {}
    for _, row in merged.iterrows():
        sid = str(row["store_id"])
        out[sid] = {k: row[k] for k in merged.columns if k != "store_id"}
    return out


def _competitor_baselines(competitor_features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-store mean count of bakeries/cafes in 1km — used to scale excess to a ratio."""
    out: dict[str, dict[str, float]] = {}
    for store_id, group in competitor_features.groupby("store_id"):
        out[str(store_id)] = {
            "bakery_1km": float(group["competitor_bakery_1km"].mean()),
            "cafe_1km": float(group["competitor_cafe_1km"].mean()),
        }
    return out


def generate_synthetic(
    *,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    stores: list[StoreSpec] | None = None,
    items: list[ItemSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible (hourly, daily) entry point."""
    bundle = generate_synthetic_bundle(
        start=start, end=end, seed=seed, stores=stores, items=items
    )
    return bundle.hourly, bundle.daily


def _simulate_item(
    store: StoreSpec,
    item: ItemSpec,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    calendar_lookup: dict,
    weather_lookup: dict,
    competitor_lookup: dict,
    competitor_baselines: dict,
    static_meta: dict,
    weights: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    start_year = dates.min().year
    comp_baseline = competitor_baselines.get(store.store_id, {})
    static_mult = _static_external_multiplier(static_meta, item)
    for date in dates:
        years_in = (date - pd.Timestamp(f"{start_year}-01-01")).days / 365.25
        trend = 1.0 + 0.03 * years_in
        cal_row = calendar_lookup[date]
        w_row = weather_lookup[(store.store_id, date)]
        comp_row = competitor_lookup.get((store.store_id, date))
        boost = (
            _calendar_boost(cal_row, item)
            * _weather_boost(w_row, item)
            * _competitor_boost(comp_row, store, item, comp_baseline)
            * static_mult
        )
        lam = _daily_lambda(store, item, date, boost, trend)
        rows.extend(_simulate_day(store, item, date, lam, rng, weights))
    return rows


def _static_external_multiplier(meta: dict, item: ItemSpec) -> float:
    """Time-independent boost combining living-pop, age, and consumption.

    Centered so a 'typical' Seoul dong yields ~1.0, clipped to [0.7, 1.3].
    """
    if not meta:
        return 1.0
    # Living population — log-scaled so a 32k vs 14k dong differs ~0.4 log units.
    daily_avg = float(meta.get("living_pop_daily_avg", 18_000.0))
    living_lift = (np.log(max(daily_avg, 1.0)) - np.log(18_000.0)) * 0.15
    # Lunch share & weekend ratio — office hub vs destination signal
    lunch_share = float(meta.get("living_pop_lunch_share", 1.10))
    weekend_ratio = float(meta.get("living_pop_weekend_ratio", 1.00))
    if item.category_id == "sandwich":
        living_lift += (lunch_share - 1.10) * 0.6
    if item.category_id in ("cake", "sweets"):
        living_lift += (weekend_ratio - 1.00) * 0.2
    # Age composition — younger demographics buy more cake/sweets, older more bread
    age_lift = 0.0
    if item.category_id in ("cake", "sweets"):
        age_lift += (float(meta.get("pop_share_20_39", 0.30)) - 0.30) * 0.6
    if item.category_id == "bread":
        age_lift += (float(meta.get("pop_share_60_plus", 0.20)) - 0.20) * 0.4
    if item.category_id in ("cake", "sweets", "pastry"):
        age_lift += (float(meta.get("pop_share_30_49_female", 0.18)) - 0.18) * 0.5
    if item.group == "A":
        age_lift += (float(meta.get("pop_share_0_9", 0.07)) - 0.07) * 0.3
    # Consumption baseline — log scale, modest lift
    consumption_log = float(meta.get("consumption_total_log", np.log(1.5e10)))
    consumption_lift = (consumption_log - np.log(1.5e10)) * 0.08
    food_retail_log = float(meta.get("consumption_food_retail_log", np.log(5e9)))
    consumption_lift += (food_retail_log - np.log(5e9)) * 0.05
    return max(0.7, min(1.3, 1.0 + living_lift + age_lift + consumption_lift))


def _competitor_boost(
    comp_row: dict | None, store: StoreSpec, item: ItemSpec, baseline: dict
) -> float:
    """Sales modulation from nearby competitive density.

    Residential stores feel competitor density most (loyal customer base is
    thin); transit hubs absorb the impact via raw foot traffic. Bakeries at
    1km matter more than cafes by a 0.7/0.3 split.
    """
    if comp_row is None or not baseline:
        return 1.0
    bakery_base = max(baseline.get("bakery_1km", 1.0), 1.0)
    cafe_base = max(baseline.get("cafe_1km", 1.0), 1.0)
    excess_bakery = comp_row.get("competitor_bakery_1km", 0) / bakery_base - 1.0
    excess_cafe = comp_row.get("competitor_cafe_1km", 0) / cafe_base - 1.0
    sensitivity = _competitor_sensitivity(store, item)
    impact = -sensitivity * (0.7 * excess_bakery + 0.3 * excess_cafe)
    return max(0.85, min(1.15, 1.0 + impact))


def _competitor_sensitivity(store: StoreSpec, item: ItemSpec) -> float:
    """How much an item's sales react to a deviation in nearby competitor density."""
    if store.profile == "residential":
        return 0.12 if item.group == "A" else 0.08
    if store.profile == "office":
        return 0.05
    return 0.04  # transit hub — high foot traffic dilutes competitor pressure


def _calendar_boost(cal_row: dict, item: ItemSpec) -> float:
    """Sales modulation by calendar context. Mirrors observable bakery patterns."""
    boost = 1.0
    if cal_row["is_public_holiday"]:
        # Public holidays nudge daily-staples down (commuters away) but specialty cakes up.
        boost *= 0.80 if item.group == "A" else 0.95
    if cal_row["is_day_before_off"]:
        boost *= 1.15  # day-before-off stockpiling
    if (cal_row["is_xmas_eve"] or cal_row["is_xmas"]) and item.category_id in ("cake", "sweets"):
        boost *= 2.5
    if (cal_row["is_valentine"] or cal_row["is_white_day"]) and item.category_id in ("cake", "sweets"):
        boost *= 2.0
    if cal_row["is_pepero"] and item.category_id == "sweets":
        boost *= 1.8
    if cal_row["is_children_day"] and item.category_id in ("cake", "sweets"):
        boost *= 1.5
    return boost




def _weather_boost(w_row: dict, item: ItemSpec) -> float:
    """Sales modulation by weather. Thresholds match weather_features.py."""
    boost = 1.0
    avg_temp = float(w_row["avg_temp"])
    if avg_temp >= 28.0:  # heatwave
        if item.group == "B" or item.category_id == "sweets":
            boost *= 1.20
        else:
            boost *= 0.95
    if avg_temp <= -5.0:  # coldsnap
        if item.category_id in ("bread", "sandwich"):
            boost *= 1.10
        else:
            boost *= 0.95
    precip = float(w_row["precipitation_mm"])
    if precip >= 10.0:
        boost *= 0.90  # heavy rain damps traffic
    elif w_row["is_rain"] == 1:
        boost *= 0.97
    snow = float(w_row["snow_depth_cm"])
    if snow >= 5.0:
        boost *= 0.85
    elif w_row["is_snow"] == 1:
        boost *= 0.95
    return boost


def _coerce_hourly_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["store_id"] = df["store_id"].astype("string")
    df["item_id"] = df["item_id"].astype("string")
    df["category_id"] = df["category_id"].astype("string")
    df["ts"] = pd.to_datetime(df["ts"])
    df["sold_units"] = df["sold_units"].astype("int32")
    df["is_open"] = df["is_open"].astype(bool)
    df["is_stockout_hour"] = df["is_stockout_hour"].astype(bool)
    df["capacity"] = df["capacity"].astype("int32")
    return df


def aggregate_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly → daily grain, preserving stockout timing as datetime."""
    df = hourly.copy()
    df["date"] = df["ts"].dt.normalize()
    keys = ["store_id", "item_id", "category_id", "date"]
    grouped = df.groupby(keys, observed=True)
    daily = grouped.agg(
        sold_units=("sold_units", "sum"),
        open_hours=("is_open", "sum"),
        stockout_hours=("is_stockout_hour", "sum"),
        capacity=("capacity", "max"),
    ).reset_index()
    daily["is_stockout"] = daily["stockout_hours"] > 0
    daily = _attach_stockout_time(df, daily, keys)
    daily = daily.drop(columns=["stockout_hours"])
    return _coerce_daily_dtypes(daily)


def _attach_stockout_time(hourly: pd.DataFrame, daily: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    so = hourly.loc[hourly["is_stockout_hour"], keys + ["ts"]]
    first = so.groupby(keys, observed=True)["ts"].min().rename("stockout_time").reset_index()
    return daily.merge(first, on=keys, how="left")


def _coerce_daily_dtypes(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["store_id"] = daily["store_id"].astype("string")
    daily["item_id"] = daily["item_id"].astype("string")
    daily["category_id"] = daily["category_id"].astype("string")
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["sold_units"] = daily["sold_units"].astype("int32")
    daily["is_stockout"] = daily["is_stockout"].astype(bool)
    if "stockout_time" not in daily.columns:
        daily["stockout_time"] = pd.NaT
    daily["stockout_time"] = pd.to_datetime(daily["stockout_time"])
    daily["open_hours"] = daily["open_hours"].astype("int16")
    daily["capacity"] = daily["capacity"].astype("int32")
    if "potential_demand" not in daily.columns:
        daily["potential_demand"] = daily["sold_units"].astype("float32")
    daily["potential_demand"] = daily["potential_demand"].astype("float32")
    daily = daily[list(DAILY_COLUMNS.keys())]
    return daily.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

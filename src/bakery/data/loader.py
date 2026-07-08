"""Real-data entry point. Replace bodies when production data arrives.

This module exists so the rest of the pipeline never imports synthetic.py
directly. The contract for both synthetic and real loaders:

- `daily` conforms to DAILY_COLUMNS (schema.py).
- `weather` conforms to WEATHER_DAILY_COLUMNS (weather.py).
- `calendar` conforms to CALENDAR_DAILY_COLUMNS (calendar.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import EXTERNAL_DATA_DIR
from ..ingest.store_mapping import load_store_mapping
from .calendar import load_calendar_from_local, validate_calendar
from .competitor import (
    build_synthetic_competitor,
    load_competitor_from_local,
    validate_competitor,
)
from .consumption import (
    build_synthetic_consumption,
    load_consumption_from_local,
    validate_consumption,
)
from .living_population import (
    build_synthetic_living_population,
    load_living_population_from_local,
    validate_living_population,
)
from .population import (
    build_synthetic_population,
    load_population_from_local,
    validate_population,
)
from .schema import validate_daily, validate_hourly
from .weather import load_weather_from_local, validate_weather


@dataclass(frozen=True)
class DailyDataset:
    """Aligned daily-grain frames for v1+ modeling."""

    daily: pd.DataFrame
    weather: pd.DataFrame
    calendar: pd.DataFrame
    competitor: pd.DataFrame
    living_population: pd.DataFrame
    population: pd.DataFrame
    consumption: pd.DataFrame


def load_dataset(
    source: str = "synthetic",
    *,
    data_dir: Path | None = None,
    external_dir: Path | None = None,
    mapping_path: Path | None = None,
) -> DailyDataset:
    """Return aligned (daily, weather, calendar) frames.

    Sources:
    - "synthetic": fully generated via synthetic.generate_synthetic_bundle.
    - "parquet":   read synthetic outputs previously written to `data_dir`.
    - "real":      sales stay synthetic (no real store data yet) but calendar
                   and weather come from `external_dir/{calendar_raw,
                   weather_observed}.parquet` ingested via the data.go.kr APIs.
                   Per-store station fan-out uses `mapping_path` (yaml).
    """
    if source == "synthetic":
        from .synthetic import generate_synthetic_bundle

        bundle = generate_synthetic_bundle()
        return DailyDataset(
            daily=bundle.daily, weather=bundle.weather,
            calendar=bundle.calendar, competitor=bundle.competitor,
            living_population=bundle.living_population,
            population=bundle.population, consumption=bundle.consumption,
        )
    if source == "parquet":
        if data_dir is None:
            raise ValueError("parquet source requires data_dir=")
        daily = pd.read_parquet(data_dir / "daily.parquet")
        validate_daily(daily)
        weather = pd.read_parquet(data_dir / "weather.parquet")
        validate_weather(weather)
        calendar = pd.read_parquet(data_dir / "calendar.parquet")
        validate_calendar(calendar)
        competitor = pd.read_parquet(data_dir / "competitor.parquet")
        validate_competitor(competitor)
        living_population = pd.read_parquet(data_dir / "living_population.parquet")
        validate_living_population(living_population)
        population = pd.read_parquet(data_dir / "population.parquet")
        validate_population(population)
        consumption = pd.read_parquet(data_dir / "consumption.parquet")
        validate_consumption(consumption)
        return DailyDataset(
            daily=daily, weather=weather, calendar=calendar, competitor=competitor,
            living_population=living_population, population=population,
            consumption=consumption,
        )
    if source == "real":
        return _load_real_dataset(
            external_dir=external_dir or EXTERNAL_DATA_DIR, mapping_path=mapping_path
        )
    raise ValueError(f"unknown source: {source}")


_INTERNAL_DAILY_PATH = Path("data/internal/bonavi_daily.parquet")


def _load_real_dataset(*, external_dir: Path, mapping_path: Path | None) -> DailyDataset:
    """Sales come from `data/internal/bonavi_daily.parquet` when present; otherwise
    fall back to synthetic-but-DGP-aligned sales so the rest of the pipeline still
    exercises calendar + weather + external sources.
    """
    from .synthetic import generate_synthetic_bundle

    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    calendar = load_calendar_from_local(external_dir / "calendar_raw.parquet", start=start, end=end)
    validate_calendar(calendar)
    mapping = load_store_mapping(mapping_path)
    weather = load_weather_from_local(
        external_dir / "weather_observed.parquet", mapping=mapping, start=start, end=end
    )
    validate_weather(weather)
    competitor_path = external_dir / "competitor_raw.parquet"
    if competitor_path.exists():
        competitor = load_competitor_from_local(competitor_path, categories=["bakery", "cafe"])
    else:
        competitor = build_synthetic_competitor(mapping=mapping)
    validate_competitor(competitor)
    dong_codes = sorted({s["admin_dong_code"] for s in mapping.values()})
    living_pop_path = external_dir / "living_population.parquet"
    if living_pop_path.exists():
        living_population = load_living_population_from_local(
            living_pop_path, admin_dong_codes=dong_codes, start=start, end=end,
        )
    else:
        living_population = build_synthetic_living_population(start, end, mapping=mapping)
    validate_living_population(living_population)
    population_path = external_dir / "population.parquet"
    if population_path.exists():
        population = load_population_from_local(population_path, admin_dong_codes=dong_codes)
    else:
        population = build_synthetic_population(mapping=mapping)
    validate_population(population)
    consumption_path = external_dir / "consumption.parquet"
    if consumption_path.exists():
        consumption = load_consumption_from_local(consumption_path, admin_dong_codes=dong_codes)
    else:
        consumption = build_synthetic_consumption(mapping=mapping)
    validate_consumption(consumption)
    if _INTERNAL_DAILY_PATH.exists():
        daily = pd.read_parquet(_INTERNAL_DAILY_PATH)
        # Clip to the training window so backtest folds line up with external data.
        daily = daily[(daily["date"] >= start) & (daily["date"] <= end)].reset_index(drop=True)
        validate_daily(daily)
    else:
        bundle = generate_synthetic_bundle(
            start=str(start.date()), end=str(end.date()),
            weather=weather, calendar=calendar, competitor=competitor,
            living_population=living_population, population=population, consumption=consumption,
            store_mapping=mapping,
        )
        daily = bundle.daily
    return DailyDataset(
        daily=daily, weather=weather, calendar=calendar, competitor=competitor,
        living_population=living_population, population=population, consumption=consumption,
    )


def load_daily(source: str = "synthetic", path: Path | None = None) -> pd.DataFrame:
    """Daily-only convenience for callers that don't need calendar/weather."""
    if source == "parquet":
        if path is None:
            raise ValueError("parquet source requires path=")
        daily = pd.read_parquet(path)
        validate_daily(daily)
        return daily
    return load_dataset(source=source).daily


def load_hourly(source: str = "synthetic", path: Path | None = None) -> pd.DataFrame:
    if source == "synthetic":
        from .synthetic import generate_synthetic

        hourly, _ = generate_synthetic()
        return hourly
    if source == "parquet":
        if path is None:
            raise ValueError("parquet source requires path=")
        hourly = pd.read_parquet(path)
        validate_hourly(hourly)
        return hourly
    raise ValueError(f"unknown source: {source}")

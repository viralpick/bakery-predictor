"""Per-store weather daily frame.

Schema is long-form keyed by `(store_id, date)`. PoC's three stores all sit in
greater Seoul, so synthetic weather shares a base seasonal frame plus a small
per-store perturbation (±0.5°C, ±3% humidity) — meant to exercise the
per-store wiring, not to claim micro-climate realism. Real loader fans out
each ASOS station to every store that maps to it.

In production, learning uses observed values and forecasting uses forecast
values — this PoC uses one observed frame for backtest and a forecast frame
(when `--use-forecast`) for predict-next-week.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.store_mapping import StationMapping

WEATHER_DAILY_COLUMNS: dict[str, str] = {
    "store_id": "string",
    "date": "datetime64[ns]",
    "avg_temp": "float32",
    "max_temp": "float32",
    "min_temp": "float32",
    "diurnal_range": "float32",
    "humidity": "float32",
    "precipitation_mm": "float32",
    "is_rain": "int8",
    "snow_depth_cm": "float32",
    "is_snow": "int8",
    "sunshine_hours": "float32",
}


def build_synthetic_weather(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    store_ids: Iterable[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Build a long-form weather frame for [start, end] × store_ids.

    Each store gets the same seasonal/precipitation base (one Seoul) but with
    small temperature/humidity perturbations so per-store wiring is exercised.
    """
    store_list = list(store_ids)
    if not store_list:
        raise ValueError("store_ids cannot be empty")
    base = _build_base_weather(start, end, seed=seed)
    frames: list[pd.DataFrame] = []
    for i, sid in enumerate(store_list):
        rng = np.random.default_rng(seed + 1000 + i)
        out = base.copy()
        n = len(out)
        # micro-climate perturbations — small enough that DGP regimes still trigger
        t_noise = rng.normal(0, 0.5, n).astype("float32")
        out["avg_temp"] = (out["avg_temp"] + t_noise).astype("float32")
        out["max_temp"] = (out["max_temp"] + t_noise).astype("float32")
        out["min_temp"] = (out["min_temp"] + t_noise).astype("float32")
        out["diurnal_range"] = (out["max_temp"] - out["min_temp"]).astype("float32")
        h_noise = rng.normal(0, 3, n)
        out["humidity"] = np.clip(out["humidity"] + h_noise, 20.0, 100.0).astype("float32")
        out.insert(0, "store_id", sid)
        frames.append(out)
    combined = pd.concat(frames, ignore_index=True)
    return _coerce_dtypes(combined)


def _build_base_weather(
    start: str | pd.Timestamp, end: str | pd.Timestamp, *, seed: int
) -> pd.DataFrame:
    """Build the seasonal base frame shared by every store (no store_id yet)."""
    dates = pd.date_range(start, end, freq="D")
    if len(dates) == 0:
        raise ValueError(f"empty date range: {start} → {end}")
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy()
    n = len(dates)

    # Annual temperature cycle: ~-2°C in Jan, ~26°C in late Jul.
    avg_temp = 12.5 - 14.0 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 1.8, n)
    diurnal = np.clip(8.0 + rng.normal(0, 1.5, n), 3.0, 15.0)
    max_temp = avg_temp + diurnal / 2
    min_temp = avg_temp - diurnal / 2

    humidity = 65 + 12 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 5, n)
    humidity = np.clip(humidity, 20.0, 100.0)

    summer_intensity = np.maximum(0.0, np.sin(2 * np.pi * (doy - 150) / 365))
    rain_prob = 0.12 + 0.38 * summer_intensity
    is_rain = (rng.random(n) < rain_prob).astype(np.int8)
    precipitation = np.where(is_rain == 1, rng.exponential(8.0, n), 0.0).astype(np.float32)

    cold = (avg_temp < 1.0).astype(int)
    snow_prob = 0.06 * cold
    is_snow = (rng.random(n) < snow_prob).astype(np.int8)
    snow_depth = np.where(is_snow == 1, rng.exponential(2.5, n), 0.0).astype(np.float32)

    base_sun = 6.0 + 3.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    sunshine = base_sun - 4.0 * is_rain - 3.0 * is_snow + rng.normal(0, 0.6, n)
    sunshine = np.clip(sunshine, 0.0, 13.0)

    return pd.DataFrame(
        {
            "date": dates,
            "avg_temp": avg_temp.astype(np.float32),
            "max_temp": max_temp.astype(np.float32),
            "min_temp": min_temp.astype(np.float32),
            "diurnal_range": (max_temp - min_temp).astype(np.float32),
            "humidity": humidity.astype(np.float32),
            "precipitation_mm": precipitation,
            "is_rain": is_rain,
            "snow_depth_cm": snow_depth,
            "is_snow": is_snow,
            "sunshine_hours": sunshine.astype(np.float32),
        }
    )


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["store_id"] = df["store_id"].astype("string")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col, dtype in WEATHER_DAILY_COLUMNS.items():
        if col in ("store_id", "date"):
            continue
        df[col] = df[col].astype(dtype)
    return df[list(WEATHER_DAILY_COLUMNS.keys())]


def load_weather_from_local(
    parquet_path: Path | str,
    *,
    mapping: dict[str, StationMapping],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Map raw multi-station ASOS parquet into long-form WEATHER_DAILY_COLUMNS.

    Stores sharing one station get an identical copy of that station's frame
    with `store_id` swapped — PoC stops there. Real micro-climate splitting
    would need per-store coordinates and a nearest-station rebuild.
    """
    raw = pd.read_parquet(parquet_path)
    raw["station_id"] = raw["station_id"].astype(int)
    frames: list[pd.DataFrame] = []
    cache: dict[int, pd.DataFrame] = {}
    for store_id, entry in mapping.items():
        station_id = int(entry["station_id"])
        if station_id not in cache:
            cache[station_id] = _build_station_frame(raw, station_id=station_id, start=start, end=end)
        frame = cache[station_id].copy()
        frame.insert(0, "store_id", store_id)
        frames.append(frame)
    if not frames:
        raise ValueError("empty store mapping")
    combined = pd.concat(frames, ignore_index=True)
    return _coerce_dtypes(combined)


def _build_station_frame(
    raw: pd.DataFrame, *, station_id: int, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """One ASOS station's daily frame in WEATHER_DAILY_COLUMNS minus store_id."""
    sub = raw[raw["station_id"] == station_id].copy()
    if sub.empty:
        raise ValueError(f"no rows for station_id={station_id}")
    sub["date"] = pd.to_datetime(sub["tm"]).dt.normalize()
    sub = sub[(sub["date"] >= pd.Timestamp(start)) & (sub["date"] <= pd.Timestamp(end))]

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(sub[col], errors="coerce") if col in sub.columns else pd.Series(dtype=float)

    snow_primary = _num("ddMes") if "ddMes" in sub.columns else None
    snow_fallback = _num("ddMefs") if "ddMefs" in sub.columns else None
    if snow_primary is not None and snow_fallback is not None:
        snow = snow_primary.fillna(snow_fallback)
    else:
        snow = snow_primary if snow_primary is not None else snow_fallback
    if snow is None:
        snow = pd.Series([0.0] * len(sub))

    out = pd.DataFrame(
        {
            "date": sub["date"].to_numpy(),
            "avg_temp": _num("avgTa").to_numpy(),
            "max_temp": _num("maxTa").to_numpy(),
            "min_temp": _num("minTa").to_numpy(),
            "humidity": _num("avgRhm").to_numpy(),
            "precipitation_mm": _num("sumRn").fillna(0.0).to_numpy(),
            "snow_depth_cm": snow.fillna(0.0).to_numpy(),
            "sunshine_hours": _num("sumSsHr").fillna(0.0).to_numpy(),
        }
    )
    full = pd.date_range(start, end, freq="D")
    out = out.set_index("date").reindex(full).rename_axis("date").reset_index()
    for col in ("avg_temp", "max_temp", "min_temp", "humidity"):
        out[col] = out[col].interpolate(limit_direction="both")
    out["precipitation_mm"] = out["precipitation_mm"].fillna(0.0)
    out["snow_depth_cm"] = out["snow_depth_cm"].fillna(0.0)
    out["sunshine_hours"] = out["sunshine_hours"].fillna(0.0)
    out["diurnal_range"] = out["max_temp"] - out["min_temp"]
    out["is_rain"] = (out["precipitation_mm"] > 0).astype("int8")
    out["is_snow"] = (out["snow_depth_cm"] > 0).astype("int8")
    cols = [c for c in WEATHER_DAILY_COLUMNS if c != "store_id"]
    return out[cols]


def load_weather_forecast_from_local(
    short_daily_path: Path | str,
    mid_daily_path: Path | str,
    observed_parquet_path: Path | str,
    *,
    mapping: dict[str, StationMapping],
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    fallback_window_days: int = 28,
) -> pd.DataFrame:
    """Long-form horizon weather frame stitching short + mid forecasts per store.

    - D+1 ~ D+3: short-term daily (TMP/REH/POP/PCP/SNO aggregated).
    - D+4 onward: mid-term daily (taMin/taMax + rnSt + wf text).
    - Columns the forecast APIs don't supply (humidity for mid-term, sunshine
      always) are filled from the trailing `fallback_window_days` of observed
      ASOS at each store's mapped station.
    """
    horizon = pd.date_range(horizon_start, horizon_end, freq="D")
    if len(horizon) == 0:
        raise ValueError(f"empty horizon: {horizon_start} → {horizon_end}")

    short_daily = _load_optional(short_daily_path)
    mid_daily = _load_optional(mid_daily_path)

    rows: list[dict] = []
    fallback_cache: dict[int, dict[str, float]] = {}
    for store_id, entry in mapping.items():
        station_id = int(entry["station_id"])
        if station_id not in fallback_cache:
            fallback_cache[station_id] = _compute_fallback(
                observed_parquet_path, station_id, horizon_start, fallback_window_days
            )
        fallback = fallback_cache[station_id]
        for d in horizon:
            row = _pick_forecast_row(
                d, short_daily, mid_daily, fallback,
                nx=int(entry["nx"]), ny=int(entry["ny"]),
                mid_land_reg_id=entry["mid_land_reg_id"],
                mid_ta_reg_id=entry["mid_ta_reg_id"],
            )
            row["store_id"] = store_id
            rows.append(row)

    out = pd.DataFrame(rows)
    out["diurnal_range"] = (out["max_temp"] - out["min_temp"]).astype(float)
    return _coerce_dtypes(out[list(WEATHER_DAILY_COLUMNS.keys())])


def _load_optional(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _compute_fallback(
    observed_path: Path | str, station_id: int, anchor: pd.Timestamp, window_days: int
) -> dict[str, float]:
    """Recent ASOS observed average — used to fill columns the forecast APIs miss."""
    p = Path(observed_path)
    if not p.exists():
        return {"humidity": 60.0, "sunshine_hours": 6.0, "precipitation_mm": 0.0, "snow_depth_cm": 0.0}
    raw = pd.read_parquet(p)
    raw = raw[raw["station_id"].astype(int) == int(station_id)].copy()
    raw["date"] = pd.to_datetime(raw["tm"]).dt.normalize()
    cutoff = anchor - pd.Timedelta(days=window_days)
    recent = raw[(raw["date"] >= cutoff) & (raw["date"] < anchor)]
    if recent.empty:
        recent = raw.tail(window_days)
    return {
        "humidity": float(pd.to_numeric(recent.get("avgRhm"), errors="coerce").mean() or 60.0),
        "sunshine_hours": float(pd.to_numeric(recent.get("sumSsHr"), errors="coerce").mean() or 6.0),
        "precipitation_mm": float(pd.to_numeric(recent.get("sumRn"), errors="coerce").fillna(0).mean()),
        "snow_depth_cm": 0.0,
    }


def _pick_forecast_row(
    date: pd.Timestamp,
    short_daily: pd.DataFrame,
    mid_daily: pd.DataFrame,
    fallback: dict[str, float],
    *,
    nx: int,
    ny: int,
    mid_land_reg_id: str,
    mid_ta_reg_id: str,
) -> dict:
    if not short_daily.empty:
        match = short_daily[
            (short_daily["nx"] == nx) & (short_daily["ny"] == ny) & (short_daily["date"] == date)
        ]
        if not match.empty:
            r = match.iloc[0]
            return _row_from_short(date, r)
    if not mid_daily.empty:
        match = mid_daily[
            (mid_daily["mid_land_reg_id"] == mid_land_reg_id)
            & (mid_daily["mid_ta_reg_id"] == mid_ta_reg_id)
            & (mid_daily["fcst_date"] == date)
        ]
        if not match.empty:
            r = match.iloc[0]
            return _row_from_mid(date, r, fallback)
    return {
        "date": date,
        "avg_temp": fallback.get("avg_temp", 15.0),
        "max_temp": fallback.get("max_temp", 20.0),
        "min_temp": fallback.get("min_temp", 10.0),
        "humidity": fallback["humidity"],
        "precipitation_mm": fallback["precipitation_mm"],
        "is_rain": 0,
        "snow_depth_cm": fallback["snow_depth_cm"],
        "is_snow": 0,
        "sunshine_hours": fallback["sunshine_hours"],
    }


def _row_from_short(date: pd.Timestamp, r: pd.Series) -> dict:
    rain = (r.get("precipitation_mm", 0) or 0) > 0 or (r.get("max_pop", 0) or 0) >= 60 or r.get("any_rain_pty", 0) == 1
    snow = (r.get("snow_depth_cm", 0) or 0) > 0 or r.get("any_snow_pty", 0) == 1
    return {
        "date": date,
        "avg_temp": r.get("avg_temp"),
        "max_temp": r.get("max_temp"),
        "min_temp": r.get("min_temp"),
        "humidity": r.get("humidity"),
        "precipitation_mm": r.get("precipitation_mm", 0.0) or 0.0,
        "is_rain": int(bool(rain)),
        "snow_depth_cm": r.get("snow_depth_cm", 0.0) or 0.0,
        "is_snow": int(bool(snow)),
        "sunshine_hours": None,
    }


def _row_from_mid(date: pd.Timestamp, r: pd.Series, fallback: dict[str, float]) -> dict:
    ta_min = r.get("taMin")
    ta_max = r.get("taMax")
    avg = (ta_min + ta_max) / 2 if pd.notna(ta_min) and pd.notna(ta_max) else None
    rn = max(r.get("rnSt_am", 0) or 0, r.get("rnSt_pm", 0) or 0)
    wf_text = f"{r.get('wf_am', '')}{r.get('wf_pm', '')}"
    has_snow_word = "눈" in wf_text
    return {
        "date": date,
        "avg_temp": avg,
        "max_temp": ta_max,
        "min_temp": ta_min,
        "humidity": fallback["humidity"],
        "precipitation_mm": fallback["precipitation_mm"] if rn >= 60 else 0.0,
        "is_rain": int(rn >= 60),
        "snow_depth_cm": fallback["snow_depth_cm"] if has_snow_word else 0.0,
        "is_snow": int(has_snow_word),
        "sunshine_hours": None,
    }


def validate_weather(df: pd.DataFrame) -> None:
    missing = set(WEATHER_DAILY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"weather frame missing columns: {sorted(missing)}")
    if df["date"].dt.normalize().ne(df["date"]).any():
        raise ValueError("weather 'date' must be normalized to midnight")
    dup = df.duplicated(subset=["store_id", "date"])
    if dup.any():
        raise ValueError(f"weather has {int(dup.sum())} duplicate (store_id, date) rows")

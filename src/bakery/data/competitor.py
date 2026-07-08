"""Competitor (bakery + cafe) location & license data.

The raw schema mirrors what `LOCALDATA` returns once we keep only the
fields we actually use for feature engineering. One row per business, with
timestamps that let us reconstruct who was operating on any given date.

  business_id    string   PK (인허가번호 또는 합성용 uuid)
  category       string   "bakery" 또는 "cafe"
  license_date   datetime 인허가일자 (영업 시작)
  close_date     datetime 폐업일자 — NaT면 현재 영업 중
  lat / lon      float64  WGS84
  business_status string  "active" / "closed" (close_date 유무로 일관)

`build_synthetic_competitor` clusters businesses around each store's coords
with realistic densities (transit/office hubs denser, residential sparser)
so the v3 model has a learnable signal before real LOCALDATA arrives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.store_mapping import StationMapping, haversine_m

COMPETITOR_RAW_COLUMNS: dict[str, str] = {
    "business_id": "string",
    "category": "string",
    "license_date": "datetime64[ns]",
    "close_date": "datetime64[ns]",
    "lat": "float64",
    "lon": "float64",
    "business_status": "string",
}

# Per-store realistic densities within ~1.5km — calibrated so the daily
# radius counts roughly match observed Seoul commercial districts.
_DEFAULT_DENSITY: dict[str, dict[str, int]] = {
    "store_A": {"bakery": 25, "cafe": 80},      # residential
    "store_B": {"bakery": 55, "cafe": 220},     # transit hub (홍대)
    "store_C": {"bakery": 45, "cafe": 140},     # office (여의도)
}
_FALLBACK_DENSITY = {"bakery": 35, "cafe": 110}


def build_synthetic_competitor(
    *,
    mapping: dict[str, StationMapping],
    license_min: str | pd.Timestamp = "2019-01-01",
    license_max: str | pd.Timestamp = "2025-12-31",
    seed: int = 42,
    cluster_radius_m: int = 1500,
    close_rate: float = 0.12,
) -> pd.DataFrame:
    """Hand-built competitor distribution near each store.

    Businesses are scattered in a 2D Gaussian around each store's coords with
    sigma scaled to `cluster_radius_m`. License dates are spread uniformly
    over [license_min, license_max]; a `close_rate` fraction get a close_date
    sampled to be after license_date but bounded by the cutoff.
    """
    rng = np.random.default_rng(seed)
    license_min_t = pd.Timestamp(license_min)
    license_max_t = pd.Timestamp(license_max)
    span_days = (license_max_t - license_min_t).days
    rows: list[dict] = []
    next_id = 0
    for store_id, entry in mapping.items():
        density = _DEFAULT_DENSITY.get(store_id, _FALLBACK_DENSITY)
        for category, count in density.items():
            for _ in range(count):
                lat, lon = _scatter_coord(entry["lat"], entry["lon"], cluster_radius_m, rng)
                license_offset = int(rng.integers(0, max(span_days, 1)))
                license_date = license_min_t + pd.Timedelta(days=license_offset)
                if rng.random() < close_rate:
                    remaining = (license_max_t - license_date).days
                    if remaining > 30:
                        close_offset = int(rng.integers(30, remaining))
                        close_date = license_date + pd.Timedelta(days=close_offset)
                    else:
                        close_date = pd.NaT
                else:
                    close_date = pd.NaT
                rows.append(
                    {
                        "business_id": f"synth-{next_id:06d}",
                        "category": category,
                        "license_date": license_date,
                        "close_date": close_date,
                        "lat": float(lat),
                        "lon": float(lon),
                        "business_status": "closed" if pd.notna(close_date) else "active",
                    }
                )
                next_id += 1
    return _coerce_dtypes(pd.DataFrame(rows))


def _scatter_coord(
    center_lat: float, center_lon: float, radius_m: int, rng: np.random.Generator
) -> tuple[float, float]:
    """2D Gaussian scatter; ~95% within `radius_m` since we use sigma=radius/2."""
    sigma = radius_m / 2
    # ~111,320 m per degree latitude; longitude scales by cos(lat).
    d_lat = rng.normal(0, sigma) / 111_320
    d_lon = rng.normal(0, sigma) / (111_320 * max(np.cos(np.radians(center_lat)), 0.1))
    return center_lat + d_lat, center_lon + d_lon


def load_competitor_from_local(
    parquet_path: Path | str,
    *,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize a raw LOCALDATA parquet into COMPETITOR_RAW_COLUMNS.

    The actual LOCALDATA OpenAPI fields vary slightly between datasets
    (제과점 vs 카페), so this loader assumes the ingest step already
    mapped them onto `business_id / category / license_date / close_date /
    lat / lon / business_status`. If `categories` is given, only those rows
    are kept.
    """
    raw = pd.read_parquet(parquet_path)
    if categories is not None:
        raw = raw[raw["category"].isin(categories)].copy()
    return _coerce_dtypes(raw)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["business_id"] = df["business_id"].astype("string")
    df["category"] = df["category"].astype("string")
    df["license_date"] = pd.to_datetime(df["license_date"]).dt.normalize()
    df["close_date"] = pd.to_datetime(df["close_date"]).dt.normalize()
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    df["business_status"] = df["business_status"].astype("string")
    return df[list(COMPETITOR_RAW_COLUMNS.keys())].reset_index(drop=True)


def validate_competitor(df: pd.DataFrame) -> None:
    missing = set(COMPETITOR_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"competitor frame missing columns: {sorted(missing)}")
    if df["business_id"].duplicated().any():
        raise ValueError(f"competitor has {int(df['business_id'].duplicated().sum())} duplicate business_id")
    bad_close = df.dropna(subset=["close_date"]).query("close_date < license_date")
    if not bad_close.empty:
        raise ValueError(f"{len(bad_close)} businesses have close_date before license_date")


def attach_distance_to_stores(
    competitor_df: pd.DataFrame, mapping: dict[str, StationMapping]
) -> pd.DataFrame:
    """Pre-compute the Haversine distance from each business to each store.

    Returns a long-form frame (store_id, business_id, category, license_date,
    close_date, distance_m) that downstream feature code can filter by
    radius + date without recomputing distances for every (store, day).
    """
    if competitor_df.empty:
        return pd.DataFrame(
            columns=["store_id", "business_id", "category", "license_date", "close_date", "distance_m"]
        )
    rows: list[pd.DataFrame] = []
    for store_id, entry in mapping.items():
        d = _distances_to_point(
            competitor_df["lat"].to_numpy(),
            competitor_df["lon"].to_numpy(),
            entry["lat"], entry["lon"],
        )
        sub = competitor_df[["business_id", "category", "license_date", "close_date"]].copy()
        sub.insert(0, "store_id", store_id)
        sub["distance_m"] = d
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _distances_to_point(
    lats: np.ndarray, lons: np.ndarray, ref_lat: float, ref_lon: float
) -> np.ndarray:
    """Vectorized haversine — same formula as ingest.store_mapping.haversine_m."""
    earth_r = 6_371_000.0
    p1 = np.radians(lats)
    p2 = np.radians(ref_lat)
    dp = np.radians(ref_lat - lats)
    dl = np.radians(ref_lon - lons)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return (2 * earth_r * np.arcsin(np.sqrt(a))).astype("float64")

"""Store → ASOS station + admin-dong mapping.

PoC default: three synthetic stores in Seoul.
  - store_A: 강남(학동) 인근, residential — 행정동 청담1동
  - store_B: 홍대입구역 인근, transit — 행정동 서교동
  - store_C: 여의도역 인근, office — 행정동 여의동

Real-data swap: override via yaml passed to `load_store_mapping(path=...)`.
`admin_dong_code` is the 행안부 행정동코드 (10-digit). Used to join
living-population / age / consumption external data, all of which are
keyed by admin dong.

ASOS station IDs from KMA: 108 = 서울, 112 = 인천, 119 = 수원, ...
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import yaml


class StationMapping(TypedDict):
    # ASOS / KMA weather
    station_id: int
    station_name: str
    nx: int
    ny: int
    mid_land_reg_id: str
    mid_ta_reg_id: str
    # Store coordinates (decimal degrees, WGS84)
    lat: float
    lon: float
    # Administrative dong (행정동) — keys living-population / age / consumption
    admin_dong_code: str
    admin_dong_name: str


DEFAULT_STATIONS: dict[str, StationMapping] = {
    "store_A": {
        "station_id": 108, "station_name": "서울",
        "nx": 60, "ny": 127,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B10101",
        "lat": 37.5160, "lon": 127.0340,
        "admin_dong_code": "11680565", "admin_dong_name": "청담동",
    },
    "store_B": {
        "station_id": 108, "station_name": "서울",
        "nx": 60, "ny": 127,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B10101",
        "lat": 37.5563, "lon": 126.9240,
        "admin_dong_code": "11440660", "admin_dong_name": "서교동",
    },
    "store_C": {
        "station_id": 108, "station_name": "서울",
        "nx": 60, "ny": 127,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B10101",
        "lat": 37.5212, "lon": 126.9239,
        "admin_dong_code": "11560540", "admin_dong_name": "여의동",
    },
    # 실 매장 — 아티제 아브뉴프랑광교점 (보나비)
    # 점포코드 1000000047, 경기도 수원시 영통구 (이의동/광교2동)
    "store_gw01": {
        "station_id": 119, "station_name": "수원",
        "nx": 60, "ny": 121,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B20601",
        "lat": 37.2853, "lon": 127.0593,
        "admin_dong_code": "41117610", "admin_dong_name": "광교2동",
    },
}


def load_store_mapping(path: Path | None = None) -> dict[str, StationMapping]:
    if path is None or not path.exists():
        return dict(DEFAULT_STATIONS)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stores = raw.get("stores", raw)
    out: dict[str, StationMapping] = {}
    for k, v in stores.items():
        out[k] = {
            "station_id": int(v["station_id"]),
            "station_name": v["station_name"],
            "nx": int(v.get("nx", 60)),
            "ny": int(v.get("ny", 127)),
            "mid_land_reg_id": str(v.get("mid_land_reg_id", "11B00000")),
            "mid_ta_reg_id": str(v.get("mid_ta_reg_id", "11B10101")),
            "lat": float(v["lat"]),
            "lon": float(v["lon"]),
            "admin_dong_code": str(v["admin_dong_code"]),
            "admin_dong_name": str(v["admin_dong_name"]),
        }
    return out


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates, in meters."""
    earth_r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * earth_r * math.asin(math.sqrt(a))


def unique_stations(mapping: dict[str, StationMapping]) -> list[StationMapping]:
    """Distinct ASOS stations to fetch (multiple stores may share one)."""
    seen: dict[int, StationMapping] = {}
    for entry in mapping.values():
        seen[entry["station_id"]] = entry
    return list(seen.values())


def unique_forecast_grids(mapping: dict[str, StationMapping]) -> list[tuple[int, int]]:
    return sorted({(s["nx"], s["ny"]) for s in mapping.values()})


def unique_mid_regions(mapping: dict[str, StationMapping]) -> list[tuple[str, str]]:
    return sorted({(s["mid_land_reg_id"], s["mid_ta_reg_id"]) for s in mapping.values()})


def unique_admin_dongs(mapping: dict[str, StationMapping]) -> list[str]:
    """Distinct admin-dong codes (10-digit) across all stores."""
    return sorted({s["admin_dong_code"] for s in mapping.values()})

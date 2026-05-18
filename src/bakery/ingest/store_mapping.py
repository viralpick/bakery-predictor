"""Store → ASOS station mapping.

PoC default assumes three synthetic stores live in greater Seoul region. When
real store coordinates arrive, override via a yaml file passed to
`load_store_mapping(path=...)`.

Station IDs are from KMA ASOS:
  108 = 서울, 112 = 인천, 119 = 수원, 143 = 대구, 156 = 광주, 159 = 부산
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import yaml


class StationMapping(TypedDict):
    station_id: int
    station_name: str
    # 단기예보 격자 좌표 (KMA Lambert grid)
    nx: int
    ny: int
    # 중기예보 예보구역 코드
    mid_land_reg_id: str  # 육상예보구역 (강수확률·날씨묘사)
    mid_ta_reg_id: str    # 기온예보구역 (최저·최고 기온)


# Forecast region codes for major regions (KMA OpenAPI 가이드 기준):
#   서울/인천/경기 육상 = "11B00000", 서울 기온 = "11B10101"
#   강원도 영서 = "11D10000", 강원도 영동 = "11D20000"
#   대전/세종/충남 = "11C20000", 충북 = "11C10000"
#   광주/전남 = "11F20000", 전북 = "11F10000"
#   대구/경북 = "11H10000", 부산/울산/경남 = "11H20000"
#   제주 = "11G00000"
# 격자 좌표(nx, ny): 서울=60,127, 인천=55,124, 수원=60,121


DEFAULT_STATIONS: dict[str, StationMapping] = {
    # PoC: all stores share Seoul observatory + grid + forecast region.
    # When real store coordinates arrive, point each store at its nearest
    # ASOS station + grid cell + forecast region.
    "store_A": {
        "station_id": 108, "station_name": "서울",
        "nx": 60, "ny": 127,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B10101",
    },
    "store_B": {
        "station_id": 108, "station_name": "서울",
        "nx": 60, "ny": 127,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B10101",
    },
    "store_C": {
        "station_id": 108, "station_name": "서울",
        "nx": 60, "ny": 127,
        "mid_land_reg_id": "11B00000", "mid_ta_reg_id": "11B10101",
    },
}


def load_store_mapping(path: Path | None = None) -> dict[str, StationMapping]:
    if path is None or not path.exists():
        return dict(DEFAULT_STATIONS)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stores = raw.get("stores", raw)
    return {
        k: {
            "station_id": int(v["station_id"]),
            "station_name": v["station_name"],
            "nx": int(v.get("nx", 60)),
            "ny": int(v.get("ny", 127)),
            "mid_land_reg_id": str(v.get("mid_land_reg_id", "11B00000")),
            "mid_ta_reg_id": str(v.get("mid_ta_reg_id", "11B10101")),
        }
        for k, v in stores.items()
    }


def unique_stations(mapping: dict[str, StationMapping]) -> list[StationMapping]:
    """Distinct stations to fetch (multiple stores may share one)."""
    seen: dict[int, StationMapping] = {}
    for entry in mapping.values():
        seen[entry["station_id"]] = entry
    return list(seen.values())


def unique_forecast_grids(mapping: dict[str, StationMapping]) -> list[tuple[int, int]]:
    """Distinct (nx, ny) grid cells for short-term forecast."""
    return sorted({(s["nx"], s["ny"]) for s in mapping.values()})


def unique_mid_regions(mapping: dict[str, StationMapping]) -> list[tuple[str, str]]:
    """Distinct (mid_land_reg_id, mid_ta_reg_id) pairs for mid-term forecast."""
    return sorted({(s["mid_land_reg_id"], s["mid_ta_reg_id"]) for s in mapping.values()})

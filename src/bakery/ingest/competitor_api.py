"""Competitor (bakery + cafe) data — combined LOCALDATA + SBIZ adapter.

Two sources fused into one `competitor_raw.parquet`:

  bakery → 행정안전부_식품_제과점영업 (data.go.kr 15155252)
           Endpoint: https://apis.data.go.kr/1741000/bakeries/info
           Per-store sigungu (OPN_ATMY_GRP_CD) pull with full licence/close
           history. TM coordinates (EPSG:5181 KATEC 중부원점) → WGS84
           via pyproj.

  cafe   → 소상공인진흥공단 상가(상권)정보 (data.go.kr B553077 sdsc2)
           Endpoint: storeListInRadius around each store's (lat, lon).
           SBIZ doesn't expose licence/close dates — those are pinned to
           2019-01-01 / NaT so the competitor_features active-count filter
           keeps every cafe alive across the training window. The
           new/closed-90d trend features are therefore meaningful for
           bakeries only (PoC limitation).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd
from pyproj import Transformer

from ..config import EXTERNAL_DATA_DIR, data_go_kr_api_key
from ..ingest.store_mapping import StationMapping, load_store_mapping
from ._http import DEFAULT_THROTTLE_SECONDS, DEFAULT_TIMEOUT, ApiError

# --- LOCALDATA (bakery, with licence history) ----------------------------

LOCALDATA_URL = "https://apis.data.go.kr/1741000/bakeries/info"
LOCALDATA_PAGE_SIZE = 100  # API cap

# Sigungu codes used by LOCALDATA (OPN_ATMY_GRP_CD). PoC matches our 3 stores'
# sigungus; real-data swap point is store_mapping → derive automatically by
# the sigungu name. Hard-coded here keeps the call path simple.
SIGUNGU_BY_DONG_PREFIX: dict[str, str] = {
    "11680": "3220000",  # 강남구
    "11440": "3130000",  # 마포구
    "11560": "3180000",  # 영등포구
    "41117": "3740000",  # 수원시 영통구 (광교)
}

_TM_TO_WGS84 = Transformer.from_crs("epsg:5181", "epsg:4326", always_xy=True)


def _fetch_localdata_page(opn_grp_cd: str, page_no: int) -> tuple[list[dict], int]:
    params = {
        "serviceKey": data_go_kr_api_key(),
        "returnType": "json",
        "pageNo": page_no,
        "numOfRows": LOCALDATA_PAGE_SIZE,
        "cond[OPN_ATMY_GRP_CD::EQ]": opn_grp_cd,
    }
    r = httpx.get(LOCALDATA_URL, params=params, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    payload = json.loads(r.text)
    response = payload.get("response") or {}
    header = response.get("header") or {}
    code = header.get("resultCode")
    if code not in ("0", "00"):
        raise ApiError(f"bakeries/info opn_grp_cd={opn_grp_cd} page={page_no}: {code} {header.get('resultMsg')}")
    body = response.get("body") or {}
    items = (body.get("items") or {}).get("item") or []
    total = int(body.get("totalCount") or 0)
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return items, total


def fetch_localdata_sigungu(opn_grp_cd: str) -> pd.DataFrame:
    """All bakery rows for one sigungu — paginated."""
    first, total = _fetch_localdata_page(opn_grp_cd, 1)
    rows = list(first)
    pages = (total + LOCALDATA_PAGE_SIZE - 1) // LOCALDATA_PAGE_SIZE
    for p in range(2, pages + 1):
        more, _ = _fetch_localdata_page(opn_grp_cd, p)
        rows.extend(more)
    return pd.DataFrame(rows)


def normalize_localdata(raw: pd.DataFrame) -> pd.DataFrame:
    """LOCALDATA bakery row → COMPETITOR_RAW_COLUMNS.

    - LCPMT_YMD (YYYY-MM-DD or YYYYMMDD) → license_date
    - CLSBIZ_YMD ("" → NaT) → close_date
    - CRD_INFO_X/Y (EPSG:5181) → lon/lat (WGS84)
    - SALS_STTS_CD "01" → active, others → closed
    """
    if raw.empty:
        return pd.DataFrame()
    license_date = pd.to_datetime(raw["LCPMT_YMD"], errors="coerce")
    close_date = pd.to_datetime(raw["CLSBIZ_YMD"].replace({"": None}), errors="coerce")
    x = pd.to_numeric(raw["CRD_INFO_X"], errors="coerce").to_numpy()
    y = pd.to_numeric(raw["CRD_INFO_Y"], errors="coerce").to_numpy()
    lon, lat = _TM_TO_WGS84.transform(x, y)
    status = raw["SALS_STTS_CD"].astype(str).map(lambda c: "active" if c == "01" else "closed")
    out = pd.DataFrame(
        {
            "business_id": raw["MNG_NO"].astype("string"),
            "category": "bakery",
            "license_date": license_date,
            "close_date": close_date,
            "lat": lat,
            "lon": lon,
            "business_status": status.astype("string"),
        }
    )
    # Drop rows with unparseable coordinates (TM 0/0 sentinel, etc.)
    return out[out["lat"].between(33.0, 39.0) & out["lon"].between(124.0, 132.0)].reset_index(drop=True)


# --- SBIZ (cafe, snapshot only) ------------------------------------------

SBIZ_BASE = "https://apis.data.go.kr/B553077/api/open/sdsc2"
SBIZ_RADIUS_M = 1000
SBIZ_SCLASS_CAFE = "I21201"


def _fetch_sbiz_page(cx: float, cy: float, sclass: str, page_no: int, num_rows: int = 1000) -> tuple[list[dict], int]:
    url = f"{SBIZ_BASE}/storeListInRadius"
    params = {
        "serviceKey": data_go_kr_api_key(),
        "type": "json",
        "radius": SBIZ_RADIUS_M,
        "cx": cx, "cy": cy,
        "indsSclsCd": sclass,
        "numOfRows": num_rows,
        "pageNo": page_no,
    }
    r = httpx.get(url, params=params, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    payload = json.loads(r.text)
    header = payload.get("header") or {}
    code = header.get("resultCode")
    if code not in ("00", "0"):
        raise ApiError(f"SBIZ storeListInRadius cx={cx} {sclass}: {code} {header.get('resultMsg')}")
    body = payload.get("body") or {}
    items = body.get("items") or []
    total = int(body.get("totalCount") or len(items))
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return items, total


def fetch_sbiz_cafe(cx: float, cy: float) -> pd.DataFrame:
    first, total = _fetch_sbiz_page(cx, cy, SBIZ_SCLASS_CAFE, 1)
    rows = list(first)
    pages = (total + 999) // 1000
    for p in range(2, pages + 1):
        more, _ = _fetch_sbiz_page(cx, cy, SBIZ_SCLASS_CAFE, p)
        rows.extend(more)
    return pd.DataFrame(rows)


def normalize_sbiz(raw: pd.DataFrame) -> pd.DataFrame:
    """SBIZ cafe → COMPETITOR_RAW_COLUMNS. license_date pinned (no licence history)."""
    if raw.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "business_id": raw["bizesId"].astype("string"),
            "category": "cafe",
            "license_date": pd.Timestamp("2019-01-01"),
            "close_date": pd.NaT,
            "lat": pd.to_numeric(raw["lat"], errors="coerce").astype("float64"),
            "lon": pd.to_numeric(raw["lon"], errors="coerce").astype("float64"),
            "business_status": "active",
        }
    )


# --- combined backfill ---------------------------------------------------

def backfill(*, out_dir: Path | None = None, mapping_path: Path | None = None) -> Path:
    """Fetch LOCALDATA bakeries (per sigungu) + SBIZ cafes (per store radius)
    and dedupe into a single `competitor_raw.parquet`.
    """
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_store_mapping(mapping_path)
    frames: list[pd.DataFrame] = []

    # bakery — LOCALDATA per unique sigungu
    sigungu_codes: set[str] = set()
    for entry in mapping.values():
        prefix = entry["admin_dong_code"][:5]
        opn_grp = SIGUNGU_BY_DONG_PREFIX.get(prefix)
        if opn_grp is None:
            raise ApiError(
                f"OPN_ATMY_GRP_CD not mapped for dong prefix {prefix}. "
                "Add to SIGUNGU_BY_DONG_PREFIX in ingest/competitor_api.py."
            )
        sigungu_codes.add(opn_grp)
    for code in sorted(sigungu_codes):
        raw = fetch_localdata_sigungu(code)
        frames.append(normalize_localdata(raw))

    # cafe — SBIZ per store radius
    for entry in mapping.values():
        raw = fetch_sbiz_cafe(cx=entry["lon"], cy=entry["lat"])
        frames.append(normalize_sbiz(raw))

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    combined = combined.drop_duplicates(subset="business_id").reset_index(drop=True)
    out_path = out_dir / "competitor_raw.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path

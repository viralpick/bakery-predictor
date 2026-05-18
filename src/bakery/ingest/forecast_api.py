"""기상청 단기/중기 예보 어댑터.

Two endpoints to cover D+1 ~ D+7:

- 단기예보 (VilageFcstInfoService_2.0/getVilageFcst):
    3-hourly forecast for D+0 ~ D+3, published 8x/day (02, 05, 08, 11, 14,
    17, 20, 23). We pull the most recent published base_time, parse the
    category-coded items, and aggregate to a daily frame.

- 중기예보 (MidFcstInfoService/{getMidLandFcst, getMidTa}):
    Day-level forecast for D+3 ~ D+10, published 2x/day (06, 18). Land
    forecast gives precipitation probability + descriptive weather; Ta
    forecast gives daily min/max temperature. Humidity / sunshine are not
    provided — the caller layers a fallback (e.g. last-28d ASOS average).

Both endpoints return small payloads (<10KB) so we don't paginate.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from ..config import EXTERNAL_DATA_DIR, data_go_kr_api_key
from ._http import DEFAULT_THROTTLE_SECONDS, DEFAULT_TIMEOUT, ApiError
from .store_mapping import (
    load_store_mapping,
    unique_forecast_grids,
    unique_mid_regions,
)

SHORT_TERM_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
MID_LAND_URL = "https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
MID_TA_URL = "https://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"

SHORT_TERM_BASE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)
MID_TERM_BASE_HOURS = (6, 18)
SHORT_TERM_PUBLISH_LAG_MINUTES = 30  # KMA가 발표 후 가용해지기까지 약 30분
MID_TERM_PUBLISH_LAG_MINUTES = 60


def latest_short_term_base(now: datetime | None = None) -> tuple[str, str]:
    """Return (base_date 'YYYYMMDD', base_time 'HHMM') of the latest issued forecast."""
    now = now or datetime.now()
    candidate = now - timedelta(minutes=SHORT_TERM_PUBLISH_LAG_MINUTES)
    for h in reversed(SHORT_TERM_BASE_HOURS):
        if candidate.hour >= h:
            return candidate.strftime("%Y%m%d"), f"{h:02d}00"
    yesterday = candidate - timedelta(days=1)
    return yesterday.strftime("%Y%m%d"), "2300"


def latest_mid_term_tmfc(now: datetime | None = None) -> str:
    """Return tmFc 'YYYYMMDDHHmm' of the latest issued mid-term forecast."""
    now = now or datetime.now()
    candidate = now - timedelta(minutes=MID_TERM_PUBLISH_LAG_MINUTES)
    for h in reversed(MID_TERM_BASE_HOURS):
        if candidate.hour >= h:
            return candidate.strftime("%Y%m%d") + f"{h:02d}00"
    yesterday = candidate - timedelta(days=1)
    return yesterday.strftime("%Y%m%d") + "1800"


def _get_json(url: str, params: dict) -> dict:
    r = httpx.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    try:
        payload = json.loads(r.text)
    except json.JSONDecodeError as exc:
        raise ApiError(f"{url} returned non-JSON body:\n{r.text[:300]}") from exc
    header = payload["response"]["header"]
    code = header.get("resultCode")
    if code != "00":
        raise ApiError(f"{url} {code}: {header.get('resultMsg')}")
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return payload


def fetch_short_term(nx: int, ny: int, base_date: str, base_time: str) -> pd.DataFrame:
    """One (nx, ny) grid cell, one publish time → flat DataFrame of fcst rows."""
    params = {
        "ServiceKey": data_go_kr_api_key(),
        "pageNo": 1,
        "numOfRows": 1000,  # 3h x ~4 days x ~10 categories ≈ 320 rows; 1000 covers it
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    payload = _get_json(SHORT_TERM_URL, params)
    items = payload["response"]["body"].get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["fcst_dt"] = pd.to_datetime(df["fcstDate"].astype(str) + df["fcstTime"].astype(str), format="%Y%m%d%H%M")
    df["nx"] = int(nx)
    df["ny"] = int(ny)
    df["base_dt"] = pd.to_datetime(base_date + base_time, format="%Y%m%d%H%M")
    return df[["base_dt", "fcst_dt", "nx", "ny", "category", "fcstValue"]]


def fetch_mid_land(reg_id: str, tm_fc: str) -> pd.DataFrame:
    params = {
        "ServiceKey": data_go_kr_api_key(),
        "pageNo": 1,
        "numOfRows": 10,
        "dataType": "JSON",
        "regId": reg_id,
        "tmFc": tm_fc,
    }
    payload = _get_json(MID_LAND_URL, params)
    items = payload["response"]["body"].get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not items:
        return pd.DataFrame()
    row = items[0]
    return _melt_mid_land(row, reg_id=reg_id, tm_fc=tm_fc)


def fetch_mid_ta(ta_reg_id: str, tm_fc: str) -> pd.DataFrame:
    params = {
        "ServiceKey": data_go_kr_api_key(),
        "pageNo": 1,
        "numOfRows": 10,
        "dataType": "JSON",
        "regId": ta_reg_id,
        "tmFc": tm_fc,
    }
    payload = _get_json(MID_TA_URL, params)
    items = payload["response"]["body"].get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not items:
        return pd.DataFrame()
    row = items[0]
    return _melt_mid_ta(row, ta_reg_id=ta_reg_id, tm_fc=tm_fc)


_MID_DAY_OFFSETS = range(3, 11)  # KMA 중기예보는 D+3~D+10


def _melt_mid_land(row: dict, *, reg_id: str, tm_fc: str) -> pd.DataFrame:
    """Mid-term land row → one row per day offset with [day_offset, rnSt_am, rnSt_pm, wf_am, wf_pm]."""
    issued = pd.to_datetime(tm_fc, format="%Y%m%d%H%M")
    rows = []
    for d in _MID_DAY_OFFSETS:
        rows.append(
            {
                "tm_fc": issued,
                "reg_id": reg_id,
                "day_offset": d,
                "fcst_date": (issued + pd.Timedelta(days=d)).normalize(),
                "rnSt_am": _to_int(row.get(f"rnSt{d}Am") or row.get(f"rnSt{d}")),
                "rnSt_pm": _to_int(row.get(f"rnSt{d}Pm") or row.get(f"rnSt{d}")),
                "wf_am": row.get(f"wf{d}Am") or row.get(f"wf{d}", ""),
                "wf_pm": row.get(f"wf{d}Pm") or row.get(f"wf{d}", ""),
            }
        )
    return pd.DataFrame(rows)


def _melt_mid_ta(row: dict, *, ta_reg_id: str, tm_fc: str) -> pd.DataFrame:
    issued = pd.to_datetime(tm_fc, format="%Y%m%d%H%M")
    rows = []
    for d in _MID_DAY_OFFSETS:
        rows.append(
            {
                "tm_fc": issued,
                "ta_reg_id": ta_reg_id,
                "day_offset": d,
                "fcst_date": (issued + pd.Timedelta(days=d)).normalize(),
                "taMin": _to_float(row.get(f"taMin{d}")),
                "taMax": _to_float(row.get(f"taMax{d}")),
            }
        )
    return pd.DataFrame(rows)


def _to_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_PCP_PATTERNS = [
    (re.compile(r"강수없음|없음"), 0.0),
    (re.compile(r"1\.?0?\s*mm\s*미만"), 0.5),  # "1.0mm 미만" → 0.5 추정
    (re.compile(r"30\s*~\s*50\s*mm"), 40.0),
    (re.compile(r"50\s*mm\s*이상"), 60.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*~\s*(\d+(?:\.\d+)?)\s*mm"), None),  # generic range
    (re.compile(r"(\d+(?:\.\d+)?)\s*mm"), None),  # generic single
]


def parse_precipitation(text: str | None) -> float:
    """KMA returns precipitation as a free-form string ('강수없음', '1.0mm', '30~50mm').
    Convert to mm as a float. Missing / unparseable → 0.0."""
    if text is None or text == "":
        return 0.0
    s = str(text).strip()
    for pattern, value in _PCP_PATTERNS:
        m = pattern.search(s)
        if m:
            if value is not None:
                return value
            groups = m.groups()
            if len(groups) == 2:
                return (float(groups[0]) + float(groups[1])) / 2
            return float(groups[0])
    return 0.0


def aggregate_short_term_to_daily(short_term: pd.DataFrame) -> pd.DataFrame:
    """Collapse 3-hourly fcst rows into per-(date, nx, ny) daily aggregates."""
    if short_term.empty:
        return pd.DataFrame()
    df = short_term.copy()
    df["date"] = df["fcst_dt"].dt.normalize()
    # category-wise pivot is messy because fcstValue dtype mixes int/text;
    # iterate per category instead.
    daily_rows: list[dict] = []
    keys = ["nx", "ny", "date"]
    for (nx, ny, date), group in df.groupby(keys, observed=True):
        cat_map = group.groupby("category")["fcstValue"].apply(list).to_dict()
        tmp = [float(v) for v in cat_map.get("TMP", []) if _to_float(v) is not None]
        tmx = [float(v) for v in cat_map.get("TMX", []) if _to_float(v) is not None]
        tmn = [float(v) for v in cat_map.get("TMN", []) if _to_float(v) is not None]
        reh = [float(v) for v in cat_map.get("REH", []) if _to_float(v) is not None]
        pop = [float(v) for v in cat_map.get("POP", []) if _to_float(v) is not None]
        pcp_total = sum(parse_precipitation(v) for v in cat_map.get("PCP", []))
        sno_total = sum(parse_precipitation(v) for v in cat_map.get("SNO", []))
        sky = cat_map.get("SKY", [])
        pty = cat_map.get("PTY", [])
        # Prefer KMA-provided TMX/TMN if available; fall back to TMP min/max.
        max_temp = float(max(tmx)) if tmx else (float(max(tmp)) if tmp else None)
        min_temp = float(min(tmn)) if tmn else (float(min(tmp)) if tmp else None)
        daily_rows.append(
            {
                "nx": int(nx),
                "ny": int(ny),
                "date": date,
                "avg_temp": float(sum(tmp) / len(tmp)) if tmp else None,
                "max_temp": max_temp,
                "min_temp": min_temp,
                "humidity": float(sum(reh) / len(reh)) if reh else None,
                "precipitation_mm": float(pcp_total),
                "snow_depth_cm": float(sno_total),
                "max_pop": float(max(pop)) if pop else None,
                "any_rain_pty": int(any(int(p) in (1, 2, 4, 5) for p in pty if _to_float(p) is not None)),
                "any_snow_pty": int(any(int(p) in (3, 6, 7) for p in pty if _to_float(p) is not None)),
                "sky_modal": _modal(sky),
            }
        )
    return pd.DataFrame(daily_rows)


def _modal(values: list) -> int | None:
    if not values:
        return None
    counts: dict = {}
    for v in values:
        if v in (None, ""):
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return int(max(counts, key=counts.get))


def backfill_forecast(
    *,
    out_dir: Path | None = None,
    mapping_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Path]:
    """Pull the latest short + mid forecast for every distinct grid/region in the
    store mapping and persist three parquet files:
      - forecast_short_term.parquet   (raw 3-hourly rows)
      - forecast_short_term_daily.parquet  (daily aggregates per grid)
      - forecast_mid_term_daily.parquet    (daily mid-term combined land+ta)
    """
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_store_mapping(mapping_path)
    base_date, base_time = latest_short_term_base(now=now)
    tm_fc = latest_mid_term_tmfc(now=now)

    short_frames: list[pd.DataFrame] = []
    for nx, ny in unique_forecast_grids(mapping):
        frame = fetch_short_term(nx, ny, base_date, base_time)
        if not frame.empty:
            short_frames.append(frame)
    short_raw = pd.concat(short_frames, ignore_index=True) if short_frames else pd.DataFrame()
    short_daily = aggregate_short_term_to_daily(short_raw)

    mid_rows: list[pd.DataFrame] = []
    for land_reg, ta_reg in unique_mid_regions(mapping):
        land = fetch_mid_land(land_reg, tm_fc)
        ta = fetch_mid_ta(ta_reg, tm_fc)
        if land.empty and ta.empty:
            continue
        merged = ta.merge(land, on=["day_offset", "fcst_date", "tm_fc"], how="outer")
        merged["mid_land_reg_id"] = land_reg
        merged["mid_ta_reg_id"] = ta_reg
        mid_rows.append(merged)
    mid_daily = pd.concat(mid_rows, ignore_index=True) if mid_rows else pd.DataFrame()

    out_paths = {}
    if not short_raw.empty:
        p = out_dir / "forecast_short_term.parquet"
        short_raw.to_parquet(p, index=False)
        out_paths["short_raw"] = p
    if not short_daily.empty:
        p = out_dir / "forecast_short_term_daily.parquet"
        short_daily.to_parquet(p, index=False)
        out_paths["short_daily"] = p
    if not mid_daily.empty:
        p = out_dir / "forecast_mid_term_daily.parquet"
        mid_daily.to_parquet(p, index=False)
        out_paths["mid_daily"] = p
    return out_paths

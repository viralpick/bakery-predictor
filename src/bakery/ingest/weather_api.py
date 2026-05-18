"""기상청 ASOS 일자료 조회 (data.go.kr 1360000/AsosDalyInfoService).

We pull observed daily values for one or more stations and write a long-form
parquet `weather_observed.parquet` with the schema KMA returns plus a
normalized `date` column. The mapping to our internal WEATHER_DAILY_COLUMNS
(avg_temp, etc.) happens in `data/weather.py::load_weather_from_local` so
that this module stays a thin API adapter.

Rate limits: 10,000 calls/day. We call once per (station, year chunk), so a
3-year backfill of 3 stations is ~9 calls — well within budget.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date as Date
from pathlib import Path

import httpx
import pandas as pd

from ..config import EXTERNAL_DATA_DIR, data_go_kr_api_key
from ._http import DEFAULT_THROTTLE_SECONDS, DEFAULT_TIMEOUT, ApiError
from .store_mapping import load_store_mapping, unique_stations

BASE_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
NUM_OF_ROWS = 999  # max allowed; ≥ days/year so one call per year-chunk


def _fetch_chunk(station_id: int, start: Date, end: Date) -> list[dict]:
    params = {
        "ServiceKey": data_go_kr_api_key(),
        "pageNo": 1,
        "numOfRows": NUM_OF_ROWS,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start.strftime("%Y%m%d"),
        "endDt": end.strftime("%Y%m%d"),
        "stnIds": str(station_id),
    }
    r = httpx.get(BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return _parse_json(r.text, station_id=station_id, start=start, end=end)


def _parse_json(text: str, *, station_id: int, start: Date, end: Date) -> list[dict]:
    """KMA occasionally returns XML even when dataType=JSON is requested
    (auth or quota errors). Surface that as ApiError with the raw body."""
    import json

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiError(
            f"ASOS station={station_id} {start}~{end} returned non-JSON body:\n{text[:300]}"
        ) from exc
    header = payload["response"]["header"]
    if header.get("resultCode") != "00":
        raise ApiError(f"ASOS {header.get('resultCode')}: {header.get('resultMsg')}")
    body = payload["response"]["body"]
    items_container = body.get("items") or {}
    items = items_container.get("item", []) if isinstance(items_container, dict) else []
    if isinstance(items, dict):
        items = [items]
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return items


def _year_chunks(start: Date, end: Date) -> Iterable[tuple[Date, Date]]:
    cur = start
    while cur <= end:
        chunk_end = min(Date(cur.year, 12, 31), end)
        yield cur, chunk_end
        cur = Date(cur.year + 1, 1, 1)


def fetch_station(station_id: int, start: Date, end: Date) -> pd.DataFrame:
    rows: list[dict] = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        rows.extend(_fetch_chunk(station_id, chunk_start, chunk_end))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["tm"]).dt.normalize()
    df["station_id"] = int(station_id)
    return df.sort_values("date").reset_index(drop=True)


def backfill(
    start: Date,
    end: Date,
    *,
    mapping_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Fetch and persist a long-form parquet `weather_observed.parquet`."""
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_store_mapping(mapping_path)
    frames = []
    for station in unique_stations(mapping):
        frame = fetch_station(station["station_id"], start, end)
        if not frame.empty:
            frame["station_name"] = station["station_name"]
            frames.append(frame)
    if not frames:
        raise RuntimeError("ASOS returned no rows for any station")
    combined = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "weather_observed.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path

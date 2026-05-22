"""서울 열린데이터광장 `SPOP_LOCAL_RESD_DONG` (행정동 단위 생활인구) 어댑터.

API:
  GET http://openapi.seoul.go.kr:8088/{KEY}/json/SPOP_LOCAL_RESD_DONG/{START}/{END}/{USE_DT}

Returns 24 hours × ~424 행정동 ≈ 10,176 rows per day. We paginate in 1000-row
chunks (11 calls per day). The dataset window is rolling — Open Data Plaza
exposes only the most recent ~2 months via OpenAPI; older data must be
downloaded as monthly CSV zips. The CLI defaults to today minus ~50 days
so it always lands inside that window.

Response columns we keep:
  STDR_DE_ID         → date (YYYYMMDD)
  TMZON_PD_SE        → hour ("00".."23")
  ADSTRD_CODE_SE     → admin_dong_code (8-digit 행정동코드)
  TOT_LVPOP_CO       → total_pop (float)
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from datetime import date as Date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from ..config import EXTERNAL_DATA_DIR, seoul_open_api_key
from ._http import DEFAULT_THROTTLE_SECONDS, DEFAULT_TIMEOUT, ApiError

DATASET = "SPOP_LOCAL_RESD_DONG"
BASE_URL = "http://openapi.seoul.go.kr:8088"
PAGE_SIZE = 1000


def _fetch_page(use_dt: str, start: int, end: int) -> tuple[list[dict], int]:
    key = seoul_open_api_key()
    url = f"{BASE_URL}/{key}/json/{DATASET}/{start}/{end}/{use_dt}"
    r = httpx.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    try:
        payload = json.loads(r.text)
    except json.JSONDecodeError as exc:
        raise ApiError(f"{DATASET} {use_dt}[{start}-{end}] non-JSON:\n{r.text[:300]}") from exc
    body = payload.get(DATASET) or {}
    result = body.get("RESULT") or {}
    code = result.get("CODE")
    if code and code != "INFO-000":
        raise ApiError(f"{DATASET} {use_dt}[{start}-{end}] {code}: {result.get('MESSAGE')}")
    rows = body.get("row") or []
    total = int(body.get("list_total_count") or 0)
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return rows, total


def fetch_one_day(use_dt: str) -> pd.DataFrame:
    """All rows for one day, paginated."""
    all_rows: list[dict] = []
    first_rows, total = _fetch_page(use_dt, 1, PAGE_SIZE)
    all_rows.extend(first_rows)
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    for p in range(2, pages + 1):
        start = (p - 1) * PAGE_SIZE + 1
        end = min(p * PAGE_SIZE, total)
        rows, _ = _fetch_page(use_dt, start, end)
        all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    return _normalize(pd.DataFrame(all_rows))


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw → (admin_dong_code, date, hour, total_pop) long-form."""
    out = pd.DataFrame(
        {
            "admin_dong_code": raw["ADSTRD_CODE_SE"].astype("string"),
            "date": pd.to_datetime(raw["STDR_DE_ID"], format="%Y%m%d").dt.normalize(),
            "hour": pd.to_numeric(raw["TMZON_PD_SE"], errors="coerce").astype("int8"),
            "total_pop": pd.to_numeric(raw["TOT_LVPOP_CO"], errors="coerce").astype("float32"),
        }
    )
    return out


def _daterange(start: Date, end: Date) -> Iterable[Date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def backfill(start: Date, end: Date, *, out_dir: Path | None = None) -> Path:
    """Fetch [start, end] inclusive → `living_population.parquet` (long-form).

    Note: Open Data Plaza retains only the last ~2 months; older dates yield
    INFO-200. For full history download monthly CSV zips from the dataset page.
    """
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for d in _daterange(start, end):
        use_dt = d.strftime("%Y%m%d")
        frame = fetch_one_day(use_dt)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"SPOP_LOCAL_RESD_DONG returned no rows for {start}~{end}")
    combined = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "living_population.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path

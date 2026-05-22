"""서울 열린데이터광장 `VwsmAdstrdNcmCnsmpW` (상권분석 소비-행정동) 어댑터.

API:
  GET http://openapi.seoul.go.kr:8088/{KEY}/json/VwsmAdstrdNcmCnsmpW/{START}/{END}/

Returns quarterly snapshots per admin dong with category-split expenditure.
We map onto our internal schema:

  STDR_YYQU_CD (YYYYQ)            → quarter (YYYYQn)
  ADSTRD_CD (8-digit 행정동)        → admin_dong_code
  EXPNDTR_TOTAMT                  → total_spend
  FDSTFFS_EXPNDTR_TOTAMT + FD_EXPNDTR_TOTAMT  → food_retail_spend
    (식료품 + 음식점 — closest proxy to bakery-adjacent spend)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

from ..config import EXTERNAL_DATA_DIR, seoul_open_api_key
from ._http import DEFAULT_THROTTLE_SECONDS, DEFAULT_TIMEOUT, ApiError

DATASET = "VwsmAdstrdNcmCnsmpW"
BASE_URL = "http://openapi.seoul.go.kr:8088"
PAGE_SIZE = 1000


def _fetch_page(start: int, end: int) -> tuple[list[dict], int]:
    key = seoul_open_api_key()
    url = f"{BASE_URL}/{key}/json/{DATASET}/{start}/{end}/"
    r = httpx.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    try:
        payload = json.loads(r.text)
    except json.JSONDecodeError as exc:
        raise ApiError(f"{DATASET} [{start}-{end}] non-JSON:\n{r.text[:300]}") from exc
    body = payload.get(DATASET) or {}
    result = body.get("RESULT") or {}
    code = result.get("CODE")
    if code and code != "INFO-000":
        raise ApiError(f"{DATASET} [{start}-{end}] {code}: {result.get('MESSAGE')}")
    rows = body.get("row") or []
    total = int(body.get("list_total_count") or 0)
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return rows, total


def fetch_all() -> pd.DataFrame:
    first_rows, total = _fetch_page(1, PAGE_SIZE)
    all_rows = list(first_rows)
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    for p in range(2, pages + 1):
        start = (p - 1) * PAGE_SIZE + 1
        end = min(p * PAGE_SIZE, total)
        rows, _ = _fetch_page(start, end)
        all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    return _normalize(pd.DataFrame(all_rows))


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw VwsmAdstrdNcmCnsmpW → (admin_dong_code, quarter, total_spend, food_retail_spend)."""
    quarter = raw["STDR_YYQU_CD"].astype(str).apply(_yyqu_to_quarter)
    total = pd.to_numeric(raw["EXPNDTR_TOTAMT"], errors="coerce").fillna(0.0)
    food = pd.to_numeric(raw.get("FD_EXPNDTR_TOTAMT", 0), errors="coerce").fillna(0.0)
    grocery = pd.to_numeric(raw.get("FDSTFFS_EXPNDTR_TOTAMT", 0), errors="coerce").fillna(0.0)
    out = pd.DataFrame(
        {
            "admin_dong_code": raw["ADSTRD_CD"].astype("string"),
            "quarter": quarter,
            "total_spend": total.astype("float64"),
            "food_retail_spend": (food + grocery).astype("float64"),
        }
    )
    return out


def _yyqu_to_quarter(yyqu: str) -> str:
    """'20241' → '2024Q1'."""
    if len(yyqu) != 5:
        return yyqu
    return f"{yyqu[:4]}Q{yyqu[4]}"


def backfill(*, out_dir: Path | None = None) -> Path:
    """Fetch every page of the dataset (full history) → `consumption.parquet`."""
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_all()
    if df.empty:
        raise RuntimeError(f"{DATASET} returned no rows")
    out_path = out_dir / "consumption.parquet"
    df.to_parquet(out_path, index=False)
    return out_path

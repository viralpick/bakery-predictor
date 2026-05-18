"""한국천문연구원 특일정보 API (data.go.kr B090041).

We fetch two operations:
- getRestDeInfo: 공휴일 (대체공휴일 포함) — `isHoliday=Y` 항목만 의미
- get24DivisionsInfo: 24절기 — 식 행사 효과 가능성, 영업 영향은 상대적으로 약함

Both return XML with `<item><locdate>YYYYMMDD</locdate><dateName>...</dateName>
<isHoliday>Y|N</isHoliday><dateKind>NN</dateKind></item>` rows, paged by
(solYear, solMonth).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import EXTERNAL_DATA_DIR, data_go_kr_api_key
from ._http import get_xml, iter_items

BASE_URL = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService"
NUM_OF_ROWS = 100  # safely > any single month's special-day count


def _fetch_month(operation: str, year: int, month: int) -> list[dict]:
    url = f"{BASE_URL}/{operation}"
    params = {
        "ServiceKey": data_go_kr_api_key(),
        "solYear": f"{year:04d}",
        "solMonth": f"{month:02d}",
        "numOfRows": NUM_OF_ROWS,
        "pageNo": 1,
    }
    root = get_xml(url, params)
    return iter_items(root)


def fetch_holidays(start_year: int, end_year: int) -> pd.DataFrame:
    """Return [date, name, is_holiday, kind, source='holiday'] rows."""
    rows = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            rows.extend(_fetch_month("getRestDeInfo", year, month))
    return _normalize(rows, source="holiday")


def fetch_divisions(start_year: int, end_year: int) -> pd.DataFrame:
    """Return [date, name, is_holiday, kind, source='division'] rows."""
    rows = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            rows.extend(_fetch_month("get24DivisionsInfo", year, month))
    return _normalize(rows, source="division")


def _normalize(rows: list[dict], *, source: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "name", "is_holiday", "kind", "source"])
    df = pd.DataFrame(rows)
    df = df.rename(columns={"locdate": "date", "dateName": "name", "dateKind": "kind"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.normalize()
    df["is_holiday"] = df["isHoliday"].fillna("N").map({"Y": True, "N": False}).astype(bool)
    df["source"] = source
    keep = ["date", "name", "is_holiday", "kind", "source"]
    return df[keep].sort_values("date").reset_index(drop=True)


def backfill(start_year: int, end_year: int, *, out_dir: Path | None = None) -> Path:
    """Fetch and persist both operations to a single parquet file."""
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    holidays = fetch_holidays(start_year, end_year)
    divisions = fetch_divisions(start_year, end_year)
    combined = pd.concat([holidays, divisions], ignore_index=True).sort_values(
        ["date", "source"]
    ).reset_index(drop=True)
    out_path = out_dir / "calendar_raw.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path

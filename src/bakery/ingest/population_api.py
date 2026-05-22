"""행정안전부 `admmSexdAgePpltn` (행정동 성/연령별 주민등록 인구수) 어댑터.

End Point: https://apis.data.go.kr/1741000/admmSexdAgePpltn/selectAdmmSexdAgePpltn

Mandatory parameters (from swagger):
  serviceKey  - DATA_GO_KR_API_KEY (활용신청 후 동일 키 사용)
  admmCd      - 10-digit 행안부 행정기관코드. With `lv=3`, the first 4 digits
                are read as the sigungu and the response contains ALL admin
                dongs in that sigungu (last 6 digits are ignored).
  srchFrYm    - YYYYMM, earliest 2022.10
  srchToYm    - YYYYMM, must be within 3 months of srchFrYm
  lv          - 3 = 읍면동 단위 (we want this), 7 = 단일 읍면동
  type        - 'json'

Strategy: derive the 4-digit sigungu prefix from each store's 8-digit
`admin_dong_code` (SPOP/소비와 동일 체계의 앞 4자리 = 시군구), call
`lv=3` per sigungu, then match `dongNm` back to our stores'
`admin_dong_name` to attach the 8-digit code on the result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

from ..config import EXTERNAL_DATA_DIR, admin_pop_api_key
from ..ingest.store_mapping import StationMapping, load_store_mapping
from ._http import DEFAULT_THROTTLE_SECONDS, DEFAULT_TIMEOUT, ApiError

URL = "https://apis.data.go.kr/1741000/admmSexdAgePpltn/selectAdmmSexdAgePpltn"
MAX_ROWS = 100  # swagger-enforced cap

_AGE_BIN_TO_COHORTS: dict[str, list[int]] = {
    "0_9": [0],
    "10_19": [10],
    "20_29": [20],
    "30_39": [30],
    "40_49": [40],
    "50_59": [50],
    "60_plus": [60, 70, 80, 90, 100],
}


def _fetch_page(*, sigungu_prefix: str, srch_ym: str, page_no: int) -> tuple[list[dict], int]:
    params = {
        "serviceKey": admin_pop_api_key(),
        "type": "json",
        "lv": 3,
        "admmCd": f"{sigungu_prefix}000000",
        "srchFrYm": srch_ym,
        "srchToYm": srch_ym,
        "numOfRows": MAX_ROWS,
        "pageNo": page_no,
    }
    r = httpx.get(URL, params=params, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    payload = json.loads(r.text)
    body = payload.get("Response") or {}
    head = body.get("head") or {}
    code = head.get("resultCode")
    if code not in ("0", "00"):
        raise ApiError(
            f"admmSexdAgePpltn sigungu={sigungu_prefix} ym={srch_ym} "
            f"resultCode={code} msg={head.get('resultMsg')}"
        )
    items = body.get("items") or {}
    if isinstance(items, dict):
        rows = items.get("item")
        rows = [rows] if isinstance(rows, dict) else (rows or [])
    elif isinstance(items, list):
        rows = items
    else:
        rows = []
    total = int(head.get("totalCount") or len(rows))
    time.sleep(DEFAULT_THROTTLE_SECONDS)
    return rows, total


def fetch_for_sigungu(sigungu_prefix: str, srch_ym: str) -> pd.DataFrame:
    """One sigungu, one month — paginated."""
    first_rows, total = _fetch_page(sigungu_prefix=sigungu_prefix, srch_ym=srch_ym, page_no=1)
    all_rows = list(first_rows)
    pages = (total + MAX_ROWS - 1) // MAX_ROWS
    for p in range(2, pages + 1):
        rows, _ = _fetch_page(sigungu_prefix=sigungu_prefix, srch_ym=srch_ym, page_no=p)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def normalize(raw: pd.DataFrame, *, mapping: dict[str, StationMapping]) -> pd.DataFrame:
    """Raw → (admin_dong_code (8-digit), ym, age_bin, sex, population).

    Joins raw rows to our stores via `sggNm + dongNm` (the 10-digit `admmCd`
    is a different coding system from the 8-digit ADSTRD code used by
    SPOP/Consumption, so we use the human-readable dong name as the bridge).
    """
    if raw.empty:
        return pd.DataFrame()
    name_to_code: dict[tuple[str, str], str] = {}
    for entry in mapping.values():
        sgg_prefix = entry["admin_dong_code"][:4]
        name_to_code[(sgg_prefix, entry["admin_dong_name"])] = entry["admin_dong_code"]
    raw = raw.copy()
    raw["_sgg_prefix"] = raw["admmCd"].astype("string").str[:4]
    raw["admin_dong_code"] = raw.apply(
        lambda r: name_to_code.get((r["_sgg_prefix"], r["dongNm"])), axis=1
    )
    raw = raw[raw["admin_dong_code"].notna()]
    if raw.empty:
        return pd.DataFrame()
    # Sum tong/ban rows within each (dong, statsYm) — for lv=3 these are
    # already aggregated but we group defensively in case of duplicates.
    long_rows: list[dict] = []
    for sex_key, prefix in (("M", "male"), ("F", "feml")):
        for age_bin, cohort_starts in _AGE_BIN_TO_COHORTS.items():
            cols = [f"{prefix}{start}AgeNmprCnt" for start in cohort_starts]
            cols = [c for c in cols if c in raw.columns]
            if not cols:
                continue
            count = raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            grouped = pd.DataFrame(
                {
                    "admin_dong_code": raw["admin_dong_code"].astype("string"),
                    "ym": raw["statsYm"].astype("string"),
                    "population": count,
                }
            ).groupby(["admin_dong_code", "ym"], as_index=False)["population"].sum()
            for _, row in grouped.iterrows():
                long_rows.append(
                    {
                        "admin_dong_code": str(row["admin_dong_code"]),
                        "ym": _yyyymm_to_label(row["ym"]),
                        "age_bin": age_bin,
                        "sex": sex_key,
                        "population": int(row["population"]),
                    }
                )
    return pd.DataFrame(long_rows)


def _yyyymm_to_label(ym: str) -> str:
    s = str(ym)
    if len(s) == 6:
        return f"{s[:4]}-{s[4:]}"
    return s


def backfill(
    *,
    out_dir: Path | None = None,
    srch_ym: str | None = None,
    mapping_path: Path | None = None,
) -> Path:
    """Fetch one monthly snapshot covering all our stores' sigungus → `population.parquet`.

    `srch_ym` defaults to the previous full month (YYYYMM); the API requires
    a window within the last few months of available data.
    """
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if srch_ym is None:
        srch_ym = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y%m")
    mapping = load_store_mapping(mapping_path)
    sigungu_prefixes = sorted({entry["admin_dong_code"][:4] for entry in mapping.values()})

    frames: list[pd.DataFrame] = []
    for prefix in sigungu_prefixes:
        frame = fetch_for_sigungu(prefix, srch_ym)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"admmSexdAgePpltn returned no rows for ym={srch_ym}")
    raw = pd.concat(frames, ignore_index=True)
    out = normalize(raw, mapping=mapping)
    if out.empty:
        raise RuntimeError(
            f"normalize produced no rows. raw dongs: {raw['dongNm'].unique().tolist()[:10]}"
        )
    out_path = out_dir / "population.parquet"
    out.to_parquet(out_path, index=False)
    return out_path

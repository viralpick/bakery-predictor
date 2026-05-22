"""Seoul living-population CSV history ingester.

OA-14991 OpenAPI only retains the last ~2 months. The full history is
exposed as monthly zips on the dataset's "파일" tab:
`LOCAL_PEOPLE_DONG_YYYYMM.zip`. Drop them into
`data/external/living_pop_zips/` and run `bakery ingest-living-pop-csv`.

Each zip contains a single CSV; column names vary slightly across vintages
(2017~2022 vs 2023+) so we map candidate Korean/English headers onto our
internal schema:

  기준일ID / STDR_DE_ID            → date
  시간대구분 / TMZON_PD_SE          → hour
  행정동코드 / ADSTRD_CODE_SE       → admin_dong_code
  총생활인구수 / TOT_LVPOP_CO       → total_pop

Existing rows in `living_population.parquet` are merged with the new ones
(dedup on (admin_dong_code, date, hour)) so re-runs are idempotent.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from ..config import EXTERNAL_DATA_DIR
from ..ingest.store_mapping import load_store_mapping

ZIP_DIR_DEFAULT = EXTERNAL_DATA_DIR / "living_pop_zips"

# Column candidates (encountered shapes across LOCAL_PEOPLE_DONG vintages).
_DATE_CANDIDATES = ["기준일ID", "STDR_DE_ID", "stdr_de_id"]
_HOUR_CANDIDATES = ["시간대구분", "TMZON_PD_SE", "tmzon_pd_se"]
_DONG_CANDIDATES = ["행정동코드", "ADSTRD_CODE_SE", "adstrd_code_se", "adstrd_cd"]
_TOTPOP_CANDIDATES = ["총생활인구수", "TOT_LVPOP_CO", "tot_lvpop_co"]

_CSV_ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def ingest_zip_dir(
    *,
    zip_dir: Path | None = None,
    admin_dong_codes: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Read every `LOCAL_PEOPLE_DONG_*.zip` in `zip_dir`, normalize, merge with
    any existing `living_population.parquet`, and persist."""
    zip_dir = zip_dir or ZIP_DIR_DEFAULT
    if not zip_dir.exists():
        raise FileNotFoundError(f"zip dir not found: {zip_dir}")
    zips = sorted(zip_dir.glob("LOCAL_PEOPLE_DONG_*.zip"))
    if not zips:
        raise FileNotFoundError(f"no LOCAL_PEOPLE_DONG_*.zip in {zip_dir}")
    if admin_dong_codes is None:
        mapping = load_store_mapping()
        admin_dong_codes = sorted({s["admin_dong_code"] for s in mapping.values()})
    frames: list[pd.DataFrame] = []
    for path in zips:
        frame = _read_zip(path, admin_dong_codes=admin_dong_codes)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no rows ingested from any zip")
    fresh = pd.concat(frames, ignore_index=True)

    out_path = out_path or (EXTERNAL_DATA_DIR / "living_population.parquet")
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = fresh
    combined = (
        combined.drop_duplicates(subset=["admin_dong_code", "date", "hour"])
        .sort_values(["admin_dong_code", "date", "hour"])
        .reset_index(drop=True)
    )
    combined.to_parquet(out_path, index=False)
    return out_path


def _read_zip(path: Path, *, admin_dong_codes: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"no .csv inside {path}")
        # If multiple CSVs (rare), read them all and concatenate.
        chunks: list[pd.DataFrame] = []
        for name in csv_names:
            with zf.open(name) as f:
                raw_bytes = f.read()
            df = _decode_csv(raw_bytes)
            chunks.append(df)
    raw = pd.concat(chunks, ignore_index=True)
    return _normalize(raw, admin_dong_codes=admin_dong_codes)


def _decode_csv(raw_bytes: bytes) -> pd.DataFrame:
    """LOCAL_PEOPLE_DONG CSVs have inconsistent encodings (utf-8 BOM, cp949, etc.).
    Strip the UTF-8 BOM at byte level (so it doesn't survive a cp949 fallback
    as garbled characters), then try encodings; finally normalize column names."""
    if raw_bytes[:3] == b"\xef\xbb\xbf":
        raw_bytes = raw_bytes[3:]
    last_exc: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            # index_col=False keeps every column in the data — without this,
            # 2024+ CSVs trigger pandas' "ragged header" heuristic and shift
            # every data value one column to the right.
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, dtype=str, index_col=False)
            df.columns = [_clean_col(c) for c in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise ValueError(f"could not decode CSV with any of {_CSV_ENCODINGS}") from last_exc


def _clean_col(name: str) -> str:
    """Strip any leading non-Korean/ASCII junk (residual BOM, quotes) and outer quotes."""
    name = name.strip().strip('"').strip("'").strip()
    # If anything still hangs off the front (e.g. residual cp949-decoded BOM),
    # drop until we reach a Hangul or ASCII alphanumeric.
    while name and not (name[0].isalnum() or "가" <= name[0] <= "힣"):
        name = name[1:]
    return name


def _normalize(raw: pd.DataFrame, *, admin_dong_codes: list[str]) -> pd.DataFrame:
    if raw.empty:
        return raw
    date_col = _first_in(raw.columns, _DATE_CANDIDATES)
    hour_col = _first_in(raw.columns, _HOUR_CANDIDATES)
    dong_col = _first_in(raw.columns, _DONG_CANDIDATES)
    pop_col = _first_in(raw.columns, _TOTPOP_CANDIDATES)
    if not all((date_col, hour_col, dong_col, pop_col)):
        raise ValueError(
            f"living_population CSV missing expected columns. Found: {list(raw.columns)[:15]}"
        )
    raw = raw.copy()
    raw["admin_dong_code"] = raw[dong_col].astype("string")
    raw = raw[raw["admin_dong_code"].isin(admin_dong_codes)]
    if raw.empty:
        return raw
    out = pd.DataFrame(
        {
            "admin_dong_code": raw["admin_dong_code"],
            "date": pd.to_datetime(raw[date_col], format="%Y%m%d").dt.normalize(),
            "hour": pd.to_numeric(raw[hour_col], errors="coerce").astype("int8"),
            "total_pop": pd.to_numeric(raw[pop_col], errors="coerce").astype("float32"),
        }
    )
    return out.dropna(subset=["date", "hour", "total_pop"]).reset_index(drop=True)


def _first_in(actual: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in actual:
            return c
    return None

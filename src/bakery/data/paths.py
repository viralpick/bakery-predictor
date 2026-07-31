"""데이터 파일 위치의 단일 출처. 하드코딩 리터럴 대신 dataset(name)을 쓴다.

레이어: raw(불변 원본) / interim(소스별 클린) / processed(canonical 사용데이터).
Task 1 인벤토리 분류에 대응.
"""
from __future__ import annotations

from pathlib import Path

from bakery.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

_INTERNAL = PROCESSED_DIR / "internal"
_EXTERNAL = PROCESSED_DIR / "external"
_RAW_INTERNAL = RAW_DIR / "internal"
_RAW_EXTERNAL = RAW_DIR / "external"

# name -> absolute Path. Task 1 인벤토리와 1:1 대응.
_DATASETS: dict[str, Path] = {
    # --- raw sources (불변) ---
    "sales_xlsx": _RAW_INTERNAL / "보나비 판매 데이터_20260721.xlsx",
    "master_xlsx": _RAW_INTERNAL / "보나비 데이터_20260526.xlsx",
    "legacy_xlsx_0520": _RAW_INTERNAL / "보나비 데이터_20260520.xlsx",
    "display_time_xls": _RAW_INTERNAL / "수원광교점 - 브레드 진열 시간(보안 해제 완료).xls",
    # ★2026 상반기 재고·영업시간·품절 (master의 상위집합 — 2021~2025 값 충돌 0건 실측).
    #   재고정보에 신규 컬럼 QT_MAKE(제시량)/QT_ADD(추가량)이 추가돼 있다.
    #   ⚠️영업시간 SALE_TIME 포맷이 master(HHMM)와 다르다(YYYYMMDDHHMMSS) — per-file 파싱 필요.
    "additional_xlsx": _RAW_INTERNAL / "보나비 추가 데이터_20260721.xlsx",
    # 아띠제 제공 브레드 배수 마스터(맞춤수량) + 대체품 매핑
    "ai_production_xlsx": _RAW_INTERNAL / "ai생산량 정보전달(수원광교점) 암호해제완료.xlsx",
    # --- interim ---
    "sales_lines_clean": INTERIM_DIR / "sales_lines_clean.parquet",
    # --- processed / internal (rebuild-deterministic) ---
    "bonavi_daily": _INTERNAL / "bonavi_daily.parquet",
    "multistore_daily": _INTERNAL / "multistore_daily.parquet",
    "bonavi_receipts": _INTERNAL / "bonavi_receipts.parquet",
    # waste_alpha_4stores: 유일하게 src/가 소비하는 data/internal/v2/ 파일
    # (cli.py:1616 CLOSING_DEMAND_WASTE_PARQUET, 읽기는 :1630, :1713). Task 5에서 이관.
    "waste_alpha_4stores": _INTERNAL / "waste_alpha_4stores.parquet",
    # --- processed / external (move-only) ---
    "weather_observed": _EXTERNAL / "weather_observed.parquet",
    "calendar_raw": _EXTERNAL / "calendar_raw.parquet",
    "competitor_raw": _EXTERNAL / "competitor_raw.parquet",
    "consumption": _EXTERNAL / "consumption.parquet",
    "population": _EXTERNAL / "population.parquet",
    "living_population": _EXTERNAL / "living_population.parquet",
    "forecast_short_term": _EXTERNAL / "forecast_short_term.parquet",
    "forecast_short_term_daily": _EXTERNAL / "forecast_short_term_daily.parquet",
    "forecast_mid_term_daily": _EXTERNAL / "forecast_mid_term_daily.parquet",
}

# living_pop_zips: 동적으로 이름 붙는 zip들의 디렉터리라 dataset(name) 1:1 매핑에 안 맞음.
# 유일한 src 소비처=ingest/living_population_csv.py:32 ZIP_DIR_DEFAULT. Task 5에서 재배선.
# Task 1 인벤토리 new_path: data/raw/external/living_pop_zips/
LIVING_POP_ZIPS_DIR = _RAW_EXTERNAL / "living_pop_zips"


def dataset(name: str) -> Path:
    """등록된 데이터 파일의 절대 경로. 미등록 이름은 KeyError."""
    if name not in _DATASETS:
        raise KeyError(
            f"unknown dataset '{name}'. known: {sorted(_DATASETS)}"
        )
    return _DATASETS[name]


def list_datasets() -> list[str]:
    return sorted(_DATASETS)

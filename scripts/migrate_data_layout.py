"""1회성 데이터 레이아웃 이동 스크립트 (Task 3).

data/{internal,external} 평면 구조 -> data/{raw,interim,processed}/{internal,external}
로 물리 이동(byte-preserving `shutil.move`)하고, scripts/의 하드코딩 옛 경로가 깨지지
않도록 옛 위치에 `os.symlink`를 남긴다.

분류 근거: docs/superpowers/plans/data-inventory-2026-07-25.md (Task 1 인벤토리).
- registry 등재 파일(17종, `bakery.data.paths._DATASETS`) -> `paths.dataset(name)`으로 목적지 확정.
- registry 밖(scripts 전용 v2/ 테이블 9종 + 파생 3종 + orphan 1종) -> DIRECT_MOVES에 직접 명시.
- `living_pop_zips/` 디렉토리 -> `paths.LIVING_POP_ZIPS_DIR`.
- cruft(.pre-*-bak 2종, item_active_stats.parquet, .DS_Store 2종) -> `data/_archive/`,
  심링크 없음(참조 0건 확인, grep 근거는 인벤토리 §3.5 및 Task 3 리포트 참조).

재실행 안전(idempotent): 옛 경로가 이미 심링크면 skip.
`v2/sales.parquet`은 재빌드 스크립트가 현재 파일을 재현하지 못함(인벤토리 §Step2) —
반드시 byte-preserving move만 하고 재생성 시도 금지.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from bakery.data import paths

_ROOT = paths.PROJECT_ROOT
_ARCHIVE = paths.DATA_DIR / "_archive"

# --- registry 등재 데이터셋: paths.dataset(name)이 목적지 ---
REGISTRY_MOVES: dict[str, str] = {
    "data/internal/보나비 데이터_20260520.xlsx": "legacy_xlsx_0520",
    "data/internal/보나비 데이터_20260526.xlsx": "master_xlsx",
    "data/internal/보나비 판매 데이터_20260721.xlsx": "sales_xlsx",
    "data/internal/수원광교점 - 브레드 진열 시간(보안 해제 완료).xls": "display_time_xls",
    "data/internal/sales_lines_clean.parquet": "sales_lines_clean",
    "data/internal/bonavi_daily.parquet": "bonavi_daily",
    "data/internal/bonavi_receipts.parquet": "bonavi_receipts",
    # waste_alpha_4stores: v2/ 안에 있었지만 src/bakery/cli.py가 실제로 읽는 유일한 v2 파일.
    # registry 새 경로는 v2/ 밖(data/processed/internal/) — Task 5에서 cli.py 배선 이관 예정,
    # 그 전까지는 옛 v2/ 위치의 심링크가 하위호환을 담당한다.
    "data/internal/v2/waste_alpha_4stores.parquet": "waste_alpha_4stores",
    "data/external/weather_observed.parquet": "weather_observed",
    "data/external/calendar_raw.parquet": "calendar_raw",
    "data/external/competitor_raw.parquet": "competitor_raw",
    "data/external/consumption.parquet": "consumption",
    "data/external/population.parquet": "population",
    "data/external/living_population.parquet": "living_population",
    "data/external/forecast_short_term.parquet": "forecast_short_term",
    "data/external/forecast_short_term_daily.parquet": "forecast_short_term_daily",
    "data/external/forecast_mid_term_daily.parquet": "forecast_mid_term_daily",
}

# --- registry 밖 이동: 옛 경로(str) -> 새 절대 경로(Path), 직접 명시 ---
_V2_INTERIM = paths.INTERIM_DIR / "v2"
_V2_PROCESSED_INTERNAL = paths.PROCESSED_DIR / "internal" / "v2"

DIRECT_MOVES: dict[str, Path] = {
    # move-only orphan (재빌드 스크립트 없음, 삭제 금지) — 인벤토리 §3.2
    "data/internal/bonavi_daily_2026h1_covariate.parquet": (
        paths.INTERIM_DIR / "bonavi_daily_2026h1_covariate.parquet"
    ),
    # v2/ scripts-only interim 테이블 (sales.parquet은 move-only, 재빌드 금지 — §Step2)
    "data/internal/v2/sales_p1.parquet": _V2_INTERIM / "sales_p1.parquet",
    "data/internal/v2/sales_p2.parquet": _V2_INTERIM / "sales_p2.parquet",
    "data/internal/v2/sales.parquet": _V2_INTERIM / "sales.parquet",
    "data/internal/v2/inventory.parquet": _V2_INTERIM / "inventory.parquet",
    "data/internal/v2/items.parquet": _V2_INTERIM / "items.parquet",
    "data/internal/v2/stores.parquet": _V2_INTERIM / "stores.parquet",
    "data/internal/v2/hours.parquet": _V2_INTERIM / "hours.parquet",
    "data/internal/v2/stockout.parquet": _V2_INTERIM / "stockout.parquet",
    "data/internal/v2/discount_codes.parquet": _V2_INTERIM / "discount_codes.parquet",
    # v2/ scripts-only processed/internal 파생 테이블
    "data/internal/v2/daily_4stores.parquet": _V2_PROCESSED_INTERNAL / "daily_4stores.parquet",
    "data/internal/v2/daily_normal_vs_bulk.parquet": (
        _V2_PROCESSED_INTERNAL / "daily_normal_vs_bulk.parquet"
    ),
    # cruft 아님(3개 스크립트가 여전히 참조 — 인벤토리 §3.6)
    "data/internal/v2/sales_with_bulk_flag.parquet": (
        _V2_PROCESSED_INTERNAL / "sales_with_bulk_flag.parquet"
    ),
}

# --- 디렉토리 이동 ---
DIR_MOVES: dict[str, Path] = {
    "data/external/living_pop_zips": paths.LIVING_POP_ZIPS_DIR,
}

# --- cruft: data/_archive/로 격리, 심링크 없음(참조 0건 확인됨) ---
ARCHIVE_MOVES: dict[str, Path] = {
    "data/internal/bonavi_daily.parquet.pre-v2-bak": (
        _ARCHIVE / "internal" / "bonavi_daily.parquet.pre-v2-bak"
    ),
    "data/internal/bonavi_receipts.parquet.pre-v2-bak": (
        _ARCHIVE / "internal" / "bonavi_receipts.parquet.pre-v2-bak"
    ),
    "data/internal/v2/sales.parquet.pre-new-bak": (
        _ARCHIVE / "internal" / "v2" / "sales.parquet.pre-new-bak"
    ),
    "data/internal/v2/item_active_stats.parquet": (
        _ARCHIVE / "internal" / "v2" / "item_active_stats.parquet"
    ),
    "data/internal/.DS_Store": _ARCHIVE / "internal" / ".DS_Store",
    "data/external/.DS_Store": _ARCHIVE / "external" / ".DS_Store",
}


def _move_with_symlink(old: Path, new: Path) -> str:
    """old -> new byte-preserving 이동 후 old 위치에 심링크. idempotent.

    idempotency는 "이미 심링크"인 경우만 skip으로 커버한다 — old가 심링크도 아니고
    실체 파일도 아닌(완전히 없는) 경우엔 skip(missing)만 반환하고 복구(재심링크)하지 않는다.
    """
    if old.is_symlink():
        return "skip(already-symlinked)"
    if not old.exists():
        return "skip(missing)"
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    os.symlink(new, old)
    return "moved+symlinked"


def _move_to_archive(old: Path, new: Path) -> str:
    """cruft 격리: byte-preserving 이동, 심링크 없음."""
    if not old.exists():
        return "skip"
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    return "archived"


def migrate() -> dict[str, str]:
    results: dict[str, str] = {}
    for old_str, name in REGISTRY_MOVES.items():
        old = _ROOT / old_str
        new = paths.dataset(name)
        results[old_str] = _move_with_symlink(old, new)
    for old_str, new in DIRECT_MOVES.items():
        old = _ROOT / old_str
        results[old_str] = _move_with_symlink(old, new)
    for old_str, new in DIR_MOVES.items():
        old = _ROOT / old_str
        results[old_str] = _move_with_symlink(old, new)
    for old_str, new in ARCHIVE_MOVES.items():
        old = _ROOT / old_str
        results[old_str] = _move_to_archive(old, new)
    return results


if __name__ == "__main__":
    for old_str, status in migrate().items():
        print(f"{status:24s} {old_str}")

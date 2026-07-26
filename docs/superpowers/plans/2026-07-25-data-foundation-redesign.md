# 데이터 파운데이션 재설계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 원본(Excel)↔파생(parquet) 위치를 `data/{raw,interim,processed}`로 분리하고, 경로를 named registry로 중앙화하며, 재현 가능한 단일진입 파이프라인·외부 갱신 CLI·크로스체크 커버리지 리포트를 세우고, 매진률 진입점 희석을 교정한다.

**Architecture:** `src/bakery`(프리미티브)는 유지. 신규 `src/bakery/data/paths.py`(경로 단일 출처)·`pipeline.py`(raw→interim→processed 오케스트레이터, 기존 loader 호출만)·`src/bakery/ingest/refresh.py`(외부 갱신)를 추가한다. 재배치는 byte-preserving(수치 불가침)이라 테스트+byte-identity로 보증하고, `build-data`는 내부 결정적 테이블에 rtol=1e-9 진단 게이트를 가진다.

**Tech Stack:** Python 3.11, uv, pandas, pyarrow, typer(CLI), pytest, LightGBM(기존 모델, 미변경).

## Global Constraints

- **재구현 금지** — 기존 `bonavi_loader`/`bonavi_loader_v2`/`build_*`/`ingest/*_api` 로직을 호출만 한다. 모델/피처/평가 로직 미변경.
- **원본 불변** — `data/raw/`는 read-only. 파이프라인은 raw를 읽되 절대 쓰지 않는다.
- **탐지 우선** — 크로스소스 화해(closing 통일 등)는 하지 않는다. 리포트로 표면화만.
- **측정헌장 2번** — 검열(censored) 데이터를 0으로 깔지 않는다(매진률 fix).
- **테스트 규칙** — 기대값 아는 단언은 정확값 `==`. 비결정/타임스탬프만 느슨한 단언, 이유 주석.
- **커밋 트레일러** — `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_015231oEXPA9S82TZf9Ue933`.
- **pytest** — 카운트 필요 시 `uv run pytest --color=no` (repo addopts에 `-q` 있어 `-q` 추가 시 요약 사라짐).
- **브랜치** — `feat/data-foundation-redesign` (스펙 커밋됨).

---

## Task 1: 경로 참조 & 아티팩트 인벤토리 (discovery)

목적: 파일을 하나라도 옮기기 전에 (a) 모든 데이터 경로 참조를 열거하고 (b) 각 processed 아티팩트를 **재빌드-결정적 / move-only / cruft**로 분류한다. `no-yolo` 규칙 준수의 전제.

**Files:**
- Create: `docs/superpowers/plans/data-inventory-2026-07-25.md` (durable 산출물)

**Interfaces:**
- Produces: 아티팩트 분류표(파일명 → {layer, class, builder, consumers}) — Task 2/3가 이 표를 registry·이동 매핑의 근거로 씀.

- [ ] **Step 1: 데이터 경로 리터럴 전수 열거**

Run:
```bash
grep -rnE "data/(internal|external)[^\"')]*\.(parquet|xlsx|xls)" src/ scripts/ tests/ experiments/ > /tmp/data_refs.txt
grep -rnE "data/internal|data/external" experiments/ tests/ | grep -iE "\.parquet|\.xls" >> /tmp/data_refs.txt
wc -l /tmp/data_refs.txt
```
Expected: 참조 목록 파일. src/ vs scripts/ vs tests/ 분리 확인.

- [ ] **Step 2: 각 데이터 파일의 소비처/빌더 조사**

`data/internal`, `data/internal/v2`, `data/external`의 모든 파일에 대해:
- **소비처**: `grep -rln "<filename>" src/ scripts/ tests/`
- **빌더**: 어느 CLI 커맨드/스크립트가 생성하는가 (`format-bonavi`, `format-bonavi-v2`, `ingest-*`, `scripts/convert_xlsx_20260526.py` 등).
- 특히 **`v2/` 테이블(inventory/items/stores/stockout/sales/discount/hours)의 빌더**를 확정한다 (src 소비처 0으로 확인됨 — scripts 전용인지, orphan인지 판정).

- [ ] **Step 3: 분류표 작성 + 커밋**

`docs/superpowers/plans/data-inventory-2026-07-25.md`에 표 작성. 각 파일:
- **layer**: raw / interim / processed(internal|external)
- **class**:
  - `rebuild-deterministic` — Excel/clean에서 결정적 재생성 (bonavi_daily, bonavi_receipts, sales_lines_clean)
  - `move-only` — 외부 API 유래 or 재생성 불가 orphan (weather_observed, calendar_raw, competitor_raw, consumption, population, living_population, forecast_*)
  - `cruft` — 소비처 0 & 대체 가능 (`*.pre-*-bak`, 필요 시 `sales_p1/p2`, `sales_with_bulk_flag`)
  - `raw-source` — 불변 원본 (0520.xlsx, 0526.xlsx, 0721.xlsx, 진열시간.xls, living_pop_zips)
- **consumers**: src/scripts/tests 중 어디서 읽는가
- **new_path**: 이동 후 위치

```bash
git add docs/superpowers/plans/data-inventory-2026-07-25.md
git commit -m "docs: 데이터 아티팩트 인벤토리·분류표 (Task 1)"
```

Expected: 모든 데이터 파일이 정확히 한 class로 분류됨. 애매한 파일(예: `waste_alpha_4stores.parquet`, `bonavi_daily_2026h1_covariate.parquet`)은 소비처 grep 근거와 함께 판정.

---

## Task 2: paths.py — 경로 단일 출처 registry

**Files:**
- Create: `src/bakery/data/paths.py`
- Test: `tests/test_data_paths.py`

**Interfaces:**
- Consumes: `bakery.config.PROJECT_ROOT`
- Produces:
  - `RAW_DIR: Path`, `INTERIM_DIR: Path`, `PROCESSED_DIR: Path`
  - `dataset(name: str) -> Path` — 이름→경로. 미등록 이름은 `KeyError`.
  - `list_datasets() -> list[str]`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_data_paths.py
from pathlib import Path
import pytest
from bakery.data import paths


def test_layer_roots_under_data():
    assert paths.RAW_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.INTERIM_DIR == paths.PROJECT_ROOT / "data" / "interim"
    assert paths.PROCESSED_DIR == paths.PROJECT_ROOT / "data" / "processed"


def test_dataset_resolves_internal_processed():
    assert paths.dataset("bonavi_daily") == (
        paths.PROCESSED_DIR / "internal" / "bonavi_daily.parquet"
    )
    assert paths.dataset("bonavi_receipts") == (
        paths.PROCESSED_DIR / "internal" / "bonavi_receipts.parquet"
    )


def test_dataset_resolves_interim_and_external():
    assert paths.dataset("sales_lines_clean") == (
        paths.INTERIM_DIR / "sales_lines_clean.parquet"
    )
    assert paths.dataset("weather_observed") == (
        paths.PROCESSED_DIR / "external" / "weather_observed.parquet"
    )


def test_dataset_resolves_raw_sources():
    assert paths.dataset("sales_xlsx") == (
        paths.RAW_DIR / "internal" / "보나비 판매 데이터_20260721.xlsx"
    )
    assert paths.dataset("master_xlsx") == (
        paths.RAW_DIR / "internal" / "보나비 데이터_20260526.xlsx"
    )


def test_unknown_dataset_raises_keyerror_with_known_names():
    with pytest.raises(KeyError, match="unknown dataset"):
        paths.dataset("does_not_exist")
    assert "bonavi_daily" in paths.list_datasets()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_data_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: bakery.data.paths`.

- [ ] **Step 3: paths.py 구현**

```python
# src/bakery/data/paths.py
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
    # --- interim ---
    "sales_lines_clean": INTERIM_DIR / "sales_lines_clean.parquet",
    # --- processed / internal (rebuild-deterministic) ---
    "bonavi_daily": _INTERNAL / "bonavi_daily.parquet",
    "bonavi_receipts": _INTERNAL / "bonavi_receipts.parquet",
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
# NOTE: Task 1 인벤토리에서 추가로 분류된 파일(waste_alpha_4stores 등)은
# 여기에 등록한다. 등록 항목은 인벤토리 분류표를 단일 근거로 한다.


def dataset(name: str) -> Path:
    """등록된 데이터 파일의 절대 경로. 미등록 이름은 KeyError."""
    if name not in _DATASETS:
        raise KeyError(
            f"unknown dataset '{name}'. known: {sorted(_DATASETS)}"
        )
    return _DATASETS[name]


def list_datasets() -> list[str]:
    return sorted(_DATASETS)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_data_paths.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/paths.py tests/test_data_paths.py
git commit -m "feat(data): 경로 단일 출처 registry paths.py (Task 2)"
```

---

## Task 3: 물리 이동 + 하위호환 심링크 shim

목적: raw/interim/processed 레이아웃으로 파일을 옮기고, scripts가 참조하는 옛 경로에 심링크를 남겨 안 깨지게 한다. **이동은 byte-preserving** — 수치 불가침을 byte-identity로 확인.

**Files:**
- Create: `scripts/migrate_data_layout.py` (1회성 이동 헬퍼)
- Test: `tests/test_data_layout_migration.py`

**Interfaces:**
- Consumes: `paths.dataset(...)`, Task 1 인벤토리 분류표.
- Produces: `data/{raw,interim,processed}/...` 배치 + `data/internal/*`·`data/external/*` 옛 경로 심링크.

- [ ] **Step 1: 이동 전 byte-identity 스냅샷 채취**

Run:
```bash
mkdir -p /tmp/data_baseline_hashes
find data/internal data/external -maxdepth 2 -type f \( -name "*.parquet" -o -name "*.xlsx" -o -name "*.xls" \) -exec shasum -a256 {} \; | sort > /tmp/data_baseline_hashes/before.txt
wc -l /tmp/data_baseline_hashes/before.txt
```
Expected: 이동 대상 파일들의 sha256 목록.

- [ ] **Step 2: 실패 테스트 작성 (이동 후 상태 단언)**

```python
# tests/test_data_layout_migration.py
"""이동 후 레이아웃/심링크/byte-identity 회귀 테스트.
데이터가 gitignored라 로컬 환경 의존 → 파일 부재 시 skip."""
from pathlib import Path
import pytest
from bakery.data import paths

_REQUIRED = ["bonavi_daily", "bonavi_receipts", "sales_lines_clean",
             "weather_observed", "calendar_raw", "sales_xlsx", "master_xlsx"]


@pytest.mark.parametrize("name", _REQUIRED)
def test_dataset_exists_at_new_location(name):
    p = paths.dataset(name)
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated in this environment")
    assert p.exists(), f"{name} not at {p}"


def test_no_parquet_left_in_flat_internal_root():
    flat = paths.DATA_DIR / "internal"
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    # 옛 평면 루트엔 심링크만 허용, 실체 parquet 금지
    reals = [f for f in flat.glob("*.parquet") if f.is_file() and not f.is_symlink()]
    assert reals == [], f"real parquet still in flat root: {reals}"


def test_legacy_symlink_resolves(tmp_path):
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    legacy = paths.DATA_DIR / "internal" / "bonavi_daily.parquet"
    assert legacy.exists()  # 심링크 통해 해석
    assert legacy.resolve() == paths.dataset("bonavi_daily").resolve()
```

- [ ] **Step 3: 이동 실패 확인**

Run: `uv run pytest tests/test_data_layout_migration.py -v`
Expected: FAIL 또는 skip (아직 이동 안 함).

- [ ] **Step 4: 이동 스크립트 작성·실행**

`scripts/migrate_data_layout.py` — Task 1 분류표대로 `shutil.move` 후 옛 경로에 `os.symlink`. raw-source는 raw/internal로, rebuild-deterministic은 processed/internal로, move-only는 processed/external로, interim은 interim으로, cruft는 `data/_archive/`로 격리(삭제는 Task 이후 사용자 확인).

```python
# scripts/migrate_data_layout.py (핵심 로직)
import os, shutil
from pathlib import Path
from bakery.data import paths

MOVES = {  # 현재경로 -> paths.dataset 이름
    "data/internal/bonavi_daily.parquet": "bonavi_daily",
    "data/internal/bonavi_receipts.parquet": "bonavi_receipts",
    "data/internal/sales_lines_clean.parquet": "sales_lines_clean",
    "data/internal/보나비 판매 데이터_20260721.xlsx": "sales_xlsx",
    "data/internal/보나비 데이터_20260526.xlsx": "master_xlsx",
    "data/internal/보나비 데이터_20260520.xlsx": "legacy_xlsx_0520",
    "data/internal/수원광교점 - 브레드 진열 시간(보안 해제 완료).xls": "display_time_xls",
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
# living_pop_zips → raw/external/, data/internal/v2/·*.pre-*-bak → data/_archive/ (별도)


def migrate(make_symlink: bool = True) -> None:
    for old_str, name in MOVES.items():
        old = paths.PROJECT_ROOT / old_str
        new = paths.dataset(name)
        if not old.exists() or old.is_symlink():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        if make_symlink:  # scripts 하위호환 (gitignored 로컬 전용)
            os.symlink(new, old)


if __name__ == "__main__":
    migrate()
```

Run: `uv run python scripts/migrate_data_layout.py`
그다음 `living_pop_zips` 이동 + `v2/`·`*.pre-*-bak` → `data/_archive/` 이동을 스크립트에 포함해 실행.

- [ ] **Step 5: byte-identity 확인**

Run:
```bash
find data/raw data/interim data/processed -type f \( -name "*.parquet" -o -name "*.xlsx" -o -name "*.xls" \) -exec shasum -a256 {} \; | sort | awk '{print $1}' > /tmp/data_baseline_hashes/after_hashes.txt
awk '{print $1}' /tmp/data_baseline_hashes/before.txt | sort > /tmp/data_baseline_hashes/before_hashes.txt
comm -3 /tmp/data_baseline_hashes/before_hashes.txt /tmp/data_baseline_hashes/after_hashes.txt && echo "BYTE-IDENTICAL: all hashes matched"
```
Expected: `comm -3` 출력 없음 (해시 완전 일치) → "BYTE-IDENTICAL" 출력. cruft/archive 파일은 before 목록에서 제외했으므로 불일치 없음.

- [ ] **Step 6: 이동 테스트 통과 확인**

Run: `uv run pytest tests/test_data_layout_migration.py -v`
Expected: PASS (skip 아님 — 로컬 데이터 존재).

- [ ] **Step 7: 커밋**

```bash
git add scripts/migrate_data_layout.py tests/test_data_layout_migration.py
git commit -m "feat(data): raw/interim/processed 물리 이동 + 하위호환 심링크 (Task 3)"
```

---

## Task 4: src 내부 loader 소비처 마이그레이션

**Files:**
- Modify: `src/bakery/data/bonavi_loader_v2.py:41-45` (모듈 경로 상수)
- Modify: `src/bakery/data/bonavi_loader.py` (data/internal 6곳)
- Modify: `src/bakery/data/loader.py` (2곳), `src/bakery/features/category_aggregate.py` (5곳), `src/bakery/analysis/discount.py` (2곳)
- Test: `tests/test_data_paths_wiring.py`

**Interfaces:**
- Consumes: `paths.dataset(...)`
- Produces: 내부 loader가 registry 경로를 기본값으로 사용 (하드코딩 리터럴 제거).

- [ ] **Step 1: 실패 테스트 작성 (기본 경로가 registry를 가리킴)**

```python
# tests/test_data_paths_wiring.py
from bakery.data import bonavi_loader_v2 as v2
from bakery.data import paths


def test_bonavi_loader_v2_constants_use_registry():
    assert v2.NEW_SALES_XLSX == paths.dataset("sales_xlsx")
    assert v2.MASTER_XLSX == paths.dataset("master_xlsx")
    assert v2.CLEAN_PARQUET == paths.dataset("sales_lines_clean")
    assert v2.OUT_DEFAULT == paths.dataset("bonavi_daily")
    assert v2.RECEIPTS_DEFAULT == paths.dataset("bonavi_receipts")
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_data_paths_wiring.py -v`
Expected: FAIL — 상수가 아직 `Path("data/internal/...")` 리터럴.

- [ ] **Step 3: 경로 상수 교체**

`src/bakery/data/bonavi_loader_v2.py:41-45`:
```python
from bakery.data import paths

NEW_SALES_XLSX = paths.dataset("sales_xlsx")
MASTER_XLSX = paths.dataset("master_xlsx")
CLEAN_PARQUET = paths.dataset("sales_lines_clean")
OUT_DEFAULT = paths.dataset("bonavi_daily")
RECEIPTS_DEFAULT = paths.dataset("bonavi_receipts")
```
`bonavi_loader.py`·`loader.py`·`category_aggregate.py`·`discount.py`의 `data/internal|external` 리터럴을 Task 1 인벤토리 매핑대로 `paths.dataset(...)`로 교체. (순환 import 주의: `paths`는 `config`만 의존하므로 `data/*`에서 안전.)

- [ ] **Step 4: wiring 테스트 + 전체 스위트 통과 확인**

Run:
```bash
uv run pytest tests/test_data_paths_wiring.py -v
uv run pytest --color=no
```
Expected: wiring PASS. 전체 스위트 통과 (leakage 포함). 실패 시 해당 소비처 경로 매핑 수정 (self-heal 3회 이내).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/ src/bakery/features/category_aggregate.py src/bakery/analysis/discount.py tests/test_data_paths_wiring.py
git commit -m "feat(data): 내부 loader 소비처를 paths registry로 (Task 4)"
```

---

## Task 5: cli.py + config 소비처 마이그레이션

**Files:**
- Modify: `src/bakery/cli.py` (data/internal·external 25곳 + `format-bonavi-v2` 기본 인자 등)
- Modify: `src/bakery/config.py:15` (`EXTERNAL_DATA_DIR` → paths alias)
- Test: 기존 CLI smoke (신규 테스트는 최소)

**Interfaces:**
- Consumes: `paths.dataset(...)`, `paths.PROCESSED_DIR`
- Produces: CLI 명령이 registry 경로 사용. `config.EXTERNAL_DATA_DIR`는 하위호환 alias.

- [ ] **Step 1: config alias 교체**

`src/bakery/config.py:15`:
```python
# 하위호환 alias — 신규 코드는 bakery.data.paths 를 쓴다.
from bakery.data import paths as _paths  # noqa: E402 (config는 최소 의존 유지)
EXTERNAL_DATA_DIR = _paths.PROCESSED_DIR / "external"
```
순환 주의: `paths`가 `config.PROJECT_ROOT`를 import → `config`가 `paths`를 import하면 순환. **해결**: `EXTERNAL_DATA_DIR`를 `paths.py`에 정의하고 `config`는 `paths`를 import하지 않는다. 대신 `config.EXTERNAL_DATA_DIR = PROCESSED_DIR/"external"`를 직접 계산(PROJECT_ROOT는 config 소유). → config.py는 `PROJECT_ROOT / "data" / "processed" / "external"`로 직접 재정의.

```python
# src/bakery/config.py:15 (순환 없는 형태)
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "external"
```

- [ ] **Step 2: cli.py 리터럴 교체**

`cli.py`의 `data/internal`·`data/external` 25곳을 `paths.dataset(...)`로. `format-bonavi-v2`의 기본 인자(`sales_xlsx=Path("data/internal/...")`)도 `paths.dataset("sales_xlsx")`로. `EXTERNAL_DATA_DIR / "..."` 사용부는 그대로 두거나 `paths.dataset(...)`로 (동일 위치 해석).

- [ ] **Step 3: 전체 스위트 + CLI smoke 통과 확인**

Run:
```bash
uv run pytest --color=no
uv run bakery harness-run experiments/gwangyo_default.yaml 2>&1 | tail -20
```
Expected: 전체 스위트 통과. harness-run이 데이터 로드·백테스트 정상 수행(경로 오류 없음). WAPE ~8.03 sanity.

- [ ] **Step 4: 커밋**

```bash
git add src/bakery/cli.py src/bakery/config.py
git commit -m "feat(data): cli+config 소비처를 paths registry로 (Task 5)"
```

---

## Task 6: tests fixture 마이그레이션

**Files:**
- Modify: `tests/test_inventory_loader.py`, `tests/test_forecast_pipeline.py`, 그 외 Task 1에서 발견된 data 경로 fixture.

**Interfaces:**
- Consumes: `paths.dataset(...)`

- [ ] **Step 1: 테스트 fixture 경로 교체**

`tests/`의 `data/internal|external` 리터럴을 `paths.dataset(...)` 또는 tmp fixture로 교체. 데이터 부재 시 skip 가드 유지.

- [ ] **Step 2: 전체 스위트 통과 확인**

Run: `uv run pytest --color=no`
Expected: 통과. `grep -rn "data/internal\|data/external" tests/` → 0곳(또는 skip 가드 주석만).

- [ ] **Step 3: 커밋**

```bash
git add tests/
git commit -m "test(data): fixture 경로를 paths registry로 (Task 6)"
```

---

## Task 7: build-data 파이프라인 (internal orchestrator) + 동등성 진단

**Files:**
- Create: `src/bakery/data/pipeline.py`
- Test: `tests/test_build_data_pipeline.py`

**Interfaces:**
- Consumes: `bonavi_loader_v2.convert_sales_to_parquet`, `bonavi_loader_v2.build_v2`, `paths`.
- Produces:
  - `build_internal(reconvert: bool = False, out_root: Path | None = None) -> dict[str, Path]` — raw→interim→processed/internal 재생성, {name: path} 반환.
  - `equivalence_diff(rebuilt: dict[str, Path], reference: dict[str, Path]) -> dict[str, float]` — 재생성 vs 참조 최대 수치 diff.

- [ ] **Step 1: 실패 테스트 작성 (동등성 진단)**

```python
# tests/test_build_data_pipeline.py
import numpy as np, pandas as pd, pytest
from bakery.data import pipeline, paths


@pytest.mark.slow
def test_build_internal_reproduces_bonavi_daily(tmp_path):
    """clean→daily 결정적 재생성이 on-disk와 rtol=1e-9 일치 (2026-07-25 진단으로 확인됨)."""
    if not paths.dataset("sales_lines_clean").exists():
        pytest.skip("interim clean parquet 부재")
    rebuilt = pipeline.build_internal(reconvert=False, out_root=tmp_path)
    regen = pd.read_parquet(rebuilt["bonavi_daily"])
    disk = pd.read_parquet(paths.dataset("bonavi_daily"))
    keys = ["date", "item_id", "store_id"]
    r = regen.sort_values(keys).reset_index(drop=True)
    d = disk.sort_values(keys).reset_index(drop=True)
    assert list(r.columns) == list(d.columns)
    for c in r.select_dtypes(include=[np.number]).columns:
        max_diff = (r[c].fillna(-9e9) - d[c].fillna(-9e9)).abs().max()
        assert max_diff <= 1e-9, f"{c} diverged by {max_diff}"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_build_data_pipeline.py -v -m slow`
Expected: FAIL — `bakery.data.pipeline` 없음.

- [ ] **Step 3: pipeline.py 구현 (호출만, 재구현 금지)**

```python
# src/bakery/data/pipeline.py
"""raw → interim → processed 단일진입 오케스트레이터. 기존 loader 호출만."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.data import bonavi_loader_v2 as v2
from bakery.data import paths


def build_internal(reconvert: bool = False,
                   out_root: Path | None = None) -> dict[str, Path]:
    """내부 결정적 테이블 재생성. out_root 주면 그 밑에(테스트/진단용), 없으면 registry 위치."""
    clean = paths.dataset("sales_lines_clean")
    if reconvert or not clean.exists():
        v2.convert_sales_to_parquet(paths.dataset("sales_xlsx"), clean)
    daily = (out_root / "bonavi_daily.parquet") if out_root else paths.dataset("bonavi_daily")
    receipts = (out_root / "bonavi_receipts.parquet") if out_root else paths.dataset("bonavi_receipts")
    v2.build_v2(clean_parquet=clean, master_xlsx=paths.dataset("master_xlsx"),
                out_path=daily, receipts_path=receipts)
    return {"bonavi_daily": daily, "bonavi_receipts": receipts}


def equivalence_diff(rebuilt: dict[str, Path],
                     reference: dict[str, Path]) -> dict[str, float]:
    """재생성 vs 참조 테이블의 최대 수치 diff (0이면 완전 일치)."""
    out: dict[str, float] = {}
    for name, path in rebuilt.items():
        r = pd.read_parquet(path)
        d = pd.read_parquet(reference[name])
        keys = [c for c in ("date", "item_id", "store_id") if c in r.columns]
        r = r.sort_values(keys).reset_index(drop=True)
        d = d.sort_values(keys).reset_index(drop=True)
        max_diff = 0.0
        for c in r.select_dtypes(include=[np.number]).columns:
            max_diff = max(max_diff,
                           float((r[c].fillna(-9e9) - d[c].fillna(-9e9)).abs().max()))
        out[name] = max_diff
    return out
```

- [ ] **Step 4: 동등성 진단 테스트 통과 확인**

Run: `uv run pytest tests/test_build_data_pipeline.py -v -m slow`
Expected: PASS — bonavi_daily rtol=1e-9 일치.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/pipeline.py tests/test_build_data_pipeline.py
git commit -m "feat(data): build-data 내부 오케스트레이터 + 동등성 진단 (Task 7)"
```

---

## Task 8: `bakery build-data` CLI

**Files:**
- Modify: `src/bakery/cli.py` (신규 `@app.command("build-data")`)
- Test: `tests/test_cli_build_data.py`

**Interfaces:**
- Consumes: `pipeline.build_internal`, `pipeline.equivalence_diff`.

- [ ] **Step 1: 실패 테스트 작성 (idempotency)**

```python
# tests/test_cli_build_data.py
import pandas as pd, pytest
from bakery.data import pipeline, paths


@pytest.mark.slow
def test_build_internal_is_idempotent(tmp_path):
    if not paths.dataset("sales_lines_clean").exists():
        pytest.skip("interim 부재")
    a = pipeline.build_internal(out_root=tmp_path / "a")
    b = pipeline.build_internal(out_root=tmp_path / "b")
    diff = pipeline.equivalence_diff(a, {k: v for k, v in b.items()})
    assert all(v == 0.0 for v in diff.values()), diff
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_cli_build_data.py -v -m slow`
Expected: PASS 여부 무관하게 우선 로직 확인 — build_internal 존재하므로 통과할 수도. CLI 커맨드 부재는 Step 4에서 확인.

- [ ] **Step 3: CLI 커맨드 구현**

```python
# src/bakery/cli.py 에 추가
@app.command("build-data")
def cmd_build_data(reconvert: bool = False, diagnose: bool = True) -> None:
    """raw → interim → processed/internal 재생성. diagnose면 on-disk와 rtol 진단."""
    from .data import pipeline, paths
    import tempfile
    from pathlib import Path
    if diagnose:
        with tempfile.TemporaryDirectory() as td:
            rebuilt = pipeline.build_internal(reconvert=reconvert, out_root=Path(td))
            ref = {k: paths.dataset(k) for k in rebuilt}
            diff = pipeline.equivalence_diff(rebuilt, ref)
            for name, d in diff.items():
                tag = "OK" if d <= 1e-9 else "DRIFT"
                console.print(f"[{'green' if d<=1e-9 else 'red'}]{tag}[/] {name}: max_diff={d:g}")
            console.print("[dim]DRIFT는 §6 커버리지 리포트로 표면화(재배치 차단 아님)[/]")
    else:
        out = pipeline.build_internal(reconvert=reconvert)
        console.print(f"[green]wrote[/] {list(out.values())}")
```

- [ ] **Step 4: CLI smoke 통과 확인**

Run:
```bash
uv run bakery build-data --diagnose --no-reconvert 2>&1 | tail -10
uv run pytest tests/test_cli_build_data.py -v -m slow
```
Expected: `OK bonavi_daily: max_diff=0` / `OK bonavi_receipts: max_diff=0`. idempotency 테스트 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/cli.py tests/test_cli_build_data.py
git commit -m "feat(cli): bakery build-data 커맨드 + 동등성 진단 출력 (Task 8)"
```

---

## Task 9: `bakery refresh-external` CLI (병렬 가능)

**Files:**
- Create: `src/bakery/ingest/refresh.py`
- Modify: `src/bakery/cli.py` (신규 `@app.command("refresh-external")`)
- Test: `tests/test_refresh_external.py`

**Interfaces:**
- Consumes: 기존 `ingest/*_api` 함수(호출만), `paths`.
- Produces:
  - `SourceSpec` (name, dataset_key, kind="observed"|"forecast", refresh_fn).
  - `refresh_source(spec, today) -> RefreshResult(name, added_rows, last_date)` — observed=idempotent append, forecast=덮어쓰기.
  - `freshness_summary(specs) -> pd.DataFrame` (source, last_date, gap_days).

- [ ] **Step 1: 실패 테스트 작성 (idempotent append + freshness)**

```python
# tests/test_refresh_external.py
import pandas as pd
from bakery.ingest import refresh


def test_observed_appends_only_new_dates():
    existing = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "v": [1, 2]})
    fetched = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-03"]), "v": [2, 3]})
    merged = refresh.append_new_dates(existing, fetched, date_col="date")
    assert list(merged["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert merged["v"].tolist() == [1, 2, 3]  # 기존 유지, 신규만 추가


def test_append_is_idempotent_on_repeat():
    existing = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "v": [1]})
    once = refresh.append_new_dates(existing, existing, date_col="date")
    twice = refresh.append_new_dates(once, existing, date_col="date")
    assert once.equals(twice)


def test_freshness_gap_days():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-07-20"])})
    gap = refresh.gap_days(df, today=pd.Timestamp("2026-07-25"), date_col="date")
    assert gap == 5
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_refresh_external.py -v`
Expected: FAIL — `bakery.ingest.refresh` 없음.

- [ ] **Step 3: refresh.py 구현**

```python
# src/bakery/ingest/refresh.py
"""외부 8종 소스 통합 갱신. observed=idempotent append, forecast=덮어쓰기."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


def append_new_dates(existing: pd.DataFrame, fetched: pd.DataFrame,
                     date_col: str) -> pd.DataFrame:
    """기존 날짜는 보존, fetched의 신규 날짜만 추가."""
    have = set(existing[date_col])
    new = fetched[~fetched[date_col].isin(have)]
    if new.empty:
        return existing.reset_index(drop=True)
    return (pd.concat([existing, new], ignore_index=True)
            .sort_values(date_col).reset_index(drop=True))


def gap_days(df: pd.DataFrame, today: pd.Timestamp, date_col: str) -> int:
    if df.empty:
        return -1
    return int((today.normalize() - df[date_col].max().normalize()).days)


@dataclass
class SourceSpec:
    name: str
    dataset_key: str          # paths.dataset(...) 키
    kind: str                 # "observed" | "forecast"
    date_col: str = "date"
    # refresh_fn: 기존 ingest 함수를 감싼 콜러블 (신규 데이터 DataFrame 반환)


# EXTERNAL_SOURCES: 기존 ingest_* 를 SourceSpec으로 등록 (weather/calendar/consumption/
# competitor/living_population/population = observed, forecast_* = forecast).
# refresh_source/freshness_summary는 EXTERNAL_SOURCES를 순회.
```
`refresh_source`(spec별 fetch→observed는 append_new_dates·forecast는 덮어쓰기→parquet 저장)와 `freshness_summary`(gap_days 테이블)를 완성. 기존 `ingest-*` CLI가 부르던 API 함수를 그대로 호출(재구현 금지).

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_refresh_external.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: CLI 커맨드 배선**

```python
# src/bakery/cli.py
@app.command("refresh-external")
def cmd_refresh_external(source: str = "all", dry_run: bool = False) -> None:
    """외부 소스 idempotent 갱신 + freshness 요약. source=all|<name>."""
    from .ingest import refresh
    import pandas as pd
    specs = refresh.select_sources(source)
    for spec in specs:
        res = refresh.refresh_source(spec, today=pd.Timestamp.today(), dry_run=dry_run)
        console.print(f"[cyan]{res.name}[/] +{res.added_rows} rows, last={res.last_date}")
    console.print(refresh.freshness_summary(specs).to_string(index=False))
```

- [ ] **Step 6: CLI smoke (dry-run)**

Run: `uv run bakery refresh-external --source weather --dry-run 2>&1 | tail -10`
Expected: weather 소스 freshness/추가행 요약 출력, 오류 없음. (실제 API 호출은 `.env` 필요 — dry-run은 fetch 스킵하고 freshness만.)

- [ ] **Step 7: 커밋**

```bash
git add src/bakery/ingest/refresh.py src/bakery/cli.py tests/test_refresh_external.py
git commit -m "feat(ingest): refresh-external 통합 갱신 CLI (Task 9)"
```

---

## Task 10: 크로스체크 커버리지 리포트

목적: 모든 Excel(0520/0526/0721/진열시간)의 커버리지 매트릭스(source × store × category × date-range × field)를 만들어 크로스소스 갭/충돌을 표면화. 프로파일링은 codex-data-cruncher 위임, 렌더는 TDD.

**Files:**
- Create: `scripts/build_coverage_matrix.py` (codex 산출 → 리포트 렌더)
- Create: `src/bakery/data/coverage.py` (매트릭스 렌더 순수 함수)
- Test: `tests/test_coverage_report.py`

**Interfaces:**
- Consumes: codex-data-cruncher 산출 커버리지 raw(parquet/json).
- Produces: `render_coverage_matrix(cells: pd.DataFrame) -> str` (HTML), `reports/data_coverage/coverage.html`.

- [ ] **Step 1: codex-data-cruncher 위임 (프로파일링)**

⚠️ **dispatch 프롬프트에 반드시 명시** (시트2 스왑 아티팩트 재발 방지):
- **per-sheet English placeholder(row1)를 컬럼명으로** 사용 — 위치·전역 헤더 금지.
- **값 기준 판별** — `SALES_FG`는 0/1, `SALES_TIME`은 14자리(YYYYMMDDHHMMSS)로 시트2 컬럼 스왑 정정.
- **앵커 검증** — 반품 1.88%(107,543행), 광교 same-item 총량 510,585, 기존 canonical 전월 diff 0. **앵커 불일치 시 산출 신뢰 금지·재프로파일**.
- 산출: `source × store × category(대분류) × month × field(sold/production/waste/stockout/closing/display)`의 존재여부 + 행수. `reports/data_coverage/cells.parquet`.

사용자 확인 1줄 후 Task tool로 `codex-data-cruncher` 디스패치. 결과 앵커를 직접 재검증(3개 anchor).

- [ ] **Step 2: 렌더 실패 테스트 작성**

```python
# tests/test_coverage_report.py
import pandas as pd
from bakery.data import coverage


def test_render_flags_missing_cell():
    cells = pd.DataFrame({
        "source": ["0721", "0520"],
        "store": ["광교", "광교"],
        "field": ["sold", "production"],
        "month": ["2026-06", "2026-06"],
        "present": [True, False],  # 2026-06 생산라벨 없음(§5 갭)
        "rows": [1234, 0],
    })
    html = coverage.render_coverage_matrix(cells)
    assert "2026-06" in html
    assert "production" in html
    # 결측 셀이 시각적으로 표시되는지 (class="missing")
    assert html.count("missing") == 1


def test_render_lists_known_conflicts():
    cells = pd.DataFrame({"source": ["0520"], "store": ["광교"], "field": ["closing"],
                          "month": ["2025-12"], "present": [True], "rows": [10]})
    html = coverage.render_coverage_matrix(cells, conflicts=["closing 소스 불일치: category=0520 vs item=clean"])
    assert "closing 소스 불일치" in html
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/test_coverage_report.py -v`
Expected: FAIL — `bakery.data.coverage` 없음.

- [ ] **Step 4: coverage.py 렌더 구현**

```python
# src/bakery/data/coverage.py
"""커버리지 매트릭스 → 자기포함 HTML. 크로스소스 갭/충돌 표면화(탐지만)."""
from __future__ import annotations
import pandas as pd


def render_coverage_matrix(cells: pd.DataFrame,
                           conflicts: list[str] | None = None) -> str:
    pivot = cells.pivot_table(index=["source", "store", "field"],
                              columns="month", values="present", aggfunc="first")
    rows_html = []
    for idx, row in pivot.iterrows():
        tds = []
        for month, present in row.items():
            cls = "" if present else "missing"
            tds.append(f'<td class="{cls}">{month if present else "—"}</td>')
        rows_html.append(f"<tr><th>{' / '.join(map(str, idx))}</th>{''.join(tds)}</tr>")
    conflict_html = ""
    if conflicts:
        items = "".join(f"<li>{c}</li>" for c in conflicts)
        conflict_html = f"<h2>규명된 크로스소스 불일치(탐지)</h2><ul>{items}</ul>"
    return (
        "<style>.missing{background:#fdd}td,th{border:1px solid #ccc;padding:4px}</style>"
        f"{conflict_html}<table>{''.join(rows_html)}</table>"
    )
```

- [ ] **Step 5: 렌더 테스트 통과 + 실데이터 리포트 생성**

Run:
```bash
uv run pytest tests/test_coverage_report.py -v
uv run python scripts/build_coverage_matrix.py  # cells.parquet → coverage.html
```
Expected: 렌더 테스트 PASS. `reports/data_coverage/coverage.html` 생성, §5 불일치 4종(closing 소스·2026 라벨갭·미매칭 67.9%·카테고리 3v5) 셀/충돌목록에 표면화.

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/data/coverage.py scripts/build_coverage_matrix.py tests/test_coverage_report.py
git commit -m "feat(data): 크로스체크 커버리지 매트릭스 리포트 (Task 10)"
```

---

## Task 11: 매진률 진입점 fix (개별 게이트)

목적: 완제품(inventory 미커버 968/1150)을 매진률 분모에서 검열 처리해 광교 매진률 0.151(희석)→0.605(생산품목) 교정. median t(18시)는 유효하므로 rate만. **수치 변경이므로 별도 커밋·재측정.**

**Files:**
- Modify: `src/bakery/data/bonavi_loader.py` (`assign_stockout_fields`)
- Modify: `scripts/store_daily.py` (`build_store_daily` — inventory left-merge 미매칭 처리)
- Test: `tests/test_store_daily_redefine.py` (canary 해소)

**Interfaces:**
- Consumes: 기존 stockout 로직.
- Produces: 생산품목(inventory 커버) 기준 매진률. 완제품은 `is_stockout=NaN`(censored, 0으로 깔지 않음).

- [ ] **Step 1: 현행 canary 실패 재현 확인**

Run: `uv run pytest tests/test_store_daily_redefine.py -v`
Expected: FAIL (기대값 옛 0.50~0.70, 실측 0.151 — 정확한 canary). 실패 메시지 확인.

- [ ] **Step 2: 목표 동작 테스트 작성**

```python
# tests/test_store_daily_redefine.py (교정)
def test_gwangyo_stockout_rate_on_produced_items():
    """생산품목(inventory 커버) 기준 매진률 = 0.605 (완제품 검열 제외).
    완제품(생산기록 없음)은 is_stockout=NaN(censored), 분모 제외."""
    from scripts.store_daily import build_store_daily
    df = build_store_daily("1000000047", "store_gw01")
    produced = df[df["is_stockout"].notna()]      # inventory 커버 품목만
    rate = produced["is_stockout"].mean()
    assert round(rate, 3) == 0.605
    # 완제품은 검열(NaN)로 남아야 함 — 0으로 깔지 않음(헌장 2번)
    assert df["is_stockout"].isna().any()


def test_stockout_median_time_still_18h():
    """median 매진시각은 유효(희석은 rate만)."""
    from scripts.store_daily import build_store_daily
    df = build_store_daily("1000000047", "store_gw01")
    so = df[df["is_stockout"] == True]
    assert so["stockout_time"].dt.hour.median() == 18
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/test_store_daily_redefine.py -v`
Expected: FAIL — 현재 완제품이 `is_stockout=False`(0)로 깔려 rate 0.151.

- [ ] **Step 4: assign_stockout_fields / build_store_daily fix**

`assign_stockout_fields`: inventory 미커버(NaN production) 행은 `is_stockout=False` 강제 대신 **NaN 유지**(censored). `build_store_daily`: inventory left-merge 후 미매칭 행의 `is_stockout`을 `False`로 채우지 않고 NaN 보존.
```python
# bonavi_loader.assign_stockout_fields 핵심 변경
# 기존: is_stockout = (production_qty > 0) & (waste_qty <= 0)  # NaN production → False
# 변경: 생산기록 있는 행만 판정, 없으면 NaN(censored)
has_production = df["production_qty"].notna()
is_so = (df["production_qty"] > 0) & (df["waste_qty"] <= 0)
df["is_stockout"] = is_so.where(has_production, other=pd.NA)
df["stockout_time"] = df["last_sale_ts"].where(df["is_stockout"] == True)
```

- [ ] **Step 5: 테스트 통과 확인 + 소비처 재측정 영향 점검**

Run:
```bash
uv run pytest tests/test_store_daily_redefine.py -v
uv run pytest --color=no
grep -rln "is_stockout" src/ scripts/ | head   # 소비처 열거
```
Expected: canary 해소(0.605·median 18h PASS). 전체 스위트 통과. `is_stockout` 소비처 중 `NaN`을 `False`로 가정하던 곳이 있으면 확인(예측/발주 경로 영향). 영향 있으면 인벤토리에 기록하고 사용자 보고.

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/data/bonavi_loader.py scripts/store_daily.py tests/test_store_daily_redefine.py
git commit -m "fix(stockout): 매진률 완제품 검열 희석 교정 0.151→0.605 (Task 11)"
```

---

## Self-Review (작성자 체크)

**1. Spec coverage:**
- §3 3층 레이아웃 → Task 3 ✓ / §4 경로 중앙화 → Task 2,4,5,6 ✓ / §5 build-data+동등성 → Task 7,8 ✓ / §6 커버리지 리포트 → Task 10 ✓ / §7 refresh-external → Task 9 ✓ / §8 매진률 fix → Task 11 ✓ / §9 경로 전수 열거 → Task 1 ✓.
- scripts shim(사용자 결정) → Task 3 Step 4 심링크 ✓.

**2. Placeholder scan:** codex 위임(Task 10 Step 1)은 런북 스텝이라 코드 없음이 정상 — 대신 dispatch 명세·앵커 검증을 구체화. 나머지 코드 스텝은 실제 코드 포함.

**3. Type consistency:** `paths.dataset(name)`·`pipeline.build_internal`·`pipeline.equivalence_diff`·`refresh.append_new_dates`/`gap_days`·`coverage.render_coverage_matrix` — 정의(초기 Task)와 사용(후속 Task) 시그니처 일치 확인.

**주의 (구현 시):**
- Task 5 config 순환 import — `config`가 `paths`를 import하면 순환(`paths`→`config.PROJECT_ROOT`). `EXTERNAL_DATA_DIR`는 config에서 `PROJECT_ROOT` 직접 계산으로 해결(코드에 명시).
- Task 3 이동은 되돌리기 어려움 — byte-identity(Step 5) 확인 전 다음 Task 진입 금지. cruft 삭제는 `_archive/` 격리까지만, 실삭제는 사용자 확인.
- Task 11은 별도 게이트 — 재배치(Task 2-8)와 섞지 말 것. `is_stockout` NaN 전환이 예측/발주 소비처에 미치는 영향 별도 점검.

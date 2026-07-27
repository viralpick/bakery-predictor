# 신규 데이터 편입 마무리 (Phase 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 신규 0721 데이터 편입을 마무리한다 — category-level 헤드라인 closing 소스를 신규 클린 parquet로 재배선하고, 4매장 canonical daily(별도 parquet, 광교 불변)를 생성하고, 광교 헤드라인 KPI를 고객 지표 경로로 재측정한다.

**Architecture:** 광교 canonical `bonavi_daily.parquet`는 byte-identical 불변 유지(equivalence gate). 4매장은 별도 `multistore_daily.parquet`로 격리해 광교 헤드라인 오염을 구조적으로 차단. 헤드라인 재측정은 `harness-run`(category_total+event_prior) 경로 고정 — 단, 그 경로의 마감할인 closing 소스가 옛 0520 파일이던 배선 갭을 먼저 신규 소스로 고친다.

**Tech Stack:** Python 3, pandas, LightGBM, pydantic, pytest, uv, typer(CLI).

## Global Constraints

- Time leakage 금지: lag/rolling은 split 이후·cutoff 이전만. `test_split_leakage.py`/`test_features_leakage.py` 반드시 통과.
- 품절 데이터 censored: `assign_stockout_fields` companion mask(`is_stockout` boolean 불변 + `is_stockout_defined`) 로직 불변.
- Random split 금지: 재측정은 harness expanding window(n_folds=52, window_days=730, horizon_days=7).
- Synthetic↔Real 경계: 다매장 산출물도 `data/schema.py` `DAILY_COLUMNS`(10컬럼) 준수 + `validate_daily` 통과.
- MAPE 단독 금지: 재측정 메인 지표 = WAPE(총량·카테고리). MAE/RMSE 보조.
- 헤드라인 파라미터 정확 보존: ALPHA=0.8, PROD_Q=0.85, WINDOW=730, FOLDS=52, HORIZON=7.
- 매장 코드 매핑(불변): 광교=1000000047(store_gw01) / 삼성타운=1000000009(store_ss01) / 광화문=1000000485(store_gh01) / 메세나폴리스=1000000029(store_mp01).
- 라벨 가용 끝 `LABEL_END="20251231"` — 이후는 sales-only, 폐기/생산 라벨 없음. 4매장 daily도 2021~2025-12 구간.
- 테스트 단언: 기대값 아는 것은 정확값 비교(`==`, 부동소수는 rtol 명시). merge 전 전체 스위트 1회 필수.

---

### Task 0: category-level 헤드라인 closing 소스 재배선 (게이팅)

`build_category_daily`(category-level, harness 헤드라인 타깃 `adjusted_demand_unit` 생성)가 `load_sales_with_discount()`/`load_closing_returns()`를 **default args**(=`legacy_xlsx_0520`, 옛 0520 파일)로 호출한다. 이미 있는 item-level `build_item_adjusted_demand`(L162-194)와 **동일한 guarded fallback 패턴**을 복제해, 신규 클린 parquet(`sales_lines_clean`, CD_USERDEF1)이 있으면 그것을 우선한다.

**의도적 제외 (건드리지 않음):** `cli.py:1719` `_load_closing_demand_inputs`의 `load_sales_with_discount(DEFAULT_XLSX)`는 closing-demand α 리서치 전용 명령이라 헤드라인 예측 경로가 아님 — 이번 범위 밖.

**Files:**
- Modify: `src/bakery/features/category_aggregate.py:94-101` (`build_category_daily` closing 자동로드 블록)
- Test: `tests/test_category_closing_source.py` (Create)

**Interfaces:**
- Consumes: `bakery.analysis.discount.CLEAN_PARQUET_DEFAULT`(Path), `load_sales_with_discount_v2()`→`DiscountSales`, `load_closing_returns_v2()`→`DataFrame[item_id,date,ret_qty]` (모두 이미 존재, 동일 스키마·라벨, default `store_code="1000000047"` 광교).
- Produces: `build_category_daily(...)`는 시그니처 불변(`daily_raw=None, discount_rows=None, alpha=DEFAULT_ALPHA, categories=TARGET_CATEGORIES, closing_returns=None`) — 반환 타입 `CategoryDaily` 불변. 내부 closing 소스만 신규로.

- [ ] **Step 1: 현재 closing 소스가 옛 0520임을 고정하는 실패 테스트 작성**

```python
# tests/test_category_closing_source.py
"""build_category_daily의 closing 소스가 신규 클린 parquet(CD_USERDEF1)여야 한다."""
from unittest.mock import patch
import pandas as pd
from bakery.features import category_aggregate as ca


def test_build_category_daily_uses_v2_closing_when_clean_parquet_exists():
    """클린 parquet 존재 시 build_category_daily가 v2 closing 로더를 호출한다."""
    with patch("bakery.analysis.discount.CLEAN_PARQUET_DEFAULT") as mock_path, \
         patch("bakery.analysis.discount.load_sales_with_discount_v2") as mock_v2_sales, \
         patch("bakery.analysis.discount.load_closing_returns_v2") as mock_v2_ret, \
         patch("bakery.analysis.discount.load_sales_with_discount") as mock_old_sales:
        mock_path.exists.return_value = True
        # v2 sales가 빈 closing_discount를 반환하도록
        empty = pd.DataFrame({"item_id": [], "date": [], "qty": []})
        mock_v2_sales.return_value.closing_discount.return_value = empty
        mock_v2_ret.return_value = pd.DataFrame({"item_id": [], "date": [], "ret_qty": []})
        # 최소 daily_raw 주입(파일 IO 회피)
        daily_raw = pd.DataFrame({
            "store_id": ["store_gw01"], "item_id": ["x1"], "category_id": ["bread"],
            "date": pd.to_datetime(["2024-01-01"]), "sold_units": [10],
            "is_stockout": [False], "stockout_time": [pd.NaT],
            "open_hours": [12], "capacity": [20], "potential_demand": [10.0],
        })
        ca.build_category_daily(daily_raw=daily_raw)
        mock_v2_sales.assert_called_once()   # 신규 소스 사용
        mock_old_sales.assert_not_called()   # 옛 0520 미사용
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_category_closing_source.py -v`
Expected: FAIL — 현재 `build_category_daily`는 `load_sales_with_discount`(옛)를 호출하므로 `mock_old_sales.assert_not_called()`에서 실패.

- [ ] **Step 3: guarded fallback으로 재배선 (build_item_adjusted_demand 패턴 복제)**

`src/bakery/features/category_aggregate.py` L94-101의 closing 블록을 아래로 교체:

```python
    # closing (신규 클린 parquet 우선, 없으면 옛 0520 폴백 — build_item_adjusted_demand와 동일 패턴)
    if discount_rows is None:
        from bakery.analysis.discount import (
            CLEAN_PARQUET_DEFAULT,
            load_closing_returns,
            load_closing_returns_v2,
            load_sales_with_discount,
            load_sales_with_discount_v2,
        )
        if CLEAN_PARQUET_DEFAULT.exists():
            ds = load_sales_with_discount_v2()
            discount_rows = ds.closing_discount().copy()
            discount_rows["item_id"] = discount_rows["item_id"].astype(str)
            if closing_returns is None:
                closing_returns = load_closing_returns_v2()
        else:
            ds = load_sales_with_discount()
            discount_rows = ds.closing_discount().copy()
            discount_rows["item_id"] = discount_rows["item_id"].astype(str)
            if closing_returns is None:
                closing_returns = load_closing_returns()
```

- [ ] **Step 4: 새 테스트 통과 확인**

Run: `uv run pytest tests/test_category_closing_source.py -v`
Expected: PASS

- [ ] **Step 5: build_category_daily 소비 테스트 전체 실행 (값 기대 재보정 판정)**

`build_category_daily`를 소비하는 테스트 전부 실행. closing 소스 교체로 `adjusted_demand_unit` 값이 바뀌면 값 기대(anchor)를 가진 테스트가 깨질 수 있다.

Run: `uv run pytest tests/test_forecast_forward.py tests/harness/test_backtest_core_equivalence.py tests/harness/test_backtest_core_distributional.py tests/harness/test_forecasters.py -v`

판정:
- **동등성 테스트(`test_backtest_core_equivalence`)가 "추출 코어 == 원본 스크립트" 대조라면** 양쪽 다 같은 `build_category_daily`를 쓰므로 소스 교체와 무관하게 통과해야 한다. → 통과면 OK.
- **고정 WAPE/값 anchor를 assert하는 테스트가 실패하면**: 그 기대값이 옛 0520 소스 기준이므로, 신규 소스로 실제 값을 재측정해 기대값을 갱신하고, 주석에 "신규 클린 parquet closing 소스 기준(Task 0)"을 남긴다. 값이 비결정적이 아니므로 정확값(rtol 명시)으로.
- `test_store_daily_redefine`는 **이 PR 이전부터 실패하는 사전존재 이슈**(라인레벨 데이터로 is_stockout 재정의 후 기대값 미갱신) — 이 태스크와 무관. 상태만 확인하고 별도 처리.

- [ ] **Step 6: 전체 스위트 실행 (크로스파일 회귀 게이트)**

Run: `uv run pytest --color=no -x 2>&1 | tail -20`
Expected: Task 0 신규 테스트 통과 + 사전존재 `test_store_daily_redefine` 외 실패 없음. (`-q` 금지 — 이 repo addopts에 `-q` 있어 `-q` 추가 시 요약 사라짐.)

- [ ] **Step 7: Commit**

```bash
git add src/bakery/features/category_aggregate.py tests/test_category_closing_source.py
git commit -m "fix: category-level 헤드라인 closing 소스를 신규 클린 parquet로 재배선 (Phase 7 Task 0)

harness 헤드라인 타깃 adjusted_demand_unit의 마감할인 closing이 옛 0520
파일에서 오던 배선 갭 수리. build_item_adjusted_demand와 동일 guarded
fallback. 재측정 delta 정확성의 선행 게이트.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012iHLiUwcdPQUrQZqM5WUKv"
```

---

### Task 1: 4매장 multistore daily 빌더

`build_v2`(단일 매장, `store_code=DEFAULT_STORE_CODE`)를 4매장으로 확장하는 얇은 상위 함수 `build_multistore`를 만든다. 각 매장을 `build_v2`로 임시 빌드 후 concat해 `multistore_daily.parquet`로 저장. 광교 `bonavi_daily.parquet`는 건드리지 않는다.

**Files:**
- Modify: `src/bakery/data/bonavi_loader_v2.py` (`build_multistore` 추가, `build_v2` 재사용)
- Modify: `src/bakery/data/paths.py:23` (`_DATASETS`에 `multistore_daily` 등록)
- Test: `tests/test_multistore_build.py` (Create)

**Interfaces:**
- Consumes: `build_v2(store_code=..., rename_store_id=..., out_path=..., receipts_path=..., clean_parquet=..., master_xlsx=...)`→`Path` (기존). `STORE_CODE_MAPPING`(`bakery.ingest.inventory`, dict[store_id→code]).
- Produces: `build_multistore(*, clean_parquet=CLEAN_PARQUET, master_xlsx=MASTER_XLSX, out_path=paths.dataset("multistore_daily")) -> Path`. 산출 parquet 컬럼 = `DAILY_COLUMNS` 10종, store_id ∈ {store_gw01, store_ss01, store_gh01, store_mp01}.
- Produces: `paths.dataset("multistore_daily")` → `_INTERNAL / "multistore_daily.parquet"`.

- [ ] **Step 1: paths registry에 multistore_daily 등록**

`src/bakery/data/paths.py` `_DATASETS` dict에서 `"bonavi_daily"` 줄 아래에 추가:

```python
    "multistore_daily": _INTERNAL / "multistore_daily.parquet",
```

- [ ] **Step 2: 실패 테스트 작성 (4매장 포함 + 광교 정합 + 스키마)**

```python
# tests/test_multistore_build.py
"""build_multistore: 4매장 daily를 별도 parquet로. 광교 파트는 canonical과 정합."""
import numpy as np
import pandas as pd
import pytest
from bakery.data import bonavi_loader_v2 as v2
from bakery.data import paths
from bakery.data.schema import DAILY_COLUMNS, validate_daily


@pytest.fixture(scope="module")
def multistore(tmp_path_factory):
    out = tmp_path_factory.mktemp("ms") / "multistore_daily.parquet"
    v2.build_multistore(out_path=out)
    return pd.read_parquet(out)


def test_multistore_has_four_stores(multistore):
    assert set(multistore["store_id"].unique()) == {
        "store_gw01", "store_ss01", "store_gh01", "store_mp01"}


def test_multistore_schema(multistore):
    assert list(multistore.columns) == list(DAILY_COLUMNS.keys())
    validate_daily(multistore)   # raise 없으면 통과


def test_multistore_gwangyo_matches_canonical(multistore):
    """multistore의 광교 파트 == 기존 canonical bonavi_daily (max_diff=0)."""
    canon = pd.read_parquet(paths.dataset("bonavi_daily"))
    gw = multistore[multistore["store_id"] == "store_gw01"].copy()
    keys = ["date", "item_id"]
    gw = gw.sort_values(keys).reset_index(drop=True)
    canon = canon.sort_values(keys).reset_index(drop=True)
    assert len(gw) == len(canon)
    for c in gw.select_dtypes(include=[np.number]).columns:
        max_diff = float((gw[c].fillna(-9e9).to_numpy() - canon[c].fillna(-9e9).to_numpy()).max())
        assert max_diff == 0.0, f"{c} diff={max_diff}"
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/test_multistore_build.py -v`
Expected: FAIL — `build_multistore` 미정의(AttributeError).

- [ ] **Step 4: build_multistore 구현**

`src/bakery/data/bonavi_loader_v2.py`에 추가(`build_v2` 아래):

```python
def build_multistore(
    *,
    clean_parquet: Path | str = CLEAN_PARQUET,
    master_xlsx: Path | str = MASTER_XLSX,
    out_path: Path | str = paths.dataset("multistore_daily"),
) -> Path:
    """4매장 daily를 별도 parquet로. 각 매장은 build_v2 재사용, concat 후 저장.

    광교(store_gw01) 파트는 build_v2(canonical과 동일 경로)라 bonavi_daily와 정합.
    receipts는 매장별 임시경로에 쓰고 버린다(multistore는 daily만 소비).
    """
    from bakery.ingest.inventory import STORE_CODE_MAPPING

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent
    frames = []
    for store_id, store_code in STORE_CODE_MAPPING.items():
        daily_path = tmp / f"_ms_{store_id}_daily.parquet"
        receipts_path = tmp / f"_ms_{store_id}_receipts.parquet"
        build_v2(
            clean_parquet=clean_parquet, master_xlsx=master_xlsx,
            store_code=store_code, rename_store_id=store_id,
            out_path=daily_path, receipts_path=receipts_path,
        )
        frames.append(pd.read_parquet(daily_path))
        daily_path.unlink(missing_ok=True)
        receipts_path.unlink(missing_ok=True)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_parquet(out_path, index=False)
    return out_path
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/test_multistore_build.py -v`
Expected: PASS (3 테스트). 광교 정합 `max_diff=0` 통과가 핵심.

- [ ] **Step 6: Commit**

```bash
git add src/bakery/data/bonavi_loader_v2.py src/bakery/data/paths.py tests/test_multistore_build.py
git commit -m "feat: 4매장 multistore_daily 빌더 (Phase 7 Task 1)

build_v2를 4매장 루프로 확장하는 build_multistore. 별도 parquet로 격리,
광교 파트는 canonical bonavi_daily와 max_diff=0 정합.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012iHLiUwcdPQUrQZqM5WUKv"
```

---

### Task 2: CLI 배선 + 실제 생성 + 게이트/leakage 확인

`build_multistore`를 CLI에 노출하고 실제 `multistore_daily.parquet`를 생성한다. 광교 canonical 불변·게이트 통과·leakage 테스트를 확인한다.

**Files:**
- Modify: `src/bakery/cli.py` (`build-multistore` 명령 추가)
- Test: `tests/test_cli_build_multistore.py` (Create)

**Interfaces:**
- Consumes: `bonavi_loader_v2.build_multistore()`(Task 1), typer app 패턴(기존 `cmd_*` 참고).
- Produces: CLI `bakery build-multistore` → `paths.dataset("multistore_daily")` 생성.

- [ ] **Step 1: CLI 명령 실패 테스트 작성**

```python
# tests/test_cli_build_multistore.py
"""bakery build-multistore CLI이 multistore_daily.parquet를 생성한다."""
from unittest.mock import patch
from pathlib import Path
from typer.testing import CliRunner
from bakery.cli import app

runner = CliRunner()


def test_build_multistore_command_invokes_builder(tmp_path):
    fake = tmp_path / "multistore_daily.parquet"
    with patch("bakery.data.bonavi_loader_v2.build_multistore") as mock_build:
        mock_build.return_value = fake
        result = runner.invoke(app, ["build-multistore"])
    assert result.exit_code == 0
    mock_build.assert_called_once()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_cli_build_multistore.py -v`
Expected: FAIL — `build-multistore` 명령 없음(exit_code != 0).

- [ ] **Step 3: CLI 명령 구현**

`src/bakery/cli.py`에 추가(기존 `cmd_format_bonavi_v2` 근처, 동일 스타일):

```python
@app.command("build-multistore")
def cmd_build_multistore() -> None:
    """4매장 multistore_daily.parquet 생성 (광교 canonical 불변, 참조/분석용)."""
    from .data import bonavi_loader_v2 as v2
    out = v2.build_multistore()
    console.print(f"[green]wrote[/] {out}")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_cli_build_multistore.py -v`
Expected: PASS

- [ ] **Step 5: 실제 multistore 생성 + 광교 canonical 불변 확인**

```bash
# 생성 전 canonical 해시
md5 -q data/processed/internal/bonavi_daily.parquet > /tmp/canon_before.md5
uv run bakery build-multistore
# canonical 불변(건드리지 않았으므로) 확인
md5 -q data/processed/internal/bonavi_daily.parquet > /tmp/canon_after.md5
diff /tmp/canon_before.md5 /tmp/canon_after.md5 && echo "canonical UNCHANGED"
```
Expected: `canonical UNCHANGED` + `wrote .../multistore_daily.parquet`.

- [ ] **Step 6: 게이트 3종 + leakage 테스트 통과 확인**

```bash
uv run bakery build-data --diagnose      # 광교 canonical max_diff=0 유지
uv run bakery check-integrity            # fail 0
uv run bakery check-conflict             # conflicting=0
uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py -v
```
Expected: build-data `max_diff=0`, integrity fail 0, conflict 0, leakage 테스트 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/bakery/cli.py tests/test_cli_build_multistore.py
git commit -m "feat: bakery build-multistore CLI + multistore 생성 (Phase 7 Task 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012iHLiUwcdPQUrQZqM5WUKv"
```

---

### Task 3: 광교 헤드라인 KPI 재측정 (Task 0 의존)

`harness-run`(category_total+event_prior, 고객 지표 경로)으로 광교 헤드라인 KPI를 새 166 canonical + 신규 closing 소스(Task 0)로 재측정하고, 옛 146 기준 headline과 delta를 리포트한다. **item-level daily WAPE 재활용 금지.**

**전제:** Task 0 완료(closing 소스 신규). 미완료면 delta 보고 금지.

**Files:**
- Create: `reports/phase7/gwangyo_headline_remeasure.md` (delta 리포트)
- (코드 변경 없음 — 기존 `harness-run` 실행 + 결과 정리)

**Interfaces:**
- Consumes: `bakery harness-run experiments/gwangyo_default.yaml` → `reports/.../comparison.csv`, `metrics.json`(wape 등), 자동생성 HTML.

- [ ] **Step 1: harness-run 실행**

```bash
uv run bakery harness-run experiments/gwangyo_default.yaml
```
Expected: category_total 52-fold 완료, `comparison.csv`·`metrics.json`·HTML 생성. (수 분 소요.)

- [ ] **Step 2: 산출 지표 수집**

`metrics.json`에서 광교 총량 WAPE + 폐기율(waste_rate) + 매진(soldout_median/stockout_item_rate) 값을 읽는다.

Run: `uv run python -c "import json,glob; [print(p, json.load(open(p))) for p in glob.glob('reports/**/metrics.json', recursive=True)]"`

- [ ] **Step 3: delta 리포트 작성**

`reports/phase7/gwangyo_headline_remeasure.md`에 표로 기록:
- 컬럼: 지표 | 옛 146 headline(naive WAPE 8.19 / 우리 8.03 / 폐기 −33~40%) | 새 166+신규closing | delta
- **지표 경계 명시(필수)**: "이 delta는 146→166 품목/타깃정의 변화 + closing 소스 교체분이며, 날짜는 여전히 2021~2025-12로 2026 전향 검증이 아님."
- 고객 지표 경로(category_total+event_prior, adjusted_demand)임을 명시. item-level WAPE 아님.

- [ ] **Step 4: Commit**

```bash
git add reports/phase7/gwangyo_headline_remeasure.md
git commit -m "docs: 광교 헤드라인 KPI 재측정 (새 166+신규 closing, Phase 7 Task 3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012iHLiUwcdPQUrQZqM5WUKv"
```

---

### Task 4: 타 3매장 참조 예측 (타깃 아님)

multistore_daily 위에서 삼성·메세나·광화문 category_total 예측을 돌려 참조표를 만든다. **event_prior는 광교 전용이라 타매장엔 미적용**(base category_total만). 헤드라인·성공기준에 쓰지 않는다.

**Files:**
- Create: `scripts/phase7_multistore_reference.py` (참조 예측 스크립트)
- Create: `reports/phase7/multistore_reference.csv`

**Interfaces:**
- Consumes: `build_category_daily(daily_raw=<store-filtered df>, alpha=0.8)` — `daily_raw` 주입으로 store 필터(harness 침습 없이). `windowed_backtest`(harness backtest_core) 또는 기존 category_total 예측 함수.
- Produces: 매장별 총량 WAPE 참조표 CSV.

- [ ] **Step 1: 참조 스크립트 작성**

```python
# scripts/phase7_multistore_reference.py
"""타 3매장 category_total 참조 예측 (event_prior 없이). 타깃 아님·참조용."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
from bakery.data import paths
from bakery.features.category_aggregate import build_category_daily
from bakery.features.category_aggregate import build_features
from bakery.harness.backtest_core import windowed_backtest, metrics_from_preds
from bakery.harness.registry import build_forecaster

REFERENCE_STORES = ["store_ss01", "store_mp01", "store_gh01"]  # 광교 제외
ms = pd.read_parquet(paths.dataset("multistore_daily"))
rows = []
for store_id in REFERENCE_STORES:
    daily_raw = ms[ms["store_id"] == store_id].copy()
    cd = build_category_daily(daily_raw=daily_raw, alpha=0.8)
    feat = build_features(cd, target_col="adjusted_demand_unit")
    fc = build_forecaster("category_total")
    bt = windowed_backtest(
        feat, window_days=730, target_col="adjusted_demand_unit",
        n_folds=52, horizon_days=7, production_q=0.85, alpha=0.8,
        events=None, lunar_events=None, forecaster=fc,   # event_prior 미적용
    )
    m = metrics_from_preds(bt.predictions)
    rows.append({"store_id": store_id, **m})
    print(store_id, m)
out = pd.DataFrame(rows)
out.to_csv("reports/phase7/multistore_reference.csv", index=False)
print("wrote reports/phase7/multistore_reference.csv")
```

- [ ] **Step 2: 실행 (스모크)**

Run: `uv run python scripts/phase7_multistore_reference.py`
Expected: 3매장 각각 metrics dict 출력 + CSV 생성. 에러 없이 완주(build_category_daily가 store-filtered daily_raw로 동작). 실패 시 build_features/windowed_backtest 인자 시그니처를 `src/bakery/harness/runner.py` `_feat`·backtest 호출부와 대조해 정합.

- [ ] **Step 3: Commit**

```bash
git add scripts/phase7_multistore_reference.py reports/phase7/multistore_reference.csv
git commit -m "feat: 타 3매장 참조 예측 (event_prior 미적용, 타깃 아님, Phase 7 Task 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012iHLiUwcdPQUrQZqM5WUKv"
```

---

### Task 5: 정리 + 열린 결정 문서화 + 메모리 갱신

drift 상태·게이트 통과·열린 결정을 durable 문서로 남기고, 전체 스위트를 최종 확인한다.

**Files:**
- Create: `reports/phase7/finalize_summary.md`
- Modify: 메모리 `project_harness_backbone.md`, `project_new_data_20260721.md` (7단계 완료 갱신)

- [ ] **Step 1: finalize 요약 문서 작성**

`reports/phase7/finalize_summary.md`에 기록:
- 게이트 3종 통과 상태(build-data max_diff=0 / integrity fail 0 / conflict 0).
- drift 2건: 할인코드 357 미정규화(기존)·비타깃 품목 1649개(is_target_scope=False, 타깃 아님) — fail 아님, 정보성.
- 열린 결정(범위 밖·아티제 확인 대기): 원가율 미보유(폐기비용 판매가 계열)·음료 마스터 부재로 T5 blocked·진열시간 계획/실측 미확정·전향 4주 실측(2026-06-30 컷오프, 실시간 피드 필요).
- multistore_daily 생성됨(4매장, 참조용). 광교 canonical 불변.

- [ ] **Step 2: 전체 스위트 최종 실행**

Run: `uv run pytest --color=no 2>&1 | tail -20`
Expected: Task 0·1·2 신규 테스트 통과. 사전존재 `test_store_daily_redefine` 외 실패 없음. 실패 개수·이름을 요약에 기록.

- [ ] **Step 3: 메모리 갱신 (auto-memory-save 스킬 활용)**

`project_harness_backbone.md` 로드맵 7단계를 ✅완료로, `project_new_data_20260721.md`를 편입 마무리 반영으로 갱신. Current goal/Last decisions/Open risks/Next first step 4칸 갱신.

- [ ] **Step 4: Commit**

```bash
git add reports/phase7/finalize_summary.md
git commit -m "docs: Phase 7 편입 마무리 요약 + 열린 결정 문서화 (Task 5)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012iHLiUwcdPQUrQZqM5WUKv"
```

---

## Self-Review (작성자 체크)

**Spec coverage:**
- 작업 0(closing 소스 재배선) ← 스펙 작업 0 + D2 선행 게이트. ✅
- 다매장 canonical(별도 parquet, 광교 불변) ← 스펙 D1 + 작업 1. Task 1·2. ✅
- 광교 정합 max_diff=0 ← Task 1 Step 2 + Task 2 Step 5. ✅
- 헤드라인 재측정(고객 지표 경로, 경계 명시) ← 스펙 D2 + 작업 3. Task 3. ✅
- 타 3매장 참조 예측(타깃 아님) ← 스펙 D3 + 작업 4. Task 4. ✅
- 게이트 3종 통과 유지 ← Task 2 Step 6. ✅
- 열린 결정 문서화 ← 스펙 작업 5. Task 5. ✅
- leakage 테스트 통과 ← Task 2 Step 6. ✅

**Placeholder scan:** 코드 스텝 전부 실제 코드 포함. "적절한 에러처리" 류 없음. ✅

**Type consistency:** `build_multistore`(Task 1 정의) ↔ Task 2 CLI 호출 일치. `build_category_daily(daily_raw=)`(Task 4 사용) ↔ 기존 시그니처 일치. `windowed_backtest`/`metrics_from_preds`/`build_forecaster`(Task 4) ↔ harness 실제 심볼. ✅

**주의(실행자):** Task 4의 `build_features`/`windowed_backtest` 인자는 `runner.py`의 실제 호출부(`_feat`·backtest 블록)와 대조해 정합 확인 후 실행 — 시그니처가 다르면 그쪽을 진실로.

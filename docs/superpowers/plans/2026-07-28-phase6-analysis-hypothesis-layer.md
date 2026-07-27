# 데이터분석 + 가설검증 레이어 (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bakery analysis-run <yaml>` 단일 진입점으로 입력 데이터 분석 5종과 가설 14종을 각각 on/off 해서 실행하고, 결과를 자기포함 HTML 하나로 뽑는다.

**Architecture:** harness backbone(예측 평면)과 형제 구조를 미러링한다 — pydantic Spec + registry + runner + 자기포함 plotly HTML. 새 서브패키지 `src/bakery/analysis/lab/`가 오케스트레이션을 담당하고, 계산은 전부 기존 `src/bakery/analysis/` 프리미티브와 `scripts/`에서 추출한 순수함수를 **호출**한다(재구현 금지). 입력 데이터는 `bakery.data.paths.dataset()`으로만 도달한다.

**Tech Stack:** Python 3.12 / uv / pydantic v2 / pandas / plotly / pytest / typer

## Global Constraints

이 섹션은 모든 태스크의 요구사항에 암묵적으로 포함된다.

- **측정 기준 헌장 (2026-07-15 확정)** — 모든 측정·비교의 단일 기준:
  - bulk(대량예약)는 **판매·생산 둘 다 제외**. `bonavi_daily`/`bonavi_receipts`/`multistore_daily`는 이미 bulk 제외 상태(`bonavi_loader.load_sales`가 `flag_bulk_lines`로 제거). `bonavi_receipts`의 `is_bulk` 컬럼은 opt-in 진단용이며 이미 필터된 프레임에 다시 적용하지 않는다.
  - 수요 = `sold`(bulk 제외), `adjusted_demand` = 정상 + **0.8** × 마감. `alpha` 기본값은 `0.8`.
  - **`potential_demand` 사용 금지** — 오염 소스(`stockout_time` 다중이벤트 버그로 부풀려짐). `multistore_daily`/`bonavi_daily`에 컬럼이 남아 있어도 읽지 않는다.
  - 매진 2관점: ①전체매진(폐기0 or Σ발주<Σadjusted, critical) ②SKU품절(폐기0&마감0 비율).
- **이 레이어는 모델을 실행하지 않는다** — `windowed_backtest`, LightGBM fit, `predict_*` 호출 금지. 모델 예측이 필요한 가설은 `spec.predictions`가 가리키는 **harness-run 산출 artifact(`predictions.csv`)를 읽기만** 한다. artifact가 없으면 실행하지 않고 리포트에 `preds_required`로 표기한다(사용자 결정, 2026-07-28).
- **DEPRECATED 3종 이식 금지** — `diag_anchor_gh`, `diag_chuseok_gh`, `diagnose_conformal_residual`. v5 conformal 구간예측은 점추정+위험수치로 전환되며 폐기됐다. spec에서 이 이름을 쓰면 에러.
- **회귀 대조 게이트 = 동일 vintage 실측 대조** (사용자 결정, 2026-07-28). `docs/*_result.md`에 기록된 수치를 golden으로 쓰지 않는다 — Phase 7 신규데이터 편입으로 값이 이동했다(헤드라인 WAPE 8.03→7.72). 게이트 형태는 두 가지뿐:
  1. **frozen-input 대조**: 출처 스크립트가 읽던 동결 artifact(`reports/raw_adjusted_series.csv`, `reports/track3_fresh_preds.parquet`)를 추출 함수에 먹여 **이 플랜에 박힌 golden 수치**와 비교. golden은 2026-07-28에 실제 실행해 캡처한 값이다.
  2. **동일 입력 등가**: 스크립트를 추출 함수의 얇은 wrapper로 바꾼 뒤, 같은 DataFrame을 두 경로에 먹여 동일 출력 확인.
  - 레거시 EDA 스크립트(`eda01`~`eda05`)는 `data/internal/v2/` 원본 시트를 다른 필터(`FG_ITEM=='SS'`, beverage/etc 포함)로 읽으므로 **수치 등가 게이트가 불가능**하다. 이 5종은 canonical 입력 위에서의 재표현이며, 게이트는 구조 불변식(비중 합=1.0, 폐기율∈[0,1], 항등식 잔차)으로 한다. 근거를 핸들러 docstring과 리포트 note에 남긴다.
- **테스트 단언 강도** — 기대값을 아는 단언은 정확값 비교(`==`, float는 `pytest.approx(..., rel=1e-9)`). truthy(`assert x`)·부분 문자열(`in`) 금지. 예외는 비결정적 값(타임스탬프)이나 계약이 "존재 여부"인 경우만, 이유를 주석으로.
- **코드 품질** — 함수 30줄 이내(빈 줄 제외), 인자 4개 초과 시 dict/객체로 묶기, 중첩 3단계 초과 금지, guard clause 우선, 매직값 금지(상수화), 불리언은 `is_`/`has_`/`needs_` 접두어.
- **산문은 한국어** — docstring·리포트 텍스트·판정 문구 전부 한국어(코드 심볼은 영어).
- **`uv run pytest`로 실행**. repo `addopts`에 이미 `-q`가 있으므로 `-q`를 추가로 붙이지 않는다(`-qq`가 되어 passed 요약이 사라짐). 카운트가 필요하면 `--color=no`.
- **기존 시각화 자산 재사용 거부(스펙 §116 대응 결정)** — `scripts/build_dashboard.py`(1209줄)·`weekly_overlay_series.py`를 재사용하지 않는다. 근거: (1) 두 스크립트는 예측 산출물 대시보드용이라 이 레이어의 섹션 구조(항목별 제목+그래프+표+판정)와 계약이 다르다, (2) 1209줄 모놀리스에서 함수를 끌어오면 `analysis/lab`이 `scripts/`에 의존하게 되어 "paths 기반 canonical 입력만" 원칙이 깨진다, (3) 핸들러당 figure는 1~2개짜리 20줄 이하라 추출 비용이 재작성 비용보다 크다. 대신 **`fig_to_div` stateless 패턴은 harness `report.py`에서 그대로 재사용**한다(자기포함 HTML의 핵심 자산).
- **plotly 경계 결정(의도적 이탈)** — harness는 "코어는 viz 무의존"이지만 `AnalysisResult.figures`는 핸들러가 plotly Figure를 담는다. 대신 **회귀 대조·수치 테스트는 `tables`/`verdict`에만** 걸어 핸들러 검증이 plotly 없이 성립하게 한다. 이는 드리프트가 아니라 결정이다.

---

## File Structure

**신규 패키지 `src/bakery/analysis/lab/`** (프리미티브 `src/bakery/analysis/*.py`를 소비하는 오케스트레이션 레이어)

| 파일 | 책임 |
|---|---|
| `lab/__init__.py` | 공개 표면 re-export (`AnalysisSpec`, `load_analysis_spec`, `run_analysis`, `build_analysis_report`) |
| `lab/result.py` | `AnalysisResult` / `SkippedResult` / `AnalysisReport` 데이터 컨테이너 |
| `lab/spec.py` | `AnalysisSpec`(pydantic) + `load_analysis_spec` + DEPRECATED/미등록 이름 강제 |
| `lab/inputs.py` | `AnalysisInputs` — canonical 입력 lazy 로더(중복 IO 방지). `paths.dataset()`만 사용 |
| `lab/registry.py` | `Handler` 메타 + `DATA_ANALYSES` / `HYPOTHESES` 딕셔너리 + `resolve` |
| `lab/runner.py` | `run_analysis(spec) -> AnalysisReport`. 켜진 것만 실행, 스킵 사유 수집 |
| `lab/report.py` | `AnalysisReport` → 자기포함 HTML. 섹션 A(데이터분석)/B(가설검증) + on/off 표 |
| `lab/handlers/sales.py` | `sales_distribution`, `category_mix` |
| `lab/handlers/waste.py` | `waste_rate`, `waste_alpha_identity`, `overproduction_breakdown` |
| `lab/handlers/absorption.py` | `demand_absorption` |
| `lab/handlers/discount.py` | `closing_discount`, `other_discounts`, `discount_regime` |
| `lab/handlers/stockout.py` | `stockout_revenue`, `popularity_stockout` |
| `lab/handlers/substitution.py` | `substitution` |
| `lab/handlers/calendar_bias.py` | `holiday_premium`, `month_dow_adjust` |
| `lab/handlers/model_bias.py` | `seasonal_bias`, `weather_bias`, `weekday_bias`, `event_prior_validation` (preds 의존 4종) |
| `lab/handlers/basket.py` | `modeling_v4_assumptions` |

**프리미티브 승격 (신규 순수함수 — scripts에서 추출)**

| 파일 | 내용 | 출처 |
|---|---|---|
| `src/bakery/analysis/demand_absorption.py` (수정) | `placebo_absorption()` 추가 | `scripts/absorption_4stores.py:97-108` |
| `src/bakery/analysis/holiday_premium.py` (신규) | `decompose_holiday_premium()` | `scripts/holiday_premium_decompose.py` |
| `src/bakery/analysis/order_bias.py` (신규) | `isowaste_dow_gap()`, `waste_rate_of()`, `soldout_freq()`, `soldout_mag()` | `scripts/weekday_bias_isowaste.py` |
| `src/bakery/analysis/month_dow.py` (신규) | `month_dow_matrix()` | `scripts/verify_month_dow_adjust.py` |

**수정**

- `src/bakery/cli.py` — `analysis-run` 커맨드 추가 (`harness-run`은 `cli.py:98-118`)
- `scripts/absorption_4stores.py` — `placebo_results` → 프리미티브 import 하는 얇은 wrapper
- `scripts/holiday_premium_decompose.py` — 프리미티브 호출 + print만
- `scripts/weekday_bias_isowaste.py` — 프리미티브 호출 + print만
- `.claude/CLAUDE.md` — 실행 섹션에 `analysis-run` 한 줄. ※2026-07-28 인수인계 작업에서 CLAUDE.md를 backbone-first로 전면 개편하면서 이 줄을 **주석 처리 상태로 미리 넣어뒀다**(`# uv run bakery analysis-run ...  ← Phase 6 구현 중`). Task 18에서는 새로 추가하지 말고 **주석만 해제**하고 라우팅 표의 "구현 중" 표기를 지운다.
- `experiments/analysis_gwangyo.yaml` (신규), `experiments/analysis_multistore.yaml` (신규)

**테스트** — `tests/analysis_lab/` (harness는 `tests/harness/` 패턴)

`__init__.py`, `conftest.py`(`stub_inputs` 팩토리 — 실 parquet IO 없이 `AnalysisInputs`의 `cached_property`를 주입한다. 핸들러 테스트 4곳이 각자 `_StubInputs`를 정의하면 시그니처가 갈라지므로 여기 한 번만 둔다), `test_spec.py`, `test_inputs.py`, `test_registry.py`, `test_runner.py`, `test_report.py`, `test_cli_analysis.py`, `test_handlers_sales.py`, `test_handlers_waste.py`, `test_handlers_absorption.py`, `test_handlers_discount.py`, `test_handlers_stockout.py`, `test_handlers_substitution.py`, `test_handlers_calendar_bias.py`, `test_handlers_model_bias.py`, `test_handlers_basket.py`

**문서** — `docs/phase6_analysis_layer.md` (사용법 + 이식 대조 기록 + 제외 근거)

---

## Task 순서 개요

- **Task 1~5**: 프레임워크 (result → spec → inputs → registry/runner → report → CLI)
- **Task 6~9**: 이식 4형태를 각 1개씩 증명 (데이터분석 / 프리미티브직결 / 스크립트추출 / preds-artifact)
- **Task 10~17**: 나머지 항목 기계적 이식
- **Task 18**: 문서 + 전체 스위트

**첫 PR 경계**: Task 1~9까지가 단독 머지 가능한 단위(프레임워크 + 4형태 증명). Task 10 이후는 후속 PR로 쪼갤 수 있다.

---

### Task 1: 결과 컨테이너 (`result.py`)

**Files:**
- Create: `src/bakery/analysis/lab/__init__.py`
- Create: `src/bakery/analysis/lab/result.py`
- Create: `tests/analysis_lab/__init__.py`
- Test: `tests/analysis_lab/test_result.py`

**Interfaces:**
- Consumes: (없음 — 첫 태스크)
- Produces:
  - `AnalysisResult(name: str, kind: str, title: str, tables: list[tuple[str, pd.DataFrame]], figures: list, verdict: str | None = None, notes: list[str] = [])`
  - `SkippedResult(name: str, kind: str, title: str, reason: str)`
  - `AnalysisReport(name: str, spec_resolved: dict, results: list[AnalysisResult], skipped: list[SkippedResult])`
  - `KIND_DATA = "data"`, `KIND_HYPOTHESIS = "hypothesis"`
  - `REASON_OFF = "off"`, `REASON_PREDS_REQUIRED = "preds_required"`, `REASON_MULTISTORE_REQUIRED = "multistore_required"`, `REASON_SINGLE_STORE_REQUIRED = "single_store_required"`
  - `AnalysisReport.table_of(name, table_name) -> pd.DataFrame` (테스트/리포트 조회 헬퍼)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/analysis_lab/test_result.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.result import (
    KIND_DATA, KIND_HYPOTHESIS, REASON_OFF, REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED, AnalysisReport, AnalysisResult, SkippedResult,
)


def _result(name="category_mix", kind=KIND_DATA):
    return AnalysisResult(
        name=name, kind=kind, title="카테고리 매출 비중",
        tables=[("share", pd.DataFrame({"category_id": ["bread"], "share": [1.0]}))],
        figures=[],
    )


def test_verdict_defaults_to_none_and_notes_empty():
    r = _result()
    assert r.verdict is None
    assert r.notes == []


def test_notes_are_not_shared_between_instances():
    a, b = _result(), _result()
    a.notes.append("주의")
    assert b.notes == []          # 가변 기본값 공유 버그 방지


def test_kind_and_reason_constants_are_exact():
    assert KIND_DATA == "data"
    assert KIND_HYPOTHESIS == "hypothesis"
    assert REASON_OFF == "off"
    assert REASON_PREDS_REQUIRED == "preds_required"
    assert REASON_SINGLE_STORE_REQUIRED == "single_store_required"


def test_report_table_of_returns_the_named_table():
    report = AnalysisReport(
        name="analysis_gwangyo", spec_resolved={"name": "analysis_gwangyo"},
        results=[_result()],
        skipped=[SkippedResult(name="substitution", kind=KIND_HYPOTHESIS,
                               title="수요 대체", reason=REASON_OFF)],
    )
    table = report.table_of("category_mix", "share")
    assert table["share"].tolist() == [1.0]


def test_report_table_of_raises_on_unknown_name():
    report = AnalysisReport(name="x", spec_resolved={}, results=[_result()], skipped=[])
    with pytest.raises(KeyError, match="waste_rate"):
        report.table_of("waste_rate", "share")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_result.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab'`

- [ ] **Step 3: 최소 구현**

`src/bakery/analysis/lab/__init__.py`:

```python
"""데이터분석 + 가설검증 레이어(Phase 6) — analysis-run 오케스트레이션.

harness backbone(예측 평면)의 형제 표면. 이 레이어는 모델을 실행하지 않고,
canonical 입력 데이터와 (선택적으로) harness-run이 남긴 예측 artifact만 읽는다.
"""
from bakery.analysis.lab.result import (
    KIND_DATA, KIND_HYPOTHESIS, REASON_MULTISTORE_REQUIRED, REASON_OFF,
    REASON_PREDS_REQUIRED, REASON_SINGLE_STORE_REQUIRED,
    AnalysisReport, AnalysisResult, SkippedResult,
)

__all__ = [
    "KIND_DATA", "KIND_HYPOTHESIS", "REASON_OFF", "REASON_PREDS_REQUIRED",
    "REASON_MULTISTORE_REQUIRED", "REASON_SINGLE_STORE_REQUIRED",
    "AnalysisResult", "SkippedResult", "AnalysisReport",
]
```

`src/bakery/analysis/lab/result.py`:

```python
"""분석/가설 실행 결과 컨테이너 — 핸들러 반환 계약."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

KIND_DATA = "data"
KIND_HYPOTHESIS = "hypothesis"

REASON_OFF = "off"                                  # YAML에서 꺼짐
REASON_PREDS_REQUIRED = "preds_required"            # preds artifact 미지정/부재
REASON_MULTISTORE_REQUIRED = "multistore_required"  # 4매장 전용 항목인데 단매장 spec
# 광교 전용 소스(category_daily=bonavi_daily)를 쓰는 항목인데 multistore spec.
# 게이트 없이 실행하면 광교 수치가 4매장 분석으로 라벨링되는 조용한 오데이터가 된다.
REASON_SINGLE_STORE_REQUIRED = "single_store_required"


@dataclass
class AnalysisResult:
    """핸들러 1개의 산출물. tables/verdict가 회귀 대조 대상, figures는 리포트 전용."""

    name: str
    kind: str
    title: str
    tables: list[tuple[str, pd.DataFrame]]
    figures: list[Any]
    verdict: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkippedResult:
    """실행하지 않은 항목 — 은폐 방지를 위해 리포트에 사유와 함께 남는다."""

    name: str
    kind: str
    title: str
    reason: str


@dataclass
class AnalysisReport:
    name: str
    spec_resolved: dict
    results: list[AnalysisResult]
    skipped: list[SkippedResult]

    def table_of(self, name: str, table_name: str) -> pd.DataFrame:
        """실행된 항목 `name`의 `table_name` 테이블. 없으면 KeyError."""
        for result in self.results:
            if result.name != name:
                continue
            for label, table in result.tables:
                if label == table_name:
                    return table
            raise KeyError(f"{name}에 테이블 '{table_name}' 없음")
        raise KeyError(f"실행된 항목에 '{name}' 없음")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_result.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/lab/__init__.py src/bakery/analysis/lab/result.py \
        tests/analysis_lab/__init__.py tests/analysis_lab/test_result.py
git commit -m "feat(analysis-lab): AnalysisResult/SkippedResult/AnalysisReport 컨테이너"
```

---

### Task 2: Spec + 이름 강제 (`spec.py`)

**Files:**
- Create: `src/bakery/analysis/lab/spec.py`
- Modify: `src/bakery/analysis/lab/__init__.py` (re-export 추가)
- Test: `tests/analysis_lab/test_spec.py`

**Interfaces:**
- Consumes: `bakery.analysis.lab.result` (없어도 무관 — spec은 독립)
- Produces:
  - `AnalysisDataSpec(source: Literal["real"], store: str = "store_gw01")`
  - `AnalysisSpec(name: str, data: AnalysisDataSpec, predictions: Path | None = None, alpha: float = 0.8, data_analyses: dict[str, bool] = {}, hypotheses: dict[str, bool] = {}, params: dict[str, dict] = {})`
  - `load_analysis_spec(path: str | Path) -> AnalysisSpec`
  - `AnalysisSpecError(ValueError)`
  - `DEPRECATED_ANALYSES: frozenset[str]`
  - `MULTISTORE = "multistore"`, `DEFAULT_ALPHA = 0.8`
  - `AnalysisSpec.enabled(kind) -> list[str]` (그 kind에서 True인 이름들)

**주의: 미등록 이름 검증은 Task 4(registry) 이후에 배선한다.** 이 태스크에서는 `DEPRECATED_ANALYSES`와 형식 검증만. `spec.py`는 `registry.py`를 import하지 않고, 대신 `load_analysis_spec(path, *, known_names=None)`으로 알려진 이름 집합을 주입받는다(순환 import 방지 — `registry`가 `spec`을 import하지 않게 방향 고정).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/analysis_lab/test_spec.py`:

```python
from pathlib import Path

import pytest
import yaml

from bakery.analysis.lab.spec import (
    DEFAULT_ALPHA, DEPRECATED_ANALYSES, MULTISTORE,
    AnalysisSpec, AnalysisSpecError, load_analysis_spec,
)


def _write(tmp_path, body):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def _base(**over):
    body = {"name": "analysis_gwangyo", "data": {"source": "real"}}
    body.update(over)
    return body


def test_defaults_are_gwangyo_alpha_08(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base()))
    assert spec.data.store == "store_gw01"
    assert spec.alpha == DEFAULT_ALPHA == 0.8
    assert spec.predictions is None
    assert spec.data_analyses == {}
    assert spec.hypotheses == {}
    assert spec.params == {}


def test_enabled_returns_only_true_keys(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base(
        data_analyses={"category_mix": True, "waste_rate": False},
        hypotheses={"demand_absorption": True, "substitution": False},
    )))
    assert spec.enabled("data_analyses") == ["category_mix"]
    assert spec.enabled("hypotheses") == ["demand_absorption"]


def test_deprecated_conformal_names_rejected(tmp_path):
    # v5 conformal 구간예측은 점추정+위험수치 전환으로 폐기 — 실수 이식 차단
    assert DEPRECATED_ANALYSES == frozenset(
        {"diag_anchor_gh", "diag_chuseok_gh", "diagnose_conformal_residual"})
    path = _write(tmp_path, _base(hypotheses={"diag_anchor_gh": True}))
    with pytest.raises(AnalysisSpecError, match="diag_anchor_gh"):
        load_analysis_spec(path)


def test_deprecated_name_rejected_even_when_off(tmp_path):
    # off여도 spec에 남아 있으면 "이식 대상"으로 오해되므로 거부한다
    path = _write(tmp_path, _base(hypotheses={"diagnose_conformal_residual": False}))
    with pytest.raises(AnalysisSpecError, match="diagnose_conformal_residual"):
        load_analysis_spec(path)


def test_potential_demand_target_is_not_configurable(tmp_path):
    # 오염 소스 차단: target 키 자체를 받지 않는다(입력 데이터 평면은 target이 없음)
    with pytest.raises(AnalysisSpecError, match="target"):
        load_analysis_spec(_write(tmp_path, _base(target="potential_demand")))


def test_unknown_name_rejected_when_known_names_given(tmp_path):
    path = _write(tmp_path, _base(hypotheses={"demand_absorbtion": True}))   # 오타
    with pytest.raises(AnalysisSpecError, match="demand_absorbtion"):
        load_analysis_spec(path, known_names=frozenset({"demand_absorption"}))


def test_multistore_store_value_accepted(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base(data={"source": "real", "store": MULTISTORE})))
    assert spec.data.store == "multistore"


def test_synthetic_source_rejected(tmp_path):
    # 이 레이어는 실측 데이터/가설 검증 전용 — synthetic은 의미 없음
    with pytest.raises(AnalysisSpecError):
        load_analysis_spec(_write(tmp_path, _base(data={"source": "synthetic"})))


def test_predictions_path_is_parsed_as_path(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base(
        predictions="reports/gwangyo_default/category_total/predictions.csv")))
    assert spec.predictions == Path("reports/gwangyo_default/category_total/predictions.csv")


def test_alpha_out_of_range_rejected(tmp_path):
    with pytest.raises(AnalysisSpecError):
        load_analysis_spec(_write(tmp_path, _base(alpha=1.4)))


def test_spec_constructed_directly_has_same_defaults():
    spec = AnalysisSpec(name="x", data={"source": "real"})
    assert spec.alpha == 0.8
    assert spec.data.store == "store_gw01"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.spec'`

- [ ] **Step 3: 최소 구현**

`src/bakery/analysis/lab/spec.py`:

```python
"""analysis-run YAML spec — pydantic 검증 + 폐기/오타 이름 강제.

harness `config.py`의 SpecError/DEPRECATED 패턴 미러링. registry를 import하지
않고 known_names를 주입받아 순환 import를 피한다(runner가 주입한다).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

MULTISTORE = "multistore"
DEFAULT_ALPHA = 0.8            # 측정 헌장: adjusted_demand = 정상 + 0.8×마감

# v5 conformal 구간예측 계열 — 점추정+위험수치 전환으로 폐기. 이식 금지.
DEPRECATED_ANALYSES: frozenset[str] = frozenset(
    {"diag_anchor_gh", "diag_chuseok_gh", "diagnose_conformal_residual"}
)


class AnalysisSpecError(ValueError):
    """spec 형식/이름 규칙 위반."""


class AnalysisDataSpec(BaseModel):
    model_config = {"extra": "forbid"}

    source: Literal["real"] = "real"
    store: str = "store_gw01"          # store_gw01 | multistore


class AnalysisSpec(BaseModel):
    model_config = {"extra": "forbid"}   # target 등 미지원 키는 즉시 거부

    name: str
    data: AnalysisDataSpec
    predictions: Path | None = None
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    data_analyses: dict[str, bool] = Field(default_factory=dict)
    hypotheses: dict[str, bool] = Field(default_factory=dict)
    params: dict[str, dict] = Field(default_factory=dict)

    def enabled(self, kind: str) -> list[str]:
        """kind('data_analyses'|'hypotheses')에서 True인 이름들(YAML 순서 보존)."""
        section: dict[str, bool] = getattr(self, kind)
        return [name for name, is_on in section.items() if is_on]

    def all_requested(self) -> list[str]:
        return list(self.data_analyses) + list(self.hypotheses)


def load_analysis_spec(
    path: str | Path, *, known_names: frozenset[str] | None = None
) -> AnalysisSpec:
    """YAML → AnalysisSpec. 폐기 이름은 항상 거부, 미등록 이름은 known_names 주면 거부."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        spec = AnalysisSpec(**raw)
    except Exception as exc:
        raise AnalysisSpecError(str(exc)) from exc
    _enforce_names(spec, known_names)
    return spec


def _enforce_names(spec: AnalysisSpec, known_names: frozenset[str] | None) -> None:
    requested = spec.all_requested()
    deprecated = [n for n in requested if n in DEPRECATED_ANALYSES]
    if deprecated:
        raise AnalysisSpecError(
            f"{deprecated}는 DEPRECATED(v5 conformal 구간예측 폐기) — 이식 대상 아님. "
            "spec에서 키를 삭제하라(off로도 남기지 말 것)."
        )
    if known_names is None:
        return
    unknown = [n for n in requested if n not in known_names]
    if unknown:
        raise AnalysisSpecError(f"미등록 분석/가설 이름: {unknown}. registry 등록명을 확인하라.")
```

`src/bakery/analysis/lab/__init__.py`에 추가:

```python
from bakery.analysis.lab.spec import (
    DEFAULT_ALPHA, DEPRECATED_ANALYSES, MULTISTORE,
    AnalysisDataSpec, AnalysisSpec, AnalysisSpecError, load_analysis_spec,
)
```

그리고 `__all__`에 `"AnalysisDataSpec", "AnalysisSpec", "AnalysisSpecError", "load_analysis_spec", "DEPRECATED_ANALYSES", "MULTISTORE", "DEFAULT_ALPHA"` 추가.

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_spec.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/lab/spec.py src/bakery/analysis/lab/__init__.py \
        tests/analysis_lab/test_spec.py
git commit -m "feat(analysis-lab): AnalysisSpec + DEPRECATED/오타 이름 강제"
```

---

### Task 3: 입력 로더 (`inputs.py`)

**Files:**
- Create: `src/bakery/analysis/lab/inputs.py`
- Test: `tests/analysis_lab/test_inputs.py`

**Interfaces:**
- Consumes: `AnalysisSpec` (Task 2)
- Produces:
  - `AnalysisInputs.from_spec(spec) -> AnalysisInputs`
  - 속성(모두 `functools.cached_property`, 최초 접근 시 1회 IO):
    - `daily -> pd.DataFrame` — `bonavi_daily`(단매장) 또는 `multistore_daily`(4매장). `potential_demand` 컬럼은 제거해서 반환.
    - `category_daily -> pd.DataFrame` — `build_category_daily(alpha=self.alpha).df` (광교 전용, 날짜별 1행, `adjusted_demand_unit` 포함)
    - `receipts -> pd.DataFrame` — `bonavi_receipts` (광교 전용)
    - `discount_rows -> pd.DataFrame` — `load_sales_with_discount_v2(store_code=...).rows`
    - `closing_returns -> pd.DataFrame` — `load_closing_returns_v2(store_code=...)`
    - `waste -> pd.DataFrame` — `waste_alpha_4stores` 필터 + `waste_qty`/`production_qty` 리네임
    - `item_to_category -> pd.Series` — `daily`에서 유도 (index=item_id, value=category_id)
    - `predictions -> pd.DataFrame | None` — `spec.predictions` CSV. 없으면 None
    - `calendar -> pd.DataFrame` — `build_calendar_daily(daily 최소~최대)`
  - `has_predictions -> bool`, `is_multistore -> bool`
  - `STORE_CODES: dict[str, str]` (store_id → CD_PARTNER), `STORE_NAMES: dict[str, str]` (store_id → 한글명), `STORE_PRIOR_KEYS: dict[str, str]` (store_id → `STORE_EVENT_PRIORS` 키)
  - `prior_key -> str` (단매장 prior 프리셋 키. multistore면 광교)
  - `params_for(name) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/analysis_lab/test_inputs.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from bakery.analysis.lab.inputs import STORE_CODES, STORE_NAMES, AnalysisInputs
from bakery.analysis.lab.spec import AnalysisSpec


def _spec(**over):
    body = {"name": "t", "data": {"source": "real"}}
    body.update(over)
    return AnalysisSpec(**body)


def test_store_prior_keys_are_english_labels():
    from bakery.analysis.lab.inputs import STORE_PRIOR_KEYS
    from bakery.harness.event_priors import STORE_EVENT_PRIORS

    assert STORE_PRIOR_KEYS == {"store_gw01": "gwangyo", "store_ss01": "samsung",
                                "store_mp01": "mecenatpolis", "store_gh01": "gwanghwamun"}
    assert set(STORE_PRIOR_KEYS.values()) == set(STORE_EVENT_PRIORS)


def test_prior_key_falls_back_to_gwangyo_for_multistore():
    assert AnalysisInputs.from_spec(_spec()).prior_key == "gwangyo"
    assert AnalysisInputs.from_spec(
        _spec(data={"source": "real", "store": "multistore"})).prior_key == "gwangyo"
    assert AnalysisInputs.from_spec(
        _spec(data={"source": "real", "store": "store_mp01"})).prior_key == "mecenatpolis"


def test_store_codes_cover_four_stores():
    assert STORE_CODES == {
        "store_gw01": "1000000047",
        "store_ss01": "1000000009",
        "store_mp01": "1000000029",
        "store_gh01": "1000000485",
    }
    assert STORE_NAMES["store_gw01"] == "광교"


def test_is_multistore_flag():
    assert AnalysisInputs.from_spec(_spec()).is_multistore is False
    assert AnalysisInputs.from_spec(
        _spec(data={"source": "real", "store": "multistore"})).is_multistore is True


def test_has_predictions_false_when_unset():
    inputs = AnalysisInputs.from_spec(_spec())
    assert inputs.has_predictions is False
    assert inputs.predictions is None


def test_has_predictions_false_when_path_missing(tmp_path):
    inputs = AnalysisInputs.from_spec(_spec(predictions=tmp_path / "nope.csv"))
    assert inputs.has_predictions is False


def test_predictions_loaded_with_parsed_dates(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("date,fold,actual,expected,production\n"
                    "2025-12-25,0,307.6,315.8,356.7\n", encoding="utf-8")
    inputs = AnalysisInputs.from_spec(_spec(predictions=path))
    assert inputs.has_predictions is True
    preds = inputs.predictions
    assert preds["date"].tolist() == [pd.Timestamp("2025-12-25")]
    assert preds["expected"].tolist() == [315.8]


def test_params_for_returns_empty_dict_when_absent():
    inputs = AnalysisInputs.from_spec(_spec())
    assert inputs.params_for("demand_absorption") == {}


def test_params_for_returns_declared_params():
    inputs = AnalysisInputs.from_spec(_spec(params={"demand_absorption": {"close_hour": 21}}))
    assert inputs.params_for("demand_absorption") == {"close_hour": 21}


@pytest.mark.slow
def test_daily_drops_potential_demand_and_filters_store():
    # 측정 헌장: potential_demand는 오염 소스 — 로더에서 아예 제거해 소비 불가로 만든다
    inputs = AnalysisInputs.from_spec(_spec())
    daily = inputs.daily
    assert "potential_demand" not in daily.columns
    assert daily["store_id"].unique().tolist() == ["store_gw01"]
    assert daily["is_stockout"].dtype == bool


@pytest.mark.slow
def test_multistore_daily_has_four_stores():
    inputs = AnalysisInputs.from_spec(_spec(data={"source": "real", "store": "multistore"}))
    assert sorted(inputs.daily["store_id"].unique()) == [
        "store_gh01", "store_gw01", "store_mp01", "store_ss01"]


@pytest.mark.slow
def test_daily_is_cached_single_read():
    inputs = AnalysisInputs.from_spec(_spec())
    assert inputs.daily is inputs.daily        # cached_property 동일 객체


@pytest.mark.slow
def test_waste_frame_renames_without_transforming_values():
    import pandas as pd
    from bakery.data import paths

    inputs = AnalysisInputs.from_spec(_spec())
    waste = inputs.waste
    assert set(waste.columns) >= {"item_id", "date", "production_qty", "waste_qty"}
    assert waste["cd"].unique().tolist() == ["1000000047"]
    # 순수 rename 계약: made→production_qty, out→waste_qty 값 변형 없음.
    # 음수 waste_qty는 전일 재고 이월(carry-in)로 판매가 당일 생산을 초과한 실제 신호이며
    # (해당 행에서 identity_diff==0), clip하면 항등식이 깨지고 폐기율이 부풀려진다.
    raw = pd.read_parquet(paths.dataset("waste_alpha_4stores"))
    raw = raw[raw["cd"].astype(str) == "1000000047"].reset_index(drop=True)
    assert waste["production_qty"].tolist() == raw["made"].tolist()
    assert waste["waste_qty"].tolist() == raw["out"].tolist()


@pytest.mark.slow
def test_item_to_category_maps_bread():
    inputs = AnalysisInputs.from_spec(_spec())
    mapping = inputs.item_to_category
    daily = inputs.daily
    first = daily.iloc[0]
    assert mapping[first["item_id"]] == first["category_id"]
```

`@pytest.mark.slow` 마커가 `pyproject.toml`에 없으면 추가한다:

```toml
[tool.pytest.ini_options]
markers = ["slow: canonical parquet 실데이터를 읽는 테스트"]
```

(기존 `[tool.pytest.ini_options]` 블록에 `markers`만 추가. `addopts`는 건드리지 않는다.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_inputs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.inputs'`

- [ ] **Step 3: 최소 구현**

`src/bakery/analysis/lab/inputs.py`:

```python
"""canonical 입력 lazy 로더 — 핸들러가 공유하는 단일 IO 지점.

모든 경로는 `bakery.data.paths.dataset()`으로만 도달한다(레거시 data/internal/v2
직독 금지). 측정 헌장: bulk는 canonical 빌드에서 이미 제외됐고, potential_demand는
오염 소스라 여기서 컬럼째 제거해 소비 자체를 막는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from bakery.data import paths
from bakery.analysis.lab.spec import MULTISTORE, AnalysisSpec

STORE_CODES: dict[str, str] = {
    "store_gw01": "1000000047",
    "store_ss01": "1000000009",
    "store_mp01": "1000000029",
    "store_gh01": "1000000485",
}
STORE_NAMES: dict[str, str] = {
    "store_gw01": "광교",
    "store_ss01": "삼성타운",
    "store_mp01": "메세나폴리스",
    "store_gh01": "광화문",
}
# harness.event_priors.STORE_EVENT_PRIORS의 키(영문 라벨) — 한글명과 다르다.
STORE_PRIOR_KEYS: dict[str, str] = {
    "store_gw01": "gwangyo",
    "store_ss01": "samsung",
    "store_mp01": "mecenatpolis",
    "store_gh01": "gwanghwamun",
}
GWANGYO = "store_gw01"
_POLLUTED_COLUMNS = ("potential_demand",)   # 오염 소스 — 측정 헌장상 사용 금지


@dataclass
class AnalysisInputs:
    """spec이 가리키는 입력 묶음. 속성 최초 접근 시에만 IO."""

    store: str
    alpha: float
    predictions_path: Path | None = None
    params: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: AnalysisSpec) -> AnalysisInputs:
        return cls(store=spec.data.store, alpha=spec.alpha,
                   predictions_path=spec.predictions, params=dict(spec.params))

    # ---------- 메타 ----------

    @property
    def is_multistore(self) -> bool:
        return self.store == MULTISTORE

    @property
    def store_code(self) -> str:
        """단매장 CD_PARTNER. multistore면 광교(참조 매장) 코드."""
        return STORE_CODES[GWANGYO if self.is_multistore else self.store]

    @property
    def prior_key(self) -> str:
        """이벤트 prior 프리셋 키. multistore면 광교(참조 매장)."""
        return STORE_PRIOR_KEYS[GWANGYO if self.is_multistore else self.store]

    @property
    def has_predictions(self) -> bool:
        return self.predictions_path is not None and Path(self.predictions_path).exists()

    def params_for(self, name: str) -> dict:
        return self.params.get(name, {})

    # ---------- 입력 ----------

    @cached_property
    def daily(self) -> pd.DataFrame:
        """item×day 관측 daily. bulk 제외·매진 재정의 반영(canonical)."""
        key = "multistore_daily" if self.is_multistore else "bonavi_daily"
        df = pd.read_parquet(paths.dataset(key))
        df = df.drop(columns=[c for c in _POLLUTED_COLUMNS if c in df.columns])
        df["item_id"] = df["item_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        if not self.is_multistore:
            df = df[df["store_id"] == self.store]
        return df.reset_index(drop=True)

    @cached_property
    def category_daily(self) -> pd.DataFrame:
        """날짜별 카테고리 합 daily(adjusted_demand_unit 포함) — 광교 전용."""
        from bakery.features.category_aggregate import build_category_daily

        return build_category_daily(alpha=self.alpha).df

    @cached_property
    def receipts(self) -> pd.DataFrame:
        """라인레벨 영수증(광교). is_bulk는 진단용 컬럼이며 이미 필터된 프레임."""
        df = pd.read_parquet(paths.dataset("bonavi_receipts"))
        df["item_id"] = df["item_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def discount_rows(self) -> pd.DataFrame:
        from bakery.analysis.discount import load_sales_with_discount_v2

        return load_sales_with_discount_v2(store_code=self.store_code).rows

    @cached_property
    def closing_returns(self) -> pd.DataFrame:
        from bakery.analysis.discount import load_closing_returns_v2

        return load_closing_returns_v2(store_code=self.store_code)

    @cached_property
    def waste(self) -> pd.DataFrame:
        """생산/폐기/마감 실측(4매장). production_qty=made, waste_qty=out.

        waste_qty 음수는 전일 재고 이월(carry-in)로 판매가 당일 생산을 초과한 경우다
        (그 행에서도 made−(normal+closing)−out=0 항등식은 성립). 값을 clip하지 않는다 —
        clip하면 항등식이 깨지고 폐기율(1차 KPI)이 부풀려진다.
        """
        df = pd.read_parquet(paths.dataset("waste_alpha_4stores"))
        df["cd"] = df["cd"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        if not self.is_multistore:
            df = df[df["cd"] == self.store_code]
        df = df.rename(columns={"made": "production_qty", "out": "waste_qty"})
        return df.reset_index(drop=True)

    @cached_property
    def item_to_category(self) -> pd.Series:
        """item_id → category_id (canonical daily에서 유도)."""
        pairs = self.daily[["item_id", "category_id"]].drop_duplicates("item_id")
        return pairs.set_index("item_id")["category_id"]

    @cached_property
    def predictions(self) -> pd.DataFrame | None:
        """harness-run 산출 predictions.csv(date/fold/actual/expected/production). 읽기 전용."""
        if not self.has_predictions:
            return None
        df = pd.read_csv(self.predictions_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def calendar(self) -> pd.DataFrame:
        from bakery.data.calendar import build_calendar_daily

        dates = self.daily["date"]
        return build_calendar_daily(dates.min(), dates.max())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_inputs.py -v`
Expected: PASS (12 passed) — `slow` 마커 테스트는 canonical parquet을 읽으므로 수십 초 걸릴 수 있다.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/lab/inputs.py tests/analysis_lab/test_inputs.py pyproject.toml
git commit -m "feat(analysis-lab): AnalysisInputs lazy 로더(paths 기반, potential_demand 차단)"
```

---

### Task 4: Registry + Runner

**Files:**
- Create: `src/bakery/analysis/lab/registry.py`
- Create: `src/bakery/analysis/lab/runner.py`
- Create: `src/bakery/analysis/lab/handlers/__init__.py`
- Modify: `src/bakery/analysis/lab/__init__.py`
- Test: `tests/analysis_lab/test_registry.py`, `tests/analysis_lab/test_runner.py`

**Interfaces:**
- Consumes: `AnalysisSpec`/`load_analysis_spec` (Task 2), `AnalysisInputs` (Task 3), `AnalysisResult`/`SkippedResult`/`AnalysisReport` (Task 1)
- Produces:
  - `Handler(name, kind, title, fn, needs_predictions=False, needs_multistore=False, needs_single_store=False)` — frozen dataclass. `fn: Callable[[AnalysisInputs], AnalysisResult]`
  - `DATA_ANALYSES: dict[str, Handler]`, `HYPOTHESES: dict[str, Handler]`
  - `register_data(name, title, **flags)` / `register_hypothesis(name, title, **flags)` — 데코레이터
  - `all_names() -> frozenset[str]`, `resolve(name) -> Handler`
  - `run_analysis(spec, *, out_dir: Path) -> AnalysisReport`
  - `SKIP_REASON_OF: dict[str, str]` (게이트 미충족 → `REASON_*`)

**설계 결정: 데코레이터 등록.** 핸들러 모듈이 `@register_hypothesis("demand_absorption", "카테고리 총량 수요이전 흡수")`로 자기 등록하고, `registry.py`는 `handlers` 패키지를 import해 딕셔너리를 채운다. 이유 — 핸들러 추가 시 registry.py를 매번 수정하지 않아 태스크 간 충돌이 줄고, 이름/제목/게이트가 핸들러 옆에 붙어 드리프트가 없다.

- [ ] **Step 1: registry 실패 테스트 작성**

`tests/analysis_lab/test_registry.py`:

```python
import pytest

from bakery.analysis.lab.registry import (
    DATA_ANALYSES, HYPOTHESES, all_names, resolve,
)
from bakery.analysis.lab.result import KIND_DATA, KIND_HYPOTHESIS
from bakery.analysis.lab.spec import DEPRECATED_ANALYSES


def test_kinds_are_tagged_per_section():
    for name, handler in DATA_ANALYSES.items():
        assert handler.kind == KIND_DATA, name
        assert handler.name == name
    for name, handler in HYPOTHESES.items():
        assert handler.kind == KIND_HYPOTHESIS, name
        assert handler.name == name


def test_no_deprecated_name_is_registered():
    assert all_names() & DEPRECATED_ANALYSES == frozenset()


def test_data_and_hypothesis_namespaces_do_not_collide():
    assert set(DATA_ANALYSES) & set(HYPOTHESES) == set()


def test_every_handler_has_korean_title():
    for handler in list(DATA_ANALYSES.values()) + list(HYPOTHESES.values()):
        assert handler.title != ""
        assert handler.title != handler.name        # 제목은 이름 재사용 금지(한국어 산문)


def test_resolve_returns_handler_for_both_sections():
    assert resolve("category_mix").kind == KIND_DATA
    assert resolve("demand_absorption").kind == KIND_HYPOTHESIS


def test_resolve_raises_on_unknown():
    with pytest.raises(KeyError, match="nope"):
        resolve("nope")


def test_all_names_is_union_of_two_sections():
    assert all_names() == frozenset(DATA_ANALYSES) | frozenset(HYPOTHESES)
```

**주의:** Task 4 시점에는 아직 핸들러가 없어 `resolve("category_mix")`/`resolve("demand_absorption")` 테스트가 실패한다. 이 두 테스트는 Task 6·7에서 켜지도록 지금은 `@pytest.mark.xfail(reason="핸들러는 Task 6/7에서 등록", strict=True)`를 붙이고, 해당 태스크에서 마커를 제거한다(다른 5개는 빈 registry에서도 통과).

- [ ] **Step 2: registry 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.registry'`

- [ ] **Step 3: registry 구현**

`src/bakery/analysis/lab/handlers/__init__.py`:

```python
"""분석/가설 핸들러 모듈 — import만으로 registry에 자기 등록된다.

registry.load_handlers()가 이 목록을 순회한다. 새 핸들러 모듈 추가 시 여기에 이름을 넣는다.
"""

HANDLER_MODULES: tuple[str, ...] = ()
```

`src/bakery/analysis/lab/registry.py`:

```python
"""이름 → 핸들러 registry. harness registry.py의 kind/is_runnable 패턴 미러링."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_DATA, KIND_HYPOTHESIS, AnalysisResult

HandlerFn = Callable[[AnalysisInputs], AnalysisResult]


@dataclass(frozen=True)
class Handler:
    name: str
    kind: str
    title: str
    fn: HandlerFn
    needs_predictions: bool = False     # harness-run predictions.csv 필요
    needs_multistore: bool = False      # 4매장 비교 전용
    needs_single_store: bool = False    # 광교 전용 소스(category_daily) 사용 → multistore 금지


DATA_ANALYSES: dict[str, Handler] = {}
HYPOTHESES: dict[str, Handler] = {}


def _register(target: dict[str, Handler], name: str, kind: str, title: str, flags: dict):
    def deco(fn: HandlerFn) -> HandlerFn:
        if name in DATA_ANALYSES or name in HYPOTHESES:
            raise ValueError(f"핸들러 이름 중복 등록: {name}")
        target[name] = Handler(name=name, kind=kind, title=title, fn=fn, **flags)
        return fn
    return deco


def register_data(name: str, title: str, **flags):
    """입력 데이터 분석 핸들러 등록 데코레이터."""
    return _register(DATA_ANALYSES, name, KIND_DATA, title, flags)


def register_hypothesis(name: str, title: str, **flags):
    """가설 검증 핸들러 등록 데코레이터."""
    return _register(HYPOTHESES, name, KIND_HYPOTHESIS, title, flags)


def load_handlers() -> None:
    """핸들러 모듈을 import해 registry를 채운다(멱등)."""
    from bakery.analysis.lab.handlers import HANDLER_MODULES

    for module in HANDLER_MODULES:
        importlib.import_module(f"bakery.analysis.lab.handlers.{module}")


def all_names() -> frozenset[str]:
    load_handlers()
    return frozenset(DATA_ANALYSES) | frozenset(HYPOTHESES)


def resolve(name: str) -> Handler:
    load_handlers()
    if name in DATA_ANALYSES:
        return DATA_ANALYSES[name]
    if name in HYPOTHESES:
        return HYPOTHESES[name]
    raise KeyError(f"미등록 분석/가설: {name}")
```

`load_handlers()`를 `DATA_ANALYSES`/`HYPOTHESES` 직접 참조 테스트에서도 타게 하려면, `tests/analysis_lab/test_registry.py` 최상단에 `from bakery.analysis.lab.registry import load_handlers; load_handlers()`를 모듈 레벨로 한 줄 넣는다.

- [ ] **Step 4: registry 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_registry.py -v`
Expected: PASS (5 passed, 2 xfailed)

- [ ] **Step 5: runner 실패 테스트 작성**

`tests/analysis_lab/test_runner.py`:

```python
import pandas as pd
import pytest
import yaml

from bakery.analysis.lab import registry
from bakery.analysis.lab.result import (
    KIND_DATA, KIND_HYPOTHESIS, REASON_MULTISTORE_REQUIRED, REASON_OFF,
    REASON_PREDS_REQUIRED, REASON_SINGLE_STORE_REQUIRED, AnalysisResult,
)
from bakery.analysis.lab.runner import run_analysis
from bakery.analysis.lab.spec import AnalysisSpec


def _fake_result(name, kind):
    return AnalysisResult(name=name, kind=kind, title="t",
                          tables=[("x", pd.DataFrame({"v": [1]}))], figures=[],
                          verdict="지지" if kind == KIND_HYPOTHESIS else None)


@pytest.fixture
def fake_registry(monkeypatch):
    """실 핸들러 대신 계산 없는 스텁 3개를 등록해 runner 배선만 검증한다."""
    calls: list[str] = []

    def _make(name, kind):
        def fn(inputs):
            calls.append(name)
            return _fake_result(name, kind)
        return fn

    data = {"stub_data": registry.Handler("stub_data", KIND_DATA, "스텁 데이터",
                                         _make("stub_data", KIND_DATA))}
    hypo = {
        "stub_hypo": registry.Handler("stub_hypo", KIND_HYPOTHESIS, "스텁 가설",
                                      _make("stub_hypo", KIND_HYPOTHESIS)),
        "stub_preds": registry.Handler("stub_preds", KIND_HYPOTHESIS, "스텁 preds",
                                       _make("stub_preds", KIND_HYPOTHESIS),
                                       needs_predictions=True),
        "stub_ms": registry.Handler("stub_ms", KIND_HYPOTHESIS, "스텁 다매장",
                                    _make("stub_ms", KIND_HYPOTHESIS),
                                    needs_multistore=True),
        "stub_single": registry.Handler("stub_single", KIND_HYPOTHESIS, "스텁 단매장",
                                        _make("stub_single", KIND_HYPOTHESIS),
                                        needs_single_store=True),
    }
    monkeypatch.setattr(registry, "DATA_ANALYSES", data)
    monkeypatch.setattr(registry, "HYPOTHESES", hypo)
    monkeypatch.setattr(registry, "load_handlers", lambda: None)
    return calls


def _spec(**over):
    body = {"name": "t", "data": {"source": "real"}}
    body.update(over)
    return AnalysisSpec(**body)


def test_only_enabled_items_run(fake_registry, tmp_path):
    report = run_analysis(_spec(data_analyses={"stub_data": True},
                                hypotheses={"stub_hypo": False}), out_dir=tmp_path)
    assert fake_registry == ["stub_data"]
    assert [r.name for r in report.results] == ["stub_data"]


def test_off_item_is_recorded_with_reason_off(fake_registry, tmp_path):
    report = run_analysis(_spec(hypotheses={"stub_hypo": False}), out_dir=tmp_path)
    assert [(s.name, s.reason) for s in report.skipped] == [("stub_hypo", REASON_OFF)]


def test_unrequested_registry_items_are_also_listed_as_off(fake_registry, tmp_path):
    # 은폐 방지: spec에 없는 항목도 off로 리포트에 남는다
    report = run_analysis(_spec(data_analyses={"stub_data": True}), out_dir=tmp_path)
    assert {s.name for s in report.skipped} == {"stub_hypo", "stub_preds", "stub_ms",
                                               "stub_single"}
    assert {s.reason for s in report.skipped} == {REASON_OFF}


def test_preds_required_item_skipped_without_artifact(fake_registry, tmp_path):
    report = run_analysis(_spec(hypotheses={"stub_preds": True}), out_dir=tmp_path)
    assert fake_registry == []
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_preds"] == [("stub_preds", REASON_PREDS_REQUIRED)]


def test_preds_required_item_runs_with_artifact(fake_registry, tmp_path):
    preds = tmp_path / "predictions.csv"
    preds.write_text("date,fold,actual,expected,production\n2025-01-01,0,1,1,1\n",
                     encoding="utf-8")
    report = run_analysis(_spec(hypotheses={"stub_preds": True}, predictions=preds),
                          out_dir=tmp_path)
    assert fake_registry == ["stub_preds"]
    assert [r.name for r in report.results] == ["stub_preds"]


def test_multistore_item_skipped_on_single_store_spec(fake_registry, tmp_path):
    report = run_analysis(_spec(hypotheses={"stub_ms": True}), out_dir=tmp_path)
    assert fake_registry == []
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_ms"] == [("stub_ms", REASON_MULTISTORE_REQUIRED)]


def test_multistore_item_runs_on_multistore_spec(fake_registry, tmp_path):
    report = run_analysis(_spec(data={"source": "real", "store": "multistore"},
                                hypotheses={"stub_ms": True}), out_dir=tmp_path)
    assert fake_registry == ["stub_ms"]


def test_single_store_item_skipped_on_multistore_spec(fake_registry, tmp_path):
    # 광교 전용 소스를 4매장 라벨로 내보내는 조용한 오데이터 차단
    report = run_analysis(_spec(data={"source": "real", "store": "multistore"},
                                hypotheses={"stub_single": True}), out_dir=tmp_path)
    assert fake_registry == []
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_single"] == [("stub_single", REASON_SINGLE_STORE_REQUIRED)]


def test_single_store_item_runs_on_single_store_spec(fake_registry, tmp_path):
    report = run_analysis(_spec(hypotheses={"stub_single": True}), out_dir=tmp_path)
    assert fake_registry == ["stub_single"]


def test_resolved_config_written_to_out_dir(fake_registry, tmp_path):
    run_analysis(_spec(name="analysis_x", data_analyses={"stub_data": True}), out_dir=tmp_path)
    written = yaml.safe_load((tmp_path / "analysis_x" / "config_resolved.yaml").read_text())
    assert written["name"] == "analysis_x"
    assert written["alpha"] == 0.8


def test_tables_written_as_csv(fake_registry, tmp_path):
    run_analysis(_spec(name="analysis_x", data_analyses={"stub_data": True}), out_dir=tmp_path)
    csv = pd.read_csv(tmp_path / "analysis_x" / "stub_data__x.csv")
    assert csv["v"].tolist() == [1]


def test_handler_exception_becomes_skip_not_crash(fake_registry, tmp_path, monkeypatch):
    def boom(inputs):
        raise ValueError("데이터 부족")
    monkeypatch.setitem(registry.DATA_ANALYSES, "stub_data",
                        registry.Handler("stub_data", KIND_DATA, "스텁 데이터", boom))
    report = run_analysis(_spec(data_analyses={"stub_data": True}), out_dir=tmp_path)
    assert report.results == []
    assert [s.reason for s in report.skipped if s.name == "stub_data"] == ["error: 데이터 부족"]


def test_report_carries_spec_resolved(fake_registry, tmp_path):
    report = run_analysis(_spec(name="analysis_x"), out_dir=tmp_path)
    assert report.name == "analysis_x"
    assert report.spec_resolved["data"]["store"] == "store_gw01"
```

- [ ] **Step 6: runner 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.runner'`

- [ ] **Step 7: runner 구현**

`src/bakery/analysis/lab/runner.py`:

```python
"""analysis-run 실행 엔진 — 켜진 항목만 돌리고, 끈/못 돌린 항목은 사유와 함께 남긴다.

harness runner.py와 달리 모델을 실행하지 않는다(예측 artifact는 읽기 전용).
핸들러 예외는 전체 실행을 죽이지 않고 그 항목만 error 스킵으로 강등한다 —
14개 항목 중 하나가 데이터 부족으로 실패해도 나머지 리포트는 나와야 한다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from bakery.analysis.lab import registry
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import (
    REASON_MULTISTORE_REQUIRED, REASON_OFF, REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED, AnalysisReport, AnalysisResult, SkippedResult,
)
from bakery.analysis.lab.spec import AnalysisSpec


def _gate_reason(handler: registry.Handler, inputs: AnalysisInputs) -> str | None:
    """실행 전 게이트. 통과면 None, 아니면 스킵 사유."""
    if handler.needs_predictions and not inputs.has_predictions:
        return REASON_PREDS_REQUIRED
    if handler.needs_multistore and not inputs.is_multistore:
        return REASON_MULTISTORE_REQUIRED
    if handler.needs_single_store and inputs.is_multistore:
        return REASON_SINGLE_STORE_REQUIRED
    return None


def _handlers_in_order(spec: AnalysisSpec) -> list[tuple[registry.Handler, bool]]:
    """(핸들러, 켜짐여부) — registry 전체를 돌아 spec 미명시는 off로 취급한다."""
    registry.load_handlers()
    requested = {**spec.data_analyses, **spec.hypotheses}
    sections = (registry.DATA_ANALYSES, registry.HYPOTHESES)
    return [(handler, requested.get(name, False))
            for section in sections for name, handler in section.items()]


def _write_tables(result: AnalysisResult, out: Path) -> None:
    for label, table in result.tables:
        table.to_csv(out / f"{result.name}__{label}.csv", index=False)


def run_analysis(spec: AnalysisSpec, *, out_dir: Path) -> AnalysisReport:
    """spec에서 켜진 분석/가설을 실행해 AnalysisReport를 만든다."""
    inputs = AnalysisInputs.from_spec(spec)
    out = out_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump(mode="json")
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")

    results: list[AnalysisResult] = []
    skipped: list[SkippedResult] = []
    for handler, is_on in _handlers_in_order(spec):
        reason = REASON_OFF if not is_on else _gate_reason(handler, inputs)
        if reason is None:
            reason = _run_one(handler, inputs, out, results)
        if reason is not None:
            skipped.append(SkippedResult(name=handler.name, kind=handler.kind,
                                         title=handler.title, reason=reason))
    return AnalysisReport(name=spec.name, spec_resolved=resolved,
                          results=results, skipped=skipped)


def _run_one(handler: registry.Handler, inputs: AnalysisInputs,
             out: Path, results: list[AnalysisResult]) -> str | None:
    """핸들러 1개 실행. 성공하면 results에 추가하고 None, 실패하면 사유 문자열."""
    try:
        result = handler.fn(inputs)
    except Exception as exc:                     # noqa: BLE001 — 항목 단위 격리가 목적
        return f"error: {exc}"
    _write_tables(result, out)
    results.append(result)
    return None
```

`src/bakery/analysis/lab/__init__.py`에 `from bakery.analysis.lab.runner import run_analysis` 추가 + `__all__`에 `"run_analysis"`.

- [ ] **Step 8: runner 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_runner.py -v`
Expected: PASS (13 passed)

- [ ] **Step 9: 커밋**

```bash
git add src/bakery/analysis/lab/registry.py src/bakery/analysis/lab/runner.py \
        src/bakery/analysis/lab/handlers/__init__.py src/bakery/analysis/lab/__init__.py \
        tests/analysis_lab/test_registry.py tests/analysis_lab/test_runner.py
git commit -m "feat(analysis-lab): registry(데코레이터 등록) + runner(on/off·게이트·항목격리)"
```

---

### Task 5: HTML 리포트 + CLI `analysis-run`

**Files:**
- Create: `src/bakery/analysis/lab/report.py`
- Create: `experiments/analysis_gwangyo.yaml`
- Create: `experiments/analysis_multistore.yaml`
- Modify: `src/bakery/cli.py` (import 블록 + 새 커맨드)
- Modify: `src/bakery/analysis/lab/__init__.py`
- Modify: `.claude/CLAUDE.md` (실행 섹션)
- Test: `tests/analysis_lab/test_report.py`, `tests/analysis_lab/test_cli_analysis.py`

**Interfaces:**
- Consumes: `AnalysisReport` (Task 1), `run_analysis` (Task 4), `load_analysis_spec` (Task 2), `all_names` (Task 4)
- Produces:
  - `build_analysis_report(report: AnalysisReport, *, out_path: Path) -> Path`
  - `fig_to_div(fig, div_id, *, include_js, height=450) -> str` (harness `report.py:20`과 동일 계약, stateless)
  - `SKIP_LABELS: dict[str, str]` (사유 → 한국어 표기)
  - CLI: `bakery analysis-run <config> [--out reports/analysis]`

- [ ] **Step 1: report 실패 테스트 작성**

`tests/analysis_lab/test_report.py`:

```python
import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.report import build_analysis_report, fig_to_div
from bakery.analysis.lab.result import (
    KIND_DATA, KIND_HYPOTHESIS, REASON_OFF, REASON_PREDS_REQUIRED,
    AnalysisReport, AnalysisResult, SkippedResult,
)


def _report():
    share = pd.DataFrame({"category_id": ["bread", "pastry"], "share": [0.7, 0.3]})
    absorb = pd.DataFrame({"category_id": ["bread"], "beta": [-0.01], "verdict": ["absorb"]})
    return AnalysisReport(
        name="analysis_gwangyo",
        spec_resolved={"name": "analysis_gwangyo", "data": {"source": "real",
                                                            "store": "store_gw01"}},
        results=[
            AnalysisResult(name="category_mix", kind=KIND_DATA, title="카테고리 매출 비중",
                           tables=[("share", share)],
                           figures=[go.Figure(go.Bar(x=["bread"], y=[0.7]))]),
            AnalysisResult(name="demand_absorption", kind=KIND_HYPOTHESIS,
                           title="카테고리 총량 수요이전 흡수",
                           tables=[("results", absorb)],
                           figures=[go.Figure(go.Bar(x=["bread"], y=[-0.01]))],
                           verdict="지지 — 일반 카테고리 walk-away 0건",
                           notes=["censoring 무시 가정(측정 헌장)"]),
        ],
        skipped=[
            SkippedResult(name="substitution", kind=KIND_HYPOTHESIS, title="수요 대체",
                          reason=REASON_OFF),
            SkippedResult(name="weekday_bias", kind=KIND_HYPOTHESIS, title="평일 과대예측",
                          reason=REASON_PREDS_REQUIRED),
        ],
    )


def test_html_written_and_returns_path(tmp_path):
    out = tmp_path / "analysis_report.html"
    assert build_analysis_report(_report(), out_path=out) == out
    assert out.exists()


def test_both_sections_and_titles_present(tmp_path):
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "입력 데이터 분석" in html
    assert "가설 검증" in html
    assert "카테고리 매출 비중" in html
    assert "카테고리 총량 수요이전 흡수" in html


def test_verdict_and_notes_rendered(tmp_path):
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "지지 — 일반 카테고리 walk-away 0건" in html
    assert "censoring 무시 가정(측정 헌장)" in html


def test_skipped_items_are_disclosed_with_reasons(tmp_path):
    # 은폐 방지 = 성공기준. off와 preds 부재는 서로 다른 라벨로 나와야 한다.
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "substitution" in html
    assert "(off)" in html
    assert "weekday_bias" in html
    assert "(preds 필요 — 미실행)" in html


def test_plotly_js_embedded_exactly_once(tmp_path):
    # stateless fig_to_div 불변: fig가 여러 개여도 plotly.js는 1회만
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert html.count("cdn.plot.ly") == 1


def test_report_without_any_figure_still_embeds_js(tmp_path):
    report = _report()
    for result in report.results:
        result.figures = []
    out = tmp_path / "r.html"
    build_analysis_report(report, out_path=out)
    html = out.read_text(encoding="utf-8")
    assert html.count("cdn.plot.ly") == 1        # on/off 요약표가 첫 fig 역할


def test_fig_to_div_include_js_toggle():
    fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
    assert "cdn.plot.ly" in fig_to_div(fig, "d1", include_js=True)
    assert "cdn.plot.ly" not in fig_to_div(fig, "d2", include_js=False)
```

- [ ] **Step 2: report 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.report'`

- [ ] **Step 3: report 구현**

`src/bakery/analysis/lab/report.py`:

```python
"""AnalysisReport → 자기포함 HTML. harness report.py의 stateless fig_to_div 패턴 재사용.

섹션 A(입력 데이터 분석) / B(가설 검증) + 실행 요약표. 끈 항목·못 돌린 항목은
사유와 함께 반드시 표기한다(은폐 방지 = 성공기준).
"""
from __future__ import annotations

import html as html_lib
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from bakery.analysis.lab.result import (
    KIND_DATA, KIND_HYPOTHESIS, REASON_MULTISTORE_REQUIRED, REASON_OFF,
    REASON_PREDS_REQUIRED, REASON_SINGLE_STORE_REQUIRED, AnalysisReport, AnalysisResult,
)

SKIP_LABELS: dict[str, str] = {
    REASON_OFF: "(off)",
    REASON_PREDS_REQUIRED: "(preds 필요 — 미실행)",
    REASON_MULTISTORE_REQUIRED: "(multistore spec 필요 — 미실행)",
    REASON_SINGLE_STORE_REQUIRED: "(단매장 spec 필요 — 미실행)",
}
_TABLE_ROW_LIMIT = 200        # HTML 비대 방지. 전체는 out_dir CSV에 있다.

_HTML_SHELL = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>analysis report — {name}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}}
h1{{border-bottom:2px solid #2c3e50}} h2{{margin-top:32px;color:#2c3e50}}
h3{{margin-top:24px}} .verdict{{background:#eef6ff;padding:8px 12px;border-left:4px solid #2c3e50}}
.note{{color:#666;font-size:0.9em}} .skip{{color:#999}}
table{{border-collapse:collapse;font-size:0.9em}} td,th{{border:1px solid #ddd;padding:4px 8px}}
</style></head><body><h1>데이터분석 + 가설검증 report — {name}</h1>{body}</body></html>"""


def fig_to_div(fig: go.Figure, div_id: str, *, include_js: bool, height: int = 450) -> str:
    """Plotly fig → HTML div. include_js=True인 첫 호출만 plotly.js를 cdn으로 임베드."""
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=50), height=height,
                      autosize=True, hovermode="x unified")
    return pio.to_html(fig, include_plotlyjs=("cdn" if include_js else False),
                       div_id=div_id, full_html=False)


def _summary_fig(report: AnalysisReport) -> go.Figure:
    rows = [{"항목": r.name, "구분": r.kind, "상태": "실행"} for r in report.results]
    rows += [{"항목": s.name, "구분": s.kind, "상태": SKIP_LABELS.get(s.reason, s.reason)}
             for s in report.skipped]
    frame = pd.DataFrame(rows)
    fig = go.Figure(go.Table(
        header=dict(values=list(frame.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[frame[c].tolist() for c in frame.columns]),
    ))
    fig.update_layout(title="실행 항목 on/off 요약")
    return fig


def _table_html(label: str, table: pd.DataFrame) -> str:
    shown = table.head(_TABLE_ROW_LIMIT)
    suffix = (f"<p class='note'>상위 {_TABLE_ROW_LIMIT}행만 표시 "
              f"(전체 {len(table)}행은 CSV 참조)</p>") if len(table) > _TABLE_ROW_LIMIT else ""
    return (f"<p><b>{html_lib.escape(label)}</b></p>"
            f"{shown.to_html(index=False, border=0)}{suffix}")


def _result_html(result: AnalysisResult, figures: list[Any], js_used: bool) -> tuple[str, bool]:
    parts = [f"<h3>{html_lib.escape(result.title)} "
             f"<span class='note'>({html_lib.escape(result.name)})</span></h3>"]
    if result.verdict is not None:
        parts.append(f"<p class='verdict'><b>판정</b>: {html_lib.escape(result.verdict)}</p>")
    for index, fig in enumerate(figures):
        parts.append(fig_to_div(fig, f"{result.name}_{index}", include_js=not js_used))
        js_used = True
    for label, table in result.tables:
        parts.append(_table_html(label, table))
    for note in result.notes:
        parts.append(f"<p class='note'>⚠️ {html_lib.escape(note)}</p>")
    return "\n".join(parts), js_used


def _section_html(report: AnalysisReport, kind: str, heading: str,
                  js_used: bool) -> tuple[str, bool]:
    parts = [f"<h2>{heading}</h2>"]
    results = [r for r in report.results if r.kind == kind]
    if not results:
        parts.append("<p class='skip'>실행된 항목 없음</p>")
    for result in results:
        body, js_used = _result_html(result, result.figures, js_used)
        parts.append(body)
    skipped = [s for s in report.skipped if s.kind == kind]
    if skipped:
        items = "".join(f"<li class='skip'>{html_lib.escape(s.name)} — "
                        f"{html_lib.escape(s.title)} "
                        f"{html_lib.escape(SKIP_LABELS.get(s.reason, s.reason))}</li>"
                        for s in skipped)
        parts.append(f"<p class='note'>미실행 항목:</p><ul>{items}</ul>")
    return "\n".join(parts), js_used


def build_analysis_report(report: AnalysisReport, *, out_path: Path) -> Path:
    """AnalysisReport → 자기포함 HTML 1개."""
    store = report.spec_resolved.get("data", {}).get("store", "?")
    divs = [f"<p>데이터 소스: real / 매장: {html_lib.escape(str(store))}</p>",
            fig_to_div(_summary_fig(report), "summary", include_js=True, height=260)]
    body_a, js_used = _section_html(report, KIND_DATA, "섹션 A — 입력 데이터 분석", True)
    divs.append(body_a)
    body_b, _ = _section_html(report, KIND_HYPOTHESIS, "섹션 B — 가설 검증", js_used)
    divs.append(body_b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_HTML_SHELL.format(name=report.name, body="\n".join(divs)),
                        encoding="utf-8")
    return out_path
```

- [ ] **Step 4: report 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_report.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: CLI 실패 테스트 작성**

`tests/analysis_lab/test_cli_analysis.py` (harness `tests/harness/test_cli_harness.py` 패턴):

```python
import yaml
from typer.testing import CliRunner

from bakery.cli import app

runner = CliRunner()


def _yaml(tmp_path, body):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def test_analysis_run_all_off_produces_html(tmp_path):
    # 모든 항목 off → 계산 없이 HTML만. 실데이터 IO 없이 CLI 배선을 검증한다.
    config = _yaml(tmp_path, {"name": "analysis_smoke", "data": {"source": "real"}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 0
    html = (tmp_path / "analysis_smoke" / "analysis_report.html").read_text(encoding="utf-8")
    assert "(off)" in html


def test_analysis_run_rejects_deprecated_name(tmp_path):
    config = _yaml(tmp_path, {"name": "x", "data": {"source": "real"},
                              "hypotheses": {"diag_anchor_gh": True}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 1


def test_analysis_run_rejects_unknown_name(tmp_path):
    config = _yaml(tmp_path, {"name": "x", "data": {"source": "real"},
                              "hypotheses": {"demand_absorbtion": True}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 1


def test_shipped_gwangyo_yaml_loads_with_registry_names():
    # experiments/analysis_gwangyo.yaml의 모든 키가 registry에 실제 등록돼 있는지
    from bakery.analysis.lab.registry import all_names
    from bakery.analysis.lab.spec import load_analysis_spec

    spec = load_analysis_spec("experiments/analysis_gwangyo.yaml", known_names=all_names())
    assert spec.name == "analysis_gwangyo"
    assert spec.data.store == "store_gw01"


def test_shipped_multistore_yaml_is_multistore():
    from bakery.analysis.lab.registry import all_names
    from bakery.analysis.lab.spec import load_analysis_spec

    spec = load_analysis_spec("experiments/analysis_multistore.yaml", known_names=all_names())
    assert spec.data.store == "multistore"
```

- [ ] **Step 6: CLI 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_cli_analysis.py -v`
Expected: FAIL — `No such command 'analysis-run'` (exit_code != 0)

- [ ] **Step 7: CLI 구현**

`src/bakery/cli.py`의 harness import 줄(`from .harness import load_spec, run_experiment, build_report`, 91행) 아래에 추가:

```python
from .analysis.lab import build_analysis_report, load_analysis_spec, run_analysis
from .analysis.lab.registry import all_names as analysis_names
from .analysis.lab.spec import AnalysisSpecError
```

`cmd_harness_run` 함수(118행 끝) 바로 아래에 커맨드 추가:

```python
ANALYSIS_DIR = REPORTS_DIR / "analysis"


@app.command("analysis-run")
def cmd_analysis_run(
    config: Path,
    out: Path = ANALYSIS_DIR,
) -> None:
    """YAML 1개로 입력 데이터 분석 + 가설 검증을 실행한다 (analysis 단일 표면)."""
    try:
        spec = load_analysis_spec(config, known_names=analysis_names())
    except AnalysisSpecError as exc:
        console.print(f"[red]spec 오류[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[cyan]analysis[/] {spec.name} store={spec.data.store} "
                  f"analyses={spec.enabled('data_analyses')} "
                  f"hypotheses={spec.enabled('hypotheses')}")
    report = run_analysis(spec, out_dir=out)
    table = Table(title=f"{spec.name} — 실행 결과")
    table.add_column("항목"); table.add_column("구분"); table.add_column("판정/상태")
    for result in report.results:
        table.add_row(result.name, result.kind, result.verdict or "실행 완료")
    for skip in report.skipped:
        table.add_row(skip.name, skip.kind, skip.reason)
    console.print(table)
    path = build_analysis_report(report, out_path=out / spec.name / "analysis_report.html")
    console.print(f"[green]report[/] {path}")
```

`src/bakery/analysis/lab/__init__.py`에 `from bakery.analysis.lab.report import build_analysis_report` 추가 + `__all__`.

`experiments/analysis_gwangyo.yaml`:

```yaml
# 광교 단독 — 입력 데이터 분석 + 가설 검증
# predictions는 harness-run 산출 artifact. 지정하면 preds 의존 가설이 활성된다.
name: analysis_gwangyo
data:
  source: real
  store: store_gw01
alpha: 0.8                        # 측정 헌장: adjusted_demand = 정상 + 0.8×마감
predictions: reports/gwangyo_default/category_total/predictions.csv
params:
  # event_prior_validation A/B 대조군(layers: [] 로 돌린 harness 산출). Task 17 Step 6b에서 생성.
  event_prior_validation:
    baseline_predictions: reports/gwangyo_no_prior/category_total/predictions.csv
data_analyses:
  sales_distribution: true
  category_mix: true
  waste_rate: true
  waste_alpha_identity: false
  overproduction_breakdown: true
hypotheses:
  demand_absorption: true
  substitution: false
  stockout_revenue: true
  closing_discount: true
  other_discounts: false
  discount_regime: false
  seasonal_bias: false
  weather_bias: false
  weekday_bias: false
  holiday_premium: true
  month_dow_adjust: false
  popularity_stockout: false
  modeling_v4_assumptions: false
  event_prior_validation: false
```

`experiments/analysis_multistore.yaml`:

```yaml
# 4매장 — 매장간 비교가 필요한 항목 위주(광교 예측 보조 데이터 관점)
name: analysis_multistore
data:
  source: real
  store: multistore
alpha: 0.8
data_analyses:
  sales_distribution: true
  category_mix: true
  waste_rate: true
hypotheses:
  demand_absorption: true
  stockout_revenue: true
```

`.claude/CLAUDE.md`의 실행 블록에 한 줄 추가 (`harness-run` 관련 줄 근처):

```bash
uv run bakery analysis-run experiments/analysis_gwangyo.yaml   # 입력데이터 분석+가설검증 → analysis_report.html
```

- [ ] **Step 8: CLI 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/ -v`
Expected: PASS (전체 통과, registry의 2개는 여전히 xfailed)

- [ ] **Step 9: 실제 CLI 스모크**

Run: `uv run bakery analysis-run experiments/analysis_gwangyo.yaml --out /tmp/analysis_smoke`
Expected: 모든 항목이 아직 미등록이므로 spec 오류(exit 1)로 떨어진다 — 이는 정상이며, Task 6에서 첫 핸들러가 등록되면 통과한다. 대신 아래로 스모크한다:

Run: `printf 'name: smoke\ndata:\n  source: real\n' > /tmp/smoke.yaml && uv run bakery analysis-run /tmp/smoke.yaml --out /tmp/analysis_smoke && open /tmp/analysis_smoke/smoke/analysis_report.html`
Expected: exit 0, HTML 생성, 브라우저에 "실행 항목 on/off 요약" 표와 두 섹션이 보인다.

- [ ] **Step 10: 커밋**

```bash
git add src/bakery/analysis/lab/report.py src/bakery/analysis/lab/__init__.py \
        src/bakery/cli.py experiments/analysis_gwangyo.yaml experiments/analysis_multistore.yaml \
        .claude/CLAUDE.md tests/analysis_lab/test_report.py tests/analysis_lab/test_cli_analysis.py
git commit -m "feat(analysis-lab): 자기포함 HTML 리포트 + bakery analysis-run CLI"
```

**주의:** `experiments/analysis_gwangyo.yaml`은 아직 등록되지 않은 이름을 담고 있어 `test_shipped_gwangyo_yaml_loads_with_registry_names`가 Task 17까지 실패한다. 이 두 테스트에 `@pytest.mark.xfail(reason="핸들러 전량 등록은 Task 17에서 완료", strict=True)`를 붙이고 Task 17에서 제거한다.

---

## 이식 4형태 — Task 6~9로 각 1개씩 증명

| 형태 | 증명 태스크 | 게이트 |
|---|---|---|
| ① 데이터 분석(canonical 재표현) | Task 6 `category_mix` | 구조 불변식(합=1.0) + 합성 fixture 정확값 |
| ② 프리미티브 직결 | Task 7 `demand_absorption` | 핸들러 출력 == 프리미티브 직접 호출(정확 일치) |
| ③ 스크립트 추출(데이터 온리) | Task 8 `holiday_premium` | frozen-input golden(2026-07-28 캡처) |
| ④ preds-artifact 소비 | Task 9 `weekday_bias` | frozen-input golden(2026-07-28 캡처) |

이 4개가 통과하면 Task 10~17은 기계적 반복이다.

---

### Task 6: 형태① 증명 — `category_mix` (입력 데이터 분석)

**Files:**
- Create: `src/bakery/analysis/lab/handlers/sales.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py` (`HANDLER_MODULES`에 `"sales"` 추가)
- Modify: `tests/analysis_lab/test_registry.py` (`resolve("category_mix")` xfail 제거)
- Test: `tests/analysis_lab/test_handlers_sales.py`

**Interfaces:**
- Consumes: `register_data` (Task 4), `AnalysisInputs.daily`/`.waste` (Task 3), `AnalysisResult` (Task 1)
- Produces:
  - `category_mix(inputs) -> AnalysisResult` — tables `("share", ...)`, `("monthly_stability", ...)`
  - `median_unit_price(waste: pd.DataFrame) -> pd.Series` (index=item_id, value=unit_price 중앙값)
  - `category_share(daily, prices) -> pd.DataFrame` (cols: store_id, category_id, sold_units, share, revenue, revenue_share)
  - `monthly_share_stability(daily) -> pd.DataFrame` (cols: store_id, category_id, n_months, share_std, share_min, share_max)
  - `MONTH_STD_DDOF = 0` (관측 월 전체 = 모집단 → 값이 재현 가능한 정수 비율로 떨어짐)

- [ ] **Step 0: 공용 stub fixture 작성**

`tests/analysis_lab/conftest.py`:

```python
"""핸들러 테스트 공용 — 실 parquet IO 없이 AnalysisInputs 속성을 주입한다.

AnalysisInputs의 입력 속성은 functools.cached_property라서 __dict__에 값을 직접
넣으면 IO 없이 그 값이 쓰인다. 핸들러별로 필요한 속성만 주면 된다.
"""
import pytest

from bakery.analysis.lab.inputs import AnalysisInputs


@pytest.fixture
def stub_inputs():
    def _make(*, store="store_gw01", alpha=0.8, params=None, **attributes):
        inputs = AnalysisInputs(store=store, alpha=alpha, params=params or {})
        for name, value in attributes.items():
            inputs.__dict__[name] = value       # cached_property 사전 채우기
        return inputs
    return _make
```

핸들러 테스트는 `_StubInputs` 클래스를 각자 정의하지 않고 이 fixture를 쓴다:

```python
def test_something(stub_inputs):
    inputs = stub_inputs(daily=_daily(), waste=_waste())
```

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/analysis_lab/test_handlers_sales.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.sales import (
    category_mix, category_share, median_unit_price, monthly_share_stability,
)
from bakery.analysis.lab.result import KIND_DATA


def _daily():
    """광교 2품목 3일. bread 60 / pastry 30 (총 90) — 손계산 가능한 fixture."""
    rows = [
        ("2025-01-01", "b1", "bread", 10), ("2025-01-01", "p1", "pastry", 5),
        ("2025-01-02", "b1", "bread", 30), ("2025-01-02", "p1", "pastry", 5),
        ("2025-02-01", "b1", "bread", 20), ("2025-02-01", "p1", "pastry", 20),
    ]
    return pd.DataFrame([{"store_id": "store_gw01", "item_id": i, "category_id": c,
                          "date": pd.Timestamp(d), "sold_units": q,
                          "is_stockout": False, "stockout_time": pd.NaT}
                         for d, i, c, q in rows])


def _waste():
    """단가만 쓰는 fixture — b1=3000, p1=5000 (중앙값 계산 대상으로 중복 행 포함)."""
    return pd.DataFrame({
        "item_id": ["b1", "b1", "p1"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
        "unit_price": [3000, 3000, 5000],
        "production_qty": [12, 32, 6], "waste_qty": [2, 2, 1],
    })


def test_median_unit_price_is_per_item_median():
    prices = median_unit_price(_waste())
    assert prices["b1"] == 3000.0
    assert prices["p1"] == 5000.0


def test_category_share_units_and_revenue_exact():
    share = category_share(_daily(), median_unit_price(_waste()))
    bread = share[share["category_id"] == "bread"].iloc[0]
    pastry = share[share["category_id"] == "pastry"].iloc[0]
    assert bread["sold_units"] == 60
    assert pastry["sold_units"] == 30
    assert bread["share"] == pytest.approx(60 / 90)
    assert pastry["share"] == pytest.approx(30 / 90)
    # revenue: bread 60×3000=180000, pastry 30×5000=150000 → 총 330000
    assert bread["revenue"] == 180000.0
    assert pastry["revenue"] == 150000.0
    assert bread["revenue_share"] == pytest.approx(180000 / 330000)


def test_category_share_sums_to_one_per_store():
    share = category_share(_daily(), median_unit_price(_waste()))
    assert share.groupby("store_id")["share"].sum().iloc[0] == pytest.approx(1.0)
    assert share.groupby("store_id")["revenue_share"].sum().iloc[0] == pytest.approx(1.0)


def test_monthly_share_stability_exact():
    # 1월: bread 40/50=0.8, pastry 10/50=0.2 | 2월: 20/40=0.5, 20/40=0.5
    # ddof=0 모집단 std → |0.8-0.5|/2 = 0.15
    stability = monthly_share_stability(_daily()).set_index("category_id")
    assert stability.loc["bread", "n_months"] == 2
    assert stability.loc["bread", "share_std"] == pytest.approx(0.15)
    assert stability.loc["bread", "share_min"] == pytest.approx(0.5)
    assert stability.loc["bread", "share_max"] == pytest.approx(0.8)
    assert stability.loc["pastry", "share_std"] == pytest.approx(0.15)


def test_monthly_share_stability_groups_by_store():
    daily = _daily()
    other = daily.copy()
    other["store_id"] = "store_ss01"
    stability = monthly_share_stability(pd.concat([daily, other], ignore_index=True))
    assert sorted(stability["store_id"].unique()) == ["store_gw01", "store_ss01"]
    assert len(stability) == 4          # 2매장 × 2카테고리


def test_handler_returns_data_kind_without_verdict(stub_inputs):
    result = category_mix(stub_inputs(daily=_daily(), waste=_waste()))
    assert result.name == "category_mix"
    assert result.kind == KIND_DATA
    assert result.verdict is None       # 데이터 분석은 판정 없음
    assert [label for label, _ in result.tables] == ["share", "monthly_stability"]
    assert len(result.figures) == 2


def test_handler_notes_price_coverage(stub_inputs):
    daily = _daily()
    daily.loc[len(daily)] = {"store_id": "store_gw01", "item_id": "unknown",
                             "category_id": "bread", "date": pd.Timestamp("2025-02-02"),
                             "sold_units": 10, "is_stockout": False, "stockout_time": pd.NaT}
    result = category_mix(stub_inputs(daily=daily, waste=_waste()))
    # 단가 미매핑 10개 / 총 100개 → coverage 0.9. 은폐 방지로 note에 남긴다.
    assert result.notes == ["단가 매핑 커버리지 0.900 — 미매핑 품목의 revenue는 0으로 계산됨"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_sales.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.handlers.sales'`

- [ ] **Step 3: 최소 구현**

`src/bakery/analysis/lab/handlers/sales.py`:

```python
"""입력 데이터 분석 — 매출 분포 / 카테고리 비중.

레거시 eda01/eda03은 `data/internal/v2/` 원본 시트를 다른 필터(FG_ITEM=='SS',
beverage/etc 포함)로 읽었다. 여기서는 canonical daily(bulk 제외·5카테고리) 위에서
재표현하므로 옛 스크립트와 수치 등가가 아니다 — 게이트는 구조 불변식(비중 합=1.0)이다.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_data
from bakery.analysis.lab.result import KIND_DATA, AnalysisResult

MONTH_STD_DDOF = 0        # 관측 월 전체가 모집단 — 표본 보정 없이 재현 가능한 값


def median_unit_price(waste: pd.DataFrame) -> pd.Series:
    """item_id → 단가 중앙값. 폐기 실측 테이블이 유일한 매장 단가 소스."""
    return waste.groupby("item_id")["unit_price"].median().astype(float)


def _with_revenue(daily: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    out = daily.copy()
    out["unit_price"] = out["item_id"].map(prices)
    out["revenue"] = out["sold_units"] * out["unit_price"].fillna(0.0)
    return out


def category_share(daily: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """매장×카테고리 수량/매출 비중."""
    priced = _with_revenue(daily, prices)
    grouped = (priced.groupby(["store_id", "category_id"], observed=True)
               .agg(sold_units=("sold_units", "sum"), revenue=("revenue", "sum"))
               .reset_index())
    totals = grouped.groupby("store_id")[["sold_units", "revenue"]].transform("sum")
    grouped["share"] = grouped["sold_units"] / totals["sold_units"]
    grouped["revenue_share"] = grouped["revenue"] / totals["revenue"]
    return grouped.sort_values(["store_id", "share"], ascending=[True, False]) \
                  .reset_index(drop=True)


def monthly_share_stability(daily: pd.DataFrame) -> pd.DataFrame:
    """월별 카테고리 비중의 산포 — 믹스가 안정적이면 카테고리 합 예측이 정당하다."""
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = (frame.groupby(["store_id", "month", "category_id"], observed=True)
               ["sold_units"].sum().reset_index())
    month_total = monthly.groupby(["store_id", "month"])["sold_units"].transform("sum")
    monthly["share"] = monthly["sold_units"] / month_total
    return (monthly.groupby(["store_id", "category_id"], observed=True)["share"]
            .agg(n_months="count",
                 share_std=lambda s: s.std(ddof=MONTH_STD_DDOF),
                 share_min="min", share_max="max")
            .reset_index())


def _share_fig(share: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for category, group in share.groupby("category_id", observed=True):
        fig.add_trace(go.Bar(x=group["store_id"], y=group["share"], name=str(category)))
    fig.update_layout(title="매장별 카테고리 수량 비중", barmode="stack",
                      xaxis_title="매장", yaxis_title="비중")
    return fig


def _stability_fig(daily: pd.DataFrame) -> go.Figure:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = (frame.groupby(["store_id", "month", "category_id"], observed=True)
               ["sold_units"].sum().reset_index())
    total = monthly.groupby(["store_id", "month"])["sold_units"].transform("sum")
    monthly["share"] = monthly["sold_units"] / total
    fig = go.Figure()
    for (store, category), group in monthly.groupby(["store_id", "category_id"],
                                                    observed=True):
        fig.add_trace(go.Scatter(x=group["month"], y=group["share"], mode="lines+markers",
                                 name=f"{store}/{category}"))
    fig.update_layout(title="월별 카테고리 비중 안정성", xaxis_title="월", yaxis_title="비중")
    return fig


def _coverage_notes(daily: pd.DataFrame, prices: pd.Series) -> list[str]:
    mapped = daily["item_id"].isin(prices.index)
    total = float(daily["sold_units"].sum())
    if total == 0.0:
        return []
    coverage = float(daily.loc[mapped, "sold_units"].sum()) / total
    if coverage >= 1.0:
        return []
    return [f"단가 매핑 커버리지 {coverage:.3f} — 미매핑 품목의 revenue는 0으로 계산됨"]


@register_data("category_mix", "카테고리 매출 비중 + 월별 안정성")
def category_mix(inputs: AnalysisInputs) -> AnalysisResult:
    daily = inputs.daily
    prices = median_unit_price(inputs.waste)
    share = category_share(daily, prices)
    stability = monthly_share_stability(daily)
    return AnalysisResult(
        name="category_mix", kind=KIND_DATA, title="카테고리 매출 비중 + 월별 안정성",
        tables=[("share", share), ("monthly_stability", stability)],
        figures=[_share_fig(share), _stability_fig(daily)],
        notes=_coverage_notes(daily, prices),
    )
```

`src/bakery/analysis/lab/handlers/__init__.py`:

```python
HANDLER_MODULES: tuple[str, ...] = ("sales",)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_sales.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: registry xfail 해제 + 전체 lab 테스트**

`tests/analysis_lab/test_registry.py`의 `test_resolve_returns_handler_for_both_sections`에서 `category_mix` 관련 xfail 마커를 제거한다(`demand_absorption` 단언은 Task 7까지 유지하려면 그 줄만 남긴 별도 테스트로 분리).

Run: `uv run pytest tests/analysis_lab/ -v`
Expected: PASS

- [ ] **Step 6: 실데이터 스모크 (구조 불변식 육안 확인)**

Run:
```bash
printf 'name: t6\ndata:\n  source: real\ndata_analyses:\n  category_mix: true\n' > /tmp/t6.yaml
uv run bakery analysis-run /tmp/t6.yaml --out /tmp/analysis_t6
uv run python -c "
import pandas as pd
s = pd.read_csv('/tmp/analysis_t6/t6/category_mix__share.csv')
print(s.to_string()); print('share 합', s['share'].sum())
"
```
Expected: `share 합` = 1.0 (부동소수 오차 내), 카테고리 5종(bread/pastry/sandwich/sweets/cake)이 보인다.

- [ ] **Step 7: 커밋**

```bash
git add src/bakery/analysis/lab/handlers/sales.py src/bakery/analysis/lab/handlers/__init__.py \
        tests/analysis_lab/test_handlers_sales.py tests/analysis_lab/test_registry.py
git commit -m "feat(analysis-lab): category_mix 핸들러 — 데이터분석 형태 증명"
```

---

### Task 7: 형태② 증명 — `demand_absorption` (프리미티브 직결)

**Files:**
- Modify: `src/bakery/analysis/demand_absorption.py` (`placebo_absorption` + `GATE_CATEGORIES` 추가)
- Create: `src/bakery/analysis/lab/handlers/absorption.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Modify: `scripts/absorption_4stores.py` (`placebo_results` → 프리미티브 위임)
- Test: `tests/analysis_lab/test_handlers_absorption.py`, `tests/test_demand_absorption_placebo.py`

**Interfaces:**
- Consumes: `run_absorption`/`fit_absorption`/`build_absorption_panel`/`AbsorptionResult` (기존 `src/bakery/analysis/demand_absorption.py`), `register_hypothesis` (Task 4)
- Produces:
  - `placebo_absorption(daily, *, close_hour=DEFAULT_CLOSE_HOUR, baseline_weeks=BASELINE_WEEKS, horizon_days=PLACEBO_HORIZON_DAYS) -> list[AbsorptionResult]`
  - `PLACEBO_HORIZON_DAYS = 7`, `GATE_CATEGORIES = ("bread", "pastry")`
  - `results_to_frame(results: list[AbsorptionResult], *, arm: str) -> pd.DataFrame`
  - `absorption_verdict(results: list[AbsorptionResult]) -> str`
  - `demand_absorption(inputs) -> AnalysisResult` — tables `("results", ...)`, `("gate_summary", ...)`

- [ ] **Step 1: 프리미티브 추출 실패 테스트 작성**

`tests/test_demand_absorption_placebo.py`:

```python
import pandas as pd

from bakery.analysis.demand_absorption import (
    GATE_CATEGORIES, PLACEBO_HORIZON_DAYS, build_absorption_panel, fit_absorption,
    placebo_absorption,
)


def _daily():
    """2카테고리 × 200일. 품절강도가 총량에 미치는 영향을 회귀할 수 있는 최소 패널."""
    rows = []
    for day in range(200):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for category, items in (("bread", ["b1", "b2"]), ("pastry", ["p1", "p2"])):
            for index, item in enumerate(items):
                is_stockout = (day + index) % 3 == 0
                rows.append({
                    "store_id": "store_gw01", "item_id": item, "category_id": category,
                    "date": date, "sold_units": 10 + (day % 7) + index * 2,
                    "is_stockout": is_stockout,
                    "stockout_time": date + pd.Timedelta(hours=19) if is_stockout else pd.NaT,
                })
    return pd.DataFrame(rows)


def test_gate_categories_are_bread_and_pastry():
    assert GATE_CATEGORIES == ("bread", "pastry")
    assert PLACEBO_HORIZON_DAYS == 7


def test_placebo_shifts_treatment_forward_by_horizon():
    """placebo = 미래 d+7 품절강도로 회귀 — 허위상관 크기의 하한."""
    daily = _daily()
    results = placebo_absorption(daily)
    # 참조 계산: 패널을 만들고 처치변수만 -7 shift 후 같은 fitter를 돌린다
    panel = build_absorption_panel(daily).sort_values("date")
    panel["stockout_hours"] = (panel.groupby(["store_id", "category_id"])["stockout_hours"]
                               .shift(-PLACEBO_HORIZON_DAYS))
    panel = panel.dropna(subset=["stockout_hours"])
    expected = [fit_absorption(panel, s, c)
                for s, c in panel[["store_id", "category_id"]]
                .drop_duplicates().itertuples(index=False)]
    expected = [r for r in expected if r is not None]
    assert [r.category_id for r in results] == [r.category_id for r in expected]
    assert [r.n for r in results] == [r.n for r in expected]
    assert [r.beta for r in results] == [r.beta for r in expected]
    assert [r.verdict for r in results] == [r.verdict for r in expected]


def test_placebo_has_fewer_rows_than_real():
    daily = _daily()
    real_n = build_absorption_panel(daily).groupby("category_id").size().min()
    placebo_n = min(r.n for r in placebo_absorption(daily))
    assert placebo_n == real_n - PLACEBO_HORIZON_DAYS


def test_script_delegates_to_primitive():
    """scripts/absorption_4stores.py는 얇은 wrapper여야 한다(로직 중복 금지)."""
    import sys
    sys.path.insert(0, "scripts")
    import absorption_4stores

    assert absorption_4stores.placebo_results is placebo_absorption
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_demand_absorption_placebo.py -v`
Expected: FAIL — `ImportError: cannot import name 'GATE_CATEGORIES' from 'bakery.analysis.demand_absorption'`

- [ ] **Step 3: 프리미티브 구현**

`src/bakery/analysis/demand_absorption.py` 끝에 추가 (기존 상수 블록 아래에 `PLACEBO_HORIZON_DAYS`/`GATE_CATEGORIES` 선언):

```python
PLACEBO_HORIZON_DAYS = 7           # 미래 d+7 품절강도 = 허위상관 하한
GATE_CATEGORIES = ("bread", "pastry")   # W0 게이트 판정 대상(단일품목/시즌 카테고리 제외)
```

```python
def placebo_absorption(daily: pd.DataFrame, *, close_hour: int = DEFAULT_CLOSE_HOUR,
                       baseline_weeks: int = BASELINE_WEEKS,
                       horizon_days: int = PLACEBO_HORIZON_DAYS) -> list[AbsorptionResult]:
    """미래(d+horizon) 품절강도로 같은 회귀를 돌린다 — 허위상관/잔차 confound 크기 하한.

    실제 β와 부호·크기가 비슷하면 그 β는 인과가 아니라 confound다.
    """
    panel = build_absorption_panel(daily, close_hour=close_hour,
                                   baseline_weeks=baseline_weeks).sort_values("date")
    panel["stockout_hours"] = (panel.groupby(["store_id", "category_id"])["stockout_hours"]
                               .shift(-horizon_days))
    panel = panel.dropna(subset=["stockout_hours"])
    out: list[AbsorptionResult] = []
    pairs = panel[["store_id", "category_id"]].drop_duplicates().itertuples(index=False)
    for store_id, category_id in pairs:
        res = fit_absorption(panel, store_id, category_id)
        if res is not None:
            out.append(res)
    return out
```

`scripts/absorption_4stores.py` 수정 — import에 `placebo_absorption`, `GENERAL_CATEGORIES`를 프리미티브로 위임:

```python
from bakery.analysis.demand_absorption import (
    GATE_CATEGORIES,
    build_absorption_panel,
    fit_absorption,
    placebo_absorption,
    run_absorption,
)
```

그리고 `GENERAL_CATEGORIES = ("bread", "pastry")` 줄(29행)을 삭제하고 `GENERAL_CATEGORIES = GATE_CATEGORIES`로, `placebo_results` 함수 정의(97~108행)를 삭제하고 `placebo_results = placebo_absorption` 별칭 한 줄로 바꾼다.

- [ ] **Step 4: 프리미티브 테스트 통과 확인**

Run: `uv run pytest tests/test_demand_absorption_placebo.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 핸들러 실패 테스트 작성**

`tests/analysis_lab/test_handlers_absorption.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.demand_absorption import AbsorptionResult
from bakery.analysis.lab.handlers.absorption import (
    absorption_verdict, demand_absorption, results_to_frame,
)
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _res(category, verdict, beta=-0.01):
    return AbsorptionResult(store_id="store_gw01", category_id=category, n=100,
                            beta=beta, se=0.02, ci_low=beta - 0.03, ci_high=beta + 0.03,
                            delta=0.05, verdict=verdict)


def test_results_to_frame_columns_and_arm():
    frame = results_to_frame([_res("bread", "absorb")], arm="real")
    assert frame.columns.tolist() == ["store_id", "category_id", "n", "beta", "se",
                                      "ci_low", "ci_high", "delta", "verdict", "arm"]
    assert frame["arm"].tolist() == ["real"]
    assert frame["beta"].tolist() == [-0.01]


def test_verdict_supports_when_all_gate_categories_absorb():
    results = [_res("bread", "absorb"), _res("pastry", "absorb"), _res("cake", "walkaway")]
    # cake는 게이트 대상 아님(단일품목/시즌) → 판정에 영향 없음
    assert absorption_verdict(results) == "지지 — 게이트 카테고리 2건 전부 absorb, walk-away 0건"


def test_verdict_rejects_on_any_gate_walkaway():
    results = [_res("bread", "walkaway"), _res("pastry", "absorb")]
    assert absorption_verdict(results) == (
        "기각 — walk-away 발견: [('store_gw01', 'bread')] (게이트 2건 중 absorb 1건)")


def test_verdict_inconclusive_when_mixed_without_walkaway():
    results = [_res("bread", "inconclusive"), _res("pastry", "absorb")]
    assert absorption_verdict(results) == (
        "불확실 — 게이트 2건 중 absorb 1건 / inconclusive 1건, walk-away 0건")


def test_verdict_when_no_gate_category_present():
    assert absorption_verdict([_res("cake", "absorb")]) == (
        "불확실 — 게이트 카테고리(bread/pastry) 결과 없음")


@pytest.mark.slow
def test_handler_tables_equal_primitive_output():
    """핸들러는 프리미티브를 호출만 한다 — 재구현 드리프트가 있으면 이 테스트가 깨진다."""
    from bakery.analysis.demand_absorption import placebo_absorption, run_absorption
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = demand_absorption(inputs)
    assert result.kind == KIND_HYPOTHESIS
    table = dict(result.tables)["results"]
    expected = pd.concat([results_to_frame(run_absorption(inputs.daily), arm="real"),
                          results_to_frame(placebo_absorption(inputs.daily), arm="placebo")],
                         ignore_index=True)
    pd.testing.assert_frame_equal(table, expected)


@pytest.mark.slow
def test_handler_gate_summary_counts_match_results():
    from bakery.analysis.demand_absorption import GATE_CATEGORIES
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = demand_absorption(inputs)
    tables = dict(result.tables)
    real = tables["results"].query("arm == 'real'")
    gate = real[real["category_id"].isin(GATE_CATEGORIES)]
    summary = tables["gate_summary"]
    assert summary["n_gate"].iloc[0] == len(gate)
    assert summary["n_walkaway"].iloc[0] == int((gate["verdict"] == "walkaway").sum())
```

- [ ] **Step 6: 핸들러 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_absorption.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.handlers.absorption'`

- [ ] **Step 7: 핸들러 구현**

`src/bakery/analysis/lab/handlers/absorption.py`:

```python
"""가설 — 카테고리 총량 수요이전 흡수(W0 게이트).

품목 조기품절이 같은 카테고리 총 sold를 떨어뜨리는가(β<0=walk-away) 아니면
카테고리 안에서 흡수되는가(β≈0). 흡수면 v4 Stage1(카테고리 합)→Stage2(비율 배분)
설계가 정당하다. 계산은 전부 `bakery.analysis.demand_absorption` 프리미티브 호출.
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.demand_absorption import (
    GATE_CATEGORIES, AbsorptionResult, placebo_absorption, run_absorption,
)
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult

ARM_REAL = "real"
ARM_PLACEBO = "placebo"
_NOTE_CENSORING = ("품절일 판매량은 censored — β는 흡수의 하한 추정. "
                   "처치변수는 마감시각 기준 품절강도(시간)이다.")
_NOTE_PLACEBO = ("placebo(미래 d+7 품절강도) β가 real β와 비슷하면 그 β는 인과가 아니라 "
                 "confound다 — 두 arm을 함께 읽어야 한다.")


def results_to_frame(results: list[AbsorptionResult], *, arm: str) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(r) for r in results])
    frame["arm"] = arm
    return frame


def _gate_results(results: list[AbsorptionResult]) -> list[AbsorptionResult]:
    return [r for r in results if r.category_id in GATE_CATEGORIES]


def absorption_verdict(results: list[AbsorptionResult]) -> str:
    """게이트 카테고리(bread/pastry) 기준 판정. walk-away 1건이라도 있으면 기각."""
    gate = _gate_results(results)
    if not gate:
        return "불확실 — 게이트 카테고리(bread/pastry) 결과 없음"
    walkaways = [(r.store_id, r.category_id) for r in gate if r.verdict == "walkaway"]
    n_absorb = sum(r.verdict == "absorb" for r in gate)
    if walkaways:
        return (f"기각 — walk-away 발견: {walkaways} "
                f"(게이트 {len(gate)}건 중 absorb {n_absorb}건)")
    if n_absorb == len(gate):
        return f"지지 — 게이트 카테고리 {len(gate)}건 전부 absorb, walk-away 0건"
    n_inconclusive = len(gate) - n_absorb
    return (f"불확실 — 게이트 {len(gate)}건 중 absorb {n_absorb}건 / "
            f"inconclusive {n_inconclusive}건, walk-away 0건")


def _gate_summary(results: list[AbsorptionResult]) -> pd.DataFrame:
    gate = _gate_results(results)
    return pd.DataFrame([{
        "n_gate": len(gate),
        "n_absorb": sum(r.verdict == "absorb" for r in gate),
        "n_walkaway": sum(r.verdict == "walkaway" for r in gate),
        "n_inconclusive": sum(r.verdict == "inconclusive" for r in gate),
    }])


def _beta_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for arm, group in table.groupby("arm", observed=True):
        label = group["store_id"] + "/" + group["category_id"]
        fig.add_trace(go.Bar(
            x=label, y=group["beta"], name=str(arm),
            error_y=dict(type="data", symmetric=False,
                         array=group["ci_high"] - group["beta"],
                         arrayminus=group["beta"] - group["ci_low"]),
        ))
    fig.update_layout(title="흡수 계수 β (90% CI) — real vs placebo",
                      xaxis_title="매장/카테고리", yaxis_title="β (품절 1시간당 총량 변화)",
                      barmode="group")
    return fig


def _delta_fig(table: pd.DataFrame) -> go.Figure:
    real = table[table["arm"] == ARM_REAL]
    label = real["store_id"] + "/" + real["category_id"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=label, y=real["delta"], name="δ(등가 경계)"))
    fig.add_trace(go.Scatter(x=label, y=real["beta"].abs(), mode="markers", name="|β|"))
    fig.update_layout(title="TOST 등가 경계 δ 대비 |β|", xaxis_title="매장/카테고리",
                      yaxis_title="단위/품절시간")
    return fig


@register_hypothesis("demand_absorption", "카테고리 총량 수요이전 흡수 (W0 게이트)")
def demand_absorption(inputs: AnalysisInputs) -> AnalysisResult:
    daily = inputs.daily
    params = inputs.params_for("demand_absorption")
    real = run_absorption(daily, **params)
    placebo = placebo_absorption(daily, **params)
    table = pd.concat([results_to_frame(real, arm=ARM_REAL),
                       results_to_frame(placebo, arm=ARM_PLACEBO)], ignore_index=True)
    return AnalysisResult(
        name="demand_absorption", kind=KIND_HYPOTHESIS,
        title="카테고리 총량 수요이전 흡수 (W0 게이트)",
        tables=[("results", table), ("gate_summary", _gate_summary(real))],
        figures=[_beta_fig(table), _delta_fig(table)],
        verdict=absorption_verdict(real),
        notes=[_NOTE_CENSORING, _NOTE_PLACEBO],
    )
```

`handlers/__init__.py`: `HANDLER_MODULES: tuple[str, ...] = ("sales", "absorption")`

- [ ] **Step 8: 핸들러 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_absorption.py -v`
Expected: PASS (7 passed)

- [ ] **Step 9: 동일 vintage 대조 (출처 스크립트 vs 핸들러)**

Run:
```bash
PYTHONPATH=scripts uv run python scripts/absorption_4stores.py 2>&1 | tail -25
printf 'name: t7\ndata:\n  source: real\n  store: multistore\nhypotheses:\n  demand_absorption: true\n' > /tmp/t7.yaml
uv run bakery analysis-run /tmp/t7.yaml --out /tmp/analysis_t7
uv run python -c "
import pandas as pd
script = pd.read_csv('reports/demand_absorption/results_4stores.csv')
lab = pd.read_csv('/tmp/analysis_t7/t7/demand_absorption__results.csv')
key = ['store_id','category_id','arm']
merged = script.merge(lab, on=key, suffixes=('_script','_lab'))
merged['dbeta'] = (merged['beta_script'] - merged['beta_lab']).abs()
print(merged[key + ['beta_script','beta_lab','dbeta','verdict_script','verdict_lab']].to_string())
print('max |Δβ| =', merged['dbeta'].max())
print('판정 불일치 =', (merged['verdict_script'] != merged['verdict_lab']).sum())
"
```
Expected: 스크립트는 `data/internal/v2/` 레거시 경로 + 로컬 `apply_fixed_stockout`으로 daily를 만들고, 핸들러는 canonical `multistore_daily`(PR#39 `assign_stockout_fields` 반영)를 쓴다. 두 경로는 같은 재정의를 공유하므로 **판정(verdict) 불일치 0건**이 기대값이고, β는 소수점 차이가 날 수 있다.

판정이 하나라도 다르거나 `max |Δβ|`가 게이트 카테고리에서 0.01을 넘으면 **멈추고 원인을 규명한 뒤 진행한다**(canonical parquet을 진실로 삼되, 차이의 원인을 반드시 기록). 결과 수치를 `docs/phase6_analysis_layer.md`의 "이식 대조 기록" 표에 그대로 적는다.

- [ ] **Step 10: 커밋**

```bash
git add src/bakery/analysis/demand_absorption.py src/bakery/analysis/lab/handlers/absorption.py \
        src/bakery/analysis/lab/handlers/__init__.py scripts/absorption_4stores.py \
        tests/test_demand_absorption_placebo.py tests/analysis_lab/test_handlers_absorption.py
git commit -m "feat(analysis-lab): demand_absorption 핸들러 + placebo_absorption 프리미티브 승격"
```

---

### Task 8: 형태③ 증명 — `holiday_premium` (스크립트 추출, 데이터 온리)

**Files:**
- Create: `src/bakery/analysis/holiday_premium.py`
- Create: `src/bakery/analysis/lab/handlers/calendar_bias.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Modify: `scripts/holiday_premium_decompose.py` (프리미티브 위임 + print만)
- Test: `tests/test_holiday_premium.py`, `tests/analysis_lab/test_handlers_calendar_bias.py`

**Interfaces:**
- Consumes: `bakery.data.calendar.build_calendar_daily`, `AnalysisInputs.category_daily`/`.calendar` (Task 3), `register_hypothesis` (Task 4)
- Produces:
  - `local_dow_baseline(series, calendar, *, halfwin=HALFWIN) -> pd.Series` (index=date, value=동일요일 로컬 median)
  - 핸들러는 `needs_single_store=True` — `inputs.category_daily`가 `bonavi_daily`(광교 전용)를 읽으므로 multistore spec에서는 실행하지 않는다(광교 수치를 4매장으로 라벨링하는 오데이터 방지)
  - `decompose_holiday_premium(series, calendar, *, halfwin=HALFWIN) -> dict[str, pd.DataFrame]` — 키 `"full"`, `"by_holiday"`, `"dow_class"`, `"event_ranking"`, `"streak_buckets"`
  - `HALFWIN = 6`, `MIN_BASELINE_SAMPLES = 3`, `SERIES_VALUE_COLUMN = "adjusted_demand_unit"`
  - `holiday_premium(inputs) -> AnalysisResult`

**설계 노트:** 스크립트는 `reports/raw_adjusted_series.csv`(2026-07-16 생성, 동결)를 읽었다. 핸들러는 그 CSV 대신 `inputs.category_daily`(현 vintage `build_category_daily(alpha=0.8)`)에서 시리즈를 만든다 — 확인된 사실: 두 소스는 최대 28단위 차이가 난다(Phase 7 신규데이터 편입 전/후 vintage 차). 그래서 **회귀 게이트는 동결 CSV를 추출 함수에 먹여 이 플랜의 golden과 비교**하고, 운영 실행은 fresh 시리즈를 쓴다. 이 분리를 핸들러 note에 남긴다.

- [ ] **Step 1: frozen-input golden 테스트 작성**

`tests/test_holiday_premium.py`:

```python
"""holiday_premium 프리미티브 — 동결 입력 golden 대조.

golden은 2026-07-28에 `scripts/holiday_premium_decompose.py`를 실제 실행해 캡처한 값이다
(reports/raw_adjusted_series.csv = 2026-07-16 생성, 동결). docs 기록 수치를 쓰지 않는 이유는
Phase 7 신규데이터 편입으로 값이 이동했기 때문이다(측정 헌장/회귀 게이트 규칙).
"""
from pathlib import Path

import pandas as pd
import pytest

from bakery.analysis.holiday_premium import (
    HALFWIN, MIN_BASELINE_SAMPLES, decompose_holiday_premium, local_dow_baseline,
)
from bakery.data.calendar import build_calendar_daily

FROZEN_SERIES = Path("reports/raw_adjusted_series.csv")


@pytest.fixture(scope="module")
def frozen_tables():
    if not FROZEN_SERIES.exists():
        pytest.skip(f"{FROZEN_SERIES} 없음 — 동결 입력 대조 스킵")
    series = pd.read_csv(FROZEN_SERIES, parse_dates=["date"])[["date", "adjusted_demand_unit"]]
    calendar = build_calendar_daily(series["date"].min(), series["date"].max())
    return decompose_holiday_premium(series, calendar)


def test_constants():
    assert HALFWIN == 6
    assert MIN_BASELINE_SAMPLES == 3


def test_dow_class_golden(frozen_tables):
    """golden(2026-07-28 캡처): 평일 n=71 median 1.25 [1.10,1.38] / 주말 n=22 0.89 [0.78,1.00]."""
    dow_class = frozen_tables["dow_class"].set_index("dow_class")
    weekday = dow_class.loc["평일"]
    weekend = dow_class.loc["주말"]
    assert weekday["n"] == 71
    assert round(weekday["median_lift"], 2) == 1.25
    assert round(weekday["q25"], 2) == 1.10
    assert round(weekday["q75"], 2) == 1.38
    assert weekend["n"] == 22
    assert round(weekend["median_lift"], 2) == 0.89
    assert round(weekend["q25"], 2) == 0.78
    assert round(weekend["q75"], 2) == 1.00


def test_event_ranking_golden_top_entries(frozen_tables):
    """golden: Christmas Day 1.52(n=3) 1위, New Year's Day 1.42(n=3) 2위."""
    ranking = frozen_tables["event_ranking"]
    top = ranking.iloc[0]
    assert top["base_name"] == "Christmas Day"
    assert round(top["median_lift"], 2) == 1.52
    assert top["n_weekday"] == 3
    second = ranking.iloc[1]
    assert second["base_name"] == "New Year's Day"
    assert round(second["median_lift"], 2) == 1.42
    assert second["n_weekday"] == 3


def test_event_ranking_is_sorted_descending(frozen_tables):
    lifts = frozen_tables["event_ranking"]["median_lift"].tolist()
    assert lifts == sorted(lifts, reverse=True)


def test_by_holiday_row_count_matches_dow_class_total(frozen_tables):
    by_holiday = frozen_tables["by_holiday"]
    assert by_holiday["lift"].notna().sum() == 71 + 22


def test_streak_buckets_labels(frozen_tables):
    assert frozen_tables["streak_buckets"]["streak_bucket"].tolist() == [
        "1(고립)", "2", "3+(연휴)"]


def test_local_dow_baseline_uses_same_dow_only():
    """동일요일 ±HALFWIN 주 median. 공휴일은 baseline에서 제외된다."""
    dates = pd.date_range("2025-01-06", periods=28, freq="D")   # 월요일 시작
    series = pd.DataFrame({"date": dates, "adjusted_demand_unit": range(100, 128)})
    calendar = pd.DataFrame({"date": dates, "is_public_holiday": 0})
    baseline = local_dow_baseline(series, calendar)
    # 2025-01-13(월)의 동일요일 이웃 = 01-06(100), 01-20(114), 01-27(121) → median 114
    assert baseline[pd.Timestamp("2025-01-13")] == 114.0


def test_local_dow_baseline_nan_when_too_few_samples():
    dates = pd.date_range("2025-01-06", periods=8, freq="D")
    series = pd.DataFrame({"date": dates, "adjusted_demand_unit": range(8)})
    calendar = pd.DataFrame({"date": dates, "is_public_holiday": 0})
    baseline = local_dow_baseline(series, calendar)
    # 월요일 이웃이 1개(01-13)뿐 → MIN_BASELINE_SAMPLES=3 미달 → NaN
    assert pd.isna(baseline[pd.Timestamp("2025-01-06")])


def test_script_delegates_to_primitive():
    import sys
    sys.path.insert(0, "scripts")
    import holiday_premium_decompose

    assert holiday_premium_decompose.decompose is decompose_holiday_premium
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_holiday_premium.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.holiday_premium'`

- [ ] **Step 3: 프리미티브 구현**

`src/bakery/analysis/holiday_premium.py`:

```python
"""공휴일 프리미엄 분해 — 요일·주말·연휴·대체공휴일 축.

lift = actual / 로컬 동일요일 baseline(±HALFWIN주, 공휴일 제외) — 추세·요일 동시 통제.
출처: scripts/holiday_premium_decompose.py(2026-07-18). 스크립트는 print만 담당하고
계산은 이 모듈로 옮겼다(순수함수 = 회귀 대조 가능).

주의: 캘린더는 `bakery.data.calendar.build_calendar_daily`로 만든 것을 주입해야 한다.
`calendar_raw` parquet 직독은 2021-23 공휴일이 누락돼 프리미엄이 −18.5% 과소평가된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALFWIN = 6                      # 동일요일 baseline ±주
MIN_BASELINE_SAMPLES = 3         # 이보다 적으면 baseline NaN
SERIES_VALUE_COLUMN = "adjusted_demand_unit"
WEEKEND_START_DOW = 5            # 토=5
STREAK_BUCKET_EDGES = [0, 1, 2, 10]
STREAK_BUCKET_LABELS = ["1(고립)", "2", "3+(연휴)"]
_WEEKDAY = "평일"
_WEEKEND = "주말"


def local_dow_baseline(series: pd.DataFrame, calendar: pd.DataFrame, *,
                       halfwin: int = HALFWIN) -> pd.Series:
    """각 날짜의 '평상 동일요일' median(±halfwin주, 공휴일 제외, 자기 자신 제외)."""
    merged = series.merge(calendar, on="date", how="left")
    normal = merged[merged["is_public_holiday"] == 0].set_index("date")[SERIES_VALUE_COLUMN]
    index = normal.index
    out: dict[pd.Timestamp, float] = {}
    for date in merged["date"]:
        low, high = date - pd.Timedelta(weeks=halfwin), date + pd.Timedelta(weeks=halfwin)
        same = normal[(index >= low) & (index <= high)
                      & (index.dayofweek == date.dayofweek) & (index != date)]
        out[date] = float(same.median()) if len(same) >= MIN_BASELINE_SAMPLES else np.nan
    return pd.Series(out, name="dow_base")


def normalize_holiday_name(name: object) -> str:
    """대체공휴일을 원 명절로 통합, 영문 표기 정리."""
    if not isinstance(name, str):
        return ""
    return name.replace("Alternative holiday for ", "").replace(" (observed)", "")


def _build_full(series: pd.DataFrame, calendar: pd.DataFrame, halfwin: int) -> pd.DataFrame:
    full = series.merge(calendar, on="date", how="left")
    full["dow_base"] = full["date"].map(local_dow_baseline(series, calendar, halfwin=halfwin))
    full["lift"] = full[SERIES_VALUE_COLUMN] / full["dow_base"]
    full["dow"] = full["date"].dt.dayofweek
    full["is_weekend"] = full["dow"] >= WEEKEND_START_DOW
    full["base_name"] = full["holiday_name"].map(normalize_holiday_name)
    return full


def _by_holiday(holidays: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "base_name", "dow", "is_weekend", "lift",
               "off_streak_length", "off_position_in_streak", "is_substitute_holiday"]
    return holidays[columns].sort_values(["base_name", "date"]).reset_index(drop=True)


def _dow_class(holidays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, subset in ((_WEEKDAY, holidays[~holidays["is_weekend"]]),
                          (_WEEKEND, holidays[holidays["is_weekend"]])):
        lifts = subset["lift"].dropna()
        rows.append({"dow_class": label, "n": int(len(lifts)),
                     "median_lift": float(lifts.median()),
                     "q25": float(lifts.quantile(0.25)),
                     "q75": float(lifts.quantile(0.75))})
    return pd.DataFrame(rows)


def _event_ranking(holidays: pd.DataFrame) -> pd.DataFrame:
    weekday = holidays[~holidays["is_weekend"]]
    rows = []
    for name, group in weekday.groupby("base_name"):
        lifts = group["lift"].dropna()
        if len(lifts) == 0:
            continue
        rows.append({"base_name": name, "median_lift": float(lifts.median()),
                     "n_weekday": int(len(lifts))})
    return (pd.DataFrame(rows).sort_values("median_lift", ascending=False)
            .reset_index(drop=True))


def _streak_buckets(holidays: pd.DataFrame) -> pd.DataFrame:
    weekday = holidays[~holidays["is_weekend"]].copy()
    weekday["streak_bucket"] = pd.cut(weekday["off_streak_length"],
                                      STREAK_BUCKET_EDGES, labels=STREAK_BUCKET_LABELS)
    rows = []
    for bucket in STREAK_BUCKET_LABELS:
        lifts = weekday[weekday["streak_bucket"] == bucket]["lift"].dropna()
        rows.append({"streak_bucket": bucket, "n": int(len(lifts)),
                     "median_lift": float(lifts.median()) if len(lifts) else np.nan})
    return pd.DataFrame(rows)


def decompose_holiday_premium(series: pd.DataFrame, calendar: pd.DataFrame, *,
                              halfwin: int = HALFWIN) -> dict[str, pd.DataFrame]:
    """공휴일 프리미엄 4축 분해. series는 (date, adjusted_demand_unit) 일별 시리즈."""
    full = _build_full(series, calendar, halfwin)
    holidays = full[full["is_public_holiday"] == 1].copy()
    return {
        "full": full,
        "by_holiday": _by_holiday(holidays),
        "dow_class": _dow_class(holidays),
        "event_ranking": _event_ranking(holidays),
        "streak_buckets": _streak_buckets(holidays),
    }
```

`scripts/holiday_premium_decompose.py`를 얇은 wrapper로 교체 — `load()`/`local_dow_baseline`/`part_a`~`part_d`의 계산 부분을 삭제하고 프리미티브 결과를 print한다:

```python
"""전 공휴일 프리미엄 분해 — 요일·주말·연휴·대체공휴일 축 (광교).

계산은 `bakery.analysis.holiday_premium.decompose_holiday_premium`로 옮겼다(Phase 6).
이 스크립트는 동결 시리즈(reports/raw_adjusted_series.csv)를 읽어 표를 print하는 wrapper다.
현 vintage 실행은 `uv run bakery analysis-run experiments/analysis_gwangyo.yaml`을 쓴다.

실행: PYTHONPATH=scripts uv run python scripts/holiday_premium_decompose.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

import pandas as pd

from bakery.analysis.holiday_premium import decompose_holiday_premium as decompose
from bakery.data.calendar import build_calendar_daily

SERIES = Path("reports/raw_adjusted_series.csv")
OUT = Path("reports/holiday_premium_decompose.csv")


def run() -> None:
    series = pd.read_csv(SERIES, parse_dates=["date"])[["date", "adjusted_demand_unit"]]
    calendar = build_calendar_daily(series["date"].min(), series["date"].max())
    tables = decompose(series, calendar)
    print(f"=== 광교 공휴일 프리미엄 분해 (n_공휴일={len(tables['by_holiday'])}) ===")
    for label in ("dow_class", "event_ranking", "streak_buckets"):
        print(f"\n--- {label} ---")
        print(tables[label].to_string(index=False))
    tables["full"].to_csv(OUT, index=False)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: 프리미티브 테스트 통과 확인**

Run: `uv run pytest tests/test_holiday_premium.py -v`
Expected: PASS (9 passed) — golden 수치가 안 맞으면 **멈추고** 원인을 규명한다(가장 흔한 원인은 `build_calendar_daily` 대신 `calendar_raw` 직독).

- [ ] **Step 5: 핸들러 실패 테스트 작성**

`tests/analysis_lab/test_handlers_calendar_bias.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.calendar_bias import holiday_premium, premium_verdict
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _dow_class(weekday_median, weekend_median):
    return pd.DataFrame([
        {"dow_class": "평일", "n": 71, "median_lift": weekday_median, "q25": 1.10, "q75": 1.38},
        {"dow_class": "주말", "n": 22, "median_lift": weekend_median, "q25": 0.78, "q75": 1.00},
    ])


def test_verdict_supports_weekday_premium_without_weekend_premium():
    assert premium_verdict(_dow_class(1.25, 0.89)) == (
        "지지 — 평일 공휴일 프리미엄 +25.0%, 주말 공휴일은 −11.0%(프리미엄 없음)")


def test_verdict_reports_weekend_premium_when_present():
    assert premium_verdict(_dow_class(1.25, 1.12)) == (
        "부분 지지 — 평일 +25.0%, 주말도 +12.0%(주말 프리미엄 존재)")


def test_verdict_rejects_when_no_weekday_premium():
    assert premium_verdict(_dow_class(1.01, 0.99)) == (
        "기각 — 평일 공휴일 프리미엄 +1.0%로 미미(임계 5%)")


@pytest.mark.slow
def test_handler_uses_fresh_category_daily_not_frozen_csv():
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = holiday_premium(inputs)
    assert result.kind == KIND_HYPOTHESIS
    from bakery.analysis.lab.registry import HYPOTHESES, load_handlers
    load_handlers()
    assert HYPOTHESES["holiday_premium"].needs_single_store is True
    assert [label for label, _ in result.tables] == [
        "dow_class", "event_ranking", "streak_buckets", "by_holiday"]
    # vintage 분리를 은폐하지 않는다
    assert any("vintage" in note for note in result.notes)
```

- [ ] **Step 6: 핸들러 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_calendar_bias.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.handlers.calendar_bias'`

- [ ] **Step 7: 핸들러 구현**

`src/bakery/analysis/lab/handlers/calendar_bias.py`:

```python
"""가설 — 캘린더 축 편향(공휴일 프리미엄 / 월×요일 조정).

모델 예측을 참조하지 않는다(입력 데이터 + 캘린더만). 시리즈는 현 vintage
`build_category_daily(alpha)`에서 만들며, 출처 스크립트가 읽던 동결 CSV와는 다르다.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from bakery.analysis.holiday_premium import SERIES_VALUE_COLUMN, decompose_holiday_premium
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult

PREMIUM_THRESHOLD = 0.05      # 평일 프리미엄 판정 임계(5%)
_NOTE_VINTAGE = ("시리즈는 현 vintage build_category_daily(alpha)에서 생성 — "
                 "출처 스크립트가 읽던 reports/raw_adjusted_series.csv(2026-07-16 동결)와 "
                 "최대 28단위 차이가 있다. 동결 입력 대조는 tests/test_holiday_premium.py 참조.")
_NOTE_CALENDAR = ("캘린더는 build_calendar_daily 사용 — calendar_raw parquet 직독은 "
                  "2021-23 공휴일이 누락돼 프리미엄을 −18.5% 과소평가한다.")


def premium_verdict(dow_class: pd.DataFrame) -> str:
    """평일/주말 공휴일 median lift로 판정."""
    indexed = dow_class.set_index("dow_class")
    weekday = float(indexed.loc["평일", "median_lift"])
    weekend = float(indexed.loc["주말", "median_lift"])
    weekday_pct = (weekday - 1.0) * 100
    weekend_pct = (weekend - 1.0) * 100
    if weekday - 1.0 < PREMIUM_THRESHOLD:
        return f"기각 — 평일 공휴일 프리미엄 {weekday_pct:+.1f}%로 미미(임계 5%)"
    if weekend > 1.0:
        return (f"부분 지지 — 평일 {weekday_pct:+.1f}%, "
                f"주말도 {weekend_pct:+.1f}%(주말 프리미엄 존재)")
    return (f"지지 — 평일 공휴일 프리미엄 {weekday_pct:+.1f}%, "
            f"주말 공휴일은 {weekend_pct:+.1f}%(프리미엄 없음)")


def _series_from_category_daily(category_daily: pd.DataFrame) -> pd.DataFrame:
    return (category_daily.groupby("date", as_index=False)[SERIES_VALUE_COLUMN].sum()
            .sort_values("date").reset_index(drop=True))


def _ranking_fig(ranking: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=ranking["base_name"], y=ranking["median_lift"]))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="평일 공휴일 median lift 랭킹 (이벤트 고유성)",
                      xaxis_title="공휴일", yaxis_title="lift (평상 동일요일 대비)")
    return fig


def _dow_class_fig(dow_class: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=dow_class["dow_class"], y=dow_class["median_lift"],
        error_y=dict(type="data", symmetric=False,
                     array=dow_class["q75"] - dow_class["median_lift"],
                     arrayminus=dow_class["median_lift"] - dow_class["q25"]),
    ))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="평일 vs 주말 공휴일 프리미엄 (median, IQR)",
                      xaxis_title="구분", yaxis_title="lift")
    return fig


@register_hypothesis("holiday_premium", "공휴일 프리미엄 분해 (요일·연휴·대체 축)",
                     needs_single_store=True)
def holiday_premium(inputs: AnalysisInputs) -> AnalysisResult:
    series = _series_from_category_daily(inputs.category_daily)
    tables = decompose_holiday_premium(series, inputs.calendar,
                                       **inputs.params_for("holiday_premium"))
    return AnalysisResult(
        name="holiday_premium", kind=KIND_HYPOTHESIS,
        title="공휴일 프리미엄 분해 (요일·연휴·대체 축)",
        tables=[("dow_class", tables["dow_class"]),
                ("event_ranking", tables["event_ranking"]),
                ("streak_buckets", tables["streak_buckets"]),
                ("by_holiday", tables["by_holiday"])],
        figures=[_dow_class_fig(tables["dow_class"]), _ranking_fig(tables["event_ranking"])],
        verdict=premium_verdict(tables["dow_class"]),
        notes=[_NOTE_VINTAGE, _NOTE_CALENDAR],
    )
```

`handlers/__init__.py`: `HANDLER_MODULES: tuple[str, ...] = ("sales", "absorption", "calendar_bias")`

- [ ] **Step 8: 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_calendar_bias.py tests/test_holiday_premium.py -v`
Expected: PASS (13 passed)

- [ ] **Step 9: 스크립트 wrapper 동작 확인 + vintage 차이 기록**

Run:
```bash
PYTHONPATH=scripts uv run python scripts/holiday_premium_decompose.py 2>&1 | tail -20
printf 'name: t8\ndata:\n  source: real\nhypotheses:\n  holiday_premium: true\n' > /tmp/t8.yaml
uv run bakery analysis-run /tmp/t8.yaml --out /tmp/analysis_t8
uv run python -c "
import pandas as pd
print(pd.read_csv('/tmp/analysis_t8/t8/holiday_premium__dow_class.csv').to_string(index=False))
"
```
Expected: 스크립트(동결 CSV)는 평일 n=71 median 1.25, 핸들러(fresh vintage)는 수치가 약간 다를 수 있다. **판정 문구가 뒤집히면 멈추고 규명한다.** 두 수치를 `docs/phase6_analysis_layer.md`의 대조 표에 기록한다.

- [ ] **Step 10: 커밋**

```bash
git add src/bakery/analysis/holiday_premium.py src/bakery/analysis/lab/handlers/calendar_bias.py \
        src/bakery/analysis/lab/handlers/__init__.py scripts/holiday_premium_decompose.py \
        tests/test_holiday_premium.py tests/analysis_lab/test_handlers_calendar_bias.py
git commit -m "feat(analysis-lab): holiday_premium 이식 — 스크립트 추출 형태 증명(frozen-input golden)"
```

---

### Task 9: 형태④ 증명 — `weekday_bias` (preds artifact 소비)

**Files:**
- Create: `src/bakery/analysis/order_bias.py`
- Create: `src/bakery/analysis/lab/handlers/model_bias.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Modify: `scripts/weekday_bias_isowaste.py` (프리미티브 위임)
- Test: `tests/test_order_bias.py`, `tests/analysis_lab/test_handlers_model_bias.py`

**Interfaces:**
- Consumes: `AnalysisInputs.predictions` (Task 3), `register_hypothesis` (Task 4)
- Produces:
  - `waste_rate_of(order, actual) -> float`, `soldout_freq(order, actual) -> float`, `soldout_mag(order, actual) -> float`
  - `dow_trimmed_order(expected, base, trim, is_target_dow) -> np.ndarray`
  - `solve_base_for_waste(expected, actual, is_target_dow, *, trim, w_target) -> float`
  - `isowaste_dow_gap(preds, *, w_target, trim, target_dows=TARGET_DOWS) -> tuple[float, float]`
  - `bootstrap_gap_ci(preds, *, w_target, trim, n_boot=N_BOOT, seed=SEED, target_dows=TARGET_DOWS) -> dict[str, np.ndarray]`
  - `isowaste_grid(preds, *, w_targets=W_TARGETS, trims=E_TRIMS, n_boot=N_BOOT, seed=SEED) -> pd.DataFrame`
  - `W_TARGETS = (0.06, 0.08, 0.10)`, `E_TRIMS = (0.02, 0.03, 0.04)`, `N_BOOT = 2000`, `SEED = 42`, `TARGET_DOWS = (0, 2)`
  - `weekday_bias(inputs) -> AnalysisResult` (`needs_predictions=True`)

**게이트 근거:** 이 가설의 판정은 "동일 waste에서 요일 트림이 전역 균일 하향을 이기는가"다. `expected`가 어느 엔진에서 나왔는지에 의존하므로, **canonical harness preds로 돌린 결과는 출처 스크립트(비-canonical `store_predictive_power` 엔진 캐시)와 수치가 다르다.** 따라서 수치 게이트는 동결 캐시(`reports/track3_fresh_preds.parquet`)에만 걸고, canonical preds로는 방향/판정만 본다.

- [ ] **Step 1: frozen-input golden 테스트 작성**

`tests/test_order_bias.py`:

```python
"""order_bias 프리미티브 — 동결 preds 캐시 golden 대조.

golden은 2026-07-28에 `scripts/weekday_bias_isowaste.py`의 함수를
reports/track3_fresh_preds.parquet(2026-07-18 생성, 동결)에 직접 돌려 캡처했다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakery.analysis.order_bias import (
    E_TRIMS, N_BOOT, SEED, TARGET_DOWS, W_TARGETS,
    bootstrap_gap_ci, dow_trimmed_order, isowaste_dow_gap, isowaste_grid,
    solve_base_for_waste, soldout_freq, soldout_mag, waste_rate_of,
)

FROZEN_PREDS = Path("reports/track3_fresh_preds.parquet")


@pytest.fixture(scope="module")
def frozen_preds():
    if not FROZEN_PREDS.exists():
        pytest.skip(f"{FROZEN_PREDS} 없음 — 동결 입력 대조 스킵")
    preds = pd.read_parquet(FROZEN_PREDS)
    preds["date"] = pd.to_datetime(preds["date"])
    return preds


def test_constants():
    assert W_TARGETS == (0.06, 0.08, 0.10)
    assert E_TRIMS == (0.02, 0.03, 0.04)
    assert N_BOOT == 2000
    assert SEED == 42
    assert TARGET_DOWS == (0, 2)          # 월=0, 수=2 (연도별 부호 안정한 요일만)


def test_metric_formulas_exact():
    order = np.array([10.0, 10.0])
    actual = np.array([8.0, 12.0])
    # 초과 2 / Σactual 20 = 0.1
    assert waste_rate_of(order, actual) == 0.1
    # 부족일 1/2
    assert soldout_freq(order, actual) == 0.5
    # 부족량 2 / 20
    assert soldout_mag(order, actual) == 0.1


def test_metrics_zero_denominator_guard():
    zeros = np.array([0.0, 0.0])
    assert waste_rate_of(np.array([1.0, 1.0]), zeros) == 0.0
    assert soldout_mag(np.array([1.0, 1.0]), zeros) == 0.0


def test_dow_trimmed_order_trims_only_target_dows():
    expected = np.array([100.0, 100.0])
    is_target = np.array([True, False])
    order = dow_trimmed_order(expected, base=0.10, trim=0.04, is_target_dow=is_target)
    assert order.tolist() == [106.0, 110.0]     # 1+0.10−0.04 / 1+0.10


def test_solve_base_hits_waste_target():
    preds = pd.DataFrame({"expected": [100.0] * 10, "actual": [100.0] * 10})
    is_target = np.zeros(10, dtype=bool)
    base = solve_base_for_waste(preds["expected"].to_numpy(), preds["actual"].to_numpy(),
                                is_target, trim=0.0, w_target=0.05)
    order = dow_trimmed_order(preds["expected"].to_numpy(), base, 0.0, is_target)
    assert waste_rate_of(order, preds["actual"].to_numpy()) == pytest.approx(0.05, abs=1e-6)


def test_frozen_preds_shape_golden(frozen_preds):
    """golden: n=1090일, 월·수 비중 0.285321, base(expected) waste=0.047616."""
    assert len(frozen_preds) == 1090
    is_monwed = frozen_preds["date"].dt.dayofweek.isin(TARGET_DOWS)
    assert round(float(is_monwed.mean()), 6) == 0.285321
    base_waste = waste_rate_of(frozen_preds["expected"].to_numpy(),
                               frozen_preds["actual"].to_numpy())
    assert round(base_waste, 6) == 0.047616


@pytest.mark.parametrize("trim,expected_freq,expected_mag", [
    (0.02, -0.0009174311926605228, -0.0001567963417563184),
    (0.03, 0.00458715596330278, -4.3062759390345706e-05),
    (0.04, 0.007339449541284404, 0.0002116974752705003),
])
def test_isowaste_gap_golden_at_w006(frozen_preds, trim, expected_freq, expected_mag):
    """golden(2026-07-28, w_target=0.06). 음수=DOW 트림 우위."""
    gap_freq, gap_mag = isowaste_dow_gap(frozen_preds, w_target=0.06, trim=trim)
    assert gap_freq == pytest.approx(expected_freq, rel=1e-9)
    assert gap_mag == pytest.approx(expected_mag, rel=1e-9)


def test_bootstrap_ci_is_deterministic_for_fixed_seed(frozen_preds):
    first = bootstrap_gap_ci(frozen_preds, w_target=0.06, trim=0.03, n_boot=50, seed=SEED)
    second = bootstrap_gap_ci(frozen_preds, w_target=0.06, trim=0.03, n_boot=50, seed=SEED)
    assert first["freq"].tolist() == second["freq"].tolist()
    assert first["freq"].shape == (3,)          # [2.5, 50, 97.5] 백분위


def test_isowaste_grid_covers_full_cross_product(frozen_preds):
    grid = isowaste_grid(frozen_preds, n_boot=20)
    assert len(grid) == len(W_TARGETS) * len(E_TRIMS)
    assert grid.columns.tolist() == [
        "w_target", "trim", "gap_freq", "gap_mag",
        "freq_ci_low", "freq_median", "freq_ci_high", "winner"]


def test_script_delegates_to_primitive():
    import sys
    sys.path.insert(0, "scripts")
    import weekday_bias_isowaste

    assert weekday_bias_isowaste.isowaste_grid is isowaste_grid
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_order_bias.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.order_bias'`

- [ ] **Step 3: 프리미티브 구현**

`src/bakery/analysis/order_bias.py`:

```python
"""발주 편향 진단 — iso-waste에서 요일 트림이 전역 균일 하향을 이기는가.

출처: scripts/weekday_bias_isowaste.py(2026-07-18). 모델을 재학습하지 않고 이미
계산된 OOS 예측(expected/actual)만 재사용한다 — 발주 정책 A/B 껍질이다.

공정 비교 설계: 두 정책 모두 발주 = expected×배수이고, 같은 waste 수준에 도달하도록
base를 이분탐색으로 맞춘 뒤 매진(빈도/크기)만 비교한다.
  GLOBAL : order = expected × (1 + base)
  DOW    : order = expected × (1 + base − trim·1[대상요일])
gap = DOW − GLOBAL. 음수 = DOW 우위(같은 폐기에서 매진이 적다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W_TARGETS: tuple[float, ...] = (0.06, 0.08, 0.10)   # waste 타겟
E_TRIMS: tuple[float, ...] = (0.02, 0.03, 0.04)     # 대상요일 트림 grid
TARGET_DOWS: tuple[int, ...] = (0, 2)               # 월=0, 수=2 (부호 안정 요일만)
N_BOOT = 2000
SEED = 42
_BISECT_ITERATIONS = 45
_BISECT_LOW, _BISECT_HIGH = -0.5, 6.0
_CI_PERCENTILES = (2.5, 50, 97.5)
WINNER_DOW = "DOW 우위"
WINNER_GLOBAL = "GLOBAL 우위"
WINNER_TIE = "0포함(무차)"


def waste_rate_of(order: np.ndarray, actual: np.ndarray) -> float:
    """Σmax(order−actual,0) / Σactual."""
    denom = actual.sum()
    return float(np.maximum(order - actual, 0).sum() / denom) if denom else 0.0


def soldout_freq(order: np.ndarray, actual: np.ndarray) -> float:
    """발주 부족일 비율."""
    return float((actual > order).mean())


def soldout_mag(order: np.ndarray, actual: np.ndarray) -> float:
    """Σmax(actual−order,0) / Σactual."""
    denom = actual.sum()
    return float(np.maximum(actual - order, 0).sum() / denom) if denom else 0.0


def dow_trimmed_order(expected: np.ndarray, base: float, trim: float,
                      is_target_dow: np.ndarray) -> np.ndarray:
    """order = expected × (1 + base − trim·1[대상요일]). trim>0 = 대상요일 삭감."""
    return expected * (1.0 + base - trim * is_target_dow)


def solve_base_for_waste(expected: np.ndarray, actual: np.ndarray,
                         is_target_dow: np.ndarray, *, trim: float,
                         w_target: float) -> float:
    """주어진 trim에서 waste가 w_target이 되는 base를 이분탐색으로 찾는다."""
    def _waste_at(base: float) -> float:
        return waste_rate_of(dow_trimmed_order(expected, base, trim, is_target_dow), actual)

    low, high = _BISECT_LOW, _BISECT_HIGH
    if _waste_at(high) < w_target:
        return high
    for _ in range(_BISECT_ITERATIONS):
        mid = (low + high) / 2
        if _waste_at(mid) < w_target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _arrays(preds: pd.DataFrame, target_dows: tuple[int, ...]):
    expected = preds["expected"].to_numpy()
    actual = preds["actual"].to_numpy()
    is_target = pd.to_datetime(preds["date"]).dt.dayofweek.isin(target_dows).to_numpy()
    return expected, actual, is_target


def isowaste_dow_gap(preds: pd.DataFrame, *, w_target: float, trim: float,
                     target_dows: tuple[int, ...] = TARGET_DOWS) -> tuple[float, float]:
    """iso-waste에서 (DOW − GLOBAL) 매진 빈도·크기 gap. 음수=DOW 우위."""
    expected, actual, is_target = _arrays(preds, target_dows)
    base_global = solve_base_for_waste(expected, actual, is_target, trim=0.0, w_target=w_target)
    base_dow = solve_base_for_waste(expected, actual, is_target, trim=trim, w_target=w_target)
    order_global = dow_trimmed_order(expected, base_global, 0.0, is_target)
    order_dow = dow_trimmed_order(expected, base_dow, trim, is_target)
    return (soldout_freq(order_dow, actual) - soldout_freq(order_global, actual),
            soldout_mag(order_dow, actual) - soldout_mag(order_global, actual))


def bootstrap_gap_ci(preds: pd.DataFrame, *, w_target: float, trim: float,
                     n_boot: int = N_BOOT, seed: int = SEED,
                     target_dows: tuple[int, ...] = TARGET_DOWS) -> dict[str, np.ndarray]:
    """주(week) 블록 부트스트랩 — 요일 구조를 깨지 않으려 주 단위로 리샘플한다."""
    frame = preds.copy()
    dates = pd.to_datetime(frame["date"])
    frame["week"] = dates.dt.isocalendar().week.astype(int) + dates.dt.year * 100
    weeks = frame["week"].unique()
    groups = {week: frame[frame["week"] == week] for week in weeks}
    rng = np.random.default_rng(seed)
    freq_gaps, mag_gaps = np.empty(n_boot), np.empty(n_boot)
    for index in range(n_boot):
        picked = rng.choice(weeks, len(weeks), replace=True)
        resampled = pd.concat([groups[w] for w in picked], ignore_index=True)
        freq_gaps[index], mag_gaps[index] = isowaste_dow_gap(
            resampled, w_target=w_target, trim=trim, target_dows=target_dows)
    return {"freq": np.percentile(freq_gaps, _CI_PERCENTILES),
            "mag": np.percentile(mag_gaps, _CI_PERCENTILES)}


def _winner(ci_low: float, ci_high: float) -> str:
    if ci_high < 0:
        return WINNER_DOW
    if ci_low > 0:
        return WINNER_GLOBAL
    return WINNER_TIE


def isowaste_grid(preds: pd.DataFrame, *, w_targets: tuple[float, ...] = W_TARGETS,
                  trims: tuple[float, ...] = E_TRIMS, n_boot: int = N_BOOT,
                  seed: int = SEED) -> pd.DataFrame:
    """(waste 타겟 × 트림) 격자에서 gap + 부트스트랩 CI + 승자 판정."""
    rows = []
    for w_target in w_targets:
        for trim in trims:
            gap_freq, gap_mag = isowaste_dow_gap(preds, w_target=w_target, trim=trim)
            ci = bootstrap_gap_ci(preds, w_target=w_target, trim=trim,
                                  n_boot=n_boot, seed=seed)
            low, median, high = ci["freq"]
            rows.append({"w_target": w_target, "trim": trim,
                         "gap_freq": gap_freq, "gap_mag": gap_mag,
                         "freq_ci_low": low, "freq_median": median, "freq_ci_high": high,
                         "winner": _winner(low, high)})
    return pd.DataFrame(rows)
```

`scripts/weekday_bias_isowaste.py`를 wrapper로 교체 — `_load()`만 남기고 계산은 프리미티브 위임:

```python
"""실제 레버 검증 — 평일(월·수) 과대예측 트림이 전역 균일 하향을 iso-waste에서 이기는가.

계산은 `bakery.analysis.order_bias`로 옮겼다(Phase 6). 이 스크립트는 동결 캐시
reports/track3_fresh_preds.parquet를 읽어 격자 결과를 print하는 wrapper다.
현 vintage/canonical preds 실행은 `bakery analysis-run`(hypotheses.weekday_bias)을 쓴다.

실행: PYTHONPATH=scripts uv run python scripts/weekday_bias_isowaste.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bakery.analysis.order_bias import TARGET_DOWS, isowaste_grid, waste_rate_of

CACHE = Path("reports/track3_fresh_preds.parquet")


def _load() -> pd.DataFrame:
    preds = pd.read_parquet(CACHE)
    preds["date"] = pd.to_datetime(preds["date"])
    return preds


def main() -> None:
    preds = _load()
    is_target = preds["date"].dt.dayofweek.isin(TARGET_DOWS)
    base_waste = waste_rate_of(preds["expected"].to_numpy(), preds["actual"].to_numpy())
    print(f"[광교 3년 OOS] {len(preds)}일 · 월·수 {is_target.mean()*100:.1f}%  "
          f"base(expected) waste={base_waste*100:.1f}%")
    print("판정: 동일 waste에서 DOW(월·수 트림)−GLOBAL 매진 gap; 음수+CI0배제=DOW 우위")
    print(isowaste_grid(preds).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 프리미티브 테스트 통과 확인**

Run: `uv run pytest tests/test_order_bias.py -v`
Expected: PASS (13 passed — parametrize 3개 포함)

- [ ] **Step 5: 핸들러 실패 테스트 작성**

`tests/analysis_lab/test_handlers_model_bias.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.model_bias import weekday_bias, weekday_bias_verdict
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import HYPOTHESES, load_handlers
from bakery.analysis.lab.result import KIND_HYPOTHESIS
from bakery.analysis.lab.spec import AnalysisSpec


def _grid(winners):
    return pd.DataFrame([{"w_target": 0.06, "trim": 0.02 + 0.01 * i,
                          "gap_freq": -0.001, "gap_mag": -0.0001,
                          "freq_ci_low": -0.01, "freq_median": -0.001,
                          "freq_ci_high": -0.0005 if w == "DOW 우위" else 0.01,
                          "winner": w} for i, w in enumerate(winners)])


def test_registered_as_needs_predictions():
    load_handlers()
    assert HYPOTHESES["weekday_bias"].needs_predictions is True


def test_verdict_supports_when_any_cell_favors_dow():
    verdict = weekday_bias_verdict(_grid(["DOW 우위", "0포함(무차)", "0포함(무차)"]))
    assert verdict == "지지 — 9칸 중 1칸에서 DOW 트림 우위(CI 0 배제). center 보정 가치 있음"


def test_verdict_rejects_when_all_cells_tie():
    verdict = weekday_bias_verdict(_grid(["0포함(무차)", "0포함(무차)", "0포함(무차)"]))
    assert verdict == "기각 — 전 격자에서 CI가 0을 포함. center 보정은 전역 균일 하향을 못 이김"


def test_verdict_reports_global_advantage():
    grid = _grid(["0포함(무차)", "0포함(무차)", "0포함(무차)"])
    grid.loc[0, "winner"] = "GLOBAL 우위"
    verdict = weekday_bias_verdict(grid)
    assert verdict == "기각 — 1칸에서 GLOBAL(전역 균일) 우위, DOW 우위 0칸"


@pytest.mark.slow
def test_handler_consumes_predictions_artifact(tmp_path):
    """canonical harness preds로 실행 — 수치는 동결 캐시와 다르며 방향만 본다."""
    preds_path = "reports/gwangyo_default/category_total/predictions.csv"
    spec = AnalysisSpec(name="t", data={"source": "real"}, predictions=preds_path,
                        params={"weekday_bias": {"n_boot": 50}})
    result = weekday_bias(AnalysisInputs.from_spec(spec))
    assert result.kind == KIND_HYPOTHESIS
    grid = dict(result.tables)["isowaste_grid"]
    assert len(grid) == 9
    assert set(grid["winner"]) <= {"DOW 우위", "GLOBAL 우위", "0포함(무차)"}
    assert any("엔진" in note for note in result.notes)
```

- [ ] **Step 6: 핸들러 테스트 실패 확인**

Run: `uv run pytest tests/analysis_lab/test_handlers_model_bias.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.lab.handlers.model_bias'`

- [ ] **Step 7: 핸들러 구현**

`src/bakery/analysis/lab/handlers/model_bias.py`:

```python
"""가설 — 모델 예측 편향 진단(preds artifact 의존).

★경계: 이 레이어는 모델을 실행하지 않는다. harness-run이 남긴 predictions.csv를
읽기만 하며, spec.predictions가 없으면 runner가 preds_required로 스킵한다.

수치 게이트 주의: 출처 스크립트들은 비-canonical 엔진(store_predictive_power)의
캐시 preds를 썼다. canonical harness preds(category_total + event_prior)로 돌리면
수치가 다르므로 방향/판정만 비교 가능하다 — 동결 입력 대조는 tests/test_order_bias.py.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.order_bias import (
    TARGET_DOWS, WINNER_DOW, WINNER_GLOBAL, isowaste_grid, waste_rate_of,
)

_NOTE_ENGINE = ("preds는 harness-run(canonical category_total+event_prior) 산출물 — "
                "출처 스크립트가 쓴 비-canonical 엔진 캐시와 수치가 다르다. "
                "판정/방향만 비교하라.")
_NOTE_NO_REFIT = "모델 재학습 없음 — 이미 계산된 expected/actual에 발주 정책 껍질만 씌운 A/B다."


def weekday_bias_verdict(grid: pd.DataFrame) -> str:
    """격자에서 DOW 트림이 CI 0 배제로 이긴 칸이 하나라도 있으면 지지."""
    n_dow = int((grid["winner"] == WINNER_DOW).sum())
    n_global = int((grid["winner"] == WINNER_GLOBAL).sum())
    if n_dow > 0:
        return (f"지지 — {len(grid)}칸 중 {n_dow}칸에서 DOW 트림 우위(CI 0 배제). "
                "center 보정 가치 있음")
    if n_global > 0:
        return f"기각 — {n_global}칸에서 GLOBAL(전역 균일) 우위, DOW 우위 0칸"
    return "기각 — 전 격자에서 CI가 0을 포함. center 보정은 전역 균일 하향을 못 이김"


def _dow_bias_table(preds: pd.DataFrame) -> pd.DataFrame:
    """요일별 상대편향 — 진단 근거(음수=과대예측)."""
    frame = preds.copy()
    frame["dow"] = pd.to_datetime(frame["date"]).dt.dayofweek
    frame["rel_error"] = (frame["actual"] - frame["expected"]) / frame["actual"]
    return (frame.groupby("dow")
            .agg(n=("rel_error", "size"), rel_mean=("rel_error", "mean"),
                 rel_median=("rel_error", "median"))
            .reset_index())


def _gap_fig(grid: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for w_target, group in grid.groupby("w_target"):
        fig.add_trace(go.Scatter(
            x=group["trim"], y=group["gap_freq"], mode="lines+markers",
            name=f"waste {w_target:.0%}",
            error_y=dict(type="data", symmetric=False,
                         array=group["freq_ci_high"] - group["gap_freq"],
                         arrayminus=group["gap_freq"] - group["freq_ci_low"]),
        ))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="iso-waste 매진빈도 gap (DOW − GLOBAL), 음수=DOW 우위",
                      xaxis_title="대상요일 트림", yaxis_title="gap (빈도 차)")
    return fig


def _dow_bias_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=table["dow"], y=table["rel_mean"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="요일별 상대오차 평균 (음수=과대예측)",
                      xaxis_title="요일(월=0)", yaxis_title="(actual−expected)/actual")
    return fig


@register_hypothesis("weekday_bias", "평일(월·수) 과대예측 트림의 iso-waste 가치",
                     needs_predictions=True)
def weekday_bias(inputs: AnalysisInputs) -> AnalysisResult:
    preds = inputs.predictions
    grid = isowaste_grid(preds, **inputs.params_for("weekday_bias"))
    base_waste = waste_rate_of(preds["expected"].to_numpy(), preds["actual"].to_numpy())
    return AnalysisResult(
        name="weekday_bias", kind=KIND_HYPOTHESIS,
        title="평일(월·수) 과대예측 트림의 iso-waste 가치",
        tables=[("isowaste_grid", grid), ("dow_bias", _dow_bias_table(preds))],
        figures=[_gap_fig(grid), _dow_bias_fig(_dow_bias_table(preds))],
        verdict=weekday_bias_verdict(grid),
        notes=[_NOTE_ENGINE, _NOTE_NO_REFIT,
               f"base(expected) waste={base_waste:.3f}, 대상요일={TARGET_DOWS}(월·수)"],
    )
```

`handlers/__init__.py`: `HANDLER_MODULES: tuple[str, ...] = ("sales", "absorption", "calendar_bias", "model_bias")`

- [ ] **Step 8: 테스트 통과 확인**

Run: `uv run pytest tests/analysis_lab/ tests/test_order_bias.py -v`
Expected: PASS

- [ ] **Step 9: preds 게이트 실동작 확인 (있을 때/없을 때)**

Run:
```bash
# preds 없이 → 스킵 표기
printf 'name: t9a\ndata:\n  source: real\nhypotheses:\n  weekday_bias: true\n' > /tmp/t9a.yaml
uv run bakery analysis-run /tmp/t9a.yaml --out /tmp/analysis_t9
grep -c "preds 필요 — 미실행" /tmp/analysis_t9/t9a/analysis_report.html

# preds 있으면 → 실행
cat > /tmp/t9b.yaml <<'YAML'
name: t9b
data:
  source: real
predictions: reports/gwangyo_default/category_total/predictions.csv
params:
  weekday_bias:
    n_boot: 200
hypotheses:
  weekday_bias: true
YAML
uv run bakery analysis-run /tmp/t9b.yaml --out /tmp/analysis_t9
uv run python -c "
import pandas as pd
print(pd.read_csv('/tmp/analysis_t9/t9b/weekday_bias__isowaste_grid.csv').to_string(index=False))
"
```
Expected: 첫 실행은 `preds 필요 — 미실행`이 1회 이상, 둘째 실행은 9행 격자가 나온다. 판정이 동결 캐시 결과(트림 2%만 미세 음수, 3~4%는 양수 → 전반적으로 무차/GLOBAL)와 방향이 크게 뒤집히면 note에 기록하고 `docs/phase6_analysis_layer.md`에 남긴다.

- [ ] **Step 10: 커밋**

```bash
git add src/bakery/analysis/order_bias.py src/bakery/analysis/lab/handlers/model_bias.py \
        src/bakery/analysis/lab/handlers/__init__.py scripts/weekday_bias_isowaste.py \
        tests/test_order_bias.py tests/analysis_lab/test_handlers_model_bias.py
git commit -m "feat(analysis-lab): weekday_bias 이식 — preds artifact 소비 형태 증명"
```

- [ ] **Step 11: 첫 PR 경계 — 전체 스위트 1회**

Run: `uv run pytest --color=no 2>&1 | tail -5`
Expected: 기존 660+ passed에 신규 테스트가 더해져 전부 통과. 실패가 있으면 **여기서 멈추고** 원인을 고친다(focused 게이트는 크로스파일 import 회귀를 놓친다).

Run: `uv run pytest --collect-only -q 2>&1 | tail -3` — collection 에러 0 확인.

```bash
git log --oneline spec/phase6-analysis-layer..HEAD   # 커밋 9개 확인
```

여기까지가 **PR 1: 프레임워크 + 4형태 증명**. 리뷰/머지 후 Task 10부터 이어간다.

---

## PR 2 — 나머지 항목 기계적 이식 (Task 10~18)

각 태스크는 Task 6~9에서 증명된 형태 중 하나를 그대로 따른다. 공통 절차(매 태스크 반복):
테스트 작성 → 실패 확인 → 핸들러 구현 → 통과 확인 → `HANDLER_MODULES` 갱신 → 스모크 → 커밋.

---

### Task 10: `sales_distribution` (형태①)

**Files:**
- Modify: `src/bakery/analysis/lab/handlers/sales.py`
- Test: `tests/analysis_lab/test_handlers_sales.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: `AnalysisInputs.daily`, `median_unit_price`/`_with_revenue` (Task 6)
- Produces: `daily_totals(daily, prices) -> pd.DataFrame` (cols: store_id, date, sold_units, revenue, n_items_active), `distribution_summary(totals) -> pd.DataFrame` (cols: store_id, n_days, mean, std, p10, median, p90, cv), `sales_distribution(inputs) -> AnalysisResult`

- [ ] **Step 1: 테스트 작성**

`tests/analysis_lab/test_handlers_sales.py`에 추가:

```python
def test_daily_totals_aggregates_per_store_day():
    from bakery.analysis.lab.handlers.sales import daily_totals

    totals = daily_totals(_daily(), median_unit_price(_waste()))
    assert len(totals) == 3                                  # 3일
    first = totals[totals["date"] == pd.Timestamp("2025-01-01")].iloc[0]
    assert first["sold_units"] == 15                         # bread 10 + pastry 5
    assert first["revenue"] == 55000.0                       # 10×3000 + 5×5000
    assert first["n_items_active"] == 2


def test_distribution_summary_exact():
    from bakery.analysis.lab.handlers.sales import daily_totals, distribution_summary

    totals = daily_totals(_daily(), median_unit_price(_waste()))
    # 일별 수량 = [15, 35, 40] → mean 30, median 35
    summary = distribution_summary(totals).iloc[0]
    assert summary["n_days"] == 3
    assert summary["mean"] == pytest.approx(30.0)
    assert summary["median"] == pytest.approx(35.0)
    assert summary["std"] == pytest.approx(13.228756555322953)   # ddof=1
    assert summary["cv"] == pytest.approx(13.228756555322953 / 30.0)


def test_sales_distribution_handler_shape(stub_inputs):
    from bakery.analysis.lab.handlers.sales import sales_distribution

    result = sales_distribution(stub_inputs(daily=_daily(), waste=_waste()))
    assert result.name == "sales_distribution"
    assert result.verdict is None
    assert [label for label, _ in result.tables] == ["daily_totals", "summary"]
    assert len(result.figures) == 2
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_sales.py -v` / Expected: FAIL `ImportError: cannot import name 'daily_totals'`

- [ ] **Step 3: 구현** — `handlers/sales.py`에 추가:

```python
_PERCENTILES = (0.10, 0.50, 0.90)


def daily_totals(daily: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """매장×일 총 수량/매출 + 활성 품목 수."""
    priced = _with_revenue(daily, prices)
    return (priced.groupby(["store_id", "date"], observed=True)
            .agg(sold_units=("sold_units", "sum"), revenue=("revenue", "sum"),
                 n_items_active=("item_id", "nunique"))
            .reset_index())


def distribution_summary(totals: pd.DataFrame) -> pd.DataFrame:
    """매장별 일 매출 분포 요약 — cv(변동계수)가 예측 난이도의 1차 지표."""
    rows = []
    for store, group in totals.groupby("store_id", observed=True):
        units = group["sold_units"]
        p10, median, p90 = (float(units.quantile(q)) for q in _PERCENTILES)
        mean, std = float(units.mean()), float(units.std())
        rows.append({"store_id": store, "n_days": int(len(units)), "mean": mean,
                     "std": std, "p10": p10, "median": median, "p90": p90,
                     "cv": std / mean if mean else 0.0})
    return pd.DataFrame(rows)


def _totals_fig(totals: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for store, group in totals.groupby("store_id", observed=True):
        fig.add_trace(go.Scatter(x=group["date"], y=group["sold_units"],
                                 mode="lines", name=str(store)))
    fig.update_layout(title="매장별 일 판매량 추이", xaxis_title="날짜", yaxis_title="수량")
    return fig


def _histogram_fig(totals: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for store, group in totals.groupby("store_id", observed=True):
        fig.add_trace(go.Histogram(x=group["sold_units"], name=str(store), opacity=0.6))
    fig.update_layout(title="일 판매량 분포", barmode="overlay",
                      xaxis_title="수량", yaxis_title="일수")
    return fig


@register_data("sales_distribution", "매장별 일 판매량/매출 분포")
def sales_distribution(inputs: AnalysisInputs) -> AnalysisResult:
    prices = median_unit_price(inputs.waste)
    totals = daily_totals(inputs.daily, prices)
    return AnalysisResult(
        name="sales_distribution", kind=KIND_DATA, title="매장별 일 판매량/매출 분포",
        tables=[("daily_totals", totals), ("summary", distribution_summary(totals))],
        figures=[_totals_fig(totals), _histogram_fig(totals)],
        notes=_coverage_notes(inputs.daily, prices),
    )
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_sales.py -v` / Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/lab/handlers/sales.py tests/analysis_lab/test_handlers_sales.py
git commit -m "feat(analysis-lab): sales_distribution 핸들러"
```

---

### Task 11: 폐기 3종 — `waste_rate` / `waste_alpha_identity` / `overproduction_breakdown` (형태①)

**Files:**
- Create: `src/bakery/analysis/lab/handlers/waste.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Test: `tests/analysis_lab/test_handlers_waste.py`

**Interfaces:**
- Consumes: `AnalysisInputs.waste` (production_qty/waste_qty/normal_qty/closing_qty/unit_price/waste_cost/identity_diff), `AnalysisInputs.daily`, `AnalysisInputs.item_to_category`
- Produces:
  - `waste_rate_by_store(waste) -> pd.DataFrame` (store_id 대신 `cd`/`store` 유지, cols: cd, store, production_qty, waste_qty, waste_rate, waste_cost)
  - `waste_rate_by_item(waste, item_to_category, *, min_production=30) -> pd.DataFrame`
  - `identity_residual(waste) -> pd.DataFrame` (cols: cd, store, n_rows, n_nonzero, max_abs_diff, mean_abs_diff, zero_frac)
  - `overproduction_by_category(waste, item_to_category) -> pd.DataFrame` (cols: cd, category_id, production_qty, waste_qty, waste_rate, waste_cost, cost_share)
  - 핸들러 3개
  - `MIN_PRODUCTION_FOR_ITEM_RATE = 30`

**항등식 정의:** `waste_alpha_4stores`의 `identity_diff`는 `production − sold_total − waste`의 잔차다(빌드 시 계산됨). 이 핸들러는 잔차를 **재계산하지 않고 검증**한다: `production_qty − (normal_qty + closing_qty) − waste_qty`가 `identity_diff`와 일치하는지 확인하고, 0이 아닌 행의 비중·크기를 표로 낸다. 재계산 공식이 컬럼과 어긋나면 그 자체가 발견이므로 note에 남긴다.

**★ carry-in 음수 폐기 (2026-07-28 실측 확인, Task 3에서 발견):** `waste_qty`(=`out`)는 8,108/280,779행(2.89%, min −282, 4매장 전부)에서 **음수**다. 손상이 아니라 **전일 재고 이월** 신호다 — 판매가 당일 생산을 초과한 경우이며, 그 행에서도 `identity_diff == 0`이다(`out`은 정의상 `made − sold`). 따라서:
- **clip 금지.** 광교 폐기율이 0.12532 → 0.12933로 +3.2% 상대 부풀려진다(1차 KPI 왜곡).
- 항등식 검증은 음수 영향 없음(정의상 성립) — `identity_residual`은 그대로 둔다.
- 폐기율은 순합(net) 기준이며, 이를 **note에 명시**해 소비자가 gross로 오독하지 않게 한다.
- 매장별 carry-in 행수/합계를 `by_store` 표에 컬럼으로 실어 은폐하지 않는다.

- [ ] **Step 1: 테스트 작성**

`tests/analysis_lab/test_handlers_waste.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.waste import (
    MIN_PRODUCTION_FOR_ITEM_RATE, identity_residual, overproduction_by_category,
    overproduction_breakdown, waste_alpha_identity, waste_rate, waste_rate_by_item,
    waste_rate_by_store,
)
from bakery.analysis.lab.result import KIND_DATA


def _waste():
    """광교 2품목 2일. b1: 생산 100 폐기 10, p1: 생산 50 폐기 20."""
    return pd.DataFrame({
        "cd": ["1000000047"] * 4,
        "store": ["광교"] * 4,
        "item_id": ["b1", "b1", "p1", "p1"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
        "production_qty": [60, 40, 25, 25],
        "waste_qty": [6, 4, 10, 10],
        "normal_qty": [50.0, 34.0, 15.0, 13.0],
        "closing_qty": [4.0, 2.0, 0.0, 2.0],
        "unit_price": [3000, 3000, 5000, 5000],
        "waste_cost": [18000, 12000, 50000, 50000],
        "identity_diff": [0.0, 0.0, 0.0, 0.0],
        "sold_total": [54.0, 36.0, 15.0, 15.0],
    })


def _item_to_category():
    return pd.Series({"b1": "bread", "p1": "pastry"})


def test_min_production_constant():
    assert MIN_PRODUCTION_FOR_ITEM_RATE == 30


def test_waste_rate_by_store_exact():
    row = waste_rate_by_store(_waste()).iloc[0]
    assert row["production_qty"] == 150          # 60+40+25+25
    assert row["waste_qty"] == 30                # 6+4+10+10
    assert row["waste_rate"] == pytest.approx(0.2)
    assert row["waste_cost"] == 130000
    assert row["n_carry_in"] == 0                 # 이 fixture엔 음수 폐기 없음
    assert row["carry_in_units"] == pytest.approx(0.0)


def test_waste_rate_by_store_surfaces_carry_in_negatives():
    """음수 waste_qty(전일 재고 이월)는 clip하지 않고 순합 + 건수/합계로 노출한다."""
    frame = _waste().copy()
    frame.loc[0, "waste_qty"] = -4               # 판매가 당일 생산 초과 → carry-in
    row = waste_rate_by_store(frame).iloc[0]
    assert row["waste_qty"] == 20                 # -4+4+10+10 (clip 안 함)
    assert row["n_carry_in"] == 1
    assert row["carry_in_units"] == pytest.approx(-4.0)
    assert row["waste_rate"] == pytest.approx(20 / 150)


def test_waste_rate_by_item_exact_and_filters_low_production():
    frame = waste_rate_by_item(_waste(), _item_to_category()).set_index("item_id")
    assert frame.loc["b1", "waste_rate"] == pytest.approx(10 / 100)
    assert frame.loc["p1", "waste_rate"] == pytest.approx(20 / 50)
    assert frame.loc["b1", "category_id"] == "bread"
    # 생산 30 미만 품목은 비율이 불안정해 제외
    small = _waste().assign(production_qty=[5, 5, 25, 25])
    assert waste_rate_by_item(small, _item_to_category())["item_id"].tolist() == ["p1"]


def test_identity_residual_reports_zero_when_consistent():
    row = identity_residual(_waste()).iloc[0]
    assert row["n_rows"] == 4
    assert row["n_nonzero"] == 0
    assert row["max_abs_diff"] == pytest.approx(0.0)
    assert row["zero_frac"] == pytest.approx(1.0)


def test_identity_residual_detects_mismatch():
    broken = _waste().copy()
    broken.loc[0, "waste_qty"] = 10             # 60 − 54 − 10 = −4 잔차
    row = identity_residual(broken).iloc[0]
    assert row["n_nonzero"] == 1
    assert row["max_abs_diff"] == pytest.approx(4.0)
    assert row["zero_frac"] == pytest.approx(0.75)


def test_overproduction_by_category_cost_share_sums_to_one():
    frame = overproduction_by_category(_waste(), _item_to_category())
    assert frame["cost_share"].sum() == pytest.approx(1.0)
    bread = frame[frame["category_id"] == "bread"].iloc[0]
    assert bread["waste_cost"] == 30000
    assert bread["cost_share"] == pytest.approx(30000 / 130000)


def test_three_handlers_are_data_kind_without_verdict(stub_inputs):
    inputs = stub_inputs(waste=_waste(), item_to_category=_item_to_category())
    for handler, tables in ((waste_rate, ["by_store", "by_item"]),
                            (waste_alpha_identity, ["residual"]),
                            (overproduction_breakdown, ["by_category"])):
        result = handler(inputs)
        assert result.kind == KIND_DATA
        assert result.verdict is None
        assert [label for label, _ in result.tables] == tables
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: ...handlers.waste`

- [ ] **Step 3: 구현** — `src/bakery/analysis/lab/handlers/waste.py`:

```python
"""입력 데이터 분석 — 폐기율 / 항등식 / 과잉생산 분해.

소스는 `waste_alpha_4stores`(생산 made, 폐기 out, 정상/마감 수량, 단가, 폐기비용).
레거시 eda02/eda04/eda05는 `data/internal/v2/inventory.parquet`를 직독했다 —
수치 등가가 아니며 게이트는 구조 불변식(비율 범위, 항등식 잔차)이다.

폐기비용 주의: `waste_cost`는 판매가 기준이다. 원가율(≈0.3)을 곱하지 않은 값이므로
사업 임팩트로 인용할 때 반드시 원가율을 적용해야 한다(과대계상 방지).
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_data
from bakery.analysis.lab.result import KIND_DATA, AnalysisResult

MIN_PRODUCTION_FOR_ITEM_RATE = 30      # 생산 누적 30 미만은 비율이 불안정 → 제외
_IDENTITY_TOLERANCE = 1e-9
_NOTE_COST_BASIS = ("waste_cost는 판매가 기준 — 사업 임팩트 인용 시 원가율(≈0.3)을 곱해야 한다.")
_NOTE_CARRY_IN = ("폐기율은 **순합(net)** 기준이다. waste_qty 음수 행(전일 재고 이월로 판매가 "
                  "당일 생산을 초과, 실측 2.89%)을 clip하지 않으므로 gross 폐기보다 낮다 — "
                  "clip하면 made−sold−out=0 항등식이 깨지고 폐기율이 부풀려진다(광교 +3.2% 상대).")
_NOTE_LEGACY = ("레거시 eda02/04/05(inventory.parquet 직독)와 수치 등가가 아니다 — "
                "canonical waste_alpha_4stores 기준 재표현이다.")


def _rate(waste_qty: pd.Series, production_qty: pd.Series) -> pd.Series:
    return (waste_qty / production_qty).where(production_qty > 0, 0.0)


def waste_rate_by_store(waste: pd.DataFrame) -> pd.DataFrame:
    """매장별 폐기율(순합 기준). carry-in 음수 행수/합계를 함께 실어 은폐하지 않는다."""
    frame = waste.copy()
    frame["is_carry_in"] = frame["waste_qty"] < 0
    grouped = (frame.groupby(["cd", "store"], observed=True)
               .agg(production_qty=("production_qty", "sum"),
                    waste_qty=("waste_qty", "sum"), waste_cost=("waste_cost", "sum"),
                    n_carry_in=("is_carry_in", "sum"))
               .reset_index())
    carry_in_sum = (frame[frame["is_carry_in"]].groupby(["cd", "store"], observed=True)
                    ["waste_qty"].sum().rename("carry_in_units"))
    grouped = grouped.merge(carry_in_sum, on=["cd", "store"], how="left")
    grouped["carry_in_units"] = grouped["carry_in_units"].fillna(0.0)
    grouped["waste_rate"] = _rate(grouped["waste_qty"], grouped["production_qty"])
    return grouped


def waste_rate_by_item(waste: pd.DataFrame, item_to_category: pd.Series, *,
                       min_production: int = MIN_PRODUCTION_FOR_ITEM_RATE) -> pd.DataFrame:
    grouped = (waste.groupby(["cd", "item_id"], observed=True)
               .agg(production_qty=("production_qty", "sum"),
                    waste_qty=("waste_qty", "sum"), waste_cost=("waste_cost", "sum"))
               .reset_index())
    grouped = grouped[grouped["production_qty"] >= min_production].copy()
    grouped["category_id"] = grouped["item_id"].map(item_to_category)
    grouped["waste_rate"] = _rate(grouped["waste_qty"], grouped["production_qty"])
    return grouped.sort_values("waste_rate", ascending=False).reset_index(drop=True)


def identity_residual(waste: pd.DataFrame) -> pd.DataFrame:
    """production − (normal + closing) − waste 잔차 검증(재계산해서 대조)."""
    frame = waste.copy()
    frame["recomputed_diff"] = (frame["production_qty"]
                                - (frame["normal_qty"] + frame["closing_qty"])
                                - frame["waste_qty"])
    rows = []
    for (cd, store), group in frame.groupby(["cd", "store"], observed=True):
        diffs = group["recomputed_diff"].abs()
        n_nonzero = int((diffs > _IDENTITY_TOLERANCE).sum())
        rows.append({"cd": cd, "store": store, "n_rows": int(len(group)),
                     "n_nonzero": n_nonzero, "max_abs_diff": float(diffs.max()),
                     "mean_abs_diff": float(diffs.mean()),
                     "zero_frac": 1.0 - n_nonzero / len(group)})
    return pd.DataFrame(rows)


def overproduction_by_category(waste: pd.DataFrame,
                               item_to_category: pd.Series) -> pd.DataFrame:
    frame = waste.copy()
    frame["category_id"] = frame["item_id"].map(item_to_category)
    grouped = (frame.groupby(["cd", "category_id"], observed=True)
               .agg(production_qty=("production_qty", "sum"),
                    waste_qty=("waste_qty", "sum"), waste_cost=("waste_cost", "sum"))
               .reset_index())
    grouped["waste_rate"] = _rate(grouped["waste_qty"], grouped["production_qty"])
    total_cost = grouped.groupby("cd")["waste_cost"].transform("sum")
    grouped["cost_share"] = (grouped["waste_cost"] / total_cost).where(total_cost > 0, 0.0)
    return grouped.sort_values(["cd", "waste_cost"], ascending=[True, False]) \
                  .reset_index(drop=True)


def _bar_fig(frame: pd.DataFrame, x: str, y: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=frame[x].astype(str), y=frame[y]))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y_title)
    return fig


@register_data("waste_rate", "매장별/품목별 폐기율")
def waste_rate(inputs: AnalysisInputs) -> AnalysisResult:
    by_store = waste_rate_by_store(inputs.waste)
    by_item = waste_rate_by_item(inputs.waste, inputs.item_to_category)
    return AnalysisResult(
        name="waste_rate", kind=KIND_DATA, title="매장별/품목별 폐기율",
        tables=[("by_store", by_store), ("by_item", by_item)],
        figures=[_bar_fig(by_store, "store", "waste_rate", "매장별 폐기율", "폐기율"),
                 _bar_fig(by_item.head(20), "item_id", "waste_rate",
                          "품목별 폐기율 상위 20", "폐기율")],
        notes=[_NOTE_COST_BASIS, _NOTE_CARRY_IN, _NOTE_LEGACY],
    )


@register_data("waste_alpha_identity", "생산 = 정상+마감+폐기 항등식 잔차")
def waste_alpha_identity(inputs: AnalysisInputs) -> AnalysisResult:
    residual = identity_residual(inputs.waste)
    return AnalysisResult(
        name="waste_alpha_identity", kind=KIND_DATA,
        title="생산 = 정상+마감+폐기 항등식 잔차",
        tables=[("residual", residual)],
        figures=[_bar_fig(residual, "store", "zero_frac",
                          "매장별 항등식 성립 비율", "잔차 0 비율")],
        notes=[_NOTE_LEGACY],
    )


@register_data("overproduction_breakdown", "과잉생산 카테고리 분해")
def overproduction_breakdown(inputs: AnalysisInputs) -> AnalysisResult:
    by_category = overproduction_by_category(inputs.waste, inputs.item_to_category)
    return AnalysisResult(
        name="overproduction_breakdown", kind=KIND_DATA, title="과잉생산 카테고리 분해",
        tables=[("by_category", by_category)],
        figures=[_bar_fig(by_category, "category_id", "cost_share",
                          "카테고리별 폐기비용 비중", "비중"),
                 _bar_fig(by_category, "category_id", "waste_rate",
                          "카테고리별 폐기율", "폐기율")],
        notes=[_NOTE_COST_BASIS, _NOTE_CARRY_IN, _NOTE_LEGACY],
    )
```

`HANDLER_MODULES`에 `"waste"` 추가.

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_waste.py -v` / Expected: PASS (7 passed)

- [ ] **Step 5: 스모크**

```bash
printf 'name: t11\ndata:\n  source: real\ndata_analyses:\n  waste_rate: true\n  waste_alpha_identity: true\n  overproduction_breakdown: true\n' > /tmp/t11.yaml
uv run bakery analysis-run /tmp/t11.yaml --out /tmp/analysis_t11
uv run python -c "
import pandas as pd
print(pd.read_csv('/tmp/analysis_t11/t11/waste_rate__by_store.csv').to_string(index=False))
print(pd.read_csv('/tmp/analysis_t11/t11/waste_alpha_identity__residual.csv').to_string(index=False))
"
```
Expected: 광교 `waste_rate`가 [0, 1] 범위. `zero_frac`이 1.0보다 크게 낮으면(예: <0.9) **항등식이 깨지는 실데이터 발견**이므로 멈추고 `docs/phase6_analysis_layer.md`에 기록한 뒤 사용자에게 보고한다(도구 제작 중 실버그 포착 — 무결성 작업의 선례가 있다).

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/analysis/lab/handlers/waste.py src/bakery/analysis/lab/handlers/__init__.py \
        tests/analysis_lab/test_handlers_waste.py
git commit -m "feat(analysis-lab): 폐기 3종(waste_rate/항등식/과잉생산) 핸들러"
```

---

### Task 12: 할인 3종 — `closing_discount` / `other_discounts` / `discount_regime` (형태②)

**Files:**
- Create: `src/bakery/analysis/lab/handlers/discount.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Test: `tests/analysis_lab/test_handlers_discount.py`

**Interfaces:**
- Consumes:
  - `bakery.analysis.discount`: `discount_summary(ds)`, `label_summary(ds)`, `closing_by_category_hour(ds, item_to_category)`, `DiscountSales`, `classify_code`
  - `bakery.analysis.closing_demand`: `run_closing_demand(rows, waste, item_to_category, category="bread")` → dict(`alpha`/`depth`/`surplus`/`kink`/`panel`)
    - **실측 필드명(2026-07-28 확인)**: `AlphaEstimate(alpha_low, alpha_high, a1, a2, a3_slope, note)` / `KinkResult(n_days, base, closing_total, alpha, note)` / `DepthResult(n, slope, se, base, alpha, note)` / `SurplusResult(n, slope, se, clearance_high, note)` — **SurplusResult에는 alpha가 없다**(A3는 slope만; α 환산값은 `AlphaEstimate.a3_slope`)
  - `bakery.analysis.discount_regime`: `run_discount_regime(rows, item_to_category, category, *, cut_date, placebo_cut_dates)` → **dict** `{category, cut_date, n, closing_share, closing_intensity, placebo, verdict}`
    - `closing_share`/`closing_intensity` = `RegimeResult(beta, se, ci_low, ci_high, n, n_params, cut_date, ill_posed)` — **p_value 없음**(CI로 판정)
  - `AnalysisInputs.discount_rows`/`.waste`/`.item_to_category`
- Produces: `closing_waste_frame(waste) -> pd.DataFrame` (item_id/date/waste_qty — `run_closing_demand`의 waste 인자 계약), `alpha_verdict(alpha: AlphaEstimate) -> str`, `regime_verdict(report: dict) -> str`, `discount_hour_table(ds, item_to_category)`, `alpha_estimates_table(report: dict) -> pd.DataFrame`, 핸들러 3개
- `CLOSING_CATEGORY_DEFAULT = "bread"`

- [ ] **Step 1: 테스트 작성**

`tests/analysis_lab/test_handlers_discount.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.discount import DiscountSales
from bakery.analysis.lab.handlers.discount import (
    CLOSING_CATEGORY_DEFAULT, alpha_verdict, closing_discount, closing_waste_frame,
    discount_hour_table, other_discounts, regime_verdict,
)
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _rows():
    return pd.DataFrame({
        "receipt_id": ["r1", "r2", "r3", "r4"],
        "date": pd.to_datetime(["2025-01-01"] * 4),
        "hour": [11, 20, 20, 15],
        "minute": [0, 30, 45, 0],
        "item_id": ["b1", "b1", "p1", "p1"],
        "qty": [2, 3, 1, 4],
        "unit_price": [3000.0, 3000.0, 5000.0, 5000.0],
        "paid": [6000.0, 6300.0, 3500.0, 20000.0],
        "discount_amt": [0.0, 2700.0, 1500.0, 0.0],
        "discount_code": ["", "0069", "0069", ""],
        "label": ["none", "closing", "closing", "none"],
        "is_set": [False, False, False, False],
    })


def _waste():
    return pd.DataFrame({
        "cd": ["1000000047"] * 2, "store": ["광교"] * 2,
        "item_id": ["b1", "p1"], "date": pd.to_datetime(["2025-01-01"] * 2),
        "production_qty": [10, 8], "waste_qty": [2, 1],
        "normal_qty": [2.0, 4.0], "closing_qty": [3.0, 1.0],
        "unit_price": [3000, 5000], "waste_cost": [6000, 5000],
        "identity_diff": [0.0, 0.0], "sold_total": [5.0, 5.0],
    })


def test_closing_category_default():
    assert CLOSING_CATEGORY_DEFAULT == "bread"


def test_closing_waste_frame_matches_primitive_contract():
    """run_closing_demand는 (item_id, date, waste_qty) 컬럼을 요구한다."""
    frame = closing_waste_frame(_waste())
    assert frame.columns.tolist() == ["item_id", "date", "waste_qty"]
    assert frame["waste_qty"].tolist() == [2, 1]


def test_discount_hour_table_counts_closing_by_hour():
    ds = DiscountSales(rows=_rows())
    table = discount_hour_table(ds, pd.Series({"b1": "bread", "p1": "pastry"}))
    bread_20 = table[(table["category_id"] == "bread") & (table["hour"] == 20)].iloc[0]
    assert bread_20["qty"] == 3


def test_alpha_verdict_reports_interval():
    from bakery.analysis.closing_demand import AlphaEstimate

    alpha = AlphaEstimate(alpha_low=0.6, alpha_high=0.9, a1=0.55, a2=0.8,
                          a3_slope=0.7, note="A1 제외(저녁 상시할인)")
    assert alpha_verdict(alpha) == (
        "구간 추정 α ∈ [0.600, 0.900] (A1 0.550 / A2 0.800 / A3 0.700) "
        "— A1 제외(저녁 상시할인)")


def test_alpha_verdict_handles_missing_estimators():
    from bakery.analysis.closing_demand import AlphaEstimate

    alpha = AlphaEstimate(alpha_low=0.5, alpha_high=1.0, a1=None, a2=None,
                          a3_slope=None, note="식별 불가")
    assert alpha_verdict(alpha) == (
        "구간 추정 α ∈ [0.500, 1.000] (A1 없음 / A2 없음 / A3 없음) — 식별 불가")


def test_regime_verdict_uses_ci_and_placebo():
    from bakery.analysis.discount_regime import RegimeResult

    share = RegimeResult(beta=-0.05, se=0.01, ci_low=-0.07, ci_high=-0.03,
                         n=500, n_params=4, cut_date=pd.Timestamp("2024-01-01"),
                         ill_posed=False)
    report = {"category": "bread", "cut_date": pd.Timestamp("2024-01-01"), "n": 500,
              "closing_share": share, "closing_intensity": share,
              "placebo": [], "verdict": "shift"}
    assert regime_verdict(report) == (
        "레짐 전환 shift — closing_share β=-0.0500 CI90[-0.0700,-0.0300], "
        "placebo 0건, n=500 (cut=2024-01-01)")


def test_other_discounts_handler_shape(stub_inputs):
    inputs = stub_inputs(discount_rows=_rows(), waste=_waste(),
                         item_to_category=pd.Series({"b1": "bread", "p1": "pastry"}))
    result = other_discounts(inputs)
    assert result.kind == KIND_HYPOTHESIS
    assert [label for label, _ in result.tables] == ["by_code", "by_label", "by_hour"]
    # 마감(0069)은 제외되고 마감 외 코드만 남는다 — 이 fixture엔 마감 외 할인이 없다
    assert result.verdict == "마감 외 할인 0건 — 이 매장은 마감할인이 전부다"
    # 빈 결과에서도 라벨별 스키마가 유지돼야 한다(오라벨 방지)
    tables = dict(result.tables)
    assert tables["by_code"].columns.tolist() == ["discount_code", "n_lines", "qty"]
    assert tables["by_label"].columns.tolist() == ["label", "n_lines", "qty"]
    assert tables["by_hour"].columns.tolist() == ["discount_code", "hour", "qty"]
    assert len(tables["by_code"]) == 0
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: ...handlers.discount`

- [ ] **Step 3: 구현** — `src/bakery/analysis/lab/handlers/discount.py`:

```python
"""가설 — 마감할인 실수요 α / 마감 외 할인 분포 / 할인 레짐 전환.

계산은 `bakery.analysis.{discount, closing_demand, discount_regime}` 프리미티브 호출.
출처 스크립트: verify_closing_codes / verify_other_discounts (+ discount_regime는 신규 노출).
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from bakery.analysis.discount import (
    DiscountSales, closing_by_category_hour, discount_summary, label_summary,
)
from bakery.analysis.closing_demand import run_closing_demand
from bakery.analysis.discount_regime import run_discount_regime
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult

CLOSING_CATEGORY_DEFAULT = "bread"
CLOSING_LABEL = "closing"
_NOTE_ALPHA_STRUCTURAL = ("광교는 저녁 상시할인 구조라 structural α가 식별되지 않는다"
                          "(A1 degenerate). 증거는 높은 α 방향이며 헌장 기본값은 0.8이다.")
_NOTE_WASTE_SOURCE = "폐기는 waste_alpha_4stores(생산−판매 실측) 기준이다."


def closing_waste_frame(waste: pd.DataFrame) -> pd.DataFrame:
    """run_closing_demand의 waste 인자 계약 = (item_id, date, waste_qty)."""
    return waste[["item_id", "date", "waste_qty"]].reset_index(drop=True)


def discount_hour_table(ds: DiscountSales, item_to_category: pd.Series) -> pd.DataFrame:
    return closing_by_category_hour(ds, item_to_category)


def _estimator_text(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "없음"


def alpha_verdict(alpha) -> str:
    """AlphaEstimate(alpha_low/alpha_high/a1/a2/a3_slope/note) → 판정 문구."""
    return (f"구간 추정 α ∈ [{alpha.alpha_low:.3f}, {alpha.alpha_high:.3f}] "
            f"(A1 {_estimator_text(alpha.a1)} / A2 {_estimator_text(alpha.a2)} / "
            f"A3 {_estimator_text(alpha.a3_slope)}) — {alpha.note}")


def alpha_estimates_table(report: dict) -> pd.DataFrame:
    """추정기별 원시 산출 — SurplusResult에는 alpha가 없어 slope를 싣는다."""
    kink, depth, surplus = report["kink"], report["depth"], report["surplus"]
    return pd.DataFrame([
        {"estimator": "A1 kink", "alpha": kink.alpha, "statistic": kink.base,
         "n": kink.n_days, "note": kink.note},
        {"estimator": "A2 depth", "alpha": depth.alpha, "statistic": depth.slope,
         "n": depth.n, "note": depth.note},
        {"estimator": "A3 surplus", "alpha": report["alpha"].a3_slope,
         "statistic": surplus.slope, "n": surplus.n, "note": surplus.note},
    ])


def regime_verdict(report: dict) -> str:
    """run_discount_regime 반환 dict → 판정. p_value가 없어 CI90로 읽는다."""
    share = report["closing_share"]
    return (f"레짐 전환 {report['verdict']} — closing_share β={share.beta:.4f} "
            f"CI90[{share.ci_low:.4f},{share.ci_high:.4f}], "
            f"placebo {len(report['placebo'])}건, n={report['n']} "
            f"(cut={pd.Timestamp(report['cut_date']).date()})")


def _hour_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for category, group in table.groupby("category_id", observed=True):
        fig.add_trace(go.Bar(x=group["hour"], y=group["qty"], name=str(category)))
    fig.update_layout(title="시각별 할인 판매 수량", barmode="stack",
                      xaxis_title="시(hour)", yaxis_title="수량")
    return fig


def _panel_fig(panel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=panel["date"], y=panel["closing_qty"],
                             mode="lines", name="마감 수량"))
    fig.add_trace(go.Scatter(x=panel["date"], y=panel["waste_qty"],
                             mode="lines", name="폐기 수량"))
    fig.update_layout(title="마감 vs 폐기 추이(잉여의 두 배출구)",
                      xaxis_title="날짜", yaxis_title="수량")
    return fig


@register_hypothesis("closing_discount", "마감할인 실수요 비율 α 추정")
def closing_discount(inputs: AnalysisInputs) -> AnalysisResult:
    ds = DiscountSales(rows=inputs.discount_rows)
    params = inputs.params_for("closing_discount")
    category = params.get("category", CLOSING_CATEGORY_DEFAULT)
    report = run_closing_demand(inputs.discount_rows, closing_waste_frame(inputs.waste),
                               inputs.item_to_category, category=category)
    estimates = alpha_estimates_table(report)
    return AnalysisResult(
        name="closing_discount", kind=KIND_HYPOTHESIS, title="마감할인 실수요 비율 α 추정",
        tables=[("estimates", estimates), ("panel", report["panel"]),
                ("by_hour", discount_hour_table(ds, inputs.item_to_category))],
        figures=[_panel_fig(report["panel"]),
                 _hour_fig(discount_hour_table(ds, inputs.item_to_category))],
        verdict=alpha_verdict(report["alpha"]),
        notes=[_NOTE_ALPHA_STRUCTURAL, _NOTE_WASTE_SOURCE, f"카테고리={category}"],
    )


# 빈 결과에서도 테이블 스키마를 유지한다 — 라벨과 컬럼이 어긋나면 CSV 소비자가 깨진다.
_EMPTY_BY_HOUR = pd.DataFrame({"discount_code": pd.Series(dtype="object"),
                               "hour": pd.Series(dtype="int64"),
                               "qty": pd.Series(dtype="float64")})
_EMPTY_BY_CODE = pd.DataFrame({"discount_code": pd.Series(dtype="object"),
                               "n_lines": pd.Series(dtype="int64"),
                               "qty": pd.Series(dtype="float64")})
_EMPTY_BY_LABEL = pd.DataFrame({"label": pd.Series(dtype="object"),
                                "n_lines": pd.Series(dtype="int64"),
                                "qty": pd.Series(dtype="float64")})


def _discount_code_hour_fig(by_hour: pd.DataFrame) -> go.Figure:
    """할인코드별 시각 분포. 빈 프레임이면 빈 축만 그린다(is_empty 은폐 방지는 verdict가 담당)."""
    fig = go.Figure()
    for code, group in by_hour.groupby("discount_code", observed=True):
        fig.add_trace(go.Bar(x=group["hour"], y=group["qty"], name=str(code)))
    fig.update_layout(title="마감 외 할인코드 시각별 수량", barmode="stack",
                      xaxis_title="시(hour)", yaxis_title="수량")
    return fig


@register_hypothesis("other_discounts", "마감 외 할인코드 시각 분포")
def other_discounts(inputs: AnalysisInputs) -> AnalysisResult:
    rows = inputs.discount_rows
    others = rows[(rows["label"] != CLOSING_LABEL) & (rows["discount_amt"] > 0)]
    ds_others = DiscountSales(rows=others)
    is_empty = len(others) == 0
    by_hour = (_EMPTY_BY_HOUR.copy() if is_empty else
               others.groupby(["discount_code", "hour"], observed=True)["qty"]
               .sum().reset_index())
    verdict = ("마감 외 할인 0건 — 이 매장은 마감할인이 전부다" if is_empty
               else f"마감 외 할인 {len(others):,}건 / 코드 "
                    f"{others['discount_code'].nunique()}종 — 시각 분포로 성격 판별")
    return AnalysisResult(
        name="other_discounts", kind=KIND_HYPOTHESIS, title="마감 외 할인코드 시각 분포",
        tables=[("by_code", _EMPTY_BY_CODE.copy() if is_empty else discount_summary(ds_others)),
                ("by_label", _EMPTY_BY_LABEL.copy() if is_empty else label_summary(ds_others)),
                ("by_hour", by_hour)],
        figures=[_discount_code_hour_fig(by_hour)],
        verdict=verdict,
    )


@register_hypothesis("discount_regime", "할인 레짐 전환(마감 비중 구조변화)")
def discount_regime(inputs: AnalysisInputs) -> AnalysisResult:
    params = inputs.params_for("discount_regime")
    category = params.get("category", CLOSING_CATEGORY_DEFAULT)
    report = run_discount_regime(inputs.discount_rows, inputs.item_to_category, category,
                                 **{k: v for k, v in params.items() if k != "category"})
    rows = [{"outcome": name, "beta": result.beta, "se": result.se,
             "ci_low": result.ci_low, "ci_high": result.ci_high, "n": result.n,
             "ill_posed": result.ill_posed}
            for name, result in (("closing_share", report["closing_share"]),
                                 ("closing_intensity", report["closing_intensity"]))]
    summary = pd.DataFrame(rows)
    placebo = pd.DataFrame([{"cut_date": r.cut_date, "beta": r.beta,
                             "ci_low": r.ci_low, "ci_high": r.ci_high}
                            for r in report["placebo"]])
    betas = [abs(report["closing_share"].beta)] + [abs(r.beta) for r in report["placebo"]]
    labels = ["real β"] + [f"placebo {pd.Timestamp(r.cut_date).date()}"
                           for r in report["placebo"]]
    fig = go.Figure(go.Bar(x=labels, y=betas))
    fig.update_layout(title="레짐 전환 |β| — real vs placebo cut", yaxis_title="|β|")
    return AnalysisResult(
        name="discount_regime", kind=KIND_HYPOTHESIS,
        title="할인 레짐 전환(마감 비중 구조변화)",
        tables=[("summary", summary), ("placebo", placebo)], figures=[fig],
        verdict=regime_verdict(report),
        notes=[f"카테고리={category}",
               "placebo cut이 real과 비슷한 β를 내면 그 전환은 구조가 아니라 추세다."],
    )
```

**빈 스키마 주의:** `_EMPTY_BY_CODE`/`_EMPTY_BY_LABEL`의 컬럼은 `discount_summary`/`label_summary`의 실제 반환 컬럼과 **정확히 같아야** 한다. 구현 시 확인해 맞춘다:

```bash
uv run python -c "
from bakery.analysis.discount import DiscountSales, discount_summary, label_summary, load_sales_with_discount_v2
ds = load_sales_with_discount_v2()
print('by_code:', discount_summary(ds).columns.tolist())
print('by_label:', label_summary(ds).columns.tolist())
"
```
확인 결과와 다르면 `_EMPTY_*` 정의와 위 테스트의 기대 컬럼 리스트를 **둘 다** 그 이름으로 고친다.

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_discount.py -v` / Expected: PASS (8 passed)

- [ ] **Step 5: 출처 스크립트 대조**

```bash
PYTHONPATH=scripts uv run python scripts/verify_closing_codes.py 2>&1 | tail -20
PYTHONPATH=scripts uv run python scripts/verify_other_discounts.py 2>&1 | tail -15
printf 'name: t12\ndata:\n  source: real\nhypotheses:\n  closing_discount: true\n  other_discounts: true\n  discount_regime: true\n' > /tmp/t12.yaml
uv run bakery analysis-run /tmp/t12.yaml --out /tmp/analysis_t12
```
Expected: 마감할인 코드(0069/0077)의 시각 분포와 α 구간 방향이 스크립트 출력과 일치. 수치를 `docs/phase6_analysis_layer.md` 대조 표에 기록.

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/analysis/lab/handlers/discount.py src/bakery/analysis/lab/handlers/__init__.py \
        tests/analysis_lab/test_handlers_discount.py
git commit -m "feat(analysis-lab): 할인 3종(마감α/마감외/레짐) 핸들러"
```

---

### Task 13: `stockout_revenue` + `popularity_stockout` (형태②)

**Files:**
- Create: `src/bakery/analysis/lab/handlers/stockout.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Test: `tests/analysis_lab/test_handlers_stockout.py`

**Interfaces:**
- Consumes:
  - `bakery.analysis.self_fulfillment` — **실측 반환 컬럼(2026-07-28 확인)**:
    - `estimated_lost_demand(daily, *, hour_weights=None)` → `store_id, item_id, date, sold_units, stockout_time, potential_demand, lost_units` (손실량 컬럼은 **`lost_units`**. `potential_demand`는 이 함수가 자체 계산한 로컬 추정치이며 오염된 canonical 컬럼이 아니다 — 혼동 방지를 위해 출력 테이블에서 **드롭**한다)
    - `stockout_hour_distribution(daily, item_ids=None)` → `store_id, item_id, dow, stockout_hour_mean, stockout_hour_std, n_weeks` (**시각 히스토그램이 아니라 품목×요일 평균 매진시각**)
    - `top_self_fulfilling_items(daily, n=15)` → `store_id, item_id, sold_total, avg_stockout_rate, avg_sold_cv, avg_stockout_hour, covered_dows`
  - `bakery.analysis.popularity`: `compute_popularity_signals(daily, closing_discount, today=None)` → `item_id, category_id, days_sold, total_sold, avg_daily_sold, stockout_freq_all, stockout_days, avg_stockout_h, median_stockout_h, closing_qty, closing_days, closing_rate_per_sold, recent_avg, prior_avg, trend_pct` (**boosted 컬럼 없음** — 부스트는 `models/item_proportion`에 있다)
  - `bakery.models.item_proportion`: `compute_proportions(history, target_date)` → `adj_stockout`/`adj_trend`/`adj_closing` 등 배분 조정 컬럼, `STOCKOUT_MAX_BOOST = 0.20`
  - `AnalysisInputs.daily`/`.discount_rows`
- Produces: `lost_demand_summary(daily) -> pd.DataFrame` (cols: store_id, n_stockout_days, est_lost_units, lost_share_of_sold), `stockout_revenue_verdict(summary) -> str`, `popularity_boost_correlation(daily, closing, *, target_date) -> pd.DataFrame`, `popularity_verdict(corr) -> str`, 핸들러 2개

**`popularity_stockout` 재정의(명시):** 출처 스크립트는 **옛 매진 라벨 vs 새 라벨**로 만든 두 비율을 spearman 비교했다. canonical에는 옛(오염) 라벨이 더 이상 없으므로 A/B가 불가능하다. 대신 **원시 인기 순위(`avg_daily_sold`) vs 매진 부스트 적용 배분 순위(`compute_proportions`의 `adj_stockout` 반영값)** 를 비교해 "매진 신호가 배분 순위를 얼마나 재배열하는가"를 잰다. 이 재정의를 verdict 문구와 note에 반드시 적는다.
- `LOST_SHARE_THRESHOLD = 0.02` (2% 미만이면 "무영향")

- [ ] **Step 1: 테스트 작성**

`tests/analysis_lab/test_handlers_stockout.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.stockout import (
    LOST_SHARE_THRESHOLD, POPULARITY_CORR_THRESHOLD, lost_demand_summary,
    popularity_verdict, stockout_revenue, stockout_revenue_verdict,
)
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _daily():
    """b1이 3일 중 1일 19시 완판. 총 sold 60."""
    rows = [
        ("2025-01-01", "b1", "bread", 20, True, "2025-01-01 19:00"),
        ("2025-01-02", "b1", "bread", 20, False, None),
        ("2025-01-03", "b1", "bread", 20, False, None),
    ]
    return pd.DataFrame([{"store_id": "store_gw01", "item_id": i, "category_id": c,
                          "date": pd.Timestamp(d), "sold_units": q,
                          "is_stockout": so, "open_hours": 13.0, "capacity": q,
                          "stockout_time": pd.Timestamp(t) if t else pd.NaT}
                         for d, i, c, q, so, t in rows])


def test_threshold_constants():
    assert LOST_SHARE_THRESHOLD == 0.02
    assert POPULARITY_CORR_THRESHOLD == 0.8


def test_lost_demand_summary_counts_and_share():
    summary = lost_demand_summary(_daily()).iloc[0]
    assert summary["store_id"] == "store_gw01"
    assert summary["n_stockout_days"] == 1
    assert summary["est_lost_units"] > 0.0
    # 손실 추정치는 sold 총합 60에 대한 비율로 표현된다
    assert summary["lost_share_of_sold"] == pytest.approx(
        summary["est_lost_units"] / 60.0)


def test_stockout_revenue_verdict_no_impact_below_threshold():
    summary = pd.DataFrame([{"store_id": "store_gw01", "n_stockout_days": 1,
                             "est_lost_units": 1.0, "lost_share_of_sold": 0.01}])
    assert stockout_revenue_verdict(summary) == (
        "지지(무영향) — 매장 1곳 전부 추정 손실 비중 2% 미만 (최대 1.0%)")


def test_stockout_revenue_verdict_flags_material_store():
    summary = pd.DataFrame([
        {"store_id": "store_gw01", "n_stockout_days": 1, "est_lost_units": 1.0,
         "lost_share_of_sold": 0.01},
        {"store_id": "store_mp01", "n_stockout_days": 5, "est_lost_units": 30.0,
         "lost_share_of_sold": 0.05},
    ])
    assert stockout_revenue_verdict(summary) == (
        "부분 기각 — ['store_mp01'] 매장에서 추정 손실 비중 2% 이상 (최대 5.0%)")


def test_popularity_verdict_reports_rank_stability():
    corr = pd.DataFrame([{"pair": "raw_vs_stockout_boosted", "spearman": 0.95, "n": 100}])
    assert popularity_verdict(corr) == (
        "매진 부스트가 배분 순위를 거의 바꾸지 않음 — spearman 0.950 (n=100), "
        "부스트 기여 작음")


def test_popularity_verdict_flags_reordering():
    corr = pd.DataFrame([{"pair": "raw_vs_stockout_boosted", "spearman": 0.55, "n": 100}])
    assert popularity_verdict(corr) == (
        "매진 부스트가 배분 순위를 크게 재배열 — spearman 0.550 (n=100), "
        "임계 0.8 미만이므로 부스트 강도 검토 필요")


def test_stockout_revenue_handler_shape(stub_inputs):
    result = stockout_revenue(stub_inputs(daily=_daily()))
    assert result.kind == KIND_HYPOTHESIS
    assert [label for label, _ in result.tables] == [
        "summary", "top_self_fulfilling", "hour_distribution"]
    hours = dict(result.tables)["hour_distribution"]
    assert hours.columns.tolist() == ["store_id", "item_id", "dow",
                                      "stockout_hour_mean", "stockout_hour_std", "n_weeks"]
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: ...handlers.stockout`

- [ ] **Step 3: 구현** — `src/bakery/analysis/lab/handlers/stockout.py`:

```python
"""가설 — 매진의 매출 영향 / 매진 보정이 인기 신호를 흔드는가.

계산은 `bakery.analysis.{self_fulfillment, popularity}` 프리미티브 호출.
출처 스크립트: verify_stockout_revenue_4stores(_fixed), revalidate_popularity_stockout.

측정 헌장: 품절일 판매량은 censored — 추정 손실은 하한이다(무영향 판정은 보수적).
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from scipy.stats import spearmanr

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.self_fulfillment import (
    estimated_lost_demand, stockout_hour_distribution, top_self_fulfilling_items,
)

LOST_UNITS_COLUMN = "lost_units"                 # estimated_lost_demand의 손실량 컬럼
_LOCAL_ESTIMATE_COLUMNS = ("potential_demand",)  # 함수 내부 추정치 — 출력에서 드롭

LOST_SHARE_THRESHOLD = 0.02      # 추정 손실 비중 2% 미만 = 무영향
POPULARITY_CORR_THRESHOLD = 0.8  # 순위 상관 0.8 이상 = 신호 안정
_TOP_ITEMS = 15
_NOTE_CENSORED = ("품절일 판매량은 censored — 추정 손실은 하한이다. "
                  "따라서 '무영향' 판정은 보수적이고, '영향 있음'은 강한 신호다.")
_NOTE_LOST_MODEL = ("손실 추정은 features/potential_demand와 같은 시간가중 공식 "
                    "(estimated_lost_demand) — 모델 예측이 아니라 관측 기반 산식이다.")


def lost_demand_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """매장별 매진일 수 + 추정 손실 수량 + sold 대비 비중."""
    lost = estimated_lost_demand(daily)
    per_store_lost = lost.groupby("store_id")[LOST_UNITS_COLUMN].sum()
    rows = []
    for store, group in daily.groupby("store_id", observed=True):
        sold = float(group["sold_units"].sum())
        est_lost = float(per_store_lost.get(store, 0.0))
        rows.append({"store_id": store,
                     "n_stockout_days": int(group["is_stockout"].sum()),
                     "est_lost_units": est_lost,
                     "lost_share_of_sold": est_lost / sold if sold else 0.0})
    return pd.DataFrame(rows)


def stockout_revenue_verdict(summary: pd.DataFrame) -> str:
    material = summary[summary["lost_share_of_sold"] >= LOST_SHARE_THRESHOLD]
    max_share = float(summary["lost_share_of_sold"].max()) * 100
    if len(material) == 0:
        return (f"지지(무영향) — 매장 {len(summary)}곳 전부 추정 손실 비중 2% 미만 "
                f"(최대 {max_share:.1f}%)")
    return (f"부분 기각 — {material['store_id'].tolist()} 매장에서 추정 손실 비중 "
            f"2% 이상 (최대 {max_share:.1f}%)")


def popularity_boost_correlation(daily: pd.DataFrame, closing: pd.DataFrame, *,
                                 target_date: pd.Timestamp) -> pd.DataFrame:
    """원시 인기 순위(avg_daily_sold) vs 매진 부스트 적용 배분 순위의 spearman.

    옛/새 매진 라벨 A/B는 canonical에 옛 라벨이 없어 불가 — 대신 부스트가 순위를
    얼마나 재배열하는지를 잰다(Stage2 배분에 실제로 쓰이는 경로).
    """
    from bakery.models.item_proportion import compute_proportions

    signals = compute_popularity_signals(daily, closing, today=target_date)
    proportions = compute_proportions(daily, target_date)
    merged = signals[["item_id", "avg_daily_sold"]].merge(
        proportions[["item_id", "adj_stockout"]], on="item_id", how="inner")
    merged["boosted_rank_value"] = merged["avg_daily_sold"] * merged["adj_stockout"]
    pair = merged[["avg_daily_sold", "boosted_rank_value"]].dropna()
    rho = float(spearmanr(pair["avg_daily_sold"], pair["boosted_rank_value"]).statistic)
    return pd.DataFrame([{"pair": "raw_vs_stockout_boosted", "spearman": rho,
                          "n": int(len(pair))}])


def popularity_verdict(corr: pd.DataFrame) -> str:
    rho = float(corr["spearman"].iloc[0])
    n = int(corr["n"].iloc[0])
    if rho >= POPULARITY_CORR_THRESHOLD:
        return (f"매진 부스트가 배분 순위를 거의 바꾸지 않음 — spearman {rho:.3f} "
                f"(n={n}), 부스트 기여 작음")
    return (f"매진 부스트가 배분 순위를 크게 재배열 — spearman {rho:.3f} (n={n}), "
            "임계 0.8 미만이므로 부스트 강도 검토 필요")


def _lost_fig(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=summary["store_id"], y=summary["lost_share_of_sold"]))
    fig.add_hline(y=LOST_SHARE_THRESHOLD, line_dash="dash")
    fig.update_layout(title="매장별 추정 손실 비중(점선=2% 임계)",
                      xaxis_title="매장", yaxis_title="sold 대비 비중")
    return fig


def _hour_fig(hours: pd.DataFrame) -> go.Figure:
    """품목×요일 평균 매진시각의 요일별 분포(히스토그램이 아니라 평균값의 분포)."""
    fig = go.Figure()
    for dow, group in hours.groupby("dow", observed=True):
        fig.add_trace(go.Box(y=group["stockout_hour_mean"], name=str(dow)))
    fig.update_layout(title="요일별 평균 매진시각 분포(품목 단위)",
                      xaxis_title="요일(월=0)", yaxis_title="평균 매진시각(시)")
    return fig


@register_hypothesis("stockout_revenue", "매진의 매장 매출 영향(무영향 가정 검증)")
def stockout_revenue(inputs: AnalysisInputs) -> AnalysisResult:
    daily = inputs.daily
    summary = lost_demand_summary(daily)
    hours = stockout_hour_distribution(daily)
    return AnalysisResult(
        name="stockout_revenue", kind=KIND_HYPOTHESIS,
        title="매진의 매장 매출 영향(무영향 가정 검증)",
        tables=[("summary", summary),
                ("top_self_fulfilling", top_self_fulfilling_items(daily, n=_TOP_ITEMS)),
                ("hour_distribution", hours)],
        figures=[_lost_fig(summary), _hour_fig(hours)],
        verdict=stockout_revenue_verdict(summary),
        notes=[_NOTE_CENSORED, _NOTE_LOST_MODEL],
    )


@register_hypothesis("popularity_stockout", "매진 재정의가 인기 신호를 흔드는가")
def popularity_stockout(inputs: AnalysisInputs) -> AnalysisResult:
    from bakery.analysis.popularity import compute_popularity_signals

    closing = inputs.discount_rows
    closing = closing[closing["label"] == "closing"][["item_id", "date", "qty"]]
    target_date = pd.Timestamp(inputs.daily["date"].max())
    signals = compute_popularity_signals(inputs.daily, closing, today=target_date)
    corr = popularity_boost_correlation(inputs.daily, closing, target_date=target_date)
    fig = go.Figure(go.Bar(x=corr["pair"], y=corr["spearman"]))
    fig.add_hline(y=POPULARITY_CORR_THRESHOLD, line_dash="dash")
    fig.update_layout(title="인기 신호 순위 상관(점선=0.8 임계)", yaxis_title="spearman")
    return AnalysisResult(
        name="popularity_stockout", kind=KIND_HYPOTHESIS,
        title="매진 부스트가 배분 순위를 재배열하는가",
        tables=[("rank_correlation", corr), ("signals", signals)],
        figures=[fig], verdict=popularity_verdict(corr),
        notes=["매진 라벨은 재정의(폐기0=완판) 반영본 — 옛 92.7% 정의가 아니다.",
               ("출처 스크립트의 옛/새 라벨 A/B는 canonical에 옛(오염) 라벨이 없어 "
                "불가 — 원시 인기 vs 매진 부스트 순위 비교로 재정의했다."),
               f"부스트 상한 STOCKOUT_MAX_BOOST=0.20, 기준일={{target}}"],
    )
```

**주의:** 마지막 note의 `{target}`은 f-string 안에서 `target_date.date()`로 채운다(리터럴 중괄호를 남기지 말 것).

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_stockout.py -v` / Expected: PASS (8 passed)

- [ ] **Step 5: 출처 대조 + 커밋**

```bash
PYTHONPATH=scripts uv run python scripts/verify_stockout_revenue_fixed.py 2>&1 | tail -15
PYTHONPATH=scripts uv run python scripts/revalidate_popularity_stockout.py 2>&1 | tail -15
git add src/bakery/analysis/lab/handlers/stockout.py src/bakery/analysis/lab/handlers/__init__.py \
        tests/analysis_lab/test_handlers_stockout.py
git commit -m "feat(analysis-lab): stockout_revenue + popularity_stockout 핸들러"
```

기대: 무영향 판정이 3/4 매장(광교·광화문·삼성)에서 유지되고 메세나만 약신호 — 뒤집히면 멈추고 규명한다.

---

### Task 14: `substitution` (형태②, 무거운 항목)

**Files:**
- Create: `src/bakery/analysis/lab/handlers/substitution.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Test: `tests/analysis_lab/test_handlers_substitution.py`

**Interfaces:**
- Consumes:
  - `bakery.analysis.substitution`: `compute_substitution_matrix(daily, receipts, *, include_inter_category=True, hour_profiles=None, cutoff_threshold=...)` → `SubstitutionMatrix(coefficients, outflow_ratio, cutoffs)`, `sensitivity_summary(outflow_ratio)`
  - `bakery.analysis.mnl_substitution`: `fit_mnl_per_category(receipts, daily)` → **`MnlResult(utilities, substitution, outflow_ratio)`** (2026-07-28 확인 — `coefficients` 아님)
  - `bakery.analysis.nested_logit`: `fit_nested_logit(receipts, daily)` → **`NestedLogitResult(utilities, lambdas, substitution, outflow_ratio)`** (λ 필드명은 **`lambdas`**)
  - `bakery.analysis.substitution_did`: `compute_did_substitution(daily, receipts, hour_profiles, ...)` → `DidResult(coefficients, outflow_ratio, cutoffs)`
  - `AnalysisInputs.daily`/`.receipts`
- Produces: `hour_profiles_from_receipts(receipts, daily) -> dict[str, np.ndarray]`, `substitution_verdict(rd, did, nested) -> str`, `substitution(inputs) -> AnalysisResult`
- `LAMBDA_INDEPENDENCE_THRESHOLD = 0.95` (λ≈1 = 사실상 독립 = 대체 약함), `DID_BETA_THRESHOLD = 0.02`

**설계 노트:** `compute_did_substitution`은 `hour_profiles`(매장별 길이 24 판매 분포)를 필수 인자로 받는다. 이를 `receipts`의 `hour` 분포에서 매장별로 만든다. 단매장 spec이면 매장 1개짜리 dict가 된다.

- [ ] **Step 1: 테스트 작성**

`tests/analysis_lab/test_handlers_substitution.py`:

```python
import numpy as np
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.substitution import (
    DID_BETA_THRESHOLD, LAMBDA_INDEPENDENCE_THRESHOLD,
    hour_profiles_from_receipts, substitution_verdict,
)


def _receipts():
    rows = []
    for hour, count in ((11, 3), (15, 2), (19, 5)):
        for index in range(count):
            rows.append({"receipt_id": f"r{hour}_{index}", "date": pd.Timestamp("2025-01-01"),
                         "item_id": "b1", "hour": hour, "minute": 0, "qty": 1,
                         "timestamp": pd.Timestamp(f"2025-01-01 {hour}:00"), "is_bulk": False})
    return pd.DataFrame(rows)


def _daily():
    return pd.DataFrame([{"store_id": "store_gw01", "item_id": "b1", "category_id": "bread",
                          "date": pd.Timestamp("2025-01-01"), "sold_units": 10,
                          "is_stockout": False, "stockout_time": pd.NaT}])


def test_thresholds():
    assert LAMBDA_INDEPENDENCE_THRESHOLD == 0.95
    assert DID_BETA_THRESHOLD == 0.02


def test_hour_profiles_are_length_24_and_normalized():
    profiles = hour_profiles_from_receipts(_receipts(), _daily())
    assert list(profiles) == ["store_gw01"]
    profile = profiles["store_gw01"]
    assert profile.shape == (24,)
    assert profile.sum() == pytest.approx(1.0)
    # 11시 3/10, 15시 2/10, 19시 5/10
    assert profile[11] == pytest.approx(0.3)
    assert profile[15] == pytest.approx(0.2)
    assert profile[19] == pytest.approx(0.5)
    assert profile[0] == 0.0


def test_verdict_weak_substitution_when_lambda_near_one_and_did_zero():
    verdict = substitution_verdict(rd_mean_outflow=0.03, did_mean_beta=0.001,
                                   nested_lambda_min=0.99)
    assert verdict == (
        "기각(대체 약함) — nested λ_min 0.990(≈1=독립), DiD 평균 β 0.0010, "
        "RD 평균 유출 0.030. 카테고리는 한 묶음 수요로 취급 가능")


def test_verdict_reports_substitution_when_did_material():
    verdict = substitution_verdict(rd_mean_outflow=0.15, did_mean_beta=0.05,
                                   nested_lambda_min=0.60)
    assert verdict == (
        "지지(대체 있음) — nested λ_min 0.600, DiD 평균 β 0.0500(임계 0.02 초과), "
        "RD 평균 유출 0.150")


def test_verdict_inconclusive_when_signals_conflict():
    verdict = substitution_verdict(rd_mean_outflow=0.15, did_mean_beta=0.001,
                                   nested_lambda_min=0.60)
    assert verdict == (
        "불확실 — nested λ_min 0.600은 대체를 시사하나 DiD 평균 β 0.0010은 0에 가깝다 "
        "(RD 평균 유출 0.150)")


@pytest.mark.slow
def test_handler_produces_four_estimator_tables():
    from bakery.analysis.lab.handlers.substitution import substitution
    from bakery.analysis.lab.inputs import AnalysisInputs
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = substitution(inputs)
    assert [label for label, _ in result.tables] == [
        "rd_coefficients", "did_coefficients", "mnl", "nested_lambda", "sensitivity"]
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: ...handlers.substitution`

- [ ] **Step 3: 구현** — `src/bakery/analysis/lab/handlers/substitution.py`:

```python
"""가설 — 품목 매진 시 수요가 다른 품목으로 대체되는가(4추정기).

RD(회귀 불연속) / DiD / MNL / Nested logit을 모두 돌려 결론을 교차 확인한다.
계산은 `bakery.analysis.{substitution, substitution_did, mnl_substitution, nested_logit}` 호출.
출처 스크립트: substitution_4stores.

과거 결론(광교): MNL/Nested λ≈0.99, DiD β≈0 → 개별 substitution 효과 약함,
카테고리는 한 묶음 수요. 이 핸들러는 그 결론을 현 vintage에서 재확인하는 수단이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.mnl_substitution import fit_mnl_per_category
from bakery.analysis.nested_logit import fit_nested_logit
from bakery.analysis.substitution import compute_substitution_matrix, sensitivity_summary
from bakery.analysis.substitution_did import compute_did_substitution

LAMBDA_INDEPENDENCE_THRESHOLD = 0.95   # λ≈1 = nest 내 독립 = 대체 약함
DID_BETA_THRESHOLD = 0.02              # DiD 평균 β 임계
HOURS_IN_DAY = 24
_NOTE_RECEIPTS = ("영수증은 canonical bonavi_receipts(bulk 제외 빌드) — "
                  "is_bulk 컬럼은 진단용이며 재필터하지 않는다.")
_NOTE_COST = "4추정기 전부 실행하므로 실행 시간이 길다(수 분). 필요 없으면 off로 둔다."


def hour_profiles_from_receipts(receipts: pd.DataFrame,
                                daily: pd.DataFrame) -> dict[str, np.ndarray]:
    """매장별 시간대 판매 분포(길이 24, 합=1). DiD가 매진 노출 시간을 배분하는 데 쓴다."""
    stores = daily["store_id"].unique()
    if "store_id" in receipts.columns:
        grouped = {store: group for store, group in receipts.groupby("store_id",
                                                                    observed=True)}
    else:
        grouped = {store: receipts for store in stores}      # 단매장 receipts
    profiles: dict[str, np.ndarray] = {}
    for store in stores:
        group = grouped.get(store)
        counts = np.zeros(HOURS_IN_DAY, dtype=float)
        if group is None or len(group) == 0:
            profiles[store] = counts
            continue
        by_hour = group.groupby("hour")["qty"].sum()
        for hour, qty in by_hour.items():
            if 0 <= int(hour) < HOURS_IN_DAY:
                counts[int(hour)] = float(qty)
        total = counts.sum()
        profiles[store] = counts / total if total else counts
    return profiles


def substitution_verdict(*, rd_mean_outflow: float, did_mean_beta: float,
                         nested_lambda_min: float) -> str:
    """세 추정기 신호를 합쳐 판정. λ≈1 + DiD β≈0 = 대체 약함(카테고리 한 묶음)."""
    is_lambda_independent = nested_lambda_min >= LAMBDA_INDEPENDENCE_THRESHOLD
    is_did_material = abs(did_mean_beta) > DID_BETA_THRESHOLD
    if is_lambda_independent and not is_did_material:
        return (f"기각(대체 약함) — nested λ_min {nested_lambda_min:.3f}(≈1=독립), "
                f"DiD 평균 β {did_mean_beta:.4f}, RD 평균 유출 {rd_mean_outflow:.3f}. "
                "카테고리는 한 묶음 수요로 취급 가능")
    if is_did_material:
        return (f"지지(대체 있음) — nested λ_min {nested_lambda_min:.3f}, "
                f"DiD 평균 β {did_mean_beta:.4f}(임계 0.02 초과), "
                f"RD 평균 유출 {rd_mean_outflow:.3f}")
    return (f"불확실 — nested λ_min {nested_lambda_min:.3f}은 대체를 시사하나 "
            f"DiD 평균 β {did_mean_beta:.4f}은 0에 가깝다 "
            f"(RD 평균 유출 {rd_mean_outflow:.3f})")


def _outflow_fig(rd_outflow: pd.Series, did_outflow: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rd_outflow.index.astype(str), y=rd_outflow.values, name="RD"))
    fig.add_trace(go.Bar(x=did_outflow.index.astype(str), y=did_outflow.values, name="DiD"))
    fig.update_layout(title="품목별 유출 비율(대체 강도)", barmode="group",
                      xaxis_title="품목", yaxis_title="Σ 대체율")
    return fig


def _lambda_fig(lambdas: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=lambdas["nest"].astype(str), y=lambdas["lambda"]))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="Nested logit λ (1=nest 내 독립)",
                      xaxis_title="nest", yaxis_title="λ")
    return fig


@register_hypothesis("substitution", "품목 매진 시 수요 대체(RD/DiD/MNL/Nested)")
def substitution(inputs: AnalysisInputs) -> AnalysisResult:
    daily, receipts = inputs.daily, inputs.receipts
    profiles = hour_profiles_from_receipts(receipts, daily)
    rd = compute_substitution_matrix(daily, receipts, hour_profiles=profiles)
    did = compute_did_substitution(daily, receipts, profiles)
    mnl = fit_mnl_per_category(receipts, daily)
    nested = fit_nested_logit(receipts, daily)
    lambdas = pd.Series(nested.lambdas).rename("lambda").rename_axis("nest").reset_index()
    verdict = substitution_verdict(
        rd_mean_outflow=float(rd.outflow_ratio.mean()),
        did_mean_beta=float(did.coefficients["beta_did"].mean()),
        nested_lambda_min=float(lambdas["lambda"].min()),
    )
    return AnalysisResult(
        name="substitution", kind=KIND_HYPOTHESIS,
        title="품목 매진 시 수요 대체(RD/DiD/MNL/Nested)",
        tables=[("rd_coefficients", rd.coefficients),
                ("did_coefficients", did.coefficients),
                ("mnl", mnl.substitution),
                ("nested_lambda", lambdas),
                ("sensitivity", sensitivity_summary(rd.outflow_ratio))],
        figures=[_outflow_fig(rd.outflow_ratio, did.outflow_ratio), _lambda_fig(lambdas)],
        verdict=verdict, notes=[_NOTE_RECEIPTS, _NOTE_COST],
    )
```

**주의:** `nested.lambdas`가 dict/Series 중 무엇이든 `pd.Series(...)`로 감싸 통일한다. `mnl.substitution`/`nested.substitution`은 대체 계수 프레임, `outflow_ratio`는 index=item_id인 Series다.

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_substitution.py -v` / Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/lab/handlers/substitution.py \
        src/bakery/analysis/lab/handlers/__init__.py \
        tests/analysis_lab/test_handlers_substitution.py
git commit -m "feat(analysis-lab): substitution 핸들러(RD/DiD/MNL/Nested 4추정기)"
```

---

### Task 15: `modeling_v4_assumptions` (형태③)

**Files:**
- Create: `src/bakery/analysis/lab/handlers/basket.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py`
- Test: `tests/analysis_lab/test_handlers_basket.py`

**Interfaces:**
- Consumes: `bakery.analysis.basket_composition`: `classify_baskets(lines)`, `basket_composition_summary(lines, closing_category=None)`; `bakery.analysis.seasonal`: `filter_seasonal(df)`, `excluded_summary()`; `AnalysisInputs.receipts`/`.daily`/`.item_to_category`
- Produces: `assumption_table(daily, receipts, item_to_category) -> pd.DataFrame` (cols: assumption, statistic, value, threshold, is_supported), `v4_verdict(table) -> str`, `modeling_v4_assumptions(inputs) -> AnalysisResult`

**검증 대상 4가정** (출처: `scripts/verify_hypotheses.py`, `docs/modeling_v4.md`):

| 키 | 가정 | 통계량 | 임계 |
|---|---|---|---|
| `1-1-b` | 카테고리 총량이 품목 합보다 예측 가능(변동계수 낮음) | cv(카테고리 총량) / mean cv(품목) | < 0.7 |
| `2-1-a` | 카테고리 내 품목 비율이 시간에 안정 | 월별 비율 std 중앙값 | < 0.05 |
| `2-1-b` | 신제품이 기존 품목 비율을 크게 흔들지 않음 | 신제품 도입월 비율 변화 중앙값 | < 0.10 |
| `basket` | 바스켓에 카테고리가 섞임(한 묶음 수요 근거) | 다중 카테고리 바스켓 비율 | > 0.30 |

- [ ] **Step 1: 테스트 작성**

`tests/analysis_lab/test_handlers_basket.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.lab.handlers.basket import (
    ASSUMPTION_THRESHOLDS, modeling_v4_assumptions, v4_verdict,
)
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _table(supported):
    return pd.DataFrame([{"assumption": key, "statistic": "s", "value": 0.1,
                          "threshold": 0.5, "is_supported": flag}
                         for key, flag in supported.items()])


def test_thresholds_are_declared_for_four_assumptions():
    assert sorted(ASSUMPTION_THRESHOLDS) == ["1-1-b", "2-1-a", "2-1-b", "basket"]
    assert ASSUMPTION_THRESHOLDS["1-1-b"] == 0.7
    assert ASSUMPTION_THRESHOLDS["2-1-a"] == 0.05
    assert ASSUMPTION_THRESHOLDS["2-1-b"] == 0.10
    assert ASSUMPTION_THRESHOLDS["basket"] == 0.30


def test_verdict_supports_when_all_pass():
    verdict = v4_verdict(_table({"1-1-b": True, "2-1-a": True, "2-1-b": True,
                                "basket": True}))
    assert verdict == "지지 — 4가정 전부 통과(v4 카테고리 합 → 품목 비율 설계 정당)"


def test_verdict_lists_failed_assumptions():
    verdict = v4_verdict(_table({"1-1-b": True, "2-1-a": False, "2-1-b": False,
                                "basket": True}))
    assert verdict == "부분 지지 — 4가정 중 2건 통과, 미통과: ['2-1-a', '2-1-b']"


def test_verdict_rejects_when_none_pass():
    verdict = v4_verdict(_table({"1-1-b": False, "2-1-a": False, "2-1-b": False,
                                "basket": False}))
    assert verdict == "기각 — 4가정 전부 미통과"


@pytest.mark.slow
def test_handler_reports_all_four_assumptions():
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = modeling_v4_assumptions(inputs)
    assert result.kind == KIND_HYPOTHESIS
    table = dict(result.tables)["assumptions"]
    assert sorted(table["assumption"].tolist()) == ["1-1-b", "2-1-a", "2-1-b", "basket"]
    assert table["is_supported"].dtype == bool
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: ...handlers.basket`

- [ ] **Step 3: 구현** — `src/bakery/analysis/lab/handlers/basket.py`:

```python
"""가설 — modeling_v4 framework의 4가정(카테고리 합 → 품목 비율 3-stage 전제).

출처 스크립트: verify_hypotheses.py. 바스켓 구성은
`bakery.analysis.basket_composition` 프리미티브 호출.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from bakery.analysis.basket_composition import basket_composition_summary
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.seasonal import filter_seasonal

ASSUMPTION_THRESHOLDS: dict[str, float] = {
    "1-1-b": 0.7,     # cv(카테고리 총량) / 평균 cv(품목) — 낮을수록 총량이 안정
    "2-1-a": 0.05,    # 월별 품목 비율 std 중앙값
    "2-1-b": 0.10,    # 신제품 도입 시 기존 비율 변화 중앙값
    "basket": 0.30,   # 다중 카테고리 바스켓 비율(> 임계여야 통과)
}
NEW_ITEM_MIN_MONTHS = 2       # 도입 전후 비교에 필요한 최소 월 수
_LOWER_IS_BETTER = ("1-1-b", "2-1-a", "2-1-b")


def _cv(series: pd.Series) -> float:
    mean = float(series.mean())
    return float(series.std()) / mean if mean else 0.0


def _total_vs_item_cv(daily: pd.DataFrame) -> float:
    """카테고리 총량 cv를 품목 cv 평균으로 나눈 값(작을수록 총량이 예측 쉽다)."""
    category_daily = daily.groupby(["category_id", "date"])["sold_units"].sum()
    category_cv = category_daily.groupby("category_id").apply(_cv).mean()
    item_daily = daily.groupby(["item_id", "date"])["sold_units"].sum()
    item_cv = item_daily.groupby("item_id").apply(_cv).mean()
    return float(category_cv / item_cv) if item_cv else 0.0


def _monthly_proportion(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = (frame.groupby(["category_id", "month", "item_id"], observed=True)
               ["sold_units"].sum().reset_index())
    total = monthly.groupby(["category_id", "month"])["sold_units"].transform("sum")
    monthly["proportion"] = (monthly["sold_units"] / total).where(total > 0, 0.0)
    return monthly


def _proportion_stability(monthly: pd.DataFrame) -> float:
    stds = monthly.groupby(["category_id", "item_id"])["proportion"].std()
    return float(stds.median())


def _new_item_disruption(monthly: pd.DataFrame) -> float:
    """신제품 첫 등장 월에 기존 품목 비율이 얼마나 흔들리는지(중앙 절대변화)."""
    changes = []
    for (category,), group in monthly.groupby(["category_id"], observed=True):
        months = sorted(group["month"].unique())
        seen: set[str] = set()
        previous: pd.Series | None = None
        for month in months:
            snapshot = group[group["month"] == month].set_index("item_id")["proportion"]
            new_items = set(snapshot.index) - seen
            if previous is not None and new_items and len(seen) >= 1:
                shared = previous.index.intersection(snapshot.index).difference(new_items)
                if len(shared):
                    changes.append(float((snapshot[shared] - previous[shared]).abs().median()))
            seen |= set(snapshot.index)
            previous = snapshot
    return float(pd.Series(changes).median()) if changes else 0.0


def _multi_category_basket_share(receipts: pd.DataFrame,
                                 item_to_category: pd.Series) -> float:
    frame = receipts.copy()
    frame["category_id"] = frame["item_id"].map(item_to_category)
    per_receipt = frame.groupby("receipt_id")["category_id"].nunique()
    return float((per_receipt > 1).mean()) if len(per_receipt) else 0.0


def assumption_table(daily: pd.DataFrame, receipts: pd.DataFrame,
                     item_to_category: pd.Series) -> pd.DataFrame:
    filtered = filter_seasonal(daily)
    monthly = _monthly_proportion(filtered)
    values = {
        "1-1-b": ("cv(카테고리 총량)/평균 cv(품목)", _total_vs_item_cv(filtered)),
        "2-1-a": ("월별 품목 비율 std 중앙값", _proportion_stability(monthly)),
        "2-1-b": ("신제품 도입 시 기존 비율 변화 중앙값", _new_item_disruption(monthly)),
        "basket": ("다중 카테고리 바스켓 비율",
                   _multi_category_basket_share(receipts, item_to_category)),
    }
    rows = []
    for key, (statistic, value) in values.items():
        threshold = ASSUMPTION_THRESHOLDS[key]
        is_supported = value < threshold if key in _LOWER_IS_BETTER else value > threshold
        rows.append({"assumption": key, "statistic": statistic, "value": value,
                     "threshold": threshold, "is_supported": bool(is_supported)})
    return pd.DataFrame(rows)


def v4_verdict(table: pd.DataFrame) -> str:
    failed = table[~table["is_supported"]]["assumption"].tolist()
    n_passed = len(table) - len(failed)
    if not failed:
        return "지지 — 4가정 전부 통과(v4 카테고리 합 → 품목 비율 설계 정당)"
    if n_passed == 0:
        return "기각 — 4가정 전부 미통과"
    return f"부분 지지 — 4가정 중 {n_passed}건 통과, 미통과: {failed}"


def _assumption_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["assumption"], y=table["value"], name="관측값"))
    fig.add_trace(go.Scatter(x=table["assumption"], y=table["threshold"],
                             mode="markers", name="임계"))
    fig.update_layout(title="v4 4가정 통계량 vs 임계", xaxis_title="가정", yaxis_title="값")
    return fig


@register_hypothesis("modeling_v4_assumptions", "modeling_v4 framework 4가정")
def modeling_v4_assumptions(inputs: AnalysisInputs) -> AnalysisResult:
    table = assumption_table(inputs.daily, inputs.receipts, inputs.item_to_category)
    basket = basket_composition_summary(inputs.receipts)
    return AnalysisResult(
        name="modeling_v4_assumptions", kind=KIND_HYPOTHESIS,
        title="modeling_v4 framework 4가정",
        tables=[("assumptions", table), ("basket_composition", basket)],
        figures=[_assumption_fig(table)],
        verdict=v4_verdict(table),
        notes=["시즌 제외 품목은 filter_seasonal로 제거(광교 기준) — 계절 특수품이 "
               "비율 안정성을 왜곡하지 않게 한다.",
               "임계값은 modeling_v4 설계 문서 기준의 실무 기준선이며 통계적 검정이 아니다."],
    )
```

**주의:** `basket_composition_summary(lines, closing_category=None)`가 요구하는 `lines` 컬럼을 확인해 `inputs.receipts`로 충분한지 검증하고, 부족하면 `category_id`를 붙여 전달한다.

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/analysis_lab/test_handlers_basket.py -v` / Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/lab/handlers/basket.py src/bakery/analysis/lab/handlers/__init__.py \
        tests/analysis_lab/test_handlers_basket.py
git commit -m "feat(analysis-lab): modeling_v4_assumptions 핸들러(4가정)"
```

---

### Task 16: `month_dow_adjust` (형태③)

**Files:**
- Create: `src/bakery/analysis/month_dow.py`
- Modify: `src/bakery/analysis/lab/handlers/calendar_bias.py`
- Modify: `scripts/verify_month_dow_adjust.py` (프리미티브 위임)
- Test: `tests/test_month_dow.py`, `tests/analysis_lab/test_handlers_calendar_bias.py` (추가)

**Interfaces:**
- Consumes: `AnalysisInputs.category_daily` (`sold_total_unit`, `sold_closing`, `sold_normal_unit`, `adjusted_demand_unit` 컬럼 보유)
- Produces:
  - `month_dow_matrix(series, value_column) -> pd.DataFrame` (index=month 1~12, columns=요일 월~일, 값=일평균)
  - `adjust_effect_table(series) -> pd.DataFrame` (cols: month, dow, raw_mean, adjusted_mean, closing_mean, delta, delta_pct)
  - `MONTH_DOW_VALUE_COLUMNS = ("sold_total_unit", "adjusted_demand_unit", "sold_closing")`
  - `month_dow_adjust(inputs) -> AnalysisResult` (`needs_single_store=True` — `category_daily`가 광교 전용)

**출처와의 차이(명시):** 스크립트는 `data/internal/v2/sales.parquet` 직독 + `ALPHA = 0.5`를 썼다. 핸들러는 canonical `category_daily`(헌장 α=0.8)를 쓴다. 따라서 **수치 등가가 아니다** — 구조(12×7 매트릭스 형태·adjust 방향)만 대조하고 근거를 note에 남긴다.

- [ ] **Step 1: 테스트 작성**

`tests/test_month_dow.py`:

```python
import pandas as pd
import pytest

from bakery.analysis.month_dow import (
    MONTH_DOW_VALUE_COLUMNS, adjust_effect_table, month_dow_matrix,
)

DOW_LABELS = ["월", "화", "수", "목", "금", "토", "일"]


def _series():
    """2025-01-06(월) ~ 01-12(일) 1주 + 02-03(월) 1일."""
    dates = list(pd.date_range("2025-01-06", periods=7, freq="D")) + [pd.Timestamp("2025-02-03")]
    return pd.DataFrame({
        "date": dates,
        "sold_total_unit": [100, 110, 120, 130, 140, 200, 190, 300],
        "sold_closing": [10, 10, 10, 10, 10, 20, 20, 30],
        "adjusted_demand_unit": [92, 102, 112, 122, 132, 184, 174, 276],
    })


def test_value_columns_declared():
    assert MONTH_DOW_VALUE_COLUMNS == ("sold_total_unit", "adjusted_demand_unit",
                                       "sold_closing")


def test_month_dow_matrix_shape_and_values():
    matrix = month_dow_matrix(_series(), "sold_total_unit")
    assert matrix.columns.tolist() == DOW_LABELS
    assert matrix.index.tolist() == [1, 2]
    assert matrix.loc[1, "월"] == 100.0
    assert matrix.loc[1, "일"] == 190.0
    assert matrix.loc[2, "월"] == 300.0
    assert pd.isna(matrix.loc[2, "화"])          # 2월 화요일 관측 없음


def test_adjust_effect_table_exact():
    table = adjust_effect_table(_series())
    monday_jan = table[(table["month"] == 1) & (table["dow"] == 0)].iloc[0]
    assert monday_jan["raw_mean"] == 100.0
    assert monday_jan["adjusted_mean"] == 92.0
    assert monday_jan["closing_mean"] == 10.0
    assert monday_jan["delta"] == -8.0
    assert monday_jan["delta_pct"] == pytest.approx(-8.0)


def test_adjust_effect_table_covers_every_observed_cell():
    table = adjust_effect_table(_series())
    assert len(table) == 8          # 1월 7요일 + 2월 월요일


def test_script_delegates_to_primitive():
    import sys
    sys.path.insert(0, "scripts")
    import verify_month_dow_adjust

    assert verify_month_dow_adjust.month_dow_matrix is month_dow_matrix
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: 'bakery.analysis.month_dow'`

- [ ] **Step 3: 구현** — `src/bakery/analysis/month_dow.py`:

```python
"""월 × 요일 12×7 매트릭스 — 마감 조정(adjust) 전후 비교.

출처: scripts/verify_month_dow_adjust.py. 스크립트는 레거시 sales.parquet 직독 +
α=0.5였고, 이 모듈은 (date, 값) 시리즈를 인자로 받는 순수함수라 소스/α에 무관하다.
"""
from __future__ import annotations

import pandas as pd

DOW_LABELS: tuple[str, ...] = ("월", "화", "수", "목", "금", "토", "일")
MONTH_DOW_VALUE_COLUMNS: tuple[str, ...] = ("sold_total_unit", "adjusted_demand_unit",
                                            "sold_closing")
RAW_COLUMN = "sold_total_unit"
ADJUSTED_COLUMN = "adjusted_demand_unit"
CLOSING_COLUMN = "sold_closing"


def _with_axes(series: pd.DataFrame) -> pd.DataFrame:
    frame = series.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month
    frame["dow"] = frame["date"].dt.dayofweek
    return frame


def month_dow_matrix(series: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """월(행) × 요일(열) 일평균 매트릭스. 관측 없는 칸은 NaN."""
    frame = _with_axes(series)
    matrix = frame.groupby(["month", "dow"])[value_column].mean().unstack("dow")
    matrix = matrix.reindex(columns=range(len(DOW_LABELS)))
    matrix.columns = list(DOW_LABELS)
    return matrix


def adjust_effect_table(series: pd.DataFrame) -> pd.DataFrame:
    """월×요일 칸별 raw vs adjusted 차이 — 마감 조정이 어느 칸을 얼마나 낮추는가."""
    frame = _with_axes(series)
    table = (frame.groupby(["month", "dow"])
             .agg(raw_mean=(RAW_COLUMN, "mean"),
                  adjusted_mean=(ADJUSTED_COLUMN, "mean"),
                  closing_mean=(CLOSING_COLUMN, "mean"))
             .reset_index())
    table["delta"] = table["adjusted_mean"] - table["raw_mean"]
    table["delta_pct"] = (table["delta"] / table["raw_mean"] * 100).where(
        table["raw_mean"] > 0, 0.0)
    return table
```

`scripts/verify_month_dow_adjust.py`를 wrapper로 교체:

```python
"""검증: 광교 월 × 요일 12×7 매트릭스 — 마감 조정 전후 비교.

계산은 `bakery.analysis.month_dow`로 옮겼다(Phase 6). 현 vintage 실행은
`bakery analysis-run`(hypotheses.month_dow_adjust)을 쓴다 — 이 wrapper는
canonical category_daily로 표를 print한다(레거시 α=0.5 직독 경로는 폐기).

실행: uv run python scripts/verify_month_dow_adjust.py
"""
from __future__ import annotations

from bakery.analysis.month_dow import (
    ADJUSTED_COLUMN, RAW_COLUMN, adjust_effect_table, month_dow_matrix,
)
from bakery.features.category_aggregate import build_category_daily


def main() -> None:
    series = build_category_daily(alpha=0.8).df
    print("=== raw (sold_total_unit) — 월 × 요일 일평균 ===")
    print(month_dow_matrix(series, RAW_COLUMN).round(1).to_string())
    print("\n=== adjusted_demand_unit — 월 × 요일 일평균 ===")
    print(month_dow_matrix(series, ADJUSTED_COLUMN).round(1).to_string())
    print("\n=== 조정 효과(delta_pct 하위 10칸) ===")
    table = adjust_effect_table(series).sort_values("delta_pct")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
```

`handlers/calendar_bias.py`에 핸들러 추가:

```python
from bakery.analysis.month_dow import (
    ADJUSTED_COLUMN, RAW_COLUMN, adjust_effect_table, month_dow_matrix,
)

_NOTE_MONTH_DOW_SOURCE = ("출처 스크립트는 레거시 sales.parquet 직독 + α=0.5였다. "
                          "여기는 canonical category_daily + 헌장 α — 수치 등가가 아니라 "
                          "구조/방향만 비교 가능하다.")


def month_dow_verdict(table: pd.DataFrame) -> str:
    """조정이 특정 월×요일 칸에 편중되면 그 축에 구조가 있다는 신호."""
    worst = table.loc[table["delta_pct"].idxmin()]
    spread = float(table["delta_pct"].max() - table["delta_pct"].min())
    return (f"조정 효과 최대 칸: {int(worst['month'])}월 "
            f"{DOW_LABELS[int(worst['dow'])]}요일 {worst['delta_pct']:.1f}%, "
            f"칸간 편차 {spread:.1f}%p")


def _heatmap_fig(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns.tolist(),
                               y=matrix.index.tolist(), colorscale="Blues"))
    fig.update_layout(title=title, xaxis_title="요일", yaxis_title="월")
    return fig


@register_hypothesis("month_dow_adjust", "월×요일 매트릭스 — 마감 조정 전후",
                     needs_single_store=True)
def month_dow_adjust(inputs: AnalysisInputs) -> AnalysisResult:
    series = inputs.category_daily
    table = adjust_effect_table(series)
    raw_matrix = month_dow_matrix(series, RAW_COLUMN)
    adjusted_matrix = month_dow_matrix(series, ADJUSTED_COLUMN)
    return AnalysisResult(
        name="month_dow_adjust", kind=KIND_HYPOTHESIS,
        title="월×요일 매트릭스 — 마감 조정 전후",
        tables=[("effect", table),
                ("raw_matrix", raw_matrix.reset_index()),
                ("adjusted_matrix", adjusted_matrix.reset_index())],
        figures=[_heatmap_fig(raw_matrix, "raw 일평균 (월×요일)"),
                 _heatmap_fig(adjusted_matrix, "adjusted 일평균 (월×요일)")],
        verdict=month_dow_verdict(table),
        notes=[_NOTE_MONTH_DOW_SOURCE],
    )
```

`calendar_bias.py` 상단 import에 `DOW_LABELS`를 추가한다(`from bakery.analysis.month_dow import DOW_LABELS, ...`).

- [ ] **Step 4: 테스트 추가** — `tests/analysis_lab/test_handlers_calendar_bias.py`에 추가:

```python
def test_month_dow_verdict_names_worst_cell():
    from bakery.analysis.lab.handlers.calendar_bias import month_dow_verdict

    table = pd.DataFrame([
        {"month": 1, "dow": 0, "raw_mean": 100.0, "adjusted_mean": 92.0,
         "closing_mean": 10.0, "delta": -8.0, "delta_pct": -8.0},
        {"month": 7, "dow": 5, "raw_mean": 200.0, "adjusted_mean": 196.0,
         "closing_mean": 5.0, "delta": -4.0, "delta_pct": -2.0},
    ])
    assert month_dow_verdict(table) == (
        "조정 효과 최대 칸: 1월 월요일 -8.0%, 칸간 편차 6.0%p")
```

- [ ] **Step 5: 통과 확인** — Run: `uv run pytest tests/test_month_dow.py tests/analysis_lab/test_handlers_calendar_bias.py -v` / Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/analysis/month_dow.py src/bakery/analysis/lab/handlers/calendar_bias.py \
        scripts/verify_month_dow_adjust.py tests/test_month_dow.py \
        tests/analysis_lab/test_handlers_calendar_bias.py
git commit -m "feat(analysis-lab): month_dow_adjust 이식 + month_dow 프리미티브 승격"
```

---

### Task 17: preds 의존 3종 — `seasonal_bias` / `weather_bias` / `event_prior_validation` (형태④)

**Files:**
- Modify: `src/bakery/analysis/lab/handlers/model_bias.py`
- Modify: `src/bakery/analysis/lab/handlers/__init__.py` (변경 없음 — 이미 등록)
- Modify: `tests/analysis_lab/test_handlers_model_bias.py`
- Modify: `tests/analysis_lab/test_cli_analysis.py` (shipped YAML xfail 제거)
- Test: `tests/test_pred_bias_axes.py`

**Interfaces:**
- Consumes: `AnalysisInputs.predictions`/`.calendar`, `bakery.data.paths.dataset("weather_observed")`, `bakery.harness.event_priors.STORE_EVENT_PRIORS`
- Produces (신규 프리미티브 `src/bakery/analysis/pred_bias.py`):
  - `wpe_percent(preds) -> float` — `(Σexpected − Σactual)/Σ|actual| × 100` (음수=과소예측)
  - `stockout_rate_percent(preds) -> float` — `(actual > production).mean() × 100`
  - `bias_by_axis(preds, axis_column) -> pd.DataFrame` (cols: axis 값, n, wpe, stockout_rate)
  - `robust_z(values) -> pd.Series` — `0.6745·(x − median)/MAD`
  - `segment_contrast(preds, mask, *, n_boot=2000, seed=42) -> dict` — 세그먼트 vs 여집합 WPE 차이 부트스트랩 CI
  - `EXTREME_THRESHOLDS: dict[str, float]` — `{"heatwave_max_ta": 33.0, "coldwave_min_ta": -10.0, "heavy_rain_mm": 30.0}`
  - `SUMMER_MONTHS = (6, 7, 8, 9)`, `WINTER_MONTHS = (12, 1, 2)`, `WEEKEND_DOW = (5, 6)`
- 핸들러 3개: `seasonal_bias`, `weather_bias`, `event_prior_validation` (모두 `needs_predictions=True`)

**event_prior_validation 재정의(중요, 경계 준수):** 출처 `scripts/verify_event_prior.py`는 `model.predict_expected()`를 호출해 base vs prior A/B를 했다 — 이 레이어에서는 **모델 실행이 금지**다. 두 모드로 대체한다:
1. `params.event_prior_validation.baseline_predictions`에 `layers: []`로 돌린 harness preds 경로가 주어지면 **artifact 대 artifact A/B**(이벤트일 WPE 개선폭)를 계산한다.
2. 없으면 **단일 artifact 모드** — 이벤트일 vs 비이벤트일 WPE/매진률 대조만 한다(prior 효과가 이미 반영된 상태의 잔여 편향 진단).
리포트 note와 verdict에 어느 모드인지 반드시 표기한다.

**★ baseline artifact는 이 태스크에서 반드시 만든다(Step 5b).** 만들지 않으면 이름은 "prior 검증"인데 항상 단일 artifact 모드로만 출하되어 실제로는 "이벤트일 잔여 편향"을 재게 된다. `experiments/analysis_gwangyo.yaml`에 그 경로를 박아 A/B가 기본 동작이 되게 한다.

**`STORE_EVENT_PRIORS` 실측 구조(2026-07-28 확인)** — 추측 금지:
- 키는 **영문 라벨** `gwangyo` / `samsung` / `mecenatpolis` / `gwanghwamun` (한글명 아님 → `inputs.prior_key` 사용)
- `events` = `{이벤트명: (월, 일)}` 양력 고정 (예: `{"xmas": (12, 25), "childrens": (5, 5)}`)
- `lunar_events` = `{이벤트명: {연도: "YYYY-MM-DD"}}` (예: `{"chuseok": {2021: "2021-09-21", ...}}`)
- 광교만 `childrens` + `chuseok`가 등록돼 있다(OOS 순개선 확인된 것만 opt-in)

- [ ] **Step 1: 프리미티브 테스트 작성**

`tests/test_pred_bias_axes.py`:

```python
import numpy as np
import pandas as pd
import pytest

from bakery.analysis.pred_bias import (
    EXTREME_THRESHOLDS, SUMMER_MONTHS, WEEKEND_DOW, WINTER_MONTHS,
    bias_by_axis, robust_z, segment_contrast, stockout_rate_percent, wpe_percent,
)


def _preds():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-11", "2025-01-12"]),
        "actual": [100.0, 100.0, 200.0, 200.0],
        "expected": [90.0, 110.0, 180.0, 220.0],
        "production": [95.0, 120.0, 190.0, 250.0],
    })


def test_constants():
    assert SUMMER_MONTHS == (6, 7, 8, 9)
    assert WINTER_MONTHS == (12, 1, 2)
    assert WEEKEND_DOW == (5, 6)
    assert EXTREME_THRESHOLDS == {"heatwave_max_ta": 33.0, "coldwave_min_ta": -10.0,
                                  "heavy_rain_mm": 30.0}


def test_wpe_percent_exact():
    # Σexpected 600, Σactual 600 → 0%
    assert wpe_percent(_preds()) == 0.0
    biased = _preds().assign(expected=[80.0, 80.0, 160.0, 160.0])
    # (480 − 600)/600 × 100 = −20%
    assert wpe_percent(biased) == -20.0


def test_stockout_rate_percent_exact():
    # actual > production: 100>95 (True), 100>120 (False), 200>190 (True), 200>250 (False)
    assert stockout_rate_percent(_preds()) == 50.0


def test_bias_by_axis_groups_and_computes():
    # fixture 날짜: 01-06(월=0), 01-07(화=1), 01-11(토=5), 01-12(일=6) — 각 1건
    preds = _preds().assign(dow=lambda d: d["date"].dt.dayofweek)
    table = bias_by_axis(preds, "dow").set_index("dow")
    assert set(table.index) == {0, 1, 5, 6}
    assert table["n"].tolist() == [1, 1, 1, 1]
    # 월요일: expected 90, actual 100 → (90−100)/100×100 = −10%
    assert table.loc[0, "wpe"] == -10.0
    # 화요일: expected 110 > actual 100 → +10%
    assert table.loc[1, "wpe"] == 10.0
    # 매진률: 월 actual 100 > production 95 → 100%
    assert table.loc[0, "stockout_rate"] == 100.0
    assert table.loc[1, "stockout_rate"] == 0.0


def test_robust_z_is_zero_at_median():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = robust_z(values)
    assert z[2] == 0.0                            # median=3
    assert z[4] > 0.0


def test_robust_z_handles_zero_mad():
    z = robust_z(pd.Series([2.0, 2.0, 2.0]))
    assert z.tolist() == [0.0, 0.0, 0.0]


def test_segment_contrast_ci_is_deterministic():
    preds = _preds()
    mask = preds["date"].dt.dayofweek.isin(WEEKEND_DOW)
    first = segment_contrast(preds, mask, n_boot=50, seed=42)
    second = segment_contrast(preds, mask, n_boot=50, seed=42)
    assert first["wpe_diff"] == second["wpe_diff"]
    assert first["ci"].tolist() == second["ci"].tolist()
    assert first["n_segment"] == 2
    assert first["n_rest"] == 2
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL `ModuleNotFoundError: 'bakery.analysis.pred_bias'`

- [ ] **Step 3: 프리미티브 구현** — `src/bakery/analysis/pred_bias.py`:

```python
"""예측 편향 축 진단 프리미티브 — 이미 계산된 OOS preds만 소비한다.

출처: scripts/track3_seasonal_diagnose.py, scripts/track4_weather_diagnose.py.
모델을 실행하지 않는다 — (date, actual, expected, production) 프레임이 전부다.

WPE 부호 규약: (Σexpected − Σactual)/Σ|actual| × 100. 음수 = 과소예측(발주부족 방향).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUMMER_MONTHS: tuple[int, ...] = (6, 7, 8, 9)
WINTER_MONTHS: tuple[int, ...] = (12, 1, 2)
WEEKEND_DOW: tuple[int, ...] = (5, 6)
EXTREME_THRESHOLDS: dict[str, float] = {
    "heatwave_max_ta": 33.0,
    "coldwave_min_ta": -10.0,
    "heavy_rain_mm": 30.0,
}
SPIKE_Z = 3.0            # robust z 이상 = 산발 spike(체계적 성분과 분리)
N_BOOT = 2000
SEED = 42
_MAD_SCALE = 0.6745
_CI_PERCENTILES = (2.5, 97.5)


def wpe_percent(preds: pd.DataFrame) -> float:
    denom = preds["actual"].abs().sum()
    if denom == 0:
        return 0.0
    return float((preds["expected"] - preds["actual"]).sum() / denom * 100)


def stockout_rate_percent(preds: pd.DataFrame) -> float:
    """버퍼발주(production)가 뚫린 전체매진 비율 %."""
    if len(preds) == 0:
        return 0.0
    return float((preds["actual"] > preds["production"]).mean() * 100)


def bias_by_axis(preds: pd.DataFrame, axis_column: str) -> pd.DataFrame:
    """축(요일/월/계절/세그먼트)별 WPE + 매진률."""
    rows = []
    for value, group in preds.groupby(axis_column, observed=True):
        rows.append({axis_column: value, "n": int(len(group)),
                     "wpe": wpe_percent(group),
                     "stockout_rate": stockout_rate_percent(group)})
    return pd.DataFrame(rows)


def robust_z(values: pd.Series) -> pd.Series:
    """MAD 기반 robust z. MAD=0이면 0 시리즈(상수 입력)."""
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad == 0:
        fallback = float(values.std())
        if not fallback:
            return pd.Series(0.0, index=values.index)
        return _MAD_SCALE * (values - median) / fallback
    return _MAD_SCALE * (values - median) / mad


def segment_contrast(preds: pd.DataFrame, mask: pd.Series, *, n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    """세그먼트 vs 여집합 WPE 차이 + day-level 부트스트랩 95% CI."""
    segment, rest = preds[mask], preds[~mask]
    diff = wpe_percent(segment) - wpe_percent(rest)
    rng = np.random.default_rng(seed)
    seg_index, rest_index = segment.index.to_numpy(), rest.index.to_numpy()
    diffs = np.empty(n_boot)
    for index in range(n_boot):
        resampled_seg = segment.loc[rng.choice(seg_index, len(seg_index), replace=True)]
        resampled_rest = rest.loc[rng.choice(rest_index, len(rest_index), replace=True)]
        diffs[index] = wpe_percent(resampled_seg) - wpe_percent(resampled_rest)
    return {"wpe_diff": diff, "ci": np.percentile(diffs, _CI_PERCENTILES),
            "n_segment": int(len(segment)), "n_rest": int(len(rest))}


def is_signal(contrast: dict) -> bool:
    """CI가 0을 배제하면 신호, 포함하면 noise."""
    low, high = contrast["ci"]
    return bool(low > 0 or high < 0)
```

- [ ] **Step 4: 프리미티브 통과 확인** — Run: `uv run pytest tests/test_pred_bias_axes.py -v` / Expected: PASS (7 passed)

- [ ] **Step 5: 핸들러 3종 구현** — `handlers/model_bias.py`에 추가:

```python
from bakery.analysis.pred_bias import (
    EXTREME_THRESHOLDS, SUMMER_MONTHS, WEEKEND_DOW, WINTER_MONTHS,
    bias_by_axis, is_signal, segment_contrast, stockout_rate_percent, wpe_percent,
)

_NOTE_WPE_SIGN = "WPE 부호: (expected−actual)/Σ|actual|. 음수=과소예측(발주부족 방향)."
_SEASON_LABELS = {"summer": SUMMER_MONTHS, "winter": WINTER_MONTHS}


def _with_axes(preds: pd.DataFrame) -> pd.DataFrame:
    frame = preds.copy()
    dates = pd.to_datetime(frame["date"])
    frame["dow"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    frame["is_weekend"] = frame["dow"].isin(WEEKEND_DOW)
    return frame


def seasonal_bias_verdict(weekend: dict, summer: dict) -> str:
    parts = []
    for label, contrast in (("주말", weekend), ("여름", summer)):
        low, high = contrast["ci"]
        state = "신호" if is_signal(contrast) else "noise(CI 0 포함)"
        parts.append(f"{label} WPE 차 {contrast['wpe_diff']:+.2f}%p "
                     f"CI[{low:+.2f},{high:+.2f}] {state}")
    prefix = ("지지" if is_signal(weekend) or is_signal(summer) else "기각")
    return f"{prefix} — " + " / ".join(parts)


@register_hypothesis("seasonal_bias", "주말·여름 계절 편향(WPE 축 분해)",
                     needs_predictions=True)
def seasonal_bias(inputs: AnalysisInputs) -> AnalysisResult:
    preds = _with_axes(inputs.predictions)
    params = inputs.params_for("seasonal_bias")
    weekend = segment_contrast(preds, preds["is_weekend"], **params)
    summer = segment_contrast(preds, preds["month"].isin(SUMMER_MONTHS), **params)
    contrasts = pd.DataFrame([
        {"segment": "weekend", "wpe_diff": weekend["wpe_diff"],
         "ci_low": weekend["ci"][0], "ci_high": weekend["ci"][1],
         "n_segment": weekend["n_segment"], "is_signal": is_signal(weekend)},
        {"segment": "summer", "wpe_diff": summer["wpe_diff"],
         "ci_low": summer["ci"][0], "ci_high": summer["ci"][1],
         "n_segment": summer["n_segment"], "is_signal": is_signal(summer)},
    ])
    return AnalysisResult(
        name="seasonal_bias", kind=KIND_HYPOTHESIS, title="주말·여름 계절 편향(WPE 축 분해)",
        tables=[("by_dow", bias_by_axis(preds, "dow")),
                ("by_month", bias_by_axis(preds, "month")),
                ("contrasts", contrasts)],
        figures=[_axis_fig(bias_by_axis(preds, "dow"), "dow", "요일별 WPE"),
                 _axis_fig(bias_by_axis(preds, "month"), "month", "월별 WPE")],
        verdict=seasonal_bias_verdict(weekend, summer),
        notes=[_NOTE_ENGINE, _NOTE_WPE_SIGN],
    )


def _axis_fig(table: pd.DataFrame, axis: str, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=table[axis].astype(str), y=table["wpe"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title=title, xaxis_title=axis, yaxis_title="WPE %")
    return fig


def _weather_segments(preds: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(weather, on="date", how="left")
    merged["is_heatwave"] = merged["maxTa"] >= EXTREME_THRESHOLDS["heatwave_max_ta"]
    merged["is_coldwave"] = merged["minTa"] <= EXTREME_THRESHOLDS["coldwave_min_ta"]
    merged["is_heavy_rain"] = merged["sumRn"] >= EXTREME_THRESHOLDS["heavy_rain_mm"]
    return merged


_EXTREME_SEGMENTS: tuple[tuple[str, tuple[int, ...] | None], ...] = (
    ("is_heatwave", SUMMER_MONTHS),      # 비교군 = 동계절 비극한일
    ("is_coldwave", WINTER_MONTHS),
    ("is_heavy_rain", None),             # 강수는 전 계절
)


def _empty_contrast_row(segment: str, n_segment: int) -> dict:
    """세그먼트/여집합 한쪽이 비면 대조 불가 — noise로 두되 n을 남겨 은폐하지 않는다."""
    return {"segment": segment, "wpe_diff": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_segment": n_segment, "is_signal": False}


def _extreme_contrasts(merged: pd.DataFrame) -> pd.DataFrame:
    """극한날씨 3세그먼트 × 동계절 비교군 WPE 대조."""
    rows = []
    for segment, season in _EXTREME_SEGMENTS:
        scope = merged if season is None else merged[merged["month"].isin(season)]
        mask = scope[segment].fillna(False)
        if mask.sum() == 0 or (~mask).sum() == 0:
            rows.append(_empty_contrast_row(segment, int(mask.sum())))
            continue
        contrast = segment_contrast(scope, mask)
        rows.append({"segment": segment, "wpe_diff": contrast["wpe_diff"],
                     "ci_low": contrast["ci"][0], "ci_high": contrast["ci"][1],
                     "n_segment": contrast["n_segment"],
                     "is_signal": is_signal(contrast)})
    return pd.DataFrame(rows)


def weather_bias_verdict(contrasts: pd.DataFrame) -> str:
    signals = contrasts[contrasts["is_signal"]]["segment"].tolist()
    if not signals:
        return ("기각 — 폭염/한파/강한비 전부 CI 0 포함(noise). "
                "극한날씨 전용 feature는 정당화되지 않음")
    return f"지지 — {signals} 세그먼트에서 CI 0 배제(체계적 편향)"


@register_hypothesis("weather_bias", "극한날씨(폭염·한파·강한비) 편향",
                     needs_predictions=True)
def weather_bias(inputs: AnalysisInputs) -> AnalysisResult:
    from bakery.data import paths

    weather = pd.read_parquet(paths.dataset("weather_observed"))
    weather["date"] = pd.to_datetime(weather["date"])
    for column in ("maxTa", "minTa", "sumRn"):
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    station = inputs.params_for("weather_bias").get("station_id")
    if station is not None:
        weather = weather[weather["station_id"] == station]
    merged = _weather_segments(_with_axes(inputs.predictions),
                              weather[["date", "maxTa", "minTa", "sumRn"]])
    contrasts = _extreme_contrasts(merged)
    fig = go.Figure(go.Bar(
        x=contrasts["segment"], y=contrasts["wpe_diff"],
        error_y=dict(type="data", symmetric=False,
                     array=contrasts["ci_high"] - contrasts["wpe_diff"],
                     arrayminus=contrasts["wpe_diff"] - contrasts["ci_low"])))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="극한날씨 세그먼트 WPE 차(동계절 비극한 대비)",
                      yaxis_title="WPE 차 %p")
    return AnalysisResult(
        name="weather_bias", kind=KIND_HYPOTHESIS, title="극한날씨(폭염·한파·강한비) 편향",
        tables=[("contrasts", contrasts)], figures=[fig],
        verdict=weather_bias_verdict(contrasts),
        notes=[_NOTE_ENGINE, _NOTE_WPE_SIGN,
               f"임계: {EXTREME_THRESHOLDS} / 비교군은 동계절 비극한일이다."],
    )


def _baseline_segment_table(baseline_path, event_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """prior 없는 baseline artifact를 같은 세그먼트 축으로 집계해 A/B 대조군을 만든다."""
    baseline = pd.read_csv(baseline_path)
    baseline["date"] = pd.to_datetime(baseline["date"])
    baseline["segment"] = np.where(baseline["date"].isin(event_dates), "event", "non_event")
    return bias_by_axis(baseline, "segment").rename(
        columns={"wpe": "wpe_baseline", "stockout_rate": "stockout_rate_baseline",
                 "n": "n_baseline"})


def event_prior_verdict(table: pd.DataFrame, *, is_ab_mode: bool) -> str:
    event = table[table["segment"] == "event"].iloc[0]
    if is_ab_mode:
        return (f"A/B 모드 — 이벤트일 WPE {event['wpe']:+.2f}% "
                f"(baseline {event['wpe_baseline']:+.2f}%), "
                f"개선 {event['wpe_baseline'] - event['wpe']:+.2f}%p")
    non_event = table[table["segment"] == "non_event"].iloc[0]
    return (f"단일 artifact 모드 — 이벤트일 WPE {event['wpe']:+.2f}% vs "
            f"비이벤트일 {non_event['wpe']:+.2f}% "
            f"(prior 적용 후 잔여 편향; base 대비 개선폭은 baseline preds 필요)")


@register_hypothesis("event_prior_validation", "이벤트 prior 적용 후 이벤트일 정확도",
                     needs_predictions=True)
def event_prior_validation(inputs: AnalysisInputs) -> AnalysisResult:
    from bakery.harness.event_priors import STORE_EVENT_PRIORS

    params = inputs.params_for("event_prior_validation")
    preds = _with_axes(inputs.predictions)
    years = range(int(preds["date"].dt.year.min()), int(preds["date"].dt.year.max()) + 1)
    event_dates = event_dates_for(inputs.prior_key, years, STORE_EVENT_PRIORS)
    preds["segment"] = np.where(preds["date"].isin(event_dates), "event", "non_event")
    table = bias_by_axis(preds, "segment")
    baseline_path = params.get("baseline_predictions")
    is_ab_mode = baseline_path is not None and Path(baseline_path).exists()
    if is_ab_mode:
        table = table.merge(_baseline_segment_table(baseline_path, event_dates),
                            on="segment", how="left")
    fig = go.Figure(go.Bar(x=table["segment"], y=table["wpe"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="이벤트일 vs 비이벤트일 WPE", yaxis_title="WPE %")
    return AnalysisResult(
        name="event_prior_validation", kind=KIND_HYPOTHESIS,
        title="이벤트 prior 적용 후 이벤트일 정확도",
        tables=[("by_segment", table)], figures=[fig],
        verdict=event_prior_verdict(table, is_ab_mode=is_ab_mode),
        notes=[_NOTE_ENGINE, _NOTE_WPE_SIGN,
               ("모델을 실행하지 않으므로 base vs prior A/B는 baseline preds artifact"
                "(layers: [] 로 돌린 harness-run 산출)가 있을 때만 가능하다.")],
    )


def _solar_event_dates(events: dict, years: range) -> list[pd.Timestamp]:
    """{이벤트명: (월, 일)} → 연도별 날짜로 전개."""
    dates = []
    for month, day in events.values():
        dates += [pd.Timestamp(year=year, month=month, day=day) for year in years]
    return dates


def _lunar_event_dates(lunar_events: dict) -> list[pd.Timestamp]:
    """{이벤트명: {연도: 'YYYY-MM-DD'}} → 날짜 리스트."""
    dates = []
    for per_year in lunar_events.values():
        dates += [pd.Timestamp(value) for value in per_year.values()]
    return dates


def event_dates_for(prior_key: str, years: range, store_event_priors: dict) -> pd.DatetimeIndex:
    """등록된 prior 이벤트일만(공휴일 전체가 아니다 — prior가 실제로 손대는 날짜)."""
    config = store_event_priors.get(prior_key, {})
    dates = _solar_event_dates(config.get("events") or {}, years)
    dates += _lunar_event_dates(config.get("lunar_events") or {})
    return pd.DatetimeIndex(sorted(set(dates)))
```

`model_bias.py` 상단 import에 `import numpy as np`, `from pathlib import Path` 추가.

**주의:** `event_dates_for`는 **등록된 prior 이벤트일만** 잡는다(공휴일 전체를 합집합하면 prior가 손대지 않은 날까지 "event"로 라벨링되어 A/B가 희석된다).

- [ ] **Step 6: 핸들러 테스트 추가** — `tests/analysis_lab/test_handlers_model_bias.py`에 추가:

```python
def test_seasonal_bias_verdict_rejects_when_both_noise():
    from bakery.analysis.lab.handlers.model_bias import seasonal_bias_verdict
    import numpy as np

    noise = {"wpe_diff": 0.5, "ci": np.array([-1.0, 2.0]), "n_segment": 100, "n_rest": 900}
    assert seasonal_bias_verdict(noise, noise) == (
        "기각 — 주말 WPE 차 +0.50%p CI[-1.00,+2.00] noise(CI 0 포함) / "
        "여름 WPE 차 +0.50%p CI[-1.00,+2.00] noise(CI 0 포함)")


def test_weather_bias_verdict_rejects_when_no_signal():
    from bakery.analysis.lab.handlers.model_bias import weather_bias_verdict

    contrasts = pd.DataFrame([{"segment": s, "is_signal": False}
                              for s in ("is_heatwave", "is_coldwave", "is_heavy_rain")])
    assert weather_bias_verdict(contrasts) == (
        "기각 — 폭염/한파/강한비 전부 CI 0 포함(noise). "
        "극한날씨 전용 feature는 정당화되지 않음")


def test_event_prior_verdict_single_artifact_mode():
    from bakery.analysis.lab.handlers.model_bias import event_prior_verdict

    table = pd.DataFrame([{"segment": "event", "n": 20, "wpe": -3.0, "stockout_rate": 10.0},
                          {"segment": "non_event", "n": 980, "wpe": -0.5,
                           "stockout_rate": 5.0}])
    assert event_prior_verdict(table, is_ab_mode=False) == (
        "단일 artifact 모드 — 이벤트일 WPE -3.00% vs 비이벤트일 -0.50% "
        "(prior 적용 후 잔여 편향; base 대비 개선폭은 baseline preds 필요)")


def test_event_dates_for_expands_solar_and_lunar():
    from bakery.analysis.lab.handlers.model_bias import event_dates_for

    priors = {"gwangyo": {"events": {"xmas": (12, 25), "childrens": (5, 5)},
                          "lunar_events": {"chuseok": {2024: "2024-09-17",
                                                       2025: "2025-10-06"}}}}
    dates = event_dates_for("gwangyo", range(2024, 2026), priors)
    assert dates.tolist() == [pd.Timestamp("2024-05-05"), pd.Timestamp("2024-09-17"),
                              pd.Timestamp("2024-12-25"), pd.Timestamp("2025-05-05"),
                              pd.Timestamp("2025-10-06"), pd.Timestamp("2025-12-25")]


def test_event_dates_for_unknown_key_is_empty():
    from bakery.analysis.lab.handlers.model_bias import event_dates_for

    assert event_dates_for("nope", range(2024, 2025), {}).tolist() == []


def test_all_three_registered_as_needs_predictions():
    load_handlers()
    for name in ("seasonal_bias", "weather_bias", "event_prior_validation"):
        assert HYPOTHESES[name].needs_predictions is True, name
```

- [ ] **Step 6b: baseline preds artifact 생성 + YAML 배선 (A/B 활성화)**

`event_prior_validation`이 단일 artifact 모드로만 출하되지 않게, `layers: []`(prior 없음)로
harness를 한 번 돌려 baseline preds를 만든다.

```bash
cat > experiments/gwangyo_no_prior.yaml <<'YAML'
# event_prior 없는 baseline — analysis-run의 event_prior_validation A/B 대조군
name: gwangyo_no_prior
data:
  source: real
  store: store_gw01
forecaster: category_total
layers: []
event_priors: null
YAML
uv run bakery harness-run experiments/gwangyo_no_prior.yaml --out reports
ls -la reports/gwangyo_no_prior/category_total/predictions.csv
```

`experiments/analysis_gwangyo.yaml`에 params를 추가한다:

```yaml
params:
  event_prior_validation:
    baseline_predictions: reports/gwangyo_no_prior/category_total/predictions.csv
```

**주의:** `layers: []` + `event_priors: null`은 harness `config._enforce`에서 경고를 내지
않아야 정상이다(경고 조건은 `event_prior` in layers). 만약 `events=None`이 xmas로 몰래
fallback되면 baseline이 오염되므로, PR#61에서 심은 guard가 살아있는지 확인한다:

```bash
uv run python -c "
import pandas as pd
base = pd.read_csv('reports/gwangyo_no_prior/category_total/predictions.csv')
prior = pd.read_csv('reports/gwangyo_default/category_total/predictions.csv')
m = base.merge(prior, on='date', suffixes=('_base','_prior'))
xmas = m[pd.to_datetime(m['date']).dt.strftime('%m-%d') == '12-25']
print(xmas[['date','expected_base','expected_prior']].to_string(index=False))
print('xmas에서 두 expected가 동일하면 baseline이 오염됐다(guard 확인 필요)')
"
```
Expected: 크리스마스 행에서 `expected_base != expected_prior`. 동일하면 **멈추고** guard를 확인한다.

- [ ] **Step 7: shipped YAML xfail 해제**

`experiments/analysis_gwangyo.yaml`의 모든 키가 이제 등록됐다. `tests/analysis_lab/test_cli_analysis.py`의 두 shipped-YAML 테스트에서 `xfail` 마커를 제거하고, `tests/analysis_lab/test_registry.py`의 남은 xfail도 제거한다.

Run: `uv run pytest tests/analysis_lab/ -v`
Expected: PASS, xfail 0개

- [ ] **Step 8: 전 항목 실행 스모크**

```bash
uv run bakery analysis-run experiments/analysis_gwangyo.yaml --out reports/analysis
open reports/analysis/analysis_gwangyo/analysis_report.html
```
Expected: 켠 항목이 전부 실행되고, 끈 항목은 `(off)`로 표기된다. `error:` 스킵이 있으면 원인을 고친다(runner가 항목을 격리하므로 리포트는 나온다 — **error를 통과로 착각하지 말 것**).

`hypotheses`를 전부 `true`로 바꾼 임시 YAML로 한 번 더 돌려 14종 전부가 error 없이 완주하는지 확인한다(substitution 때문에 수 분 소요).

- [ ] **Step 9: 커밋**

```bash
git add src/bakery/analysis/pred_bias.py src/bakery/analysis/lab/handlers/model_bias.py \
        tests/test_pred_bias_axes.py tests/analysis_lab/test_handlers_model_bias.py \
        tests/analysis_lab/test_cli_analysis.py tests/analysis_lab/test_registry.py
git commit -m "feat(analysis-lab): preds 의존 3종(seasonal/weather/event_prior) 이식 + pred_bias 프리미티브"
```

---

### Task 18: 문서 + 전체 검증 마감

**Files:**
- Create: `docs/phase6_analysis_layer.md`
- Modify: `.claude/CLAUDE.md` (디렉토리 섹션에 `analysis/lab` 한 줄)
- Modify: `TODO.md` (있으면 Phase 6 완료 반영)

- [ ] **Step 1: 문서 작성** — `docs/phase6_analysis_layer.md`:

```markdown
# 데이터분석 + 가설검증 레이어 (Phase 6)

`bakery analysis-run <yaml>` 단일 진입점. 입력 데이터 분석 5종 + 가설 14종을 YAML로 on/off.

## 사용법

```bash
uv run bakery analysis-run experiments/analysis_gwangyo.yaml            # 광교
uv run bakery analysis-run experiments/analysis_multistore.yaml         # 4매장
```
산출: `reports/analysis/<name>/analysis_report.html` + 항목별 `<name>__<table>.csv`.

## 경계 (harness-run과의 분리)

| | harness-run | analysis-run |
|---|---|---|
| 대상 | 예측 성능 | 입력 데이터 + 가설 |
| 모델 실행 | 함 | **안 함** |
| preds | 생산 | `spec.predictions`로 **읽기만** |

preds 의존 4종(`seasonal_bias`/`weather_bias`/`weekday_bias`/`event_prior_validation`)은
artifact가 없으면 리포트에 `(preds 필요 — 미실행)`으로 표기되고 실행되지 않는다.

## 이식 제외 (DEPRECATED)

`diag_anchor_gh`, `diag_chuseok_gh`, `diagnose_conformal_residual` — v5 conformal
구간예측 계열. 점추정+위험수치 전환으로 폐기됐다. spec에 쓰면 `AnalysisSpecError`.

## 이식 대조 기록

| 항목 | 게이트 형태 | 출처 수치 | 핸들러 수치 | 판정 일치 |
|---|---|---|---|---|
| demand_absorption | 동일 vintage 실측 | (Task 7 Step 9 결과 기입) | | |
| holiday_premium | frozen-input golden | 평일 n=71 median 1.25 | (기입) | |
| weekday_bias | frozen-input golden | base waste 0.047616 | (기입) | |
| closing_discount | 출처 대조 | (기입) | | |
| stockout_revenue | 출처 대조 | (기입) | | |

수치 등가가 **불가능한** 항목과 근거:
- `sales_distribution`/`category_mix`/`waste_*`/`overproduction_breakdown` — 레거시 eda01~05는
  `data/internal/v2/` 원본 시트를 다른 필터로 읽었다(FG_ITEM=='SS', beverage/etc 포함).
  canonical 재표현이므로 게이트는 구조 불변식(비중 합=1.0, 폐기율∈[0,1], 항등식 잔차).
- `month_dow_adjust` — 출처는 레거시 직독 + α=0.5, 핸들러는 canonical + 헌장 α=0.8.
- preds 의존 4종 — 출처는 비-canonical 엔진(`store_predictive_power`) 캐시. canonical
  harness preds와 수치가 다르므로 동결 artifact에만 수치 게이트를 걸고 그 외엔 방향/판정만.
```

작성 시 (기입) 자리에 Task 7·8·9·12·13의 Step에서 실제로 얻은 수치를 채운다. 빈칸으로 남기지 않는다.

- [ ] **Step 2: CLAUDE.md 갱신** — 디렉토리 섹션에 추가:

```markdown
- `src/bakery/analysis/lab/` — 데이터분석+가설검증 레이어 (`analysis-run` 진입점, registry/runner/report)
```

- [ ] **Step 3: 전체 스위트 + collection 확인**

Run: `uv run pytest --color=no 2>&1 | tail -5`
Expected: 전부 통과. `pytest | tail`은 exit code를 마스킹하므로 별도로 확인한다:

Run: `uv run pytest --color=no > /tmp/pytest_full.log 2>&1; echo "exit=$?"; tail -5 /tmp/pytest_full.log`
Expected: `exit=0`

Run: `uv run pytest --collect-only -q 2>&1 | tail -3` — collection 에러 0.

- [ ] **Step 4: 3축 리뷰**

`/review-triple`로 신규 코드(재사용성/품질/효율) 리뷰. 지적 사항 반영 후 재실행.

- [ ] **Step 5: 커밋 + PR**

```bash
git add docs/phase6_analysis_layer.md .claude/CLAUDE.md
git commit -m "docs: Phase 6 analysis 레이어 사용법 + 이식 대조 기록"
git push -u origin spec/phase6-analysis-layer
gh pr create --title "데이터분석+가설검증 레이어 (Phase 6) — analysis-run 단일 표면" --body "$(cat <<'BODY'
## 요약
`bakery analysis-run <yaml>` 단일 진입점으로 입력 데이터 분석 5종 + 가설 14종을 on/off 실행,
자기포함 HTML 1개로 출력. harness backbone(예측 평면)의 형제 표면.

## 설계 결정
- **모델 실행 금지** — preds 의존 4종은 harness-run artifact를 읽기만 하고, 없으면 리포트에
  `(preds 필요 — 미실행)`으로 표기 후 스킵
- **회귀 게이트 = 동일 vintage 실측** — docs 기록 수치를 golden으로 쓰지 않음(Phase 7 데이터
  편입으로 값 이동). frozen-input golden + 동일 입력 등가 두 형태만 사용
- **DEPRECATED 3종 차단** — conformal 계열은 spec에서 에러
- 레거시 EDA 5종은 canonical 재표현이라 수치 등가 불가 → 구조 불변식 게이트 (근거 문서화)

## 검증
- 전체 스위트 통과
- 이식 대조 기록: `docs/phase6_analysis_layer.md`
BODY
)"
```

- [ ] **Step 6: 메모리 갱신**

`auto-memory-save` 스킬로 로드맵 메모리(`project_harness_backbone.md`)의 6단계를 "스펙완료·플랜만 남음" → "완료(PR 번호)"로 갱신하고, `MEMORY.md` 한 줄 요약도 함께 고친다.

---

## Self-Review (플랜 작성자 체크)

**1. 스펙 커버리지**

| 스펙 섹션 | 구현 태스크 |
|---|---|
| §3 spec.py | Task 2 |
| §3 registry.py | Task 4 |
| §3 inputs.py | Task 3 |
| §3 runner.py | Task 4 |
| §3 report.py | Task 5 |
| §3 result.py | Task 1 |
| §3 CLI `analysis-run` | Task 5 |
| §4 YAML 스키마 + `experiments/analysis_*.yaml` | Task 2 + Task 5 |
| §5 데이터 분석 5종 | Task 6(category_mix), 10(sales_distribution), 11(waste 3종) |
| §5 가설 14종 | Task 7, 8, 9, 12(3), 13(2), 14, 15, 16, 17(3) |
| §5 DEPRECATED 3종 제외 | Task 2 (`DEPRECATED_ANALYSES`) + Task 18 문서 |
| §6 HTML 2대 섹션 + off 표기 | Task 5 |
| §7 성공기준 | Task 5(HTML), Task 4(독립 on/off), Task 3(입력만), Task 18(전체 스위트) |
| §8 리스크(이식 폭·추출 동작변화·소스 정합·자산 재사용·DEPRECATED) | 각 태스크의 게이트 + Task 18 문서 |

| §116 시각화 자산 재사용 | Global Constraints — **재사용 거부 결정 + 근거 3항** (`fig_to_div` 패턴만 재사용) |

**의도적 범위 조정 2건 (갭 아님, 근거 기록됨):**
1. 스펙 §7의 "입력 데이터만 사용(모델 예측값 미참조)"은 preds 의존 4종과 충돌한다. 사용자
   결정(2026-07-28)으로 **artifact 읽기만 허용**으로 경계를 재정의했다. `event_prior_validation`은
   base vs prior A/B가 원래 모델 실행을 요구했으므로, Task 17 Step 6b에서 `layers: []` harness
   실행으로 **baseline artifact를 만들어 artifact 대 artifact A/B를 기본 동작으로** 만든다
   (baseline이 없을 때만 단일 artifact 모드로 강등되고, 그 사실이 verdict에 표기된다).
2. `popularity_stockout`은 출처의 "옛 라벨 vs 새 라벨" A/B가 canonical에 옛(오염) 라벨이
   없어 불가능하다. **원시 인기 vs 매진 부스트 배분 순위** 비교로 재정의하고 title도
   "매진 부스트가 배분 순위를 재배열하는가"로 맞췄다(Task 13에 명시).

**2. Placeholder 스캔** — TBD/TODO/"적절히 처리"/"Task N과 유사" 없음. 모든 코드 스텝에
실제 코드 블록이 있다. **미확정 필드명 4곳은 2026-07-28에 실측해 플랜에 확정 반영했다**
(추측 fallback 체인 제거):
- `AlphaEstimate(alpha_low, alpha_high, a1, a2, a3_slope, note)` / `KinkResult(..., alpha, note)` /
  `DepthResult(..., alpha, note)` / `SurplusResult(n, slope, se, clearance_high, note)` — **A3에 alpha 없음**
- `run_discount_regime` → **dict**(`closing_share`/`closing_intensity`/`placebo`/`verdict`),
  `RegimeResult(beta, se, ci_low, ci_high, n, n_params, cut_date, ill_posed)` — **p_value 없음**
- `estimated_lost_demand` → `lost_units`(+ 로컬 `potential_demand`는 드롭),
  `stockout_hour_distribution` → 품목×요일 `stockout_hour_mean`(히스토그램 아님),
  `compute_popularity_signals` → `avg_daily_sold` 등 15컬럼(boosted 없음)
- `MnlResult(utilities, substitution, outflow_ratio)` / `NestedLogitResult(..., lambdas, ...)`
- `STORE_EVENT_PRIORS` 키=영문 라벨, `events={명:(월,일)}`, `lunar_events={명:{연도:날짜}}`

**3. 타입 일관성** — `AnalysisResult`/`SkippedResult`/`AnalysisReport`(Task 1), `Handler`(Task 4),
`AnalysisInputs`(Task 3), `AnalysisSpec`(Task 2) 이름이 Task 5~17에서 동일하게 쓰인다.
`register_data`/`register_hypothesis` 데코레이터 시그니처, `KIND_DATA`/`KIND_HYPOTHESIS`,
`REASON_*` 상수(4종)도 일관. `fig_to_div(fig, div_id, *, include_js, height)`는 harness와 동일 계약.
핸들러 테스트의 stub은 `tests/analysis_lab/conftest.py`의 `stub_inputs` 팩토리 하나로 통일했다.

**4. 게이트 무력화 방지** — `needs_single_store` 게이트(Task 4)로 광교 전용 소스
(`category_daily`)를 쓰는 `holiday_premium`/`month_dow_adjust`가 multistore spec에서
실행되지 않게 막았다. 게이트가 없으면 광교 수치가 4매장 분석으로 라벨링되는 조용한
오데이터가 된다.

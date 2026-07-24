# Harness Backbone Phase 1 (스파인) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cli.py 안에 갇힌 backtest 오케스트레이션 스파인을 `src/bakery/harness/`로 추출하고, YAML config 1파일=1실험으로 실행하는 단일 표면을 만든다.

**Architecture:** `src/bakery`(프리미티브)는 재구현하지 않고 호출만 한다. `harness/config.py`가 YAML→`ExperimentSpec`(canonical 강제+경고), `harness/registry.py`가 forecaster/layer 이름→객체(kind 분류), `harness/runner.py`가 spec→단계별 캐시 실행→`RunResult`. 완료 기준은 광교 lightgbm_v2 adjusted_demand backtest 숫자 재현.

**Tech Stack:** Python 3.12, pydantic v2(설정 검증), PyYAML, pandas, typer(CLI), pytest. 기존 `src/bakery/evaluation`(run_backtest/generate_time_splits/metrics)·`models`·`data.loader` 재사용.

## Global Constraints

- **측정헌장** (docs/superpowers/specs/2026-07-24-harness-backbone-design.md §5): target 기본=`adjusted_demand`; `potential_demand`는 `allow_deprecated: true` 없으면 ERROR; metric 기본 6종(wape, wpe, waste_rate, soldout_median, stockout_item_rate, shortfall_day_rate); MAPE 단독 지정 시 경고.
- **Time leakage 금지**: random split 스키마 거부. window.scheme ∈ {expanding, rolling}만.
- **재구현 금지**: harness는 `src/bakery` 심볼을 호출만 한다. 모델/피처/평가 로직을 harness에 복제하지 않는다.
- **테스트 단언**: 기대값 아는 단언은 정확값 `==`. 부동소수는 `pytest.approx`.
- **pytest 실행**: 이 repo addopts에 `-q` 있음. 카운트 필요 시 `uv run pytest --color=no` (추가 `-q` 금지 — passed 요약 사라짐).
- Phase 1 범위는 **스파인만**. report.py / viz / eda.py는 Phase 2~3 (본 계획 밖).

---

## File Structure

- `src/bakery/harness/__init__.py` — 공개 심볼 re-export (ExperimentSpec, load_spec, run_experiment, RunResult)
- `src/bakery/harness/config.py` — `ExperimentSpec`(pydantic) + `load_spec(path)` + canonical 검증/경고
- `src/bakery/harness/registry.py` — `ForecasterKind` enum + `resolve_forecasters(names, target)` + `resolve_layers(names)` + kind 조회
- `src/bakery/harness/runner.py` — `RunResult` dataclass + `run_experiment(spec, out_dir, cache_dir)` (단계 캐시)
- `src/bakery/features/enrich.py` — `enrich_daily(ds, variants)` (cli._enrich_if_needed 이동, 순환 import 방지)
- `src/bakery/cli.py` — `harness-run` 커맨드 추가 (thin wrapper) + enrich 재-export
- `experiments/gwangyo_default.yaml` — canonical 재현용 config
- `tests/harness/test_config.py` — 스키마·강제·경고
- `tests/harness/test_registry.py` — 이름→객체·kind 분류
- `tests/harness/test_runner.py` — 단계 캐시·재개·RunResult 구조
- `tests/harness/test_reproduce_backtest.py` — ★acceptance: 기존 backtest 숫자 재현

---

## Task 1: ExperimentSpec 스키마 + canonical 강제

**Files:**
- Create: `src/bakery/harness/__init__.py`
- Create: `src/bakery/harness/config.py`
- Test: `tests/harness/test_config.py`

**Interfaces:**
- Produces:
  - `class DataSpec(source: Literal["real","synthetic","parquet"], store: str | None = None)`
  - `class WindowSpec(scheme: Literal["expanding","rolling"] = "expanding", n_splits: int = 8, horizon_days: int = 7, step_days: int = 7)`
  - `class ExperimentSpec(name: str, data: DataSpec, target: str = "adjusted_demand", forecaster: list[str], layers: list[str] = [], window: WindowSpec, metrics: list[str], closing_alpha: float = 0.8, allow_deprecated: bool = False)`
  - `DEFAULT_FORECASTERS: list[str] = ["lightgbm_v2", "distributional_total"]`
  - `DEFAULT_LAYERS: list[str] = ["event_prior"]`
  - `DEFAULT_METRICS: list[str] = ["wape","wpe","waste_rate","soldout_median","stockout_item_rate","shortfall_day_rate"]`
  - `load_spec(path: str | Path) -> ExperimentSpec` — YAML 파싱 + 검증, 경고는 `warnings.warn`
  - `class SpecError(ValueError)` — 강제 규칙 위반

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_config.py
import warnings
import pytest
import yaml
from bakery.harness.config import (
    ExperimentSpec, load_spec, SpecError,
    DEFAULT_FORECASTERS, DEFAULT_LAYERS, DEFAULT_METRICS,
)


def _write(tmp_path, body: dict):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


def test_defaults_applied_when_omitted(tmp_path):
    spec = load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real", "store": "gwangyo"}}))
    assert spec.target == "adjusted_demand"
    assert spec.forecaster == DEFAULT_FORECASTERS
    assert spec.layers == DEFAULT_LAYERS
    assert spec.metrics == DEFAULT_METRICS
    assert spec.window.scheme == "expanding"
    assert spec.closing_alpha == 0.8


def test_potential_demand_rejected_without_override(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "target": "potential_demand"}
    with pytest.raises(SpecError, match="potential_demand"):
        load_spec(_write(tmp_path, body))


def test_potential_demand_allowed_with_flag(tmp_path):
    body = {"name": "x", "data": {"source": "synthetic"},
            "target": "potential_demand", "allow_deprecated": True}
    spec = load_spec(_write(tmp_path, body))
    assert spec.target == "potential_demand"


def test_random_split_scheme_rejected(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "window": {"scheme": "random"}}
    with pytest.raises(SpecError):
        load_spec(_write(tmp_path, body))


def test_mape_only_metrics_warns(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "metrics": ["mape"]}
    with pytest.warns(UserWarning, match="MAPE"):
        spec = load_spec(_write(tmp_path, body))
    assert spec.metrics == ["mape"]


def test_deprecated_forecaster_warns(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "forecaster": ["conformal_interval"]}
    with pytest.warns(UserWarning, match="DEPRECATED"):
        load_spec(_write(tmp_path, body))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_config.py --color=no`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.harness'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/harness/__init__.py
from bakery.harness.config import (
    ExperimentSpec, DataSpec, WindowSpec, load_spec, SpecError,
    DEFAULT_FORECASTERS, DEFAULT_LAYERS, DEFAULT_METRICS,
)

__all__ = [
    "ExperimentSpec", "DataSpec", "WindowSpec", "load_spec", "SpecError",
    "DEFAULT_FORECASTERS", "DEFAULT_LAYERS", "DEFAULT_METRICS",
]
```

```python
# src/bakery/harness/config.py
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_FORECASTERS: list[str] = ["lightgbm_v2", "distributional_total"]
DEFAULT_LAYERS: list[str] = ["event_prior"]
DEFAULT_METRICS: list[str] = [
    "wape", "wpe", "waste_rate", "soldout_median",
    "stockout_item_rate", "shortfall_day_rate",
]
DEPRECATED_FORECASTERS = {"conformal_interval"}


class SpecError(ValueError):
    """canonical 강제 규칙 위반."""


class DataSpec(BaseModel):
    source: Literal["real", "synthetic", "parquet"]
    store: str | None = None


class WindowSpec(BaseModel):
    scheme: Literal["expanding", "rolling"] = "expanding"
    n_splits: int = 8
    horizon_days: int = 7
    step_days: int = 7


class ExperimentSpec(BaseModel):
    name: str
    data: DataSpec
    target: str = "adjusted_demand"
    forecaster: list[str] = Field(default_factory=lambda: list(DEFAULT_FORECASTERS))
    layers: list[str] = Field(default_factory=lambda: list(DEFAULT_LAYERS))
    window: WindowSpec = Field(default_factory=WindowSpec)
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    closing_alpha: float = 0.8
    allow_deprecated: bool = False

    @field_validator("forecaster", mode="before")
    @classmethod
    def _wrap_single(cls, v):
        return [v] if isinstance(v, str) else v


def load_spec(path: str | Path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        spec = ExperimentSpec(**raw)
    except Exception as exc:  # pydantic ValidationError → SpecError
        raise SpecError(str(exc)) from exc
    _enforce(spec)
    return spec


def _enforce(spec: ExperimentSpec) -> None:
    if spec.target == "potential_demand" and not spec.allow_deprecated:
        raise SpecError(
            "target=potential_demand는 오염 소스라 금지. allow_deprecated: true 필요 "
            "(docs/.../2026-07-10-potential-demand-audit-design.md)"
        )
    if spec.metrics == ["mape"]:
        warnings.warn("MAPE 단독은 희소 품목에서 폭발한다. WAPE 병기 권장.", UserWarning)
    for name in spec.forecaster:
        if name in DEPRECATED_FORECASTERS:
            warnings.warn(f"{name}는 DEPRECATED forecaster.", UserWarning)
```

Note: `window.scheme: "random"`은 `Literal["expanding","rolling"]`이라 pydantic이 거부 → `load_spec`이 `SpecError`로 변환하므로 `test_random_split_scheme_rejected` 통과.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_config.py --color=no`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/harness/__init__.py src/bakery/harness/config.py tests/harness/test_config.py
git commit -m "feat(harness): ExperimentSpec YAML 스키마 + canonical 강제/경고"
```

---

## Task 2: Registry — 이름→객체 + kind 분류

**Files:**
- Create: `src/bakery/harness/registry.py`
- Modify: `src/bakery/harness/__init__.py` (re-export 추가)
- Test: `tests/harness/test_registry.py`

**Interfaces:**
- Consumes: `src/bakery/models`의 `SeasonalNaive`, `MovingAverage`, `GlobalLGBM`.
- Produces:
  - `class ForecasterKind(str, Enum)`: `POINT="point_forecaster"`, `DISTRIBUTIONAL="distributional"`, `COMPOSITE="composite_pipeline"`
  - `kind_of(name: str) -> ForecasterKind` — 이름의 kind 반환, 미등록 시 `KeyError`
  - `resolve_forecasters(names: list[str], *, target: str) -> list[Forecaster]` — Phase 1은 point_forecaster만 인스턴스화; distributional/composite는 `NotImplementedError("Phase 2+")`
  - `feature_set_of(name: str) -> str | None` — lightgbm 계열의 feature_set("v0".."v3") 반환, 그 외 None. runner가 enrich variant 파싱에 재사용(드리프트 방지).
  - `LAYER_NAMES: frozenset[str]` = `{"event_prior", "decision", "conformal_order"}`
  - `resolve_layers(names: list[str]) -> list[str]` — Phase 1은 이름 검증만(미등록 시 `KeyError`), 객체화는 Phase 2+

**확정 사실 (VERIFY 완료 2026-07-24):** `GlobalLGBM.__init__(params, y_col, feature_set, drop_groups)` — target 인자명은 **`y_col`** (not `target`). seed=42 고정(정확일치 유효). `resolve_forecasters`의 `target` 파라미터는 GlobalLGBM에 `y_col=target`으로 전달.

Note (taxonomy, 설계 §4): v0~v3=`lightgbm[_v1/_v2/_v3]`(POINT), `distributional_total`(DISTRIBUTIONAL), `category_v4`(COMPOSITE). Phase 1 acceptance는 point_forecaster 경로만 필요하므로 나머지 kind는 등록만 하고 인스턴스화는 다음 Phase로 미룬다 (YAGNI).

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_registry.py
import pytest
from bakery.harness.registry import (
    ForecasterKind, kind_of, resolve_forecasters, resolve_layers, LAYER_NAMES,
)
from bakery.models.lightgbm_regressor import GlobalLGBM


def test_kind_of_classifies_by_taxonomy():
    assert kind_of("seasonal_naive") == ForecasterKind.POINT
    assert kind_of("lightgbm_v2") == ForecasterKind.POINT
    assert kind_of("distributional_total") == ForecasterKind.DISTRIBUTIONAL
    assert kind_of("category_v4") == ForecasterKind.COMPOSITE


def test_kind_of_unknown_raises():
    with pytest.raises(KeyError):
        kind_of("nonexistent_model")


def test_resolve_point_forecaster_returns_globallgbm():
    fs = resolve_forecasters(["lightgbm_v2"], target="adjusted_demand")
    assert len(fs) == 1
    assert isinstance(fs[0], GlobalLGBM)
    assert fs[0].feature_set == "v2"


def test_feature_set_of():
    from bakery.harness.registry import feature_set_of
    assert feature_set_of("lightgbm_v2") == "v2"
    assert feature_set_of("lightgbm") == "v0"
    assert feature_set_of("seasonal_naive") is None


def test_resolve_distributional_not_yet_implemented():
    with pytest.raises(NotImplementedError, match="Phase 2"):
        resolve_forecasters(["distributional_total"], target="adjusted_demand")


def test_resolve_layers_validates_names():
    assert resolve_layers(["event_prior"]) == ["event_prior"]
    with pytest.raises(KeyError):
        resolve_layers(["bogus_layer"])
    assert "decision" in LAYER_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_registry.py --color=no`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.harness.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/harness/registry.py
from __future__ import annotations

from enum import Enum

from bakery.models.base import Forecaster
from bakery.models.lightgbm_regressor import GlobalLGBM
from bakery.models.moving_average import MovingAverage
from bakery.models.seasonal_naive import SeasonalNaive


class ForecasterKind(str, Enum):
    POINT = "point_forecaster"
    DISTRIBUTIONAL = "distributional"
    COMPOSITE = "composite_pipeline"


_KIND: dict[str, ForecasterKind] = {
    "seasonal_naive": ForecasterKind.POINT,
    "moving_average": ForecasterKind.POINT,
    "lightgbm": ForecasterKind.POINT,
    "lightgbm_v1": ForecasterKind.POINT,
    "lightgbm_v2": ForecasterKind.POINT,
    "lightgbm_v3": ForecasterKind.POINT,
    "distributional_total": ForecasterKind.DISTRIBUTIONAL,
    "category_v4": ForecasterKind.COMPOSITE,
}

LAYER_NAMES: frozenset[str] = frozenset({"event_prior", "decision", "conformal_order"})

_FEATURE_SET = {
    "lightgbm": "v0", "lightgbm_v1": "v1", "lightgbm_v2": "v2", "lightgbm_v3": "v3",
}


def kind_of(name: str) -> ForecasterKind:
    return _KIND[name]


def feature_set_of(name: str) -> str | None:
    """lightgbm 계열의 feature_set을 반환, 그 외는 None."""
    return _FEATURE_SET.get(name)


def resolve_forecasters(names: list[str], *, target: str) -> list[Forecaster]:
    out: list[Forecaster] = []
    for name in names:
        k = kind_of(name)
        if k is not ForecasterKind.POINT:
            raise NotImplementedError(f"{name} ({k.value})는 Phase 2+ 구현 대상")
        out.append(_build_point(name, target))
    return out


def _build_point(name: str, target: str) -> Forecaster:
    if name == "seasonal_naive":
        return SeasonalNaive(n_weeks=4)
    if name == "moving_average":
        return MovingAverage(window=28)
    return GlobalLGBM(feature_set=_FEATURE_SET[name], y_col=target)  # 인자명 y_col (VERIFY 완료)


def resolve_layers(names: list[str]) -> list[str]:
    for name in names:
        if name not in LAYER_NAMES:
            raise KeyError(name)
    return list(names)
```

(VERIFY 완료: `GlobalLGBM.__init__`은 `y_col=` kwarg를 받음. `_build_point`는 `y_col=target`으로 전달 — 위 코드에 반영됨.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_registry.py --color=no`
Expected: PASS (5 passed)

- [ ] **Step 5: Update __init__ + commit**

```python
# src/bakery/harness/__init__.py  — append to imports and __all__
from bakery.harness.registry import (
    ForecasterKind, kind_of, feature_set_of, resolve_forecasters, resolve_layers, LAYER_NAMES,
)
```
Add `"ForecasterKind", "kind_of", "feature_set_of", "resolve_forecasters", "resolve_layers", "LAYER_NAMES"` to `__all__`.

```bash
git add src/bakery/harness/registry.py src/bakery/harness/__init__.py tests/harness/test_registry.py
git commit -m "feat(harness): forecaster/layer registry + kind taxonomy"
```

---

## Task 3: Runner — 단계별 캐시 실행 + RunResult

**Files:**
- Create: `src/bakery/harness/runner.py`
- Create: `src/bakery/features/enrich.py` (cli._enrich_if_needed 이동)
- Modify: `src/bakery/harness/__init__.py`, `src/bakery/cli.py` (enrich 재-export)
- Test: `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: `ExperimentSpec`(Task 1), `resolve_forecasters`/`feature_set_of`(Task 2); `bakery.data.loader.load_dataset`, `bakery.evaluation.split.generate_time_splits`, `bakery.evaluation.backtest.run_backtest`, `bakery.features.category_aggregate.build_item_adjusted_demand`, `bakery.features.enrich.enrich_daily`.
- Produces:
  - `STAGES: tuple[str,...]` = `("load","features","fit_predict","evaluate")` (Phase 1은 fit·predict를 run_backtest가 함께 하므로 1단계로 묶고, report는 Phase 2)
  - `class RunResult` (dataclass): `name: str`, `predictions: pd.DataFrame`, `fold_metrics: pd.DataFrame`, `demand_col: str`, `resolved: dict` (config_resolved)
  - `run_experiment(spec: ExperimentSpec, *, out_dir: Path, cache_dir: Path | None = None) -> RunResult`
  - 캐시: 각 단계 산출을 `cache_dir/<stage>_<hash>.parquet|json`. hash = 그 단계에 영향 주는 spec 필드의 안정 직렬화(`_stage_key`). config 무변경+파일 존재 시 로드.

**⚠️ Phase 1 축소 (사용자 요청 "중간부터 실행"의 부분 구현):** `--from STAGE`/`--only STAGE`(임의 단계 재개)는 **Phase 2로 연기**한다. 이유: Phase 1엔 report/eda 단계가 없어 features 캐시 히트만으로 "중간부터"의 실질 이득(evaluate/report만 재실행)이 아직 성립하지 않음. Phase 1은 **features 단계 자동 캐시**(동일 config 재실행 시 피처 재계산 스킵)까지만 제공하고, 명시적 스테이지 선택은 report 단계가 생기는 Phase 2에서 배선한다. no-op 플래그를 노출하지 않는다(동작하는 것으로 오해 방지).

Note: Phase 1의 캐시 검증은 "동일 spec 2회 실행 시 2회차가 load/features 아티팩트를 재사용한다"를 파일 mtime이 아니라 **캐시 히트 카운터**(RunResult에 부수적으로 노출하지 않고, runner 내부 `_load_or_compute`가 반환하는 hit 여부를 테스트에서 monkeypatch로 관찰)로 확인한다. 과설계를 피하려 hit 여부는 `run_experiment(..., _trace=list)` 선택 인자로 관찰 가능하게 한다.

- [ ] **Step 1: Write the failing test** (synthetic 경로로 빠르게 — real 데이터 없이 CI 가능)

```python
# tests/harness/test_runner.py
from pathlib import Path
import pandas as pd
import pytest
from bakery.harness.config import ExperimentSpec, DataSpec, WindowSpec
from bakery.harness.runner import run_experiment, RunResult, STAGES


def _synth_spec():
    return ExperimentSpec(
        name="synth_smoke",
        data=DataSpec(source="synthetic"),
        target="potential_demand",
        forecaster=["seasonal_naive"],
        layers=[],
        window=WindowSpec(scheme="expanding", n_splits=2, horizon_days=7, step_days=7),
        allow_deprecated=True,
    )


def test_run_experiment_returns_runresult(tmp_path):
    result = run_experiment(_synth_spec(), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    assert isinstance(result, RunResult)
    assert result.name == "synth_smoke"
    assert not result.predictions.empty
    assert {"date", "yhat"}.issubset(result.predictions.columns)
    assert result.demand_col == "potential_demand"


def test_run_writes_config_resolved(tmp_path):
    out = tmp_path / "out"
    run_experiment(_synth_spec(), out_dir=out, cache_dir=tmp_path / "cache")
    resolved = out / "synth_smoke" / "config_resolved.yaml"
    assert resolved.exists()


def test_cache_hit_on_second_run(tmp_path):
    cache = tmp_path / "cache"
    trace1, trace2 = [], []
    run_experiment(_synth_spec(), out_dir=tmp_path / "o1", cache_dir=cache, _trace=trace1)
    run_experiment(_synth_spec(), out_dir=tmp_path / "o2", cache_dir=cache, _trace=trace2)
    # 1회차: load/features 계산(miss). 2회차: 둘 다 캐시 히트.
    assert ("load", "miss") in trace1
    assert ("load", "hit") in trace2
    assert ("features", "hit") in trace2


def test_stages_constant():
    assert STAGES == ("load", "features", "fit_predict", "evaluate")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_runner.py --color=no`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.harness.runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/harness/runner.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from bakery.data.loader import load_dataset
from bakery.evaluation.backtest import run_backtest
from bakery.evaluation.split import generate_time_splits
from bakery.features.category_aggregate import build_item_adjusted_demand
from bakery.features.enrich import enrich_daily  # Task 5에서 cli._enrich_if_needed를 여기로 이동
from bakery.harness.config import ExperimentSpec
from bakery.harness.registry import feature_set_of, resolve_forecasters

STAGES: tuple[str, ...] = ("load", "features", "fit_predict", "evaluate")


@dataclass
class RunResult:
    name: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    demand_col: str
    resolved: dict


def _stage_key(fields: dict) -> str:
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_or_compute(stage, key, cache_dir, compute, trace):
    if cache_dir is None:
        trace.append((stage, "nocache"))
        return compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{stage}_{key}.parquet"
    if path.exists():
        trace.append((stage, "hit"))
        return pd.read_parquet(path)
    trace.append((stage, "miss"))
    df = compute()
    df.to_parquet(path)
    return df


def _demand_col_of(spec: ExperimentSpec) -> str:
    """값 계산 없이 최종 수요 컬럼명만 반환."""
    if spec.data.source == "real" and spec.target == "adjusted_demand":
        return "adjusted_demand"
    return spec.target


def _variant_set(spec: ExperimentSpec) -> list[str]:
    """lightgbm forecaster들의 feature_set variant(enrich 요구사항 결정)."""
    return sorted({fs for name in spec.forecaster if (fs := feature_set_of(name)) is not None})


def run_experiment(
    spec: ExperimentSpec, *, out_dir: Path, cache_dir: Path | None = None,
    _trace: list | None = None,
) -> RunResult:
    trace = _trace if _trace is not None else []
    variants = _variant_set(spec)
    demand_col = _demand_col_of(spec)

    ds = load_dataset(source=spec.data.source, data_dir=None)

    def _feat():
        base = enrich_daily(ds, variants) if variants else ds.daily
        if spec.data.store:
            base = base[base["store_id"] == spec.data.store].copy()
        if spec.data.source == "real" and spec.target == "adjusted_demand":
            return build_item_adjusted_demand(base, alpha=spec.closing_alpha)
        return base

    # 캐시 키에 반드시 variants·store 포함 — v0(enrich 없음)와 v2(calendar+weather)가
    # 같은 features 캐시를 공유하지 않도록(캐시 무효화 버그 방지, advisor #2).
    feat_key = _stage_key({
        "source": spec.data.source, "store": spec.data.store,
        "target": spec.target, "alpha": spec.closing_alpha, "variants": variants,
    })
    # load 단계는 DailyDataset(비-DataFrame)이라 parquet 캐시 대상이 아님 →
    # features 캐시 파일 존재로 load hit/miss를 대리 판정한다.
    load_hit = cache_dir is not None and (cache_dir / f"features_{feat_key}.parquet").exists()
    trace.append(("load", "hit" if load_hit else "miss"))
    daily = _load_or_compute("features", feat_key, cache_dir, _feat, trace)

    windows = generate_time_splits(
        daily["date"], n_splits=spec.window.n_splits,
        val_horizon_days=spec.window.horizon_days, step_days=spec.window.step_days,
    )
    forecasters = resolve_forecasters(spec.forecaster, target=demand_col)
    fold_df, pred_df = run_backtest(daily, forecasters, windows)

    out = out_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump()
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")
    pred_df.to_csv(out / "predictions.csv", index=False)
    fold_df.to_csv(out / "fold_results.csv", index=False)

    return RunResult(name=spec.name, predictions=pred_df, fold_metrics=fold_df,
                     demand_col=demand_col, resolved=resolved)
```

**이 스텝에 enrich 추출 포함** (순환 import 방지 — advisor·self-review): `cli.py`의 `_enrich_if_needed`를 `src/bakery/features/enrich.py`로 이동해 `enrich_daily(ds, variants)`로 노출하고, `cli.py`는 `from bakery.features.enrich import enrich_daily as _enrich_if_needed`로 재-export(기존 호출부·시그니처 불변). runner는 cli가 아니라 features.enrich에서 import → 순환 없음. 이동 시 `_enrich_if_needed`의 시그니처가 `(ds: DailyDataset, variants: list[str])`인지 확인하고 그대로 `enrich_daily`로 리네임.

**VERIFY signatures before running** (기존 심볼과 정확히 일치해야 함):
Run:
```bash
uv run python -c "import inspect; from bakery.data.loader import load_dataset; from bakery.evaluation.split import generate_time_splits; from bakery.evaluation.backtest import run_backtest; from bakery.features.category_aggregate import build_item_adjusted_demand; print('load', inspect.signature(load_dataset)); print('split', inspect.signature(generate_time_splits)); print('bt', inspect.signature(run_backtest)); print('adj', inspect.signature(build_item_adjusted_demand))"
```
Expected: 각 시그니처 확인. `run_backtest`가 `(daily, forecasters, windows)` 순서인지, `pred_df`에 `yhat` 컬럼이 있는지 확인 후 test의 컬럼 단언(`{"date","yhat"}`)을 실제 출력에 맞춰 조정.

또한 `_enrich_if_needed` 시그니처 확인 후 이동:
```bash
uv run python -c "import inspect; from bakery.cli import _enrich_if_needed; print(inspect.signature(_enrich_if_needed))"
```
Expected: `(ds, variants)`. 그대로 `src/bakery/features/enrich.py`의 `enrich_daily(ds, variants)`로 이동, cli.py는 재-export.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_runner.py --color=no`
Expected: PASS (4 passed). 만약 synthetic daily에 `store_id`가 없거나 컬럼명이 다르면 VERIFY 결과에 맞춰 test/impl 조정.

- [ ] **Step 5: Update __init__ + commit**

```python
# src/bakery/harness/__init__.py  — append
from bakery.harness.runner import RunResult, run_experiment, STAGES
```
Add to `__all__`: `"RunResult", "run_experiment", "STAGES"`.

```bash
git add src/bakery/harness/runner.py src/bakery/harness/__init__.py tests/harness/test_runner.py \
        src/bakery/features/enrich.py src/bakery/cli.py
git commit -m "feat(harness): 단계 캐시 runner + RunResult + enrich 스파인 추출"
```

---

## Task 4: CLI `harness run` 커맨드 (thin wrapper)

**Files:**
- Modify: `src/bakery/cli.py` (커맨드 추가, 기존 커맨드 불변)
- Test: `tests/harness/test_cli_harness.py`

**Interfaces:**
- Consumes: `load_spec`(Task 1), `run_experiment`(Task 3).
- Produces: CLI `bakery harness-run <config.yaml> [--out DIR] [--cache DIR]`. (`--from`/`--only` 스테이지 선택은 Phase 2로 연기 — Task 3 노트 참조.)

Note: typer는 서브그룹보다 단일 커맨드가 기존 패턴(app.command)과 맞으므로 `harness-run` 하이픈 커맨드로 추가한다(기존 `format-bonavi-v2` 등과 일관). 설계 문서의 `harness run` 표기는 `harness-run`으로 실현.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_cli_harness.py
import yaml
from typer.testing import CliRunner
from bakery.cli import app

runner = CliRunner()


def test_harness_run_synthetic_smoke(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(yaml.safe_dump({
        "name": "cli_smoke",
        "data": {"source": "synthetic"},
        "target": "potential_demand",
        "allow_deprecated": True,
        "forecaster": ["seasonal_naive"],
        "layers": [],
        "window": {"scheme": "expanding", "n_splits": 2},
    }), encoding="utf-8")
    result = runner.invoke(app, [
        "harness-run", str(cfg),
        "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"),
    ])  # --from/--only 없음 (Phase 2)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "cli_smoke" / "predictions.csv").exists()
    assert (tmp_path / "out" / "cli_smoke" / "config_resolved.yaml").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: FAIL — `No such command 'harness-run'` (exit_code != 0)

- [ ] **Step 3: Write minimal implementation** — add to `src/bakery/cli.py` (near other `@app.command` defs, after imports add `from bakery.harness import load_spec, run_experiment`)

```python
@app.command("harness-run")
def cmd_harness_run(
    config: Path,
    out: Path = REPORTS_DIR,
    cache: Path | None = None,
) -> None:
    """YAML 실험 config 1개를 실행한다 (harness 단일 표면)."""
    spec = load_spec(config)
    console.print(f"[cyan]harness[/] {spec.name} forecaster={spec.forecaster} target={spec.target}")
    result = run_experiment(spec, out_dir=out, cache_dir=cache)
    console.print(f"[green]wrote[/] {out}/{result.name}/predictions.csv (demand_col={result.demand_col})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py tests/harness/test_cli_harness.py
git commit -m "feat(cli): harness-run 커맨드 (YAML 실험 단일 표면)"
```

---

## Task 5: ★Acceptance — 광교 lightgbm_v2 재현

**Files:**
- Create: `experiments/gwangyo_default.yaml`
- Test: `tests/harness/test_reproduce_backtest.py`

**Interfaces:**
- Consumes: `run_experiment`(Task 3), `enrich_daily`(Task 3), 그리고 **비교 기준**으로 기존 cli 헬퍼 `_build_forecasters` + `generate_time_splits` + `run_backtest` + `build_item_adjusted_demand`.

**목표:** harness 경로와 기존 cli 경로가 **동일 fold·동일 forecaster·동일 target**에서 예측값이 일치함을 증명. 데이터가 real 단일 forecaster(lightgbm_v2)로 좁혀 비교한다.

**전제 (VERIFY 완료 2026-07-24):** real 데이터 로컬 존재 확인됨 — `load_dataset(source="real")` 성공, 25,105행, `store_id`의 유일값 = **`store_gw01`**(광교). GlobalLGBM `seed=42` 고정 → 정확일치(rtol=1e-9) 유효.
**★이 테스트는 skip하지 않는다.** 이것이 Phase 1 완료의 hard gate다(PASS 없이는 Phase 1 미완). real이 사라진 예외 상황에서만 `load_dataset` 실패로 error가 나며, 그때는 조용한 skip이 아니라 명시적 실패로 드러나야 한다.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_reproduce_backtest.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bakery.harness.config import ExperimentSpec, DataSpec, WindowSpec
from bakery.harness.runner import run_experiment
from bakery.evaluation.split import generate_time_splits
from bakery.evaluation.backtest import run_backtest
from bakery.features.category_aggregate import build_item_adjusted_demand
from bakery.data.loader import load_dataset

STORE = "store_gw01"  # 광교 (VERIFY 완료: real daily의 유일 store_id)


def test_harness_reproduces_legacy_lightgbm_v2(tmp_path):
    n_splits, horizon, step, alpha = 8, 7, 7, 0.8

    # --- 기존(legacy) 경로 재현 (enrich → store filter → adjusted_demand) ---
    ds = load_dataset(source="real", data_dir=None)
    from bakery.features.enrich import enrich_daily  # Task 3에서 이동
    from bakery.cli import _build_forecasters
    daily = enrich_daily(ds, ["v2"])
    daily = daily[daily["store_id"] == STORE].copy()
    daily = build_item_adjusted_demand(daily, alpha=alpha)
    windows = generate_time_splits(daily["date"], n_splits=n_splits,
                                   val_horizon_days=horizon, step_days=step)
    forecasters = _build_forecasters(["v2"], include_production=False,
                                     v23_target="adjusted_demand", drop_groups=frozenset())
    legacy_fold, legacy_pred = run_backtest(daily, forecasters, windows)

    # --- harness 경로 (runner가 동일 순서: enrich → store filter → adjusted_demand) ---
    spec = ExperimentSpec(
        name="gwangyo_v2_repro", data=DataSpec(source="real", store=STORE),
        target="adjusted_demand", forecaster=["lightgbm_v2"], layers=[],
        window=WindowSpec(scheme="expanding", n_splits=n_splits,
                          horizon_days=horizon, step_days=step),
        closing_alpha=alpha,
    )
    result = run_experiment(spec, out_dir=tmp_path / "out", cache_dir=None)

    # 예측값 정확 일치 (동일 fold·모델·target이므로 deterministic해야 함)
    lp = legacy_pred.sort_values(["date", "item_id"]).reset_index(drop=True)
    hp = result.predictions.sort_values(["date", "item_id"]).reset_index(drop=True)
    assert len(lp) == len(hp)
    np.testing.assert_allclose(hp["yhat"].to_numpy(), lp["yhat"].to_numpy(), rtol=1e-9)
```

Note: legacy·harness 두 경로 모두 **같은 전처리 순서**(enrich→store filter→adjusted_demand)를 밟는다. runner(Task 3)의 `_feat()`가 이미 이 순서로 구현되어 있으므로(enrich_daily → store 필터 → build_item_adjusted_demand), acceptance는 별도 runner 수정 없이 두 경로를 나란히 비교만 한다. enrich 부재로 인한 피처 불일치 리스크는 Task 3에서 이미 해소됨.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_reproduce_backtest.py --color=no`
Expected: 이 테스트 파일이 아직 없으므로 collection 후 실행. Task 3까지 완료된 상태라면 값이 일치해 PASS할 가능성이 높다 — 이 경우 "실패 후 통과" TDD 리듬 대신 **회귀 가드**로 간주(스파인이 이미 배선됐으므로 정상). 만약 FAIL이면 diff를 출력해 원인 규명(순서/필터/target 불일치).

디버깅용 실패 진단:
```python
# FAIL 시 임시 추가해 원인 파악 (통과 후 제거)
print("len legacy/harness:", len(legacy_pred), len(result.predictions))
print("cols:", sorted(result.predictions.columns))
```

- [ ] **Step 3: Run full suite to verify no regression**

Run: `uv run pytest --color=no`
Expected: 전체 PASS (기존 테스트 + harness 신규 테스트). enrich 이동(Task 3)이 cli 기존 커맨드를 깨지 않았는지 여기서 최종 확인.

- [ ] **Step 4: (조정) 컬럼명 불일치 시 test 정합**

`run_backtest` 출력 컬럼이 `yhat`/`item_id`/`date`가 아니면(Task 3 VERIFY에서 확인된 실제 이름) test의 정렬/비교 컬럼을 실제 이름으로 교체. 값 비교 자체(rtol=1e-9)는 불변.

- [ ] **Step 5: Write experiments/gwangyo_default.yaml + commit**

```yaml
# experiments/gwangyo_default.yaml
name: gwangyo_default
data:
  source: real
  store: store_gw01   # 광교 (real daily의 유일 store_id)
target: adjusted_demand
forecaster: [lightgbm_v2]
layers: []
window:
  scheme: expanding
  n_splits: 8
  horizon_days: 7
  step_days: 7
closing_alpha: 0.8
```

```bash
git add experiments/gwangyo_default.yaml tests/harness/test_reproduce_backtest.py
git commit -m "feat(harness): 광교 lightgbm_v2 재현 acceptance + 기본 experiment config"
```

---

## Self-Review 결과

**Spec coverage:**
- §3 디렉토리 → Task 1~5에서 config/registry/runner/cli/experiments 생성. ✅ (report.py/eda.py/viz는 Phase 2~3, 명시적 범위 밖)
- §4 Taxonomy → Task 2 `_KIND` 4kind 중 3개(POINT/DIST/COMPOSITE); post_layer는 `LAYER_NAMES`로 분리. ✅
- §5 canonical 강제표 → Task 1 `_enforce` (target/potential_demand/metrics/split/deprecated). ✅
- §6 단계 캐시·재개 → Task 3 STAGES + `_load_or_compute`. ⚠️ **Phase 1은 features 자동 캐시까지만** (동일 config 재실행 시 피처 재계산 스킵). 임의 스테이지 재개(`--from`/`--only`)는 report 단계가 생기는 **Phase 2로 명시 연기** — no-op 플래그를 노출하지 않음. 사용자 요청 "중간부터 실행"의 부분 구현임을 Task 3 인터페이스 노트에 표면화. ✅(축소 명시)
- §7 RunResult → Task 3. ✅  §9 Phase1 acceptance → Task 5(skip 없는 hard gate). ✅

**Placeholder scan:** 코드 스텝 모두 실제 코드 포함. VERIFY 스텝은 시그니처 불확실성을 명시적으로 처리(추측 금지 원칙). ✅

**Type consistency:** `run_experiment(spec, *, out_dir, cache_dir, _trace)` 시그니처가 Task 3 정의 = Task 4/5 호출부 일치(from_stage/only_stage 제거 반영). `resolve_forecasters(names, *, target)` Task 2 정의 = Task 3 호출 일치. `feature_set_of` Task 2 정의 = Task 3 `_variant_set` 사용 일치. `RunResult` 필드(name/predictions/fold_metrics/demand_col/resolved) Task 3~5 일관. ✅

**VERIFY 완료 (2026-07-24, 실행 전 확정):**
1. ✅ `GlobalLGBM.__init__(params, y_col, feature_set, drop_groups)` — 인자명 `y_col`, seed=42 고정(정확일치 유효). 플랜 반영 완료.
2. ✅ real 데이터 존재 — 25,105행, store_id 유일값 `store_gw01`. acceptance skip 제거, store 값 확정.
3. ⏳ `run_backtest` 반환 `pred_df` 컬럼명(`yhat`/`item_id`/`date`) — Task 3 Step 3 전 VERIFY로 확인(테스트 컬럼 단언 조정).
4. ⏳ synthetic `ds.daily`에 `store_id` 존재 여부 — Task 3 Step 4에서 확인(synthetic엔 store 필터 미적용이 기본이라 무해).
5. ✅ 순환 import 방지: `enrich_daily`를 features 모듈로 이동, cli·runner 양쪽이 거기서 import — Task 3에 통합.

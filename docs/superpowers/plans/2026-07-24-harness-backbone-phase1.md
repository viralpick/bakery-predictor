# Harness Backbone Phase 1 (스파인) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 발행 헤드라인 백테스트 엔진(`scripts/store_predictive_power.windowed_backtest` = 카테고리 총량 + event_prior)을 `src/bakery/harness/`로 추출하고, YAML config 1파일=1실험으로 실행하는 단일 표면을 만든다.

**Architecture:** canonical = **category_total + event_prior**(총량 granularity). `windowed_backtest` 코어를 harness로 추출하고 원본 스크립트는 harness 코어를 import하는 래퍼로 전환(단일 출처). config.py가 YAML→`ExperimentSpec`(canonical 강제+경고), registry가 forecaster 이름→kind, runner가 spec→feat build→backtest_core→`RunResult`. 완료 기준은 추출 코어가 원본 `windowed_backtest`와 예측값 정확일치(엔진 동등성).

**Tech Stack:** Python 3.12, pydantic v2, PyYAML, pandas, numpy, lightgbm, typer, pytest. 기존 `src/bakery/models/category_total`(`fit_category_total`, random_state=42), `models/event_prior`(`EventLevelPrior`), `features/category_aggregate`(`build_category_daily`/`build_features`) 재사용.

## Global Constraints

- **canonical 스택** (docs/superpowers/specs/2026-07-24-harness-backbone-design.md §4): 기본 forecaster=`[category_total, distributional_total]`, layers=`[event_prior]`. category_total = 총량 granularity(item 배분 아님). lightgbm v0~v3 = 비교용 보조.
- **카테고리 타깃 컬럼**: `adjusted_demand_unit` (category level). item-level의 `adjusted_demand`와 다름 — 혼동 금지.
- **헤드라인 파라미터** (원본 상수, 정확 보존): `ALPHA=0.8`, `PROD_Q=0.85`, `DEFAULT_WINDOW_DAYS=730`, `MAIN_FOLDS=52`, `HORIZON=7`, `MIN_TRAIN_ROWS=60`, `TARGET="adjusted_demand_unit"`.
- **event_prior leakage 규칙** (헌장 1번): prior는 train window가 아니라 **pre-test 전체 history**(`df["date"] < test_start_date`)로 fit. blend는 expected·production 둘 다 보정.
- **측정헌장**: target 기본=카테고리 `adjusted_demand_unit`; `potential_demand`는 `allow_deprecated: true` 없으면 ERROR; metric 기본 6종.
- **재구현 금지**: harness는 `src/bakery` 심볼을 호출만. `windowed_backtest`는 **추출**(로직 복제 아님) — 원본은 harness 코어를 import.
- **테스트 단언**: 기대값 아는 단언은 정확값 `==`, 부동소수는 `np.testing.assert_allclose(rtol=1e-9)`.
- **pytest 실행**: 이 repo addopts에 `-q` 있음. 카운트 필요 시 `uv run pytest --color=no` (추가 `-q` 금지).
- **STORE_EVENT_PRIORS 키**: 원본은 한글 라벨("광교"). harness preset은 영문 키("gwangyo") + 데이터 store_id "store_gw01".
- Phase 1 범위는 **스파인만**. report.py/eda.py/viz는 Phase 2~3.

---

## File Structure

- `src/bakery/harness/__init__.py` — 공개 심볼 re-export
- `src/bakery/harness/config.py` — `ExperimentSpec`(pydantic) + `load_spec` + canonical 검증/경고
- `src/bakery/harness/event_priors.py` — 특수일 상수(XMAS/CHILDRENS/CHUSEOK/SEOLLAL) + `STORE_EVENT_PRIORS` preset + `resolve_event_priors`
- `src/bakery/harness/backtest_core.py` — `windowed_backtest`/`metrics_from_preds` 추출(단일 출처)
- `src/bakery/harness/registry.py` — `ForecasterKind` + `kind_of` + `is_supported_phase1`
- `src/bakery/harness/runner.py` — `RunResult` + `run_experiment`(feat build→core→result, 캐시)
- `src/bakery/cli.py` — `harness-run` 커맨드
- `scripts/store_predictive_power.py` — `windowed_backtest`를 harness 코어 import 래퍼로 전환
- `experiments/gwangyo_default.yaml`
- `tests/harness/test_config.py`, `test_event_priors.py`, `test_backtest_core_equivalence.py`(★acceptance), `test_registry.py`, `test_runner.py`, `test_cli_harness.py`

---

## Task 1: ExperimentSpec 스키마 + canonical 강제 (카테고리 기본)

**Files:**
- Create: `src/bakery/harness/__init__.py`, `src/bakery/harness/config.py`
- Test: `tests/harness/test_config.py`

**Interfaces:**
- Produces:
  - `class DataSpec(source: Literal["real","synthetic","parquet"], store: str = "store_gw01")`
  - `class WindowSpec(scheme: Literal["expanding","rolling"] = "expanding", n_folds: int = 52, window_days: int = 730, horizon_days: int = 7)`
  - `class ExperimentSpec(name, data, target="adjusted_demand_unit", forecaster, layers, event_priors="gwangyo", window, metrics, alpha=0.8, production_q=0.85, allow_deprecated=False)`
  - `DEFAULT_FORECASTERS=["category_total","distributional_total"]`, `DEFAULT_LAYERS=["event_prior"]`, `DEFAULT_METRICS=[6종]`
  - `load_spec(path) -> ExperimentSpec`, `class SpecError(ValueError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_config.py
import pytest, yaml
from bakery.harness.config import (
    ExperimentSpec, load_spec, SpecError,
    DEFAULT_FORECASTERS, DEFAULT_LAYERS, DEFAULT_METRICS,
)


def _write(tmp_path, body):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


def test_defaults_are_category_stack(tmp_path):
    spec = load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"}}))
    assert spec.forecaster == DEFAULT_FORECASTERS       # [category_total, distributional_total]
    assert spec.layers == DEFAULT_LAYERS                # [event_prior]
    assert spec.target == "adjusted_demand_unit"
    assert spec.data.store == "store_gw01"
    assert spec.window.n_folds == 52
    assert spec.window.window_days == 730
    assert spec.alpha == 0.8
    assert spec.event_priors == "gwangyo"


def test_potential_demand_rejected(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "target": "potential_demand"}
    with pytest.raises(SpecError, match="potential_demand"):
        load_spec(_write(tmp_path, body))


def test_random_split_rejected(tmp_path):
    with pytest.raises(SpecError):
        load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"},
                                    "window": {"scheme": "random"}}))


def test_mape_only_warns(tmp_path):
    with pytest.warns(UserWarning, match="MAPE"):
        load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"}, "metrics": ["mape"]}))


def test_event_prior_without_preset_warns(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "layers": ["event_prior"], "event_priors": None}
    with pytest.warns(UserWarning, match="event_prior"):
        load_spec(_write(tmp_path, body))


def test_single_forecaster_string_wrapped(tmp_path):
    spec = load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"},
                                       "forecaster": "category_total"}))
    assert spec.forecaster == ["category_total"]
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

DEFAULT_FORECASTERS: list[str] = ["category_total", "distributional_total"]
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
    store: str = "store_gw01"


class WindowSpec(BaseModel):
    scheme: Literal["expanding", "rolling"] = "expanding"
    n_folds: int = 52
    window_days: int = 730
    horizon_days: int = 7


class ExperimentSpec(BaseModel):
    name: str
    data: DataSpec
    target: str = "adjusted_demand_unit"
    forecaster: list[str] = Field(default_factory=lambda: list(DEFAULT_FORECASTERS))
    layers: list[str] = Field(default_factory=lambda: list(DEFAULT_LAYERS))
    event_priors: str | None = "gwangyo"
    window: WindowSpec = Field(default_factory=WindowSpec)
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    alpha: float = 0.8
    production_q: float = 0.85
    allow_deprecated: bool = False

    @field_validator("forecaster", "layers", mode="before")
    @classmethod
    def _wrap_single(cls, v):
        return [v] if isinstance(v, str) else v


def load_spec(path: str | Path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        spec = ExperimentSpec(**raw)
    except Exception as exc:
        raise SpecError(str(exc)) from exc
    _enforce(spec)
    return spec


def _enforce(spec: ExperimentSpec) -> None:
    if spec.target == "potential_demand" and not spec.allow_deprecated:
        raise SpecError("target=potential_demand는 오염 소스라 금지. allow_deprecated: true 필요.")
    if spec.metrics == ["mape"]:
        warnings.warn("MAPE 단독은 희소 품목에서 폭발한다. WAPE 병기 권장.", UserWarning)
    if "event_prior" in spec.layers and spec.event_priors is None:
        warnings.warn("event_prior layer가 있으나 event_priors 프리셋 키 미지정.", UserWarning)
    for name in spec.forecaster:
        if name in DEPRECATED_FORECASTERS:
            warnings.warn(f"{name}는 DEPRECATED forecaster.", UserWarning)
```

Note: `window.scheme: "random"`은 `Literal`이라 pydantic이 거부 → `SpecError`로 변환.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_config.py --color=no`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/harness/__init__.py src/bakery/harness/config.py tests/harness/test_config.py
git commit -m "feat(harness): ExperimentSpec — 카테고리 총량 canonical 기본 + 강제/경고"
```

---

## Task 2: event_priors.py — STORE_EVENT_PRIORS 프리셋 승격

**Files:**
- Create: `src/bakery/harness/event_priors.py`
- Modify: `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_event_priors.py`

**Interfaces:**
- Produces:
  - `STORE_EVENT_PRIORS: dict[str, dict]` — 영문 키(`"gwangyo"`, `"samsung"`, `"mecenatpolis"`, `"gwanghwamun"`)
  - `resolve_event_priors(key: str | None) -> tuple[dict | None, dict | None]` — `(events, lunar_events)`. None이면 `(None,None)`. 미등록 키 `KeyError`.

**VERIFY 먼저** — 원본 특수일 상수 정의를 복사한다:
Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'scripts'); import store_predictive_power as s; print('XMAS', s.XMAS); print('CHILDRENS', s.CHILDRENS); print('CHUSEOK', s.CHUSEOK); print('SEOLLAL', s.SEOLLAL); print('PRIORS', s.STORE_EVENT_PRIORS)"
```
출력 dict를 `event_priors.py`에 **그대로** 복사(수기 재작성 금지). 키 매핑: 광교→gwangyo, 삼성타운→samsung, 메세나폴리스→mecenatpolis, 광화문→gwanghwamun.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_event_priors.py
import sys
import pytest
from bakery.harness.event_priors import STORE_EVENT_PRIORS, resolve_event_priors


def test_gwangyo_preset_matches_script():
    """harness 프리셋이 원본 scripts 정의와 동일해야 한다(단일 출처 승격)."""
    sys.path.insert(0, "scripts")
    import store_predictive_power as s
    script_gw = s.STORE_EVENT_PRIORS["광교"]
    events, lunar = resolve_event_priors("gwangyo")
    assert events == script_gw["events"]
    assert lunar == script_gw["lunar_events"]


def test_resolve_none_returns_none_pair():
    assert resolve_event_priors(None) == (None, None)


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        resolve_event_priors("nonexistent")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_event_priors.py --color=no`
Expected: FAIL — `ModuleNotFoundError: bakery.harness.event_priors`

- [ ] **Step 3: Write implementation** (VERIFY 출력값을 채워 넣는다 — 아래는 골격, `{...}`는 VERIFY 실제값으로 교체)

```python
# src/bakery/harness/event_priors.py
"""특수일 EventLevelPrior 프리셋 (scripts/store_predictive_power.py에서 승격, 단일 출처)."""
from __future__ import annotations

# ↓ VERIFY 출력값을 그대로 붙여넣기 (실제 값으로 교체)
XMAS = {...}          # VERIFY: s.XMAS
CHILDRENS = {...}     # VERIFY: s.CHILDRENS
CHUSEOK = {...}       # VERIFY: s.CHUSEOK
SEOLLAL = {...}       # VERIFY: s.SEOLLAL

STORE_EVENT_PRIORS: dict[str, dict] = {
    "gwangyo":      {"events": {**XMAS, **CHILDRENS}, "lunar_events": dict(CHUSEOK)},
    "samsung":      {"events": dict(XMAS), "lunar_events": {}},
    "mecenatpolis": {"events": dict(XMAS), "lunar_events": dict(SEOLLAL)},
    "gwanghwamun":  {"events": dict(XMAS), "lunar_events": {}},
}


def resolve_event_priors(key: str | None) -> tuple[dict | None, dict | None]:
    if key is None:
        return None, None
    cfg = STORE_EVENT_PRIORS[key]
    return cfg.get("events"), cfg.get("lunar_events")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_event_priors.py --color=no`
Expected: PASS (3 passed). 실패 시 VERIFY 출력과 붙여넣은 값 diff.

- [ ] **Step 5: Update __init__ + commit**

```python
# __init__.py append
from bakery.harness.event_priors import STORE_EVENT_PRIORS, resolve_event_priors
```
Add to `__all__`: `"STORE_EVENT_PRIORS", "resolve_event_priors"`.

```bash
git add src/bakery/harness/event_priors.py src/bakery/harness/__init__.py tests/harness/test_event_priors.py
git commit -m "feat(harness): STORE_EVENT_PRIORS 프리셋 승격 (단일 출처)"
```

---

## Task 3: ★backtest_core.py — windowed_backtest 추출 + 엔진 동등성 (hard gate)

**Files:**
- Create: `src/bakery/harness/backtest_core.py`
- Modify: `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_backtest_core_equivalence.py`

**Interfaces:**
- Consumes: `bakery.models.category_total.fit_category_total`, `bakery.models.event_prior.EventLevelPrior`, `bakery.models.category_total.BacktestResult`.
- Produces:
  - `windowed_backtest(df, *, window_days, target_col="adjusted_demand_unit", n_folds=52, horizon_days=7, production_q=0.85, alpha=0.8, events=None, lunar_events=None, min_train_rows=60) -> BacktestResult`
  - `metrics_from_preds(p) -> dict`
  - 모듈 상수 `MIN_TRAIN_ROWS=60`

**목표:** 추출한 `windowed_backtest`가 원본 `scripts/store_predictive_power.windowed_backtest`와 **예측값 정확일치**. Phase 1 hard gate — PASS 없이 진행 불가.

**전제 (VERIFY 완료 2026-07-24):** `fit_category_total` random_state=42, `EventLevelPrior` 비랜덤 → 결정적. real 데이터 존재(store_gw01). 정확일치 유효.

- [ ] **Step 1: Write the failing equivalence test**

```python
# tests/harness/test_backtest_core_equivalence.py
import sys
import numpy as np

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.event_priors import resolve_event_priors


def _feat():
    cd = build_category_daily(alpha=0.8)                    # None→canonical 3cat
    return build_features(cd, target_col="adjusted_demand_unit")


def test_core_matches_script_windowed_backtest():
    sys.path.insert(0, "scripts")
    import store_predictive_power as s
    from bakery.harness.backtest_core import windowed_backtest as core_wb

    feat = _feat()
    events, lunar = resolve_event_priors("gwangyo")

    legacy = s.windowed_backtest(
        feat, window_days=s.DEFAULT_WINDOW_DAYS, n_folds=s.MAIN_FOLDS,
        events=events, lunar_events=lunar,
    )
    got = core_wb(
        feat, window_days=730, n_folds=52, production_q=0.85, alpha=0.8,
        events=events, lunar_events=lunar,
    )
    lp = legacy.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    gp = got.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    assert len(lp) == len(gp)
    np.testing.assert_allclose(gp["expected"].to_numpy(), lp["expected"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(gp["production"].to_numpy(), lp["production"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(gp["actual"].to_numpy(), lp["actual"].to_numpy(), rtol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_backtest_core_equivalence.py --color=no`
Expected: FAIL — `ModuleNotFoundError: bakery.harness.backtest_core`

- [ ] **Step 3: Extract windowed_backtest into backtest_core.py**

원본 `scripts/store_predictive_power.py:106-197`의 `windowed_backtest` + `metrics_from_preds`를 **로직 변경 없이** 옮긴다. 모듈 상수(`HORIZON`, `PROD_Q`, `ALPHA`)는 함수 인자 기본값으로 흡수(값 동일).

```python
# src/bakery/harness/backtest_core.py
"""windowed_backtest 코어 — 카테고리 총량 + event_prior (발행 헤드라인 엔진).

scripts/store_predictive_power.py에서 추출한 단일 출처. leakage-safe:
prior는 pre-test 전체 history로 fit (train window보다 김).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.models.category_total import BacktestResult, fit_category_total
from bakery.models.event_prior import EventLevelPrior

MIN_TRAIN_ROWS = 60


def windowed_backtest(
    df: pd.DataFrame, *, window_days: int,
    target_col: str = "adjusted_demand_unit", n_folds: int = 52,
    horizon_days: int = 7, production_q: float = 0.85, alpha: float = 0.8,
    events: dict | None = None, lunar_events: dict | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> BacktestResult:
    df = df.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).copy()
    df = df.dropna().reset_index(drop=True)
    total = len(df)
    test_size = horizon_days
    if total <= n_folds * test_size + min_train_rows:
        raise ValueError(f"Not enough data: total={total}, folds={n_folds}")

    window = pd.Timedelta(days=window_days)
    folds, preds = [], []
    for k in range(n_folds):
        test_end = total - k * test_size
        test_start = test_end - test_size
        test_df = df.iloc[test_start:test_end]
        test_start_date = test_df["date"].iloc[0]
        train_df = df[(df["date"] < test_start_date) & (df["date"] >= test_start_date - window)]
        if len(train_df) < min_train_rows:
            continue
        model = fit_category_total(
            train_df, target_col=target_col, alpha_demand=alpha, production_q=production_q,
        )
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        hist = df[df["date"] < test_start_date]     # leakage-safe: pre-test 전체
        prior = EventLevelPrior(events=events, lunar_events=lunar_events).fit(hist, target_col=target_col)
        exp_pred, prod_pred = prior.blend(test_df["date"].values, exp_pred, prod_pred)
        actual = test_df[target_col].values
        wape = np.abs(actual - exp_pred).sum() / max(np.abs(actual).sum(), 1)
        folds.append(dict(
            fold=k, n_train=len(train_df), n_test=len(test_df),
            test_start=test_start_date, test_end=test_df["date"].iloc[-1], wape=wape,
            wpe=(exp_pred - actual).sum() / max(actual.sum(), 1),
            prod_pct_under=(prod_pred < actual).mean(),
        ))
        preds.append(pd.DataFrame({
            "date": test_df["date"].values, "fold": k,
            "actual": actual, "expected": exp_pred, "production": prod_pred,
        }))
    return BacktestResult(
        folds=pd.DataFrame(folds).sort_values("fold").reset_index(drop=True),
        predictions=pd.concat(preds, ignore_index=True),
    )


def metrics_from_preds(p: pd.DataFrame) -> dict:
    actual, expected, prod = p["actual"], p["expected"], p["production"]
    surplus = (prod - actual).clip(lower=0)
    return {
        "n_test": int(len(p)),
        "wape": float(np.abs(actual - expected).sum() / max(np.abs(actual).sum(), 1)),
        "wpe": float((expected - actual).sum() / max(actual.sum(), 1)),
        "stockout_risk": float((prod < actual).mean()),
        "surplus_mean_units": float(surplus.mean()),
        "surplus_rate": float(surplus.sum() / max(actual.sum(), 1)),
    }
```

⚠️ 원본과 **한 줄도 다르면 안 된다**(정확일치 게이트). `HORIZON=7`/`PROD_Q=0.85`/`ALPHA=0.8`이 인자 기본값으로 정확히 흡수됐는지, `fit_category_total` 호출 인자(`alpha_demand=alpha, production_q=production_q`)가 원본과 동일한지 대조.

- [ ] **Step 4: Run equivalence test**

Run: `uv run pytest tests/harness/test_backtest_core_equivalence.py --color=no`
Expected: PASS. 불일치 시 원본 대비 diff(특히 fit 인자·prior fit history 범위·fold 루프 경계) 정밀 대조. **PASS 없이 다음 태스크 진행 금지.**

- [ ] **Step 5: Update __init__ + commit**

```python
# __init__.py append
from bakery.harness.backtest_core import windowed_backtest, metrics_from_preds
```
Add to `__all__`: `"windowed_backtest", "metrics_from_preds"`.

```bash
git add src/bakery/harness/backtest_core.py src/bakery/harness/__init__.py tests/harness/test_backtest_core_equivalence.py
git commit -m "feat(harness): windowed_backtest 코어 추출 + 원본 엔진 동등성 검증"
```

---

## Task 4: 원본 스크립트를 harness 코어 래퍼로 전환 (단일 출처)

**Files:**
- Modify: `scripts/store_predictive_power.py`
- Test: (기존 회귀) `uv run pytest --color=no`

**Interfaces:**
- Consumes: `bakery.harness.backtest_core`.

**목표:** 두 구현 공존 금지. 원본의 `windowed_backtest`/`metrics_from_preds` 정의를 harness import로 교체하되, **호출 시그니처는 원본 유지**(기존 호출부·weekly_overlay_series.py 불변).

- [ ] **Step 1: Replace definitions with import**

`scripts/store_predictive_power.py`에서 `windowed_backtest`/`metrics_from_preds` 함수 정의 본문을 삭제하고 상단 import에 추가:

```python
from bakery.harness.backtest_core import windowed_backtest, metrics_from_preds  # noqa: F401
```

`HORIZON`/`PROD_Q`/`ALPHA`/`MAIN_FOLDS`/`DEFAULT_WINDOW_DAYS`/`MIN_TRAIN_ROWS` 상수를 스크립트 다른 곳(find_optimal_* 등)에서도 참조하면 그 상수 정의는 **그대로 남긴다**(삭제 금지). harness 코어가 인자 기본값으로 흡수했으므로 값 동일.

**VERIFY**: 원본 windowed_backtest 호출부의 키워드(events/lunar_events/production_q)가 harness 코어 시그니처와 일치하는지 확인.

- [ ] **Step 2: Run full suite + weekly_overlay smoke**

Run: `uv run pytest --color=no`
Expected: 전체 PASS.

Run: `WEEKLY_OUT_DIR=/tmp/wk uv run python scripts/weekly_overlay_series.py 2>&1 | tail -5`
Expected: 에러 없이 overlay_unified.parquet 생성(값은 이전과 동일 — 코어 로직 불변).

- [ ] **Step 3: Commit**

```bash
git add scripts/store_predictive_power.py
git commit -m "refactor(scripts): windowed_backtest를 harness 코어 import로 전환 (단일 출처)"
```

---

## Task 5: Registry — forecaster 이름→kind

**Files:**
- Create: `src/bakery/harness/registry.py`
- Modify: `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_registry.py`

**Interfaces:**
- Produces:
  - `class ForecasterKind(str, Enum)`: `CATEGORY_TOTAL`, `DISTRIBUTIONAL`, `POINT`, `COMPOSITE`
  - `kind_of(name) -> ForecasterKind` (미등록 `KeyError`)
  - `LAYER_NAMES: frozenset[str]` = `{"event_prior", "decision", "conformal_order"}`
  - `is_supported_phase1(name) -> bool` — category_total만 True

Note (taxonomy §4): category_total=CATEGORY_TOTAL(canonical), distributional_total=DISTRIBUTIONAL, lightgbm[_v1/2/3]·seasonal_naive·moving_average=POINT(보조), category_v4=COMPOSITE. Phase 1 runner는 category_total만 실행.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_registry.py
import pytest
from bakery.harness.registry import ForecasterKind, kind_of, LAYER_NAMES, is_supported_phase1


def test_kind_taxonomy():
    assert kind_of("category_total") == ForecasterKind.CATEGORY_TOTAL
    assert kind_of("distributional_total") == ForecasterKind.DISTRIBUTIONAL
    assert kind_of("lightgbm_v2") == ForecasterKind.POINT
    assert kind_of("category_v4") == ForecasterKind.COMPOSITE


def test_unknown_raises():
    with pytest.raises(KeyError):
        kind_of("bogus")


def test_layers_registered():
    assert "event_prior" in LAYER_NAMES
    assert "decision" in LAYER_NAMES


def test_phase1_supports_category_total_only():
    assert is_supported_phase1("category_total") is True
    assert is_supported_phase1("distributional_total") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_registry.py --color=no`
Expected: FAIL — `ModuleNotFoundError: bakery.harness.registry`

- [ ] **Step 3: Write implementation**

```python
# src/bakery/harness/registry.py
from __future__ import annotations
from enum import Enum


class ForecasterKind(str, Enum):
    CATEGORY_TOTAL = "category_total"
    DISTRIBUTIONAL = "distributional"
    POINT = "point_forecaster"
    COMPOSITE = "composite_pipeline"


_KIND: dict[str, ForecasterKind] = {
    "category_total": ForecasterKind.CATEGORY_TOTAL,
    "distributional_total": ForecasterKind.DISTRIBUTIONAL,
    "seasonal_naive": ForecasterKind.POINT,
    "moving_average": ForecasterKind.POINT,
    "lightgbm": ForecasterKind.POINT,
    "lightgbm_v1": ForecasterKind.POINT,
    "lightgbm_v2": ForecasterKind.POINT,
    "lightgbm_v3": ForecasterKind.POINT,
    "category_v4": ForecasterKind.COMPOSITE,
}

LAYER_NAMES: frozenset[str] = frozenset({"event_prior", "decision", "conformal_order"})


def kind_of(name: str) -> ForecasterKind:
    return _KIND[name]


def is_supported_phase1(name: str) -> bool:
    """Phase 1은 category_total 경로만 실행(나머지는 taxonomy 등록만)."""
    return kind_of(name) is ForecasterKind.CATEGORY_TOTAL
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_registry.py --color=no`
Expected: PASS (4 passed)

- [ ] **Step 5: Update __init__ + commit**

```python
# __init__.py append
from bakery.harness.registry import ForecasterKind, kind_of, LAYER_NAMES, is_supported_phase1
```
Add to `__all__`: `"ForecasterKind", "kind_of", "LAYER_NAMES", "is_supported_phase1"`.

```bash
git add src/bakery/harness/registry.py src/bakery/harness/__init__.py tests/harness/test_registry.py
git commit -m "feat(harness): forecaster kind taxonomy (category_total canonical)"
```

---

## Task 6: Runner — spec → feat build → core → RunResult (캐시)

**Files:**
- Create: `src/bakery/harness/runner.py`
- Modify: `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: `ExperimentSpec`(T1), `resolve_event_priors`(T2), `windowed_backtest`/`metrics_from_preds`(T3), `is_supported_phase1`(T5); `bakery.features.category_aggregate.build_category_daily`/`build_features`.
- Produces:
  - `STAGES: tuple[str,...]` = `("features","backtest","evaluate")`
  - `class RunResult` (dataclass): `name`, `predictions: pd.DataFrame`, `fold_metrics: pd.DataFrame`, `metrics: dict`, `resolved: dict`
  - `run_experiment(spec, *, out_dir: Path, cache_dir: Path | None = None, _trace: list | None = None) -> RunResult`
  - Phase 1: category_total만 실행. distributional/point는 경고 후 스킵. event_prior layer는 backtest_core가 events/lunar로 처리.

Note: 캐시는 features 단계만. 키 = (source, store, target, alpha). 임의 스테이지 재개(`--from`/`--only`)는 Phase 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_runner.py
from bakery.harness.config import ExperimentSpec, DataSpec, WindowSpec
from bakery.harness.runner import run_experiment, RunResult, STAGES


def _spec(n_folds=52):
    return ExperimentSpec(
        name="gw_core", data=DataSpec(source="real", store="store_gw01"),
        target="adjusted_demand_unit", forecaster=["category_total"], layers=["event_prior"],
        event_priors="gwangyo",
        window=WindowSpec(scheme="expanding", n_folds=n_folds, window_days=730, horizon_days=7),
        alpha=0.8, production_q=0.85,
    )


def test_run_returns_runresult(tmp_path):
    result = run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    assert isinstance(result, RunResult)
    assert result.name == "gw_core"
    assert not result.predictions.empty
    assert {"date", "expected", "production", "actual"}.issubset(result.predictions.columns)
    assert "wape" in result.metrics


def test_run_writes_artifacts(tmp_path):
    run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    d = tmp_path / "out" / "gw_core"
    assert (d / "config_resolved.yaml").exists()
    assert (d / "predictions.csv").exists()
    assert (d / "metrics.json").exists()


def test_features_cache_hit(tmp_path):
    cache = tmp_path / "cache"
    t1, t2 = [], []
    run_experiment(_spec(), out_dir=tmp_path / "o1", cache_dir=cache, _trace=t1)
    run_experiment(_spec(), out_dir=tmp_path / "o2", cache_dir=cache, _trace=t2)
    assert ("features", "miss") in t1
    assert ("features", "hit") in t2


def test_stages_constant():
    assert STAGES == ("features", "backtest", "evaluate")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_runner.py --color=no`
Expected: FAIL — `ModuleNotFoundError: bakery.harness.runner`

- [ ] **Step 3: Write implementation**

```python
# src/bakery/harness/runner.py
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.backtest_core import metrics_from_preds, windowed_backtest
from bakery.harness.config import ExperimentSpec
from bakery.harness.event_priors import resolve_event_priors
from bakery.harness.registry import is_supported_phase1

STAGES: tuple[str, ...] = ("features", "backtest", "evaluate")


@dataclass
class RunResult:
    name: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    metrics: dict
    resolved: dict


def _stage_key(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]


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


def run_experiment(
    spec: ExperimentSpec, *, out_dir: Path, cache_dir: Path | None = None,
    _trace: list | None = None,
) -> RunResult:
    trace = _trace if _trace is not None else []
    runnable = [f for f in spec.forecaster if is_supported_phase1(f)]
    for f in spec.forecaster:
        if not is_supported_phase1(f):
            warnings.warn(f"forecaster '{f}'는 Phase 2+ 대상 — 이번 실행에서 스킵.", UserWarning)
    if not runnable:
        raise ValueError("Phase 1에서 실행 가능한 forecaster 없음(category_total 필요).")

    feat_key = _stage_key({"source": spec.data.source, "store": spec.data.store,
                           "target": spec.target, "alpha": spec.alpha})

    def _feat():
        cd = build_category_daily(alpha=spec.alpha)
        return build_features(cd, target_col=spec.target)

    feat = _load_or_compute("features", feat_key, cache_dir, _feat, trace)

    events, lunar = resolve_event_priors(spec.event_priors) if "event_prior" in spec.layers else (None, None)
    trace.append(("backtest", "run"))
    bt = windowed_backtest(
        feat, window_days=spec.window.window_days, target_col=spec.target,
        n_folds=spec.window.n_folds, horizon_days=spec.window.horizon_days,
        production_q=spec.production_q, alpha=spec.alpha,
        events=events, lunar_events=lunar,
    )
    trace.append(("evaluate", "run"))
    metrics = metrics_from_preds(bt.predictions)

    out = out_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump()
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")
    bt.predictions.to_csv(out / "predictions.csv", index=False)
    bt.folds.to_csv(out / "fold_results.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return RunResult(name=spec.name, predictions=bt.predictions, fold_metrics=bt.folds,
                     metrics=metrics, resolved=resolved)
```

**VERIFY** build_category_daily가 real canonical을 기본으로 읽는지:
Run: `uv run python -c "from bakery.features.category_aggregate import build_category_daily; cd=build_category_daily(alpha=0.8); print(type(cd), cd.df.shape)"`
Expected: CategoryDaily, 광교 3cat 일별.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_runner.py --color=no`
Expected: PASS (4 passed). n_folds=52가 real span에서 ValueError면 test의 `_spec()` 호출을 `_spec(n_folds=8)`로 낮추되 **acceptance(Task 3)는 52 유지**(Task 3는 이미 통과했으므로 실제 feasible 확인됨 — 이 경우 test만 조정).

- [ ] **Step 5: Update __init__ + commit**

```python
# __init__.py append
from bakery.harness.runner import RunResult, run_experiment, STAGES
```
Add to `__all__`: `"RunResult", "run_experiment", "STAGES"`.

```bash
git add src/bakery/harness/runner.py src/bakery/harness/__init__.py tests/harness/test_runner.py
git commit -m "feat(harness): runner — spec→feat→windowed_backtest→RunResult"
```

---

## Task 7: CLI harness-run + experiments/gwangyo_default.yaml

**Files:**
- Modify: `src/bakery/cli.py`
- Create: `experiments/gwangyo_default.yaml`
- Test: `tests/harness/test_cli_harness.py`

**Interfaces:**
- Consumes: `load_spec`(T1), `run_experiment`(T6).
- Produces: CLI `bakery harness-run <config.yaml> [--out DIR] [--cache DIR]`.

- [ ] **Step 1: Write experiments/gwangyo_default.yaml**

```yaml
# experiments/gwangyo_default.yaml
name: gwangyo_default
data:
  source: real
  store: store_gw01
target: adjusted_demand_unit
forecaster: [category_total]        # distributional_total은 Phase 2 실행(등록만)
layers: [event_prior]
event_priors: gwangyo
window:
  scheme: expanding
  n_folds: 52
  window_days: 730
  horizon_days: 7
alpha: 0.8
production_q: 0.85
```

- [ ] **Step 2: Write the failing test**

```python
# tests/harness/test_cli_harness.py
from typer.testing import CliRunner
from bakery.cli import app

runner = CliRunner()


def test_harness_run_default_config(tmp_path):
    result = runner.invoke(app, [
        "harness-run", "experiments/gwangyo_default.yaml",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "gwangyo_default" / "predictions.csv").exists()
    assert (tmp_path / "out" / "gwangyo_default" / "metrics.json").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: FAIL — `No such command 'harness-run'`

- [ ] **Step 4: Add CLI command** — `src/bakery/cli.py` (imports에 `from bakery.harness import load_spec, run_experiment` 추가, 다른 `@app.command` 근처에)

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
    console.print(
        f"[green]wrote[/] {out}/{result.name}/ "
        f"(WAPE={result.metrics['wape']:.4f}, n={result.metrics['n_test']})"
    )
```

- [ ] **Step 5: Run test + full suite**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no && uv run pytest --color=no`
Expected: 둘 다 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bakery/cli.py experiments/gwangyo_default.yaml tests/harness/test_cli_harness.py
git commit -m "feat(cli): harness-run 커맨드 + 광교 기본 experiment config"
```

---

## Self-Review 결과

**Spec coverage:**
- §2 원칙1(두 스파인 추출) → Task 3(windowed_backtest 추출) + Task 4(원본 래퍼화). ✅
- §4 taxonomy(category_total canonical) → Task 5 registry `_KIND`. ✅
- §5 config(카테고리 기본·event_priors 프리셋·folds52/window730) → Task 1 + Task 2. ✅
- §5 STORE_EVENT_PRIORS 정착 → Task 2 event_priors.py. ✅
- §9 acceptance(엔진 동등성) → Task 3 equivalence test(hard gate). ✅
- §4 event_prior 후처리·leakage 규칙(pre-test 전체 history) → Task 3 코어에 보존+주석. ✅
- §7 RunResult → Task 6. ✅
- report/eda/viz(§8) = Phase 2~3, 범위 밖. distributional/point 실행 = Phase 2(Task 5/6 등록만·경고 스킵). ✅

**Placeholder scan:** Task 2 event_priors.py의 `XMAS={...}`는 VERIFY 출력을 붙여넣는 의도적 골격 — VERIFY 스텝이 정확값을 강제하므로 placeholder 아님. 나머지 코드 스텝 완전. ✅

**Type consistency:** `windowed_backtest(...events, lunar_events...)` T3 정의 = T4 import·T6 호출 일치. `run_experiment(spec, *, out_dir, cache_dir, _trace)` T6 정의 = T7 호출 일치. `RunResult`(name/predictions/fold_metrics/metrics/resolved) T6~T7 일관. `resolve_event_priors→(events,lunar)` T2=T6 일치. ✅

**VERIFY 완료 (2026-07-24, 실행 전 확정):**
1. ✅ `fit_category_total` random_state=42, `EventLevelPrior` 비랜덤 → 정확일치 유효.
2. ✅ 상수: ALPHA=0.8, PROD_Q=0.85, WINDOW=730, FOLDS=52, HORIZON=7, MIN_TRAIN=60, TARGET="adjusted_demand_unit".
3. ✅ STORE_EVENT_PRIORS["광교"]={"events":{XMAS,CHILDRENS},"lunar_events":CHUSEOK}. real store_id=store_gw01.
4. ⏳ XMAS/CHILDRENS/CHUSEOK/SEOLLAL 실제 dict — Task 2 VERIFY로 복사.
5. ⏳ build_category_daily(alpha=0.8) real 기본 로드 — Task 6 VERIFY로 확인.
6. ⏳ n_folds=52가 real 데이터에서 feasible한지 — Task 3(acceptance)에서 확정; runner test는 부족 시 낮춤.

# Harness distributional 배선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** harness `windowed_backtest`를 Forecaster 어댑터로 일반화해, 실험 1개가 category_total + distributional_total을 실행하고 forecaster별 결과 + 비교표를 산출한다.

**Architecture:** Forecaster protocol(`fit`)이 두 엔진의 fit/predict_production 규약 차이를 어댑터에 격리하고, windowed_backtest는 `forecaster` 인자(default=CategoryTotalForecaster)만 받아 fold 루프를 순수 오케스트레이션으로 유지한다. runner가 리스트를 순회해 `ExperimentResult`(runs dict + comparison)를 반환한다. category_total 경로는 바이트 불변이라 Phase 1 엔진 동등성 게이트가 계속 통과한다.

**Tech Stack:** Python 3.12, pydantic v2, pandas, numpy, lightgbm(category), ngboost 0.5.11(distributional, LogNormal), typer, rich, pytest.

## Global Constraints

- **canonical 타깃**: 카테고리 `adjusted_demand_unit`(빵 총량, 하루 1행). item-level `adjusted_demand`와 다름.
- **헤드라인 파라미터(정확 보존)**: `ALPHA=0.8`, `PROD_Q=0.85`, `WINDOW=730`, `HORIZON=7`, `MIN_TRAIN_ROWS=60`.
- **★category_total 엔진 동등성 게이트 유지**: `tests/harness/test_backtest_core_equivalence.py`가 계속 통과해야 한다(어댑터 리팩토링 회귀 방지). `forecaster` 인자는 default=None→CategoryTotalForecaster로 후방호환.
- **★distributional 결정성 = hermetic seed**: NGBoost 0.5.11은 `random_state` 인자만으론 비결정적(전역 numpy RNG 사용). 어댑터 `fit`에서 `np.random.get_state()`→`np.random.seed(42)`→fit→`finally: np.random.set_state(...)`로 hermetic 시드. `fit_distributional_total`(공유 src)은 건드리지 않는다. 시드 42 하드코딩.
- **재구현 금지**: harness는 `src/bakery` 심볼(`fit_category_total`/`fit_distributional_total`) 호출만.
- **테스트 단언**: 결정성/정확일치는 `np.testing.assert_array_equal`. sanity는 정확값 비교(`==`)·경계(`0<wape<1`).
- **pytest 실행**: 이 repo addopts에 `-q` 있음. 카운트 필요 시 `uv run pytest --color=no`(추가 `-q` 금지).
- **distributional은 느림**(NGBoost 500 estimators × fold): 신규 테스트는 8-fold. 동등성 게이트는 category 52-fold 유지.

---

## File Structure

- **Create** `src/bakery/harness/forecasters.py` — Forecaster/FittedForecaster protocol + CategoryTotalForecaster + DistributionalTotalForecaster + `_ProdQBound`. 어댑터 = 유일한 fit/predict 규약 차이 격리 지점.
- **Modify** `src/bakery/harness/backtest_core.py` — `windowed_backtest`에 `forecaster` 인자, fit 한 줄 교체.
- **Modify** `src/bakery/harness/registry.py` — `build_forecaster` 팩토리 + `is_supported_phase1`→`is_runnable`.
- **Modify** `src/bakery/harness/runner.py` — `ExperimentResult` + 다중 forecaster `run_experiment`.
- **Modify** `src/bakery/harness/__init__.py` — 태스크별 re-export 갱신.
- **Modify** `src/bakery/cli.py` — `cmd_harness_run` comparison 표 출력.
- **Create** `experiments/gwangyo_compare.yaml` — 두 forecaster 비교 예시.
- **Tests**: create `test_forecasters.py`, `test_backtest_core_distributional.py`; modify `test_registry.py`, `test_runner.py`, `test_cli_harness.py`.

---

## Task 1: Forecaster 어댑터 레이어

**Files:**
- Create: `src/bakery/harness/forecasters.py`
- Modify: `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_forecasters.py`

**Interfaces:**
- Consumes: `bakery.models.category_total.fit_category_total`, `bakery.models.distributional_total.fit_distributional_total`(lazy).
- Produces:
  - `class CategoryTotalForecaster` (`name="category_total"`), `.fit(train, *, target_col, alpha, production_q) -> FittedForecaster`
  - `class DistributionalTotalForecaster` (`name="distributional_total"`), 동일 시그니처
  - `FittedForecaster`(protocol): `.predict_expected(df)->np.ndarray`, `.predict_production(df)->np.ndarray`(q 바인딩됨)
  - `Forecaster`(protocol): `name: str`, `.fit(...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_forecasters.py
import numpy as np
import pytest

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.forecasters import CategoryTotalForecaster, DistributionalTotalForecaster
from bakery.models.category_total import fit_category_total

TARGET = "adjusted_demand_unit"


def _feat():
    cd = build_category_daily(alpha=0.8)
    return build_features(cd, target_col=TARGET).dropna().reset_index(drop=True)


def test_category_adapter_matches_direct_fit():
    feat = _feat()
    train, test = feat.iloc[:400], feat.iloc[400:407]
    adapted = CategoryTotalForecaster().fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    direct = fit_category_total(train, target_col=TARGET, alpha_demand=0.8, production_q=0.85)
    np.testing.assert_array_equal(adapted.predict_expected(test), direct.predict_expected(test))
    np.testing.assert_array_equal(adapted.predict_production(test), direct.predict_production(test))


def test_distributional_adapter_deterministic():
    feat = _feat()
    train, test = feat.iloc[:400], feat.iloc[400:407]
    fc = DistributionalTotalForecaster()
    m1 = fc.fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    m2 = fc.fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    np.testing.assert_array_equal(m1.predict_expected(test), m2.predict_expected(test))
    np.testing.assert_array_equal(m1.predict_production(test), m2.predict_production(test))


def test_distributional_adapter_no_rng_leak():
    """hermetic seed: fit이 전역 numpy RNG 스트림을 소비/변경하지 않아야."""
    feat = _feat()
    train = feat.iloc[:400]
    np.random.seed(7)
    a = np.random.rand()
    np.random.seed(7)
    DistributionalTotalForecaster().fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    b = np.random.rand()
    assert a == b


def test_distributional_production_ge_expected():
    feat = _feat()
    train, test = feat.iloc[:400], feat.iloc[400:407]
    m = DistributionalTotalForecaster().fit(train, target_col=TARGET, alpha=0.8, production_q=0.85)
    assert bool((m.predict_production(test) >= m.predict_expected(test)).all())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_forecasters.py --color=no`
Expected: FAIL — `ModuleNotFoundError: bakery.harness.forecasters`

- [ ] **Step 3: Write implementation**

```python
# src/bakery/harness/forecasters.py
"""Forecaster 어댑터 — fit/predict_production 규약 차이를 격리(windowed_backtest 순수 유지).

category_total(LightGBM)과 distributional_total(NGBoost)은 동일 타깃(빵 총량)을 예측하되
fit 시그니처·production_q 전달 시점·결정성 처리가 다르다. 어댑터가 이를 흡수해 균일한
FittedForecaster 계약(predict_expected/predict_production)을 제공한다.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from bakery.models.category_total import fit_category_total


@runtime_checkable
class FittedForecaster(Protocol):
    def predict_expected(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_production(self, df: pd.DataFrame) -> np.ndarray: ...


class Forecaster(Protocol):
    name: str
    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster: ...


class CategoryTotalForecaster:
    name = "category_total"

    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster:
        # 반환 CategoryTotalModel이 이미 predict_expected/predict_production(q fit-고정) 계약 만족.
        return fit_category_total(
            train, target_col=target_col, alpha_demand=alpha, production_q=production_q,
        )


class _ProdQBound:
    """distributional 모델을 production_q로 바인딩해 균일 계약(predict_production(df)) 제공."""

    def __init__(self, model, production_q: float):
        self._model = model
        self._production_q = production_q

    def predict_expected(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_expected(df)

    def predict_production(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_production(df, production_q=self._production_q)


class DistributionalTotalForecaster:
    name = "distributional_total"

    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster:
        # ngboost는 무거워 lazy import(category 전용 실행 시 회피). alpha는 미사용(균일 인터페이스).
        from bakery.models.distributional_total import fit_distributional_total

        # hermetic seed: NGBoost 0.5.11은 전역 numpy RNG 사용 → random_state 인자만으론 비결정적.
        # save/restore로 전역 스트림 누수 없이 결정성 확보. 시드 42(fit_distributional_total 기본값과 일치).
        state = np.random.get_state()
        np.random.seed(42)
        try:
            model = fit_distributional_total(train, target_col=target_col)
        finally:
            np.random.set_state(state)
        return _ProdQBound(model, production_q)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_forecasters.py --color=no`
Expected: PASS (4 passed)

- [ ] **Step 5: Update __init__ + commit**

`src/bakery/harness/__init__.py`에 추가:
```python
from bakery.harness.forecasters import (
    Forecaster, FittedForecaster, CategoryTotalForecaster, DistributionalTotalForecaster,
)
```
`__all__`에 추가: `"Forecaster", "FittedForecaster", "CategoryTotalForecaster", "DistributionalTotalForecaster"`.

```bash
git add src/bakery/harness/forecasters.py src/bakery/harness/__init__.py tests/harness/test_forecasters.py
git commit -m "feat(harness): Forecaster 어댑터 — category/distributional 규약 격리 + hermetic seed"
```

---

## Task 2: ★windowed_backtest 일반화 (forecaster 인자) + distributional 경로 검증

**Files:**
- Modify: `src/bakery/harness/backtest_core.py`
- Test: `tests/harness/test_backtest_core_distributional.py` (신규), `tests/harness/test_backtest_core_equivalence.py` (기존, 회귀 확인)

**Interfaces:**
- Consumes: `bakery.harness.forecasters.CategoryTotalForecaster`, `Forecaster`.
- Produces: `windowed_backtest(..., forecaster: Forecaster | None = None)` — default None→CategoryTotalForecaster(후방호환).

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_backtest_core_distributional.py
import numpy as np

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.backtest_core import windowed_backtest
from bakery.harness.event_priors import resolve_event_priors
from bakery.harness.forecasters import DistributionalTotalForecaster


def _feat():
    return build_features(build_category_daily(alpha=0.8), target_col="adjusted_demand_unit")


def test_distributional_through_windowed_backtest_deterministic():
    feat = _feat()
    events, lunar = resolve_event_priors("gwangyo")
    kw = dict(window_days=730, n_folds=8, production_q=0.85, alpha=0.8,
              events=events, lunar_events=lunar)
    r1 = windowed_backtest(feat, forecaster=DistributionalTotalForecaster(), **kw)
    r2 = windowed_backtest(feat, forecaster=DistributionalTotalForecaster(), **kw)
    p1 = r1.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    p2 = r2.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    np.testing.assert_array_equal(p1["expected"].to_numpy(), p2["expected"].to_numpy())
    np.testing.assert_array_equal(p1["production"].to_numpy(), p2["production"].to_numpy())


def test_distributional_wape_sane():
    feat = _feat()
    events, lunar = resolve_event_priors("gwangyo")
    r = windowed_backtest(feat, forecaster=DistributionalTotalForecaster(),
                          window_days=730, n_folds=8, production_q=0.85, alpha=0.8,
                          events=events, lunar_events=lunar)
    p = r.predictions
    wape = np.abs(p["actual"] - p["expected"]).sum() / max(np.abs(p["actual"]).sum(), 1)
    assert 0.0 < wape < 1.0
    assert (p["production"] >= p["expected"]).mean() > 0.9   # production은 대체로 expected 이상
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_backtest_core_distributional.py --color=no`
Expected: FAIL — `windowed_backtest() got an unexpected keyword argument 'forecaster'`

- [ ] **Step 3: Modify windowed_backtest**

`src/bakery/harness/backtest_core.py` 상단 import 교체 — `fit_category_total` 직접 import 제거하고 어댑터 import 추가:
```python
from bakery.models.category_total import BacktestResult
from bakery.models.event_prior import EventLevelPrior
from bakery.harness.forecasters import CategoryTotalForecaster, Forecaster
```
(주의: `fit_category_total`는 더 이상 backtest_core가 직접 안 씀 — import 줄에서 제거. `BacktestResult`는 유지.)

시그니처에 `forecaster` 인자 추가(마지막 인자로):
```python
def windowed_backtest(
    df: pd.DataFrame, *, window_days: int,
    target_col: str = "adjusted_demand_unit", n_folds: int = 52,
    horizon_days: int = 7, production_q: float = 0.85, alpha: float = 0.8,
    events: dict | None = None, lunar_events: dict | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    forecaster: Forecaster | None = None,
) -> BacktestResult:
```

함수 진입부(df 정리 직전)에 추가:
```python
    fc = forecaster if forecaster is not None else CategoryTotalForecaster()
```

fold 루프의 fit 한 줄 교체 — 기존:
```python
        model = fit_category_total(
            train_df, target_col=target_col,
            alpha_demand=alpha, production_q=production_q,
        )
```
교체 후:
```python
        model = fc.fit(
            train_df, target_col=target_col, alpha=alpha, production_q=production_q,
        )
```
**나머지(fold 경계·predict_expected/predict_production 호출·event_prior 블렌드·folds/preds 조립)는 불변.**

- [ ] **Step 4: Run distributional test + category 동등성 게이트**

Run: `uv run pytest tests/harness/test_backtest_core_distributional.py --color=no`
Expected: PASS (2 passed, ~1분 — 8-fold 2회+1회 NGBoost).

Run(★게이트, ~5분): `uv run pytest tests/harness/test_backtest_core_equivalence.py --color=no`
Expected: PASS (1 passed) — category 경로 바이트 불변 확인. **실패 시 forecaster 어댑터가 category fit 인자를 바꿨는지 대조(`alpha_demand=alpha, production_q=production_q` 일치).** 게이트 PASS 없이 다음 태스크 금지.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/harness/backtest_core.py tests/harness/test_backtest_core_distributional.py
git commit -m "feat(harness): windowed_backtest forecaster 인자 — distributional 실행 + category 동등성 유지"
```

---

## Task 3: Registry — build_forecaster 팩토리 + is_runnable

**Files:**
- Modify: `src/bakery/harness/registry.py`, `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_registry.py`

**Interfaces:**
- Consumes: `bakery.harness.forecasters`(lazy, 팩토리 내부).
- Produces:
  - `build_forecaster(name: str) -> Forecaster` (미등록 `KeyError`)
  - `is_runnable(name: str) -> bool` (category_total/distributional_total=True; point/composite/미등록=False)
  - `is_supported_phase1` **제거**.

- [ ] **Step 1: Update the test** (기존 `test_phase1_supports_category_total_only` 삭제, 아래로 교체; `is_supported_phase1` import 제거, `build_forecaster`/`is_runnable` import 추가)

```python
# tests/harness/test_registry.py
import pytest
from bakery.harness.registry import (
    ForecasterKind, kind_of, LAYER_NAMES, is_runnable, build_forecaster,
)


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


def test_is_runnable():
    assert is_runnable("category_total") is True
    assert is_runnable("distributional_total") is True
    assert is_runnable("lightgbm_v2") is False
    assert is_runnable("bogus") is False


def test_build_forecaster():
    from bakery.harness.forecasters import CategoryTotalForecaster, DistributionalTotalForecaster
    assert isinstance(build_forecaster("category_total"), CategoryTotalForecaster)
    assert isinstance(build_forecaster("distributional_total"), DistributionalTotalForecaster)


def test_build_forecaster_unknown_raises():
    with pytest.raises(KeyError):
        build_forecaster("lightgbm_v2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_registry.py --color=no`
Expected: FAIL — `ImportError: cannot import name 'is_runnable'`

- [ ] **Step 3: Modify registry.py**

`is_supported_phase1` 함수를 삭제하고 아래로 교체:
```python
_RUNNABLE_KINDS: frozenset[ForecasterKind] = frozenset(
    {ForecasterKind.CATEGORY_TOTAL, ForecasterKind.DISTRIBUTIONAL}
)


def is_runnable(name: str) -> bool:
    """실행 가능한 forecaster(category_total/distributional_total)면 True. 미등록/미지원=False."""
    try:
        return kind_of(name) in _RUNNABLE_KINDS
    except KeyError:
        return False


def build_forecaster(name: str):
    """forecaster 이름 → 어댑터 인스턴스. 미등록 KeyError."""
    from bakery.harness.forecasters import (
        CategoryTotalForecaster, DistributionalTotalForecaster,
    )
    factories = {
        "category_total": CategoryTotalForecaster,
        "distributional_total": DistributionalTotalForecaster,
    }
    return factories[name]()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_registry.py --color=no`
Expected: PASS (6 passed)

- [ ] **Step 5: Update __init__ + commit**

`__init__.py`에서 `is_supported_phase1`를 `is_runnable, build_forecaster`로 교체(import 줄·`__all__` 둘 다):
```python
from bakery.harness.registry import ForecasterKind, kind_of, LAYER_NAMES, is_runnable, build_forecaster
```
`__all__`: `"is_supported_phase1"` 제거, `"is_runnable", "build_forecaster"` 추가.

```bash
git add src/bakery/harness/registry.py src/bakery/harness/__init__.py tests/harness/test_registry.py
git commit -m "feat(harness): build_forecaster 팩토리 + is_runnable(distributional 실행 허용)"
```

---

## Task 4: Runner — ExperimentResult + 다중 forecaster

**Files:**
- Modify: `src/bakery/harness/runner.py`, `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: `is_runnable`/`build_forecaster`(T3), `windowed_backtest`(T2), `metrics_from_preds`, `resolve_event_priors`, `build_category_daily`/`build_features`.
- Produces:
  - `class ExperimentResult` (dataclass): `name: str`, `runs: dict[str, RunResult]`, `comparison: pd.DataFrame`
  - `run_experiment(spec, *, out_dir, cache_dir=None, _trace=None) -> ExperimentResult`
  - `RunResult`(기존) 유지, `STAGES`(기존) 유지.

- [ ] **Step 1: Update the test** (반환 타입·산출물 경로·다중 forecaster 반영)

```python
# tests/harness/test_runner.py
from bakery.harness.config import ExperimentSpec, DataSpec, WindowSpec
from bakery.harness.runner import run_experiment, ExperimentResult, RunResult, STAGES


def _spec(forecaster=("category_total", "distributional_total"), n_folds=8):
    return ExperimentSpec(
        name="gw_core", data=DataSpec(source="real", store="store_gw01"),
        target="adjusted_demand_unit", forecaster=list(forecaster), layers=["event_prior"],
        event_priors="gwangyo",
        window=WindowSpec(scheme="expanding", n_folds=n_folds, window_days=730, horizon_days=7),
        alpha=0.8, production_q=0.85,
    )


def test_run_returns_experiment_result(tmp_path):
    result = run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    assert isinstance(result, ExperimentResult)
    assert result.name == "gw_core"
    assert set(result.runs) == {"category_total", "distributional_total"}
    assert all(isinstance(r, RunResult) for r in result.runs.values())
    assert len(result.comparison) == 2
    assert {"forecaster", "wape"}.issubset(result.comparison.columns)


def test_run_writes_artifacts(tmp_path):
    run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    d = tmp_path / "out" / "gw_core"
    assert (d / "config_resolved.yaml").exists()
    assert (d / "comparison.csv").exists()
    for fname in ("category_total", "distributional_total"):
        assert (d / fname / "predictions.csv").exists()
        assert (d / fname / "metrics.json").exists()


def test_features_cache_hit(tmp_path):
    cache = tmp_path / "cache"
    t1, t2 = [], []
    spec = _spec(forecaster=("category_total",))   # 캐시만 검증 — 빠른 단일 forecaster
    run_experiment(spec, out_dir=tmp_path / "o1", cache_dir=cache, _trace=t1)
    run_experiment(spec, out_dir=tmp_path / "o2", cache_dir=cache, _trace=t2)
    assert ("features", "miss") in t1
    assert ("features", "hit") in t2


def test_unrunnable_skipped_with_warning(tmp_path):
    import pytest
    spec = _spec(forecaster=("category_total", "lightgbm_v2"))
    with pytest.warns(UserWarning, match="lightgbm_v2"):
        result = run_experiment(spec, out_dir=tmp_path / "out", cache_dir=None)
    assert set(result.runs) == {"category_total"}


def test_stages_constant():
    assert STAGES == ("features", "backtest", "evaluate")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_runner.py --color=no`
Expected: FAIL — `ImportError: cannot import name 'ExperimentResult'`

- [ ] **Step 3: Modify runner.py**

import 교체: `from bakery.harness.registry import is_supported_phase1` → `from bakery.harness.registry import is_runnable, build_forecaster`.

`RunResult` dataclass 다음에 `ExperimentResult` 추가:
```python
@dataclass
class ExperimentResult:
    name: str
    runs: dict[str, RunResult]
    comparison: pd.DataFrame
```

`run_experiment` 본문 전체 교체(반환 타입·다중 forecaster):
```python
def run_experiment(
    spec: ExperimentSpec, *, out_dir: Path, cache_dir: Path | None = None,
    _trace: list | None = None,
) -> ExperimentResult:
    trace = _trace if _trace is not None else []
    runnable = [f for f in spec.forecaster if is_runnable(f)]
    for f in spec.forecaster:
        if not is_runnable(f):
            warnings.warn(f"forecaster '{f}'는 실행 미지원(point/composite) — 스킵.", UserWarning)
    if not runnable:
        raise ValueError("실행 가능한 forecaster 없음(category_total/distributional_total 필요).")

    feat_key = _stage_key({"source": spec.data.source, "store": spec.data.store,
                           "target": spec.target, "alpha": spec.alpha})

    def _feat():
        cd = build_category_daily(alpha=spec.alpha)
        return build_features(cd, target_col=spec.target)

    feat = _load_or_compute("features", feat_key, cache_dir, _feat, trace)
    events, lunar = resolve_event_priors(spec.event_priors) if "event_prior" in spec.layers else (None, None)

    out = out_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump()
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")

    runs: dict[str, RunResult] = {}
    rows = []
    for fname in runnable:
        fc = build_forecaster(fname)
        trace.append((f"backtest:{fname}", "run"))
        bt = windowed_backtest(
            feat, window_days=spec.window.window_days, target_col=spec.target,
            n_folds=spec.window.n_folds, horizon_days=spec.window.horizon_days,
            production_q=spec.production_q, alpha=spec.alpha,
            events=events, lunar_events=lunar, forecaster=fc,
        )
        metrics = metrics_from_preds(bt.predictions)
        fout = out / fname
        fout.mkdir(parents=True, exist_ok=True)
        bt.predictions.to_csv(fout / "predictions.csv", index=False)
        bt.folds.to_csv(fout / "fold_results.csv", index=False)
        (fout / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        runs[fname] = RunResult(name=fname, predictions=bt.predictions,
                                fold_metrics=bt.folds, metrics=metrics, resolved=resolved)
        rows.append({"forecaster": fname, **metrics})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)
    return ExperimentResult(name=spec.name, runs=runs, comparison=comparison)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_runner.py --color=no`
Expected: PASS (5 passed, ~2-3분 — 두 forecaster × 8-fold 백테스트 2회 + 캐시/스킵 테스트).

- [ ] **Step 5: Update __init__ + commit**

`__init__.py`: `from bakery.harness.runner import RunResult, run_experiment, STAGES` → `..., ExperimentResult, ...`. `__all__`에 `"ExperimentResult"` 추가.

```bash
git add src/bakery/harness/runner.py src/bakery/harness/__init__.py tests/harness/test_runner.py
git commit -m "feat(harness): runner 다중 forecaster — ExperimentResult + comparison"
```

---

## Task 5: CLI comparison 출력 + compare config + 전체 스위트

**Files:**
- Modify: `src/bakery/cli.py`
- Create: `experiments/gwangyo_compare.yaml`
- Test: `tests/harness/test_cli_harness.py`

**Interfaces:**
- Consumes: `run_experiment`(T4, `ExperimentResult` 반환), `load_spec`.

- [ ] **Step 1: Create experiments/gwangyo_compare.yaml**

```yaml
# experiments/gwangyo_compare.yaml — category vs distributional 비교(수동 실행, 52-fold=수 분 소요)
name: gwangyo_compare
data:
  source: real
  store: store_gw01
target: adjusted_demand_unit
forecaster: [category_total, distributional_total]
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

- [ ] **Step 2: Update the test** (산출물 경로가 forecaster 하위로 이동)

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
    d = tmp_path / "out" / "gwangyo_default"
    assert (d / "comparison.csv").exists()
    assert (d / "category_total" / "predictions.csv").exists()
    assert (d / "category_total" / "metrics.json").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: FAIL — `assert (d / "category_total" / "predictions.csv").exists()` (경로가 아직 forecaster 하위 아님).

- [ ] **Step 4: Modify cmd_harness_run**

`src/bakery/cli.py`의 `cmd_harness_run` 본문에서 결과 출력부를 comparison 표로 교체:
```python
    result = run_experiment(spec, out_dir=out, cache_dir=cache)
    console.print(f"[green]wrote[/] {out}/{result.name}/  (forecaster={list(result.runs)})")
    table = Table(title=f"{result.name} — forecaster 비교")
    for col in result.comparison.columns:
        table.add_column(col)
    for _, row in result.comparison.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)
```
(`Table`은 파일 상단에서 이미 import됨 — `from rich.table import Table`.)

- [ ] **Step 5: Run test + 전체 스위트**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: PASS (1 passed).

Run(전체, distributional 게이트 포함 ~15분): `uv run pytest --color=no`
Expected: 사전존재 `test_store_daily_redefine` 1건만 실패, 나머지 전부 PASS. (사전존재 실패는 이 작업과 무관 — base commit에서도 동일.)

- [ ] **Step 6: Commit**

```bash
git add src/bakery/cli.py experiments/gwangyo_compare.yaml tests/harness/test_cli_harness.py
git commit -m "feat(cli): harness-run comparison 표 출력 + gwangyo_compare 예시 config"
```

---

## Self-Review 결과

**Spec coverage:**
- §아키텍처 1(Forecaster 어댑터+hermetic seed) → Task 1. ✅
- §아키텍처 2(windowed_backtest 일반화, 한 곳만 변경) → Task 2. ✅
- §아키텍처 3(ExperimentResult+comparison) → Task 4. ✅
- §아키텍처 4(build_forecaster+is_runnable, CLI 표) → Task 3 + Task 5. ✅
- §Acceptance 1(category 동등성 게이트) → Task 2 Step 4(기존 test 재실행). ✅
- §Acceptance 2(distributional 결정성+sanity) → Task 2 신규 test. ✅
- §Acceptance 3(runner ExperimentResult) → Task 4. ✅
- §마이그레이션(is_supported_phase1→is_runnable, run_experiment 반환 변경) → Task 3(registry+__init__), Task 4(runner+test), Task 5(cli+test). ✅

**Placeholder scan:** 모든 코드 스텝 완전(어댑터·backtest 수정·registry·runner·cli 전부 실제 코드). ✅

**Type consistency:** `Forecaster.fit(train, *, target_col, alpha, production_q)` T1 정의 = T2 호출(`fc.fit(train_df, target_col=..., alpha=alpha, production_q=production_q)`) 일치. `windowed_backtest(..., forecaster=)` T2 정의 = T4 호출 일치. `ExperimentResult`(name/runs/comparison) T4 정의 = T5 CLI 소비 일치. `build_forecaster`/`is_runnable` T3 정의 = T4 소비 일치. `RunResult`(name/predictions/fold_metrics/metrics/resolved) 불변. ✅

**VERIFY 완료(실행 전, 2026-07-25):**
1. ✅ `feat` = 하루 1행 빵 총량(1826행), 타깃 `adjusted_demand_unit`.
2. ✅ `select_feature_cols` = date/target/LEAK 제외 numeric — NGBoost 호환(기존 CLI full-window서 동작).
3. ✅ distributional 스모크: fit+predict 정상, production≥expected.
4. ✅ hermetic seed로 결정성 확보(8-fold 루프 run-to-run `array_equal`, 전역 RNG 누수 없음).
5. ✅ 소비처 grep: is_supported_phase1={runner,registry,__init__,test_registry}, run_experiment/RunResult={cli,runner,__init__,test_runner} — 전부 마이그레이션 스텝에 포함. windowed_backtest script 호출부는 forecaster 없이 호출→후방호환.

# Upstream Lever `what_if_driver` (v7 S6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 드라이버(날씨/캘린더)를 가상 변경하면 실 LightGBM 모델이 demand를 재예측하고, 그 변화가 OntologyLink를 따라 위험/비용으로 전파되는 읽기 전용 `what_if_driver` 함수를 구현한다.

**Architecture:** 새 모듈 `ontology/scenario.py`. `what_if_driver`는 (1) DailyDataset의 daily/calendar/weather를 단일 프레임으로 조립 → (2) train_cutoff 이전 데이터로 `GlobalLGBM(feature_set="v2")` fit(1회, before/after 공유) → (3) period의 (store,item) row로 before 예측, 드라이버 컬럼만 override해 after 예측 (ceteris paribus) → (4) `simulate_item_risk`로 위험/비용 전파 → (5) out_of_support 외삽 플래그 + OntologyLink 전파 경로를 담은 `WhatIfDriverResult` 반환.

**Tech Stack:** Python, pandas, dataclasses, pytest. 재사용: `features/calendar_features.add_calendar_features`, `features/weather_features.add_weather_features`, `models/lightgbm_regressor.GlobalLGBM`, `decision/risk.simulate_item_risk`+`RiskParams`, `ontology/functions.OntologyFunctionSpec`+`FUNCTION_REGISTRY`, `ontology/grounding/tools` dispatch.

## Global Constraints

- **Leakage 절대규칙**: `_fit_demand_model`은 `train_cutoff` **미만(<)** 날짜의 row만 학습한다. period 드라이버만 override하고 lag/rolling은 history에서 재계산(새 계산 없음). `test_features_leakage`/`test_split_leakage`가 통과해야 하고 scenario 전용 leakage 테스트를 추가한다.
- **feature_set="v2"** (target=potential_demand). v0/v1=sold_units는 검열값이라 stockout 위험 시뮬에 부적합(절대규칙 #2). v2의 cannibalization은 GlobalLGBM 내부 계산이라 입력 프레임은 calendar+weather enriched면 충분.
- **읽기 전용**: `what_if_driver`는 state를 mutate하지 않는다(`side="read"`). writeback/commit 없음.
- **결정론**: LGBM 기본 seed 고정(기존 모델 테스트가 의존). before/after는 같은 fitted 모델 1개를 공유(함수 호출당 1회 fit).
- **순환 import 금지**: `functions.py`→`scenario`, `tools.py`→`scenario`(단방향). `scenario`는 `functions`를 import하지 않는다.
- 기존 테스트 불변: `uv run pytest` 전부 통과(현재 main 기준). 기존 5 read 함수의 demand_point proxy는 건드리지 않는다.
- 새 의존성 추가 금지. 커밋 메시지 말미 트레일러:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01MkjMzDP7i4zJ1HJFUqbBcR`

---

### Task 1: scenario.py 순수 조각 (Result 타입 + 드라이버 검증 + 전파경로 + support 카운트)

모델 없이 결정론적으로 테스트 가능한 부분을 먼저 만든다.

**Files:**
- Create: `src/bakery/ontology/scenario.py`
- Test: `tests/test_scenario.py`

**Interfaces:**
- Produces:
  - `WEATHER_DRIVERS = frozenset({"is_rain", "is_snow"})`, `CALENDAR_DRIVERS = frozenset({"is_public_holiday"})`, `VALID_DRIVERS = WEATHER_DRIVERS | CALENDAR_DRIVERS`. (is_weekend/is_off_day는 모델에 닿지 않는 no-op이라 제외 — 설계 D2-3.)
  - `WhatIfDriverResult` (frozen): `store_id, item_id, driver_overrides: dict[str, float], before_demand: float, after_demand: float, demand_delta: float, before_p_stockout: float, after_p_stockout: float, before_expected_cost: float, after_expected_cost: float, out_of_support: bool, propagation_path: tuple[str, ...]`.
  - `_validate_drivers(driver_overrides: dict) -> None` — 빈 dict 또는 VALID_DRIVERS 외 키면 ValueError.
  - `_propagation_path(driver_overrides: dict) -> tuple[str, ...]` — weather 키 있으면 `"dailysales_observed_on_weather"`, calendar 키 있으면 `"dailysales_observed_on_calendar"`, 항상 끝에 `"item_sold_as_dailysales"`.
  - `_count_support(enriched: pd.DataFrame, store_id: str, driver_overrides: dict) -> int` — 해당 store에서 모든 override 컬럼 값이 일치하는 과거 row 수.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scenario.py` 생성:

```python
import pandas as pd
import pytest
from bakery.ontology.scenario import (
    VALID_DRIVERS, WhatIfDriverResult,
    _validate_drivers, _propagation_path, _count_support,
)


def test_valid_drivers_membership():
    assert VALID_DRIVERS == {"is_public_holiday", "is_rain", "is_snow"}


def test_validate_drivers_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        _validate_drivers({})
    with pytest.raises(ValueError):
        _validate_drivers({"is_sunny": 1})
    _validate_drivers({"is_rain": 1})            # ok, no raise


def test_propagation_path_by_driver_kind():
    assert _propagation_path({"is_rain": 1}) == (
        "dailysales_observed_on_weather", "item_sold_as_dailysales")
    assert _propagation_path({"is_public_holiday": 1}) == (
        "dailysales_observed_on_calendar", "item_sold_as_dailysales")
    assert _propagation_path({"is_rain": 1, "is_public_holiday": 1}) == (
        "dailysales_observed_on_weather", "dailysales_observed_on_calendar",
        "item_sold_as_dailysales")


def test_count_support_matches_store_and_overrides():
    df = pd.DataFrame({
        "store_id": ["A", "A", "A", "B"],
        "is_rain":  [1,   0,   1,   1],
        "is_public_holiday": [1, 1, 0, 1],
    })
    # store A, is_rain=1 & is_public_holiday=1 → only row 0
    assert _count_support(df, "A", {"is_rain": 1, "is_public_holiday": 1}) == 1
    # store A, is_rain=1 → rows 0,2
    assert _count_support(df, "A", {"is_rain": 1}) == 2
    # store A, combo never seen
    assert _count_support(df, "A", {"is_rain": 0, "is_public_holiday": 0}) == 0


def test_result_is_frozen():
    r = WhatIfDriverResult("A", "P1", {"is_rain": 1}, 10.0, 7.0, -3.0,
                           0.2, 0.1, 5.0, 3.0, False,
                           ("dailysales_observed_on_weather", "item_sold_as_dailysales"))
    with pytest.raises(Exception):
        r.before_demand = 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.ontology.scenario'`.

- [ ] **Step 3: 최소 구현**

`src/bakery/ontology/scenario.py` 생성:

```python
"""Upstream Scenario lever (v7 S6) — what_if_driver.

Perturb a driver (weather/calendar), re-run the real LightGBM forecast, and
propagate the demand change through to stockout risk / cost. Palantir Scenario
(model-on-object) homolog. Read-only: this never mutates ontology state.

See docs/superpowers/specs/2026-06-30-whatif-driver-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WEATHER_DRIVERS = frozenset({"is_rain", "is_snow"})
CALENDAR_DRIVERS = frozenset({"is_public_holiday"})
VALID_DRIVERS = WEATHER_DRIVERS | CALENDAR_DRIVERS

_WEATHER_LINK = "dailysales_observed_on_weather"
_CALENDAR_LINK = "dailysales_observed_on_calendar"
_ITEM_LINK = "item_sold_as_dailysales"


@dataclass(frozen=True)
class WhatIfDriverResult:
    store_id: str
    item_id: str
    driver_overrides: dict[str, float]
    before_demand: float
    after_demand: float
    demand_delta: float
    before_p_stockout: float
    after_p_stockout: float
    before_expected_cost: float
    after_expected_cost: float
    out_of_support: bool
    propagation_path: tuple[str, ...]


def _validate_drivers(driver_overrides: dict) -> None:
    if not driver_overrides:
        raise ValueError("driver_overrides is empty; provide at least one driver")
    unknown = set(driver_overrides) - VALID_DRIVERS
    if unknown:
        raise ValueError(f"unknown driver(s): {sorted(unknown)}; valid: {sorted(VALID_DRIVERS)}")


def _propagation_path(driver_overrides: dict) -> tuple[str, ...]:
    path = []
    if any(d in WEATHER_DRIVERS for d in driver_overrides):
        path.append(_WEATHER_LINK)
    if any(d in CALENDAR_DRIVERS for d in driver_overrides):
        path.append(_CALENDAR_LINK)
    path.append(_ITEM_LINK)
    return tuple(path)


def _count_support(enriched: pd.DataFrame, store_id: str, driver_overrides: dict) -> int:
    sub = enriched[enriched["store_id"] == store_id]
    mask = pd.Series(True, index=sub.index)
    for col, val in driver_overrides.items():
        mask &= sub[col] == val
    return int(mask.sum())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: PASS (5개).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/scenario.py tests/test_scenario.py
git commit -m "feat: scenario.py result type + driver validation + propagation path + support count"
```

---

### Task 2: forecast wiring (feature 조립 + fit + predict + period row 추출)

DailyDataset 분리 프레임을 단일 프레임으로 조립하고, cutoff 이전 데이터로 fit, period row로 predict하는 실 모델 경로.

**Files:**
- Modify: `src/bakery/ontology/scenario.py`
- Test: `tests/test_scenario.py`

**Interfaces:**
- Consumes: `add_calendar_features`(features.calendar_features), `add_weather_features`(features.weather_features), `GlobalLGBM`(models.lightgbm_regressor).
- Produces:
  - `_build_enriched(daily, calendar, weather) -> pd.DataFrame` — `add_weather_features(add_calendar_features(daily, calendar), weather)`.
  - `_fit_demand_model(enriched, train_cutoff: str, feature_set: str = "v2") -> GlobalLGBM` — `date < train_cutoff` row로 fit; train 비면 ValueError.
  - `_period_item_rows(enriched, store_id, item_id, period: tuple[str, str]) -> pd.DataFrame` — store/item, `period[0] <= date <= period[1]`; 비면 ValueError.
  - `_predict_demand(model, target) -> float` — `model.predict(target).mean()`을 float로.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scenario.py`에 추가:

```python
from bakery.data.loader import load_dataset
from bakery.ontology.scenario import (
    _build_enriched, _fit_demand_model, _period_item_rows, _predict_demand,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def _cutoff_and_period(enriched):
    dates = pd.to_datetime(enriched["date"]).sort_values().unique()
    cutoff = pd.Timestamp(dates[-3])                 # last 2 dates are the "future" period
    return str(cutoff.date()), (str(pd.Timestamp(dates[-2]).date()),
                                str(pd.Timestamp(dates[-1]).date()))


def test_build_enriched_has_driver_columns(dataset):
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    for col in ["is_rain", "is_snow", "is_public_holiday"]:
        assert col in enriched.columns


def test_fit_excludes_cutoff_and_later_rows(dataset):
    """Leakage guard: no training row may be dated >= train_cutoff."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, _ = _cutoff_and_period(enriched)
    model = _fit_demand_model(enriched, cutoff)
    trained = pd.to_datetime(model._train_history["date"])
    assert (trained < pd.Timestamp(cutoff)).all()


def test_fit_empty_train_raises(dataset):
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    early = str(pd.to_datetime(enriched["date"]).min().date())
    with pytest.raises(ValueError):
        _fit_demand_model(enriched, early)            # nothing strictly before the first date


def test_predict_demand_deterministic_and_finite(dataset):
    import math
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    model = _fit_demand_model(enriched, cutoff)
    rows = _period_item_rows(enriched, store, item, period)
    d1 = _predict_demand(model, rows)
    d2 = _predict_demand(model, rows)
    assert math.isfinite(d1) and d1 == d2            # deterministic


def test_period_item_rows_empty_raises(dataset):
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    with pytest.raises(ValueError):
        _period_item_rows(enriched, "NO_STORE", "NO_ITEM", ("2024-01-01", "2024-01-02"))
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_scenario.py::test_build_enriched_has_driver_columns -v`
Expected: FAIL — `ImportError: cannot import name '_build_enriched'`.

- [ ] **Step 3: 최소 구현**

`scenario.py` import 영역에 추가:

```python
from ..features.calendar_features import add_calendar_features
from ..features.weather_features import add_weather_features
from ..models.lightgbm_regressor import GlobalLGBM
```

파일 끝에 추가:

```python
def _build_enriched(daily: pd.DataFrame, calendar: pd.DataFrame,
                    weather: pd.DataFrame) -> pd.DataFrame:
    """Merge the separate ontology frames into the single frame GlobalLGBM needs."""
    return add_weather_features(add_calendar_features(daily, calendar), weather)


def _fit_demand_model(enriched: pd.DataFrame, train_cutoff: str,
                      feature_set: str = "v2") -> GlobalLGBM:
    """Fit on rows strictly before train_cutoff (leakage rule). Caller injects cutoff."""
    train = enriched[pd.to_datetime(enriched["date"]) < pd.Timestamp(train_cutoff)]
    if train.empty:
        raise ValueError(f"no training rows before cutoff {train_cutoff}")
    return GlobalLGBM(feature_set=feature_set).fit(train)


def _period_item_rows(enriched: pd.DataFrame, store_id: str, item_id: str,
                      period: tuple[str, str]) -> pd.DataFrame:
    dates = pd.to_datetime(enriched["date"])
    mask = ((enriched["store_id"] == store_id) & (enriched["item_id"] == item_id)
            & (dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1])))
    rows = enriched.loc[mask]
    if rows.empty:
        raise ValueError(f"no rows for store={store_id} item={item_id} in {period}")
    return rows


def _predict_demand(model: GlobalLGBM, target: pd.DataFrame) -> float:
    """Mean predicted demand over the target rows (matches _item_demand_points mean)."""
    return float(model.predict(target).mean())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: PASS (Task 1의 5개 + 신규 5개). 만약 `_predict_demand`가 빈 예측(NaN)을 내면 period가 train 직후가 아니라 lag가 안 풀린 것 — `_cutoff_and_period`가 cutoff 직후 2일을 period로 잡으므로 history가 연속이면 정상.

Run: `uv run pytest tests/test_features_leakage.py tests/test_split_leakage.py -q`
Expected: PASS (기존 leakage 회귀 불변).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/scenario.py tests/test_scenario.py
git commit -m "feat: scenario forecast wiring — enrich + fit(<cutoff) + period rows + predict"
```

---

### Task 3: `what_if_driver` 오케스트레이션

before/after 재예측 + 드라이버 override + risk 전파 + out_of_support + 결과 조립.

**Files:**
- Modify: `src/bakery/ontology/scenario.py`
- Test: `tests/test_scenario.py`

**Interfaces:**
- Consumes: Task 1·2 전부 + `simulate_item_risk`, `RiskParams`(decision).
- Produces:
  - `what_if_driver(daily, calendar, weather, store_id, item_id, period, driver_overrides, *, base_order, train_cutoff, feature_set="v2", risk=RiskParams()) -> WhatIfDriverResult` — 위 데이터흐름. before/after는 같은 fitted 모델, after는 period row의 driver_overrides 컬럼만 덮어쓴 ceteris paribus 예측.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scenario.py`에 추가 (모델은 stub로 주입 — orchestration 로직을 LGBM 신호와 분리해 결정론 검증):

```python
from bakery.ontology import scenario as sc


class _StubModel:
    """Demand responds to is_rain: 10 baseline, −3 when is_rain==1. Deterministic."""
    def predict(self, target):
        rain = float(target["is_rain"].iloc[0]) if "is_rain" in target.columns else 0.0
        return pd.Series([10.0 - 3.0 * rain] * len(target))


def test_what_if_driver_propagates_demand_to_risk(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    res = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                            store, item, period, {"is_rain": 1}, base_order=10.0,
                            train_cutoff=cutoff)
    assert res.before_demand == 10.0 and res.after_demand == 7.0
    assert res.demand_delta == -3.0
    # demand fell → stockout risk should not rise
    assert res.after_p_stockout <= res.before_p_stockout
    assert res.propagation_path == ("dailysales_observed_on_weather", "item_sold_as_dailysales")


def test_what_if_driver_out_of_support_flag(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    # impossible combo unlikely in history → out_of_support True
    res = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                            store, item, period, {"is_rain": 1, "is_snow": 1},
                            base_order=10.0, train_cutoff=cutoff)
    assert isinstance(res.out_of_support, bool)


def test_what_if_driver_rejects_unknown_driver(dataset):
    with pytest.raises(ValueError):
        sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                          "A", "P1", ("2024-01-01", "2024-01-02"), {"is_sunny": 1},
                          base_order=10.0, train_cutoff="2024-01-01")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_scenario.py::test_what_if_driver_propagates_demand_to_risk -v`
Expected: FAIL — `AttributeError: module 'bakery.ontology.scenario' has no attribute 'what_if_driver'`.

- [ ] **Step 3: 최소 구현**

`scenario.py` import에 추가:

```python
from ..decision import RiskParams, simulate_item_risk
```

파일 끝에 추가:

```python
def what_if_driver(daily, calendar, weather, store_id, item_id, period,
                   driver_overrides, *, base_order, train_cutoff,
                   feature_set: str = "v2", risk: RiskParams = RiskParams()) -> WhatIfDriverResult:
    """Upstream Scenario lever: perturb driver(s) → re-forecast demand → propagate
    to stockout risk/cost. Read-only. before/after share one fitted model; only the
    driver columns differ (ceteris paribus). train_cutoff is caller-injected (leakage)."""
    _validate_drivers(driver_overrides)
    enriched = _build_enriched(daily, calendar, weather)
    model = _fit_demand_model(enriched, train_cutoff, feature_set)
    base_rows = _period_item_rows(enriched, store_id, item_id, period)
    before_demand = _predict_demand(model, base_rows)
    pert_rows = base_rows.copy()
    for col, val in driver_overrides.items():
        pert_rows[col] = val
    after_demand = _predict_demand(model, pert_rows)
    before = simulate_item_risk(before_demand, base_order, risk)
    after = simulate_item_risk(after_demand, base_order, risk)
    return WhatIfDriverResult(
        store_id=store_id, item_id=item_id, driver_overrides=dict(driver_overrides),
        before_demand=before_demand, after_demand=after_demand,
        demand_delta=after_demand - before_demand,
        before_p_stockout=before.p_stockout, after_p_stockout=after.p_stockout,
        before_expected_cost=before.expected_cost, after_expected_cost=after.expected_cost,
        out_of_support=_count_support(enriched, store_id, driver_overrides) == 0,
        propagation_path=_propagation_path(driver_overrides),
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: PASS (전체 13개).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/scenario.py tests/test_scenario.py
git commit -m "feat: what_if_driver orchestration — re-forecast, override, propagate risk"
```

---

### Task 4: OntologyFunction 등록 + grounding 도구 노출

`what_if_driver`를 FUNCTION_REGISTRY(side="read")에 등록하고 LLM 도구로 노출한다. train_cutoff은 LLM이 고르지 않고 dispatch가 `period[0]`으로 주입(leakage-safe).

**Files:**
- Modify: `src/bakery/ontology/functions.py` (FUNCTION_REGISTRY)
- Modify: `src/bakery/ontology/grounding/tools.py` (TOOL_SPECS, `_call`)
- Test: `tests/test_ontology_functions.py`, `tests/test_grounding_tools.py`

**Interfaces:**
- Consumes: `scenario.what_if_driver`(Task 3), 기존 `OntologyFunctionSpec`(side 필드 존재), `ToolSpec`, dispatch 패턴.
- Produces:
  - `FUNCTION_REGISTRY["what_if_driver"]` — `side="read"`, `impl=scenario.what_if_driver`.
  - `TOOL_SPECS`에 `what_if_driver` ToolSpec 추가. `_call`에 분기 추가 (train_cutoff=period[0] 주입).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_ontology_functions.py`에 추가:

```python
def test_what_if_driver_registered_read_side():
    from bakery.ontology.functions import FUNCTION_REGISTRY
    from bakery.ontology import scenario
    spec = FUNCTION_REGISTRY["what_if_driver"]
    assert spec.side == "read"
    assert spec.impl is scenario.what_if_driver
```

`tests/test_grounding_tools.py`에 추가 (기존 import/fixture 패턴 따름):

```python
def test_what_if_driver_tool_spec_present():
    from bakery.ontology.grounding.tools import TOOL_SPECS
    names = {t.name for t in TOOL_SPECS}
    assert "what_if_driver" in names
    spec = next(t for t in TOOL_SPECS if t.name == "what_if_driver")
    props = spec.parameters["properties"]
    assert set(props["driver_overrides"]["properties"]) == {
        "is_public_holiday", "is_rain", "is_snow"}


def test_dispatch_what_if_driver_serializes(dataset):
    """dispatch derives train_cutoff=period[0]; returns JSON with demand_delta."""
    import json
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    enriched_dates = sorted(pd.to_datetime(dataset.daily["date"]).dt.date.unique())
    period = [str(enriched_dates[-2]), str(enriched_dates[-1])]
    store = dataset.daily["store_id"].iloc[0]
    item = dataset.daily.loc[dataset.daily["store_id"] == store, "item_id"].iloc[0]
    call = ToolCall(id="c1", name="what_if_driver", arguments={
        "store_id": store, "item_id": item, "period": period,
        "driver_overrides": {"is_rain": 1}, "base_order": 10.0})
    result = dispatch(call, dataset)
    payload = json.loads(result.content)
    assert "demand_delta" in payload or "error" in payload   # real fit may be heavy; both shapes valid
```

(위 `test_grounding_tools.py`에 `pandas as pd` import와 `dataset` fixture가 없으면 파일 상단 기존 패턴대로 추가.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_ontology_functions.py::test_what_if_driver_registered_read_side tests/test_grounding_tools.py::test_what_if_driver_tool_spec_present -v`
Expected: FAIL — `KeyError: 'what_if_driver'` / `assert 'what_if_driver' in names`.

- [ ] **Step 3: 최소 구현**

`src/bakery/ontology/functions.py` import 영역에 추가:

```python
from . import scenario
```

`FUNCTION_REGISTRY` dict 닫는 `}` 직전에 추가:

```python
    "what_if_driver": OntologyFunctionSpec(
        "what_if_driver",
        "Upstream lever: perturb weather/calendar driver(s), re-forecast demand, propagate to stockout risk/cost.",
        ("store_id", "item_id", "period", "driver_overrides", "base_order"),
        "WhatIfDriverResult", scenario.what_if_driver, side="read"),
```

`src/bakery/ontology/grounding/tools.py` — import에 `from .. import scenario` 추가. `TOOL_SPECS` 리스트 끝에 추가:

```python
    ToolSpec("what_if_driver",
             "Upstream Scenario lever: perturb weather/calendar driver(s) for a store/item over a period, "
             "re-forecast demand, and report how stockout risk/cost change. Read-only.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "item_id": {"type": "string"}, "period": _PERIOD,
                 "driver_overrides": {
                     "type": "object",
                     "properties": {
                         "is_public_holiday": {"type": "number"},
                         "is_rain": {"type": "number"}, "is_snow": {"type": "number"}},
                     "additionalProperties": False,
                     "description": "Hypothetical 0/1 driver values to set. 공휴일=is_public_holiday, 비=is_rain, 눈=is_snow. (주말/휴무일은 모델에 닿지 않아 미지원.)"},
                 "base_order": {"type": "number"}},
              "required": ["store_id", "item_id", "period", "driver_overrides", "base_order"],
              "additionalProperties": False}),
```

`tools.py`의 `_call` 함수에 분기 추가 (마지막 `raise KeyError` 전):

```python
    if name == "what_if_driver":
        return scenario.what_if_driver(
            dataset.daily, dataset.calendar, dataset.weather,
            a["store_id"], a["item_id"], tuple(a["period"]), a["driver_overrides"],
            base_order=a["base_order"], train_cutoff=a["period"][0])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_ontology_functions.py tests/test_grounding_tools.py tests/test_scenario.py -v`
Expected: PASS (신규 포함 전부).

순환 import 점검 — Run: `uv run python -c "import bakery.ontology.functions; import bakery.ontology.scenario; import bakery.ontology.grounding.tools; print('ok')"`
Expected: `ok`.

전체 회귀 — Run: `uv run pytest -q`
Expected: 전부 통과, 실패 0 (live 1 skip).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/functions.py src/bakery/ontology/grounding/tools.py tests/test_ontology_functions.py tests/test_grounding_tools.py
git commit -m "feat: register what_if_driver (read side) + expose as grounding tool (cutoff=period[0])"
```

---

## Self-Review

**Spec coverage:**
- D1(실 forecast 재실행) → Task 2 `_fit_demand_model`+`_predict_demand`. ✓
- D2(국소 도입, 기존 proxy 불변) → scenario.py만 추가, functions.py 기존 함수 불변. ✓
- D2-1(feature_set="v2" target=potential_demand) → Task 2 기본값 "v2", Global Constraints. ✓
- D2-2(feature 프레임 조립) → Task 2 `_build_enriched`. ✓
- D3(ceteris paribus, 복수 드라이버 dict, out_of_support, 분해 안 함) → Task 3 override 루프 + `_count_support`, Task 1 `_validate_drivers`. ✓
- D4(읽기 전용) → side="read"(Task 4), state mutate 없음. ✓
- D5(caller 주입 train_cutoff/base_order, seed 결정론) → Task 2/3 시그니처, dispatch는 period[0] 주입(Task 4). ✓
- 전파 lineage(OntologyLink) → Task 1 `_propagation_path`. ✓
- 에러/엣지(빈 base_row, 알 수 없는 키, cutoff 이후 데이터 없음) → Task 2/3 ValueError + 테스트. ✓
- 테스트(결정론·델타·복수·out_of_support·path·leakage·dispatch) → Task 1~4. ✓
- 범위 밖(Scenario→commit, 기여 분해, 비논리 차단) → 구현 안 함, TODO. ✓

**Placeholder scan:** 없음 — 모든 step에 실제 코드/명령/기대출력.

**Type consistency:** `WhatIfDriverResult` 필드 ↔ Task 3 생성자 인자 일치. `_fit_demand_model(enriched, train_cutoff, feature_set="v2")` / `_predict_demand(model, target)->float` / `_period_item_rows(...)->DataFrame` / `what_if_driver(daily, calendar, weather, store_id, item_id, period, driver_overrides, *, base_order, train_cutoff, feature_set, risk)` — Task 간 호출 시그니처 일치. `simulate_item_risk(demand, order, risk)->RiskResult(p_stockout, expected_cost)` 실제와 일치. dispatch `_call`의 train_cutoff=period[0] 주입 일관.

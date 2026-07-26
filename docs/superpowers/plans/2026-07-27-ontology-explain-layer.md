# 온톨로지 2층 설명 레이어 (5b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 5a `ForwardForecast` 중간값을 소비해 카테고리 총량·품목 발주를 실제 수식으로 분해 서술하는 grounded 설명 함수 2개 + 도구 등록 + Q셋 delta 측정을 구현한다.

**Architecture:** 신규 `src/bakery/ontology/explain.py`에 `explain_category_total`·`explain_item_order`를 두고, 둘 다 `forecast_forward()`를 호출해 lineage 스타일 분해 DataFrame을 반환한다. `grounding/tools.py`에 ToolSpec 2개 + dispatch, `grounding/questions.py`에 Q 2개 + forward 컨텍스트 + gold 분기를 추가한다. 새 모델링 없음 — 5a 엔진이 낸 값을 분해만.

**Tech Stack:** Python, pandas, pytest, uv. 대상 = real 소스(광교 `store_gw01`). LLM eval은 키 없이 CI 통과(mock/skip).

## Global Constraints

- **faithfulness (reconcile)**: `explain_item_order` 최종(라운딩 전) == seam `our_order`; `explain_category_total` prior_prod == seam `category_totals.prior_prod`; event_prior 기여 == `prior_median − base_median`(실제 blend 차이). 이상화된 "고정 N개 룰" 라벨 금지.
- **보존식**: 체인 각 단계 재구성 == 최종(총량 레벨 `prior_median + 버퍼 = prior_prod`, 품목 레벨 `prior_prod × proportion = our_order`, `ceil_3(our_order) = final`).
- **apply_policy 이중계상 금지**: seam our_order가 분위수 버퍼를 이미 포함 — explain_item_order는 apply_policy 퍼센트 safety를 얹지 않는다.
- **기존 `explain_order`(v6 apply_policy 경로) 무변경**: `functions.py`의 explain_order·그 테스트 그대로.
- **결정론 gold**: `forecast_forward(use_forecast=False)` + 고정 seed. 수동 라벨 0.
- **배수 라운딩**: `decision/policy.py::_round_up_to_unit(qty, unit=3)` 재사용(3/6/9). 라벨된 가정(품목별 배수는 후속).
- **테스트 게이트 = focused만**: 각 태스크는 해당 focused 파일만 실행(전체 스위트는 22분→백그라운드→커밋 막힘). 전체 스위트는 컨트롤러가 최종 1회. 성공 판정은 `pytest > out 2>&1; echo EXIT=$?`로 실제 exit 확인(`| tail` 금지).
- **순환의존 금지**: `explain.py`는 `forecast.forward`·`models.item_proportion`·`decision.policy`에 의존, `grounding`을 import하지 않는다. `functions.py`를 import하지 않는다.
- **커밋**: 각 태스크 끝 커밋, 브랜치 `spec/ontology-explain-layer-5b`. trailer 2줄(Co-Authored-By / Claude-Session).
- **pytest**: `uv run pytest ... --color=no`(이 repo addopts `-q`, `-q` 추가 금지).

---

### Task 1: `explain.py` — 두 설명 함수 (핵심)

`ForwardForecast`를 소비해 총량·품목 발주를 lineage 스타일로 분해. 프레임 주입(`daily`)으로 결정론 테스트.

**Files:**
- Create: `src/bakery/ontology/explain.py`
- Create: `tests/test_ontology_explain.py`

**Interfaces:**
- Consumes: `bakery.forecast.forward.forecast_forward(store_id, *, daily, horizon_days, use_forecast) -> ForwardForecast` (5a). `ForwardForecast.category_totals`=[date, base_median, base_prod, prior_median, prior_prod], `.proportions`=compute_proportions 출력([date, item_id, proportion, base_sold, adj_trend, adj_stockout, adj_closing, adj_new, ...]), `.item_quantities`=[store_id, item_id, category_id, date, demand_point, our_order]. `bakery.decision.policy._round_up_to_unit(qty, unit) -> float`.
- Produces:
  - `explain_category_total(store_id, *, daily, date, horizon_days=7, use_forecast=False) -> pd.DataFrame` cols [store_id, date, step, value, detail]
  - `explain_item_order(store_id, item_id, *, daily, date, horizon_days=7, round_unit=3, use_forecast=False) -> pd.DataFrame` cols [store_id, item_id, date, step, value, detail]
  - `BATCH_ROUND_UNIT = 3` (모듈 상수, 라벨된 가정)

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_ontology_explain.py`:

```python
import pandas as pd
import pytest

from bakery.forecast.forward import forecast_forward
from bakery.forecast.loaders import load_real_daily
from bakery.ontology.explain import (
    BATCH_ROUND_UNIT, explain_category_total, explain_item_order,
)

STORE = "store_gw01"


@pytest.fixture(scope="module")
def daily():
    return load_real_daily(STORE)


@pytest.fixture(scope="module")
def ff(daily):
    return forecast_forward(STORE, daily=daily, horizon_days=7, use_forecast=False)


@pytest.fixture(scope="module")
def target_date(ff):
    return str(pd.to_datetime(ff.item_quantities["date"]).min().date())


def test_category_total_reconciles_with_seam(daily, ff, target_date):
    """explain_category_total의 prior_prod 단계 == seam category_totals.prior_prod."""
    rows = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False)
    ct = ff.category_totals[pd.to_datetime(ff.category_totals["date"]) == pd.Timestamp(target_date)].iloc[0]
    got = rows.set_index("step")["value"]
    assert got["base_median"] == pytest.approx(ct["base_median"], rel=1e-9)
    assert got["prior_median"] == pytest.approx(ct["prior_median"], rel=1e-9)
    assert got["prior_prod"] == pytest.approx(ct["prior_prod"], rel=1e-9)
    # event_prior 기여 = prior_median − base_median (실제 blend 차이, "룰" 아님)
    assert got["event_prior"] == pytest.approx(ct["prior_median"] - ct["base_median"], rel=1e-9)


def test_category_total_conservation(daily, target_date):
    """base_median + event_prior = prior_median; prior_median + buffer = prior_prod."""
    got = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False).set_index("step")["value"]
    assert got["base_median"] + got["event_prior"] == pytest.approx(got["prior_median"], rel=1e-9)
    assert got["prior_median"] + got["quantile_buffer"] == pytest.approx(got["prior_prod"], rel=1e-9)


def test_item_order_reconciles_with_seam(daily, ff, target_date):
    """explain_item_order의 item_order 단계(라운딩 전) == seam our_order."""
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    seam_order = float(iq_d[iq_d["item_id"].astype(str) == item]["our_order"].iloc[0])
    rows = explain_item_order(STORE, item, daily=daily, date=target_date, use_forecast=False).set_index("step")["value"]
    assert rows["item_order"] == pytest.approx(seam_order, rel=1e-9)


def test_item_order_conservation(daily, ff, target_date):
    """category_total × proportion = item_order; ceil_3(item_order) = final."""
    import math
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    got = explain_item_order(STORE, item, daily=daily, date=target_date, round_unit=3, use_forecast=False).set_index("step")["value"]
    assert got["category_total"] * got["proportion"] == pytest.approx(got["item_order"], rel=1e-9)
    assert got["final"] == pytest.approx(math.ceil(got["item_order"] / 3) * 3, rel=1e-9)


def test_batch_round_unit_default():
    assert BATCH_ROUND_UNIT == 3
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_ontology_explain.py --color=no > /tmp/5b_t1.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t1.txt`
Expected: FAIL (`ImportError: cannot import name 'explain_category_total'`)

- [ ] **Step 3: 구현**

`src/bakery/ontology/explain.py`:

```python
"""2층 설명 레이어 (v7 5b) — explain_category_total / explain_item_order.

5a forecast_forward seam의 중간값을 소비해 "왜 이 수량?"을 총량·품목 두 층위로
분해 서술한다. 새 모델링 없음 — 엔진이 낸 실제 값을 충실히 분해한다.

★faithfulness: event_prior 기여는 prior_median − base_median(실제 blend 차이)이지
"크리스마스=고정 N개 룰"이 아니다(레벨-앵커 블렌드). apply_policy 퍼센트 safety를
얹지 않는다 — seam our_order가 이미 분위수 버퍼(q0.85)를 총량 레벨에 포함하므로
이중계상이 된다. 기존 explain_order(v6 apply_policy)는 다른 발주 철학, 무변경.

See docs/superpowers/specs/2026-07-27-ontology-explain-layer-design.md.
"""

from __future__ import annotations

import pandas as pd

from ..decision.policy import _round_up_to_unit
from ..forecast.forward import forecast_forward

BATCH_ROUND_UNIT = 3  # 라벨된 가정: 아띠제 배수생산(3/6/9). 품목별 배수는 후속.


def _forward_at_date(store_id, daily, date, horizon_days, use_forecast):
    """forecast_forward를 돌려 요청 date의 category_totals 1행 + item_quantities 슬라이스."""
    ff = forecast_forward(store_id, daily=daily, horizon_days=horizon_days, use_forecast=use_forecast)
    ts = pd.Timestamp(date)
    ct = ff.category_totals[pd.to_datetime(ff.category_totals["date"]) == ts]
    if ct.empty:
        raise ValueError(f"date {date} not in forward horizon for {store_id}")
    iq = ff.item_quantities[pd.to_datetime(ff.item_quantities["date"]) == ts].copy()
    iq["item_id"] = iq["item_id"].astype(str)
    return ct.iloc[0], iq


def explain_category_total(store_id, *, daily, date, horizon_days=7, use_forecast=False):
    """카테고리 생산총량 분해: base 예측 → event_prior 보정 → 분위수 버퍼 → prior_prod.

    event_prior = prior_median − base_median (실제 blend 차이, 특수일 레벨-앵커).
    """
    ct, _ = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    base_median = float(ct["base_median"])
    prior_median = float(ct["prior_median"])
    prior_prod = float(ct["prior_prod"])
    event_prior = prior_median - base_median
    buffer = prior_prod - prior_median
    rows = [
        ("base_median", base_median, "Stage1 카테고리 수요 예측 (event_prior 이전)"),
        ("event_prior", event_prior, "특수일 레벨-앵커 블렌드 보정 (prior_median − base_median)"),
        ("prior_median", prior_median, "보정된 카테고리 수요 (q0.5)"),
        ("quantile_buffer", buffer, "생산 분위수 버퍼 (q0.85 − q0.5)"),
        ("prior_prod", prior_prod, "카테고리 생산총량"),
    ]
    return pd.DataFrame(
        [{"store_id": store_id, "date": date, "step": s, "value": v, "detail": d} for s, v, d in rows]
    )


def explain_item_order(store_id, item_id, *, daily, date, horizon_days=7,
                       round_unit=BATCH_ROUND_UNIT, use_forecast=False):
    """품목 생산량 분해(통합 단일 체인): 카테고리 총량 × 품목 비중 → 배수 라운딩.

    category_total(prior_prod) × proportion = item_order(=seam our_order) →
    ceil_round_unit = final. apply_policy 미사용(이중계상 방지).
    """
    ct, iq = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    row = iq[iq["item_id"] == str(item_id)]
    if row.empty:
        raise ValueError(f"item {item_id} not in forward forecast for {store_id} at {date}")
    prior_prod = float(ct["prior_prod"])
    item_order = float(row["our_order"].iloc[0])
    proportion = item_order / prior_prod if prior_prod > 0 else 0.0
    final = _round_up_to_unit(item_order, round_unit)
    rows = [
        ("category_total", prior_prod, "카테고리 생산총량 (prior_prod)"),
        ("proportion", proportion, "품목 비중 (base_sold×adj_trend×adj_stockout×adj_closing×adj_new / Σ)"),
        ("item_order", item_order, "품목 생산량 (= 총량 × 비중)"),
        ("final", final, f"배수 라운딩 (ceil to {round_unit})"),
    ]
    return pd.DataFrame(
        [{"store_id": store_id, "item_id": str(item_id), "date": date,
          "step": s, "value": v, "detail": d} for s, v, d in rows]
    )
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_ontology_explain.py --color=no > /tmp/5b_t1.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t1.txt`
Expected: EXIT=0, 5 passed. 실패 시: reconcile(item_order==our_order)가 어긋나면 proportion 재계산이 seam과 다른 것 — 중단하고 forecast_forward 산출과 대조.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/explain.py tests/test_ontology_explain.py
git commit -m "feat(ontology): explain_category_total/explain_item_order 2층 설명 (5b Task1)"
```

---

### Task 2: 도구 등록 (grounded surface)

두 설명 함수를 `grounding/tools.py`의 ToolSpec + dispatch에 배선.

**Files:**
- Modify: `src/bakery/ontology/grounding/tools.py`
- Modify: `tests/test_grounding_tools.py`

**Interfaces:**
- Consumes: `explain.explain_category_total`, `explain.explain_item_order` (Task 1). `DailyDataset.daily` (dispatch가 주입).
- Produces: TOOL_SPECS에 `explain_category_total`·`explain_item_order` 항목; `_call`이 두 이름 처리.

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_grounding_tools.py`에 추가(기존 dispatch 테스트 패턴 따름):

```python
def test_dispatch_explain_category_total(real_dataset):
    """explain_category_total 도구가 dispatch되어 분해 행을 JSON으로 반환."""
    import json
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    ds = real_dataset
    date = _forward_date(ds)   # 헬퍼: 마지막 관측일 다음날 (아래 정의 또는 fixture)
    call = ToolCall(id="c1", name="explain_category_total",
                    arguments={"store_id": "store_gw01", "date": date})
    res = dispatch(call, ds)
    rows = json.loads(res.content)
    steps = [r["step"] for r in rows]
    assert steps == ["base_median", "event_prior", "prior_median", "quantile_buffer", "prior_prod"]


def test_dispatch_explain_item_order(real_dataset):
    import json
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    ds = real_dataset
    date = _forward_date(ds)
    # forward our_order 최대 품목을 결정론 선택
    from bakery.forecast.forward import forecast_forward
    iq = forecast_forward("store_gw01", daily=ds.daily, horizon_days=7, use_forecast=False).item_quantities
    iq_d = iq[iq["date"].astype(str).str.startswith(date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    call = ToolCall(id="c2", name="explain_item_order",
                    arguments={"store_id": "store_gw01", "item_id": item, "date": date})
    res = dispatch(call, ds)
    rows = json.loads(res.content)
    assert [r["step"] for r in rows] == ["category_total", "proportion", "item_order", "final"]
```

(`real_dataset` fixture·`_forward_date` 헬퍼는 기존 grounding 테스트의 real dataset 로딩 패턴 재사용. 없으면 `DailyDataset` real 로드 + 마지막 관측일+1로 정의.)

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_grounding_tools.py -k explain --color=no > /tmp/5b_t2.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t2.txt`
Expected: FAIL (도구 미등록 → dispatch가 KeyError / error JSON)

- [ ] **Step 3: 구현**

`grounding/tools.py` 상단 import에 추가: `from .. import explain`.

`TOOL_SPECS` 리스트에 2개 추가(what_if_driver 뒤):

```python
    ToolSpec("explain_category_total",
             "Decompose a store's forward category production total: Stage1 base forecast "
             "→ event_prior anchor adjustment (actual blend delta, not a fixed rule) "
             "→ production quantile buffer → prior_prod.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"},
                 "date": {"type": "string", "description": "forward horizon date YYYY-MM-DD"}},
              "required": ["store_id", "date"], "additionalProperties": False}),
    ToolSpec("explain_item_order",
             "Decompose one item's forward production: category total × item proportion "
             "(base×trend×stockout×closing×new factors) → item order → batch rounding (3/6/9).",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "item_id": {"type": "string"},
                 "date": {"type": "string", "description": "forward horizon date YYYY-MM-DD"}},
              "required": ["store_id", "item_id", "date"], "additionalProperties": False}),
```

`_call`에 분기 추가(what_if_driver 앞/뒤). ⚠️ Task 1 시그니처에서 `daily`가 keyword-only(`*, daily`)이고 store_id/item_id가 positional이므로 아래처럼 정확히 맞춘다:

```python
    if name == "explain_category_total":
        return explain.explain_category_total(
            a["store_id"], daily=dataset.daily, date=a["date"], use_forecast=False)
    if name == "explain_item_order":
        return explain.explain_item_order(
            a["store_id"], a["item_id"], daily=dataset.daily, date=a["date"], use_forecast=False)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_grounding_tools.py -k explain --color=no > /tmp/5b_t2.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t2.txt`
Expected: EXIT=0, 2 passed.

- [ ] **Step 5: 기존 grounding_tools 회귀**

Run: `uv run pytest tests/test_grounding_tools.py --color=no > /tmp/5b_t2b.txt 2>&1; echo "EXIT=$?"; tail -5 /tmp/5b_t2b.txt`
Expected: EXIT=0, all passed.

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/ontology/grounding/tools.py tests/test_grounding_tools.py
git commit -m "feat(grounding): explain_category_total/explain_item_order 도구 등록 (5b Task2)"
```

---

### Task 3: Q셋 delta (forward 컨텍스트 + gold)

두 설명 질문을 Q셋에 추가. forward 컨텍스트 + 결정론 gold + 채점기 재사용.

**Files:**
- Modify: `src/bakery/ontology/grounding/questions.py`
- Modify: `tests/test_grounding_questions.py`

**Interfaces:**
- Consumes: `explain.explain_category_total`·`explain.explain_item_order` (Task 1), 기존 `build_gold`/`QUESTIONS`/`_ctx` 구조. `forecast_forward` (item 선택·forward date).
- Produces: `QUESTIONS`에 `q_explain_total`(NUMERIC)·`q_explain_item`(DECOMPOSITION); `build_gold`이 두 source_fn 처리; `_forward_ctx(dataset, horizon_days=7) -> (store, date)` 헬퍼.

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_grounding_questions.py`에 추가:

```python
def test_gold_explain_total_matches_function(real_dataset):
    """q_explain_total gold == explain_category_total prior_prod (결정론)."""
    from bakery.ontology.grounding.questions import QUESTIONS, build_gold
    from bakery.ontology import explain
    from bakery.ontology.grounding.questions import _forward_ctx
    q = next(q for q in QUESTIONS if q.id == "q_explain_total")
    gold = build_gold(q, real_dataset)
    store, date = _forward_ctx(real_dataset)
    rows = explain.explain_category_total(store, daily=real_dataset.daily, date=date, use_forecast=False)
    expected = float(rows.set_index("step")["value"]["prior_prod"])
    assert gold["answer_value"] == pytest.approx(expected, rel=1e-9)


def test_gold_explain_item_matches_function(real_dataset):
    """q_explain_item gold == explain_item_order final for the deterministic item."""
    from bakery.ontology.grounding.questions import QUESTIONS, build_gold
    q = next(q for q in QUESTIONS if q.id == "q_explain_item")
    gold = build_gold(q, real_dataset)
    assert "item_id" in gold and "order_qty" in gold
    assert gold["order_qty"] > 0
```

(`real_dataset` fixture = 기존 grounding 테스트 재사용.)

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_grounding_questions.py -k explain --color=no > /tmp/5b_t3.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t3.txt`
Expected: FAIL (q_explain_total 없음 / _forward_ctx 없음)

- [ ] **Step 3: 구현**

`questions.py` 상단 import에 `from .. import explain` + `from ...forecast.forward import forecast_forward` 추가.

`_ctx` 아래에 forward 컨텍스트 헬퍼 추가:

```python
def _forward_ctx(dataset: DailyDataset, horizon_days: int = 7):
    """explain 질문용 forward 컨텍스트: (store, 마지막 관측일 다음날).

    _ctx(historical)와 달리 forward 대상. forecast_forward가 결정론이라 gold 재현 가능.
    """
    store = sorted(dataset.daily["store_id"].unique())[0]
    dd = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    first_future = (dd.max() + pd.Timedelta(days=1)).date()
    return store, str(first_future)


def _forward_top_item(dataset: DailyDataset, store: str, date: str) -> str:
    """forward our_order 최대 품목(결정론 선택)."""
    iq = forecast_forward(store, daily=dataset.daily, horizon_days=7,
                          use_forecast=False).item_quantities
    iq_d = iq[iq["date"].astype(str).str.startswith(date)].copy()
    iq_d["item_id"] = iq_d["item_id"].astype(str)
    return str(iq_d.sort_values(["our_order", "item_id"], ascending=[False, True]).iloc[0]["item_id"])
```

`QUESTIONS` 리스트에 추가:

```python
    Question("q_explain_total",
             "다음주 이 매장의 빵 카테고리 생산총량은? base 예측·특수일 보정·분위수 버퍼로 분해하면?",
             NUMERIC, "explain_category_total", {}),
    Question("q_explain_item",
             "다음주 이 매장에서 가장 많이 생산하는 품목의 생산량은? 카테고리 총량과 품목 비중으로 분해하면?",
             DECOMPOSITION, "explain_item_order", {}),
```

`build_gold`에 두 분기 추가(마지막 `raise KeyError` 앞):

```python
    if question.source_fn == "explain_category_total":
        store, date = _forward_ctx(dataset)
        rows = explain.explain_category_total(store, daily=dataset.daily, date=date, use_forecast=False)
        value = float(rows.set_index("step")["value"]["prior_prod"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"answer_value": value}
    if question.source_fn == "explain_item_order":
        store, date = _forward_ctx(dataset)
        item = _forward_top_item(dataset, store, date)
        rows = explain.explain_item_order(store, item, daily=dataset.daily, date=date, use_forecast=False)
        value = float(rows.set_index("step")["value"]["final"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"item_id": item, "order_qty": value}
```

⚠️ `target_period()`(questions.py:68, "same basis as build_gold's gold")도 확인: 신규 질문이 그 함수를 타면 forward 컨텍스트로 분기해야 grounded arm이 같은 date로 도구를 호출한다. `target_period`가 source_fn별 분기라면 explain 2종에 `_forward_ctx` 기반 컨텍스트를 반환하도록 추가(historical 11종은 무변경). 없으면 이 태스크에서 배선.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_grounding_questions.py -k explain --color=no > /tmp/5b_t3.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t3.txt`
Expected: EXIT=0, 2 passed.

- [ ] **Step 5: 전체 grounding 회귀(키 없이 CI 경로)**

Run: `uv run pytest tests/test_grounding_questions.py tests/test_grounding_scorer.py tests/test_grounding_run.py --color=no > /tmp/5b_t3b.txt 2>&1; echo "EXIT=$?"; tail -8 /tmp/5b_t3b.txt`
Expected: EXIT=0, all passed(기존 11 Q + 신규 2, LLM live는 skip).

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/ontology/grounding/questions.py tests/test_grounding_questions.py
git commit -m "feat(grounding): q_explain_total/q_explain_item Q셋 + forward 컨텍스트 gold (5b Task3)"
```

---

## Self-Review 체크

- **Spec coverage**: §3(설명 함수)=Task1, §4(도구)=Task2, §5(Q셋 forward 컨텍스트+gold)=Task3, §6 acceptance=Task1(reconcile/보존식)+Task3(gold 결정론·무회귀). §2 비목표(explain_order 무변경)=어느 태스크도 functions.py explain_order 안 건드림. ✅
- **Placeholder scan**: Task1 전체 코드, Task2·3 전체 코드+⚠️ 배선 주의. `real_dataset`/`_forward_date` fixture는 "기존 grounding 테스트 재사용"으로 명시(구현자가 실제 fixture명 확인). ✅
- **Type consistency**: explain 함수 시그니처(store_id 위치·daily keyword-only·date)가 Task1 정의 ↔ Task2 dispatch ↔ Task3 build_gold에서 일치. `_forward_ctx` 반환 (store, date), `category_totals` 컬럼(base_median/prior_median/prior_prod), OUTPUT_SCHEMAS(NUMERIC=answer_value / DECOMPOSITION=item_id+order_qty) 재사용 일치. ✅
- **Faithfulness**: reconcile 테스트(item_order==seam our_order, event_prior==prior_median−base_median)가 Task1에 있음, apply_policy 미사용(체인에 없음). ✅

## Execution Handoff

1. **Subagent-Driven (권장)** — 태스크별 fresh subagent + 2단계 리뷰. faithfulness reconcile을 태스크 경계에서 검증.
2. **Inline Execution** — 배치 + 체크포인트.

어느 방식으로 진행할까요?

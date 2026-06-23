# Grounding Eval (S3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** with-AOS(grounded, 함수호출) vs without-AOS(rag-only) grounding 정답률 delta를 측정하는 self-contained eval 하네스를 구축한다.

**Architecture:** provider 중립 `LLMClient` Protocol 위에 OpenAI(gpt-5-mini) 어댑터를 둔다. Q셋·gold·채점은 결정론(키 불필요), LLM 호출만 provider 의존. 두 arm은 동일 모델, 도구 유무만 다르다. 설계 = `docs/superpowers/specs/2026-06-23-grounding-eval-design.md`.

**Tech Stack:** Python 3.x, `openai` SDK(`chat.completions.parse`), pandas, pytest. 기존 `src/bakery/ontology/`(schema.py, functions.py — 이미 머지됨) 재사용.

## Global Constraints

- 신규 모델링 0 — `functions.py`의 OntologyFunction만 호출(절대규칙: post-prediction read-only).
- **CI는 키 없이 통과** — LLM 실호출 테스트는 `@pytest.mark.skipif(no key)`. Q셋·gold·채점·변환은 키 없이 단위테스트.
- 양 arm 동일 모델·파라미터, 도구 유무만 차이(공정성 §8).
- gold는 OntologyFunction 직접 호출로 생성(결정론, 수동 라벨 0).
- 의존성 추가: `openai`를 `pyproject.toml`에. provider 전환을 위해 import는 OpenAIClient 내부에서만.
- 새 모듈은 `src/bakery/ontology/grounding/` 아래. provider 의존 코드는 `llm.py`의 `OpenAIClient`에만 격리.

---

## File Structure

```
src/bakery/ontology/grounding/
  __init__.py     # 공개 API
  llm.py          # 중립 타입(ToolSpec/Message/ToolCall/ToolResult/LLMResponse) + LLMClient Protocol + OpenAIClient + make_llm_client
  tools.py        # 5개 OntologyFunction의 ToolSpec(JSON schema) 정의 + name→(impl, dataset 바인딩) 디스패치
  questions.py    # Question dataclass + 사전등록 Q셋 + gold 생성기
  arms.py         # run_grounded / run_rag_only + 공통 tool-loop
  scorer.py       # 유형별 grade + evaluate + EvalReport
  run.py          # 실측 엔트리 (키 필요)
tests/
  test_grounding_tools.py      # ToolSpec·디스패치 (키 불필요)
  test_grounding_questions.py  # Q셋·gold (키 불필요)
  test_grounding_scorer.py     # 유형별 채점 (키 불필요)
  test_grounding_arms.py       # FakeLLMClient로 tool-loop (키 불필요)
```

---

### Task 1: 중립 타입 + LLMClient Protocol

**Files:**
- Create: `src/bakery/ontology/grounding/__init__.py`
- Create: `src/bakery/ontology/grounding/llm.py`

**Interfaces:**
- Produces: `ToolSpec(name, description, parameters)`, `ToolCall(id, name, arguments)`, `ToolResult(call_id, content)`, `Message(role, content, tool_calls, tool_call_id)`, `LLMResponse(text, tool_calls, parsed)`, `LLMClient` Protocol with `generate(messages, *, tools=None, output_schema=None) -> LLMResponse`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounding_llm_types.py
from bakery.ontology.grounding.llm import ToolSpec, ToolCall, Message, LLMResponse

def test_toolspec_is_frozen_dataclass():
    t = ToolSpec(name="f", description="d", parameters={"type": "object", "properties": {}})
    assert t.name == "f"
    assert t.parameters["type"] == "object"

def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.tool_calls == []
    assert m.tool_call_id is None

def test_llmresponse_holds_tool_calls_and_parsed():
    r = LLMResponse(text=None, tool_calls=[ToolCall(id="1", name="f", arguments={"x": 1})], parsed=None)
    assert r.tool_calls[0].arguments == {"x": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_llm_types.py -v`
Expected: FAIL — `ModuleNotFoundError: bakery.ontology.grounding.llm`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/ontology/grounding/llm.py
"""Provider-neutral LLM interface (docs/.../grounding-eval-design.md §4).

Higher layers (questions/arms/scorer) use only the neutral types here; each
provider's quirks (tool format, structured-output format, params) live inside
its adapter. Today: OpenAI(gpt-5-mini). Future: an Anthropic adapter slots in
without touching anything above this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict          # JSON Schema for the tool's arguments


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str


@dataclass
class Message:
    role: str                              # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None        # set on role="tool" replies


@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    parsed: dict | None                    # structured-output result, if requested


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        output_schema: dict | None = None,
    ) -> LLMResponse: ...
```

```python
# src/bakery/ontology/grounding/__init__.py
"""Grounding eval (S3) — with/without AOS delta. See design doc."""

from .llm import LLMClient, LLMResponse, Message, ToolCall, ToolResult, ToolSpec

__all__ = ["LLMClient", "LLMResponse", "Message", "ToolCall", "ToolResult", "ToolSpec"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding_llm_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/__init__.py src/bakery/ontology/grounding/llm.py tests/test_grounding_llm_types.py
git commit -m "feat: grounding LLMClient protocol + neutral types (S3 task 1)"
```

---

### Task 2: ToolSpec 정의 + dataset 디스패치

**Files:**
- Create: `src/bakery/ontology/grounding/tools.py`
- Create: `tests/test_grounding_tools.py`

**Interfaces:**
- Consumes: `ToolSpec`, `ToolCall`, `ToolResult` (Task 1); `functions.py` 5함수, `bakery.data.loader.DailyDataset`.
- Produces: `TOOL_SPECS: list[ToolSpec]` (LLM에 노출할 5개 도구 JSON schema), `dispatch(call: ToolCall, dataset) -> ToolResult` (LLM의 tool 인자 → 실제 함수 호출 + dataset 주입 → JSON 결과 문자열).

**왜:** `OntologyFunctionSpec.params`는 문자열 튜플이라 JSON schema가 없다. 또 함수는 `daily` DataFrame을 받지만 LLM엔 비즈니스 인자만 노출하고 dataset은 `dispatch`가 주입한다. 그래서 도구 정의·바인딩을 여기 명시한다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounding_tools.py
import json
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import ToolCall
from bakery.ontology.grounding.tools import TOOL_SPECS, dispatch


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def test_tool_specs_cover_five_functions():
    names = {t.name for t in TOOL_SPECS}
    assert names == {
        "rank_stockout_risk", "explain_order", "what_if",
        "waste_cost", "demand_diff_by_condition",
    }
    for t in TOOL_SPECS:
        assert t.parameters["type"] == "object"
        assert "properties" in t.parameters


def test_dispatch_rank_stockout_risk_returns_json(dataset):
    store = dataset.daily["store_id"].iloc[0]
    import pandas as pd
    dates = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    call = ToolCall(id="c1", name="rank_stockout_risk",
                    arguments={"store_id": store, "period": [str(dates.min().date()), str(dates.max().date())], "k": 3})
    result = dispatch(call, dataset)
    payload = json.loads(result.content)
    assert result.call_id == "c1"
    assert len(payload) <= 3


def test_dispatch_unknown_tool_errors(dataset):
    result = dispatch(ToolCall(id="c2", name="nope", arguments={}), dataset)
    assert "error" in json.loads(result.content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: ...grounding.tools`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/ontology/grounding/tools.py
"""LLM-facing tool definitions for the 5 OntologyFunctions, plus the dispatch
that binds the dataset (which the LLM never sees) and calls the real function.

The LLM is shown only business arguments (store_id, period, item_id, ...);
dispatch() injects the daily/weather/calendar frames and serializes the result
to a JSON string for the tool-result turn.
"""

from __future__ import annotations

import json

from ...data.loader import DailyDataset
from .. import functions as fn
from .llm import ToolCall, ToolResult, ToolSpec

_PERIOD = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
           "description": "[start_date, end_date] as YYYY-MM-DD"}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec("rank_stockout_risk", "Top-k items by stockout probability for a store over a period.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "period": _PERIOD, "k": {"type": "integer"}},
              "required": ["store_id", "period", "k"], "additionalProperties": False}),
    ToolSpec("explain_order", "Decision lineage breaking down one item's recommended order.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "item_id": {"type": "string"}, "period": _PERIOD},
              "required": ["store_id", "item_id", "period"], "additionalProperties": False}),
    ToolSpec("what_if", "Downstream lever: risk/cost delta when an order qty changes.",
             {"type": "object", "properties": {
                 "demand_point": {"type": "number"}, "base_order": {"type": "number"},
                 "delta_order": {"type": "number"}},
              "required": ["demand_point", "base_order", "delta_order"], "additionalProperties": False}),
    ToolSpec("waste_cost", "Aggregate leftover (capacity-sold) cost for a store/period.",
             {"type": "object", "properties": {"store_id": {"type": "string"}, "period": _PERIOD},
              "required": ["store_id", "period"], "additionalProperties": False}),
    ToolSpec("demand_diff_by_condition", "Mean daily sales when a 0/1 condition is on vs off.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "condition_col": {"type": "string"},
                 "frame": {"type": "string", "enum": ["calendar", "weather"]}},
              "required": ["store_id", "condition_col", "frame"], "additionalProperties": False}),
]


def _to_jsonable(obj):
    """DataFrame → records, dataclass-ish → dict, else as-is."""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def dispatch(call: ToolCall, dataset: DailyDataset) -> ToolResult:
    """Run one tool call against the real OntologyFunction, return JSON result."""
    a = call.arguments
    try:
        result = _call(call.name, a, dataset)
        content = json.dumps(_to_jsonable(result), default=str)
    except Exception as exc:                       # surfaced to the model as a tool error
        content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return ToolResult(call_id=call.id, content=content)


def _call(name: str, a: dict, dataset: DailyDataset):
    if name == "rank_stockout_risk":
        return fn.rank_stockout_risk(dataset.daily, a["store_id"], tuple(a["period"]), a["k"])
    if name == "explain_order":
        return fn.explain_order(dataset.daily, a["store_id"], a["item_id"], tuple(a["period"]))
    if name == "what_if":
        return fn.what_if(a["demand_point"], a["base_order"], a["delta_order"])
    if name == "waste_cost":
        return fn.waste_cost(dataset.daily, a["store_id"], tuple(a["period"]))
    if name == "demand_diff_by_condition":
        frame = dataset.calendar if a["frame"] == "calendar" else dataset.weather
        return fn.demand_diff_by_condition(dataset.daily, frame, a["store_id"], a["condition_col"])
    raise KeyError(f"unknown tool: {name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding_tools.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/tools.py tests/test_grounding_tools.py
git commit -m "feat: grounding ToolSpec defs + dataset dispatch (S3 task 2)"
```

---

### Task 3: Q셋 + gold 생성기

**Files:**
- Create: `src/bakery/ontology/grounding/questions.py`
- Create: `tests/test_grounding_questions.py`

**Interfaces:**
- Consumes: `functions.py`, `DailyDataset`, `_call` from tools.py (reuse for gold).
- Produces: `Question(id, text, grader_type, source_fn, fn_kwargs, tolerance, output_schema)`, `QUESTIONS: list[Question]` (사전등록 ~12개), `build_gold(question, dataset) -> dict` (정답을 OntologyFunction 직접호출로 결정론 생성, grader_type별 정규화 형태로).

**grader_type별 gold 형태:** numeric → `{"answer_value": float}`; ranking → `{"top_items": [str]}`; decomposition → `{"order_qty": float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounding_questions.py
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.questions import QUESTIONS, build_gold, Question


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def test_questions_registered_and_typed():
    assert 8 <= len(QUESTIONS) <= 16
    valid = {"numeric", "ranking", "decomposition"}
    assert all(q.grader_type in valid for q in QUESTIONS)
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)  # unique ids


def test_build_gold_is_deterministic(dataset):
    for q in QUESTIONS:
        g1 = build_gold(q, dataset)
        g2 = build_gold(q, dataset)
        assert g1 == g2                       # determinism
        if q.grader_type == "numeric":
            assert "answer_value" in g1
        elif q.grader_type == "ranking":
            assert isinstance(g1["top_items"], list)
        elif q.grader_type == "decomposition":
            assert "order_qty" in g1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_questions.py -v`
Expected: FAIL — `ModuleNotFoundError: ...grounding.questions`

- [ ] **Step 3: Write minimal implementation**

Pick concrete `store_id`/`item_id`/`period` from the synthetic dataset at import time is fragile; instead store *symbolic* kwargs and resolve against the dataset in `build_gold`. Use the first store and its full date range, and the top item by sales for item-specific questions.

```python
# src/bakery/ontology/grounding/questions.py
"""Pre-registered grounding question set + deterministic gold generator.

Questions are fixed in code (fairness §8: no cherry-picking). Gold answers are
produced by calling the OntologyFunction directly — never hand-labeled — so the
eval is reproducible and the grounded arm's job is to reach the same number
through tool calls, while the rag-only arm must guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import functions as fn
from ...data.loader import DailyDataset


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    grader_type: str            # numeric | ranking | decomposition
    source_fn: str
    fn_kwargs: dict = field(default_factory=dict)
    tolerance: float = 0.05     # relative, numeric only


def _ctx(dataset: DailyDataset):
    """Resolve a stable (store, period, top_item) from the dataset."""
    daily = dataset.daily
    store = daily["store_id"].iloc[0]
    dd = pd.to_datetime(daily.loc[daily["store_id"] == store, "date"])
    period = (str(dd.min().date()), str(dd.max().date()))
    sub = daily[daily["store_id"] == store]
    top_item = sub.groupby("item_id", observed=True)["sold_units"].sum().idxmax()
    return store, period, top_item


QUESTIONS: list[Question] = [
    Question("q_rank_top3", "광교에서 매진 위험이 가장 높은 상위 3개 품목은?",
             "ranking", "rank_stockout_risk", {"k": 3}),
    Question("q_rank_top5", "매진 위험 상위 5개 품목은?", "ranking", "rank_stockout_risk", {"k": 5}),
    Question("q_waste", "이 기간 이 매장의 폐기(capacity-sold) 수량 합계는?",
             "numeric", "waste_cost", {}),
    Question("q_diff_weekend", "주말일 때와 아닐 때 일 판매량 평균 차이는?",
             "numeric", "demand_diff_by_condition", {"condition_col": "is_weekend", "frame": "calendar"}),
    Question("q_diff_rain", "비 올 때와 안 올 때 일 판매량 평균 차이는?",
             "numeric", "demand_diff_by_condition", {"condition_col": "is_rain", "frame": "weather"}),
    Question("q_order_top", "상위 품목의 권장 발주량은?", "decomposition", "explain_order", {}),
    Question("q_whatif_up", "수요 30, 발주 30에서 발주를 10 늘리면 기대비용은?",
             "numeric", "what_if", {"demand_point": 30.0, "base_order": 30.0, "delta_order": 10.0}),
    Question("q_whatif_down", "수요 30, 발주 40에서 발주를 -10 줄이면 기대비용은?",
             "numeric", "what_if", {"demand_point": 30.0, "base_order": 40.0, "delta_order": -10.0}),
    Question("q_rank_top1", "매진 위험이 가장 높은 1개 품목은?", "ranking", "rank_stockout_risk", {"k": 1}),
    Question("q_diff_offday", "휴무일 여부에 따른 일 판매량 평균 차이는?",
             "numeric", "demand_diff_by_condition", {"condition_col": "is_off_day", "frame": "calendar"}),
]


def build_gold(question: Question, dataset: DailyDataset) -> dict:
    store, period, top_item = _ctx(dataset)
    k = question.fn_kwargs
    if question.source_fn == "rank_stockout_risk":
        ranked = fn.rank_stockout_risk(dataset.daily, store, period, k["k"])
        return {"top_items": list(ranked["item_id"])}
    if question.source_fn == "waste_cost":
        return {"answer_value": float(fn.waste_cost(dataset.daily, store, period)["waste_cost"])}
    if question.source_fn == "demand_diff_by_condition":
        frame = dataset.calendar if k["frame"] == "calendar" else dataset.weather
        out = fn.demand_diff_by_condition(dataset.daily, frame, store, k["condition_col"])
        return {"answer_value": float(out["diff"])}
    if question.source_fn == "explain_order":
        lin = fn.explain_order(dataset.daily, store, top_item, period)
        return {"order_qty": float(lin["contribution"].sum())}
    if question.source_fn == "what_if":
        r = fn.what_if(k["demand_point"], k["base_order"], k["delta_order"])
        return {"answer_value": float(r.new_expected_cost)}
    raise KeyError(question.source_fn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding_questions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/questions.py tests/test_grounding_questions.py
git commit -m "feat: pre-registered Q-set + deterministic gold generator (S3 task 3)"
```

---

### Task 4: 채점기 + EvalReport

**Files:**
- Create: `src/bakery/ontology/grounding/scorer.py`
- Create: `tests/test_grounding_scorer.py`

**Interfaces:**
- Consumes: `Question` (Task 3).
- Produces: `grade(question, answer: dict, gold: dict) -> bool` (유형별); `QResult(id, grader_type, grounded_ok, rag_ok)`; `EvalReport(results, grounded_accuracy, rag_accuracy, delta)`; `summarize(results) -> EvalReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounding_scorer.py
from bakery.ontology.grounding.questions import Question
from bakery.ontology.grounding.scorer import grade, summarize, QResult


def _q(gt):
    return Question(id="x", text="", grader_type=gt, source_fn="f")


def test_numeric_within_tolerance():
    q = Question(id="x", text="", grader_type="numeric", source_fn="f", tolerance=0.05)
    assert grade(q, {"answer_value": 102.0}, {"answer_value": 100.0}) is True
    assert grade(q, {"answer_value": 120.0}, {"answer_value": 100.0}) is False


def test_numeric_zero_gold_exact():
    q = Question(id="x", text="", grader_type="numeric", source_fn="f", tolerance=0.05)
    assert grade(q, {"answer_value": 0.0}, {"answer_value": 0.0}) is True
    assert grade(q, {"answer_value": 1.0}, {"answer_value": 0.0}) is False


def test_ranking_top1_match():
    q = _q("ranking")
    assert grade(q, {"top_items": ["A", "B"]}, {"top_items": ["A", "C"]}) is True   # top-1 match
    assert grade(q, {"top_items": ["B", "A"]}, {"top_items": ["A", "C"]}) is False


def test_decomposition_qty_match():
    q = _q("decomposition")
    assert grade(q, {"order_qty": 23.0}, {"order_qty": 23.0}) is True
    assert grade(q, {"order_qty": 25.0}, {"order_qty": 23.0}) is False


def test_malformed_answer_is_wrong():
    q = _q("numeric")
    assert grade(q, {}, {"answer_value": 5.0}) is False


def test_summarize_computes_delta():
    results = [QResult("a", "numeric", True, False), QResult("b", "ranking", True, True)]
    rep = summarize(results)
    assert rep.grounded_accuracy == 1.0
    assert rep.rag_accuracy == 0.5
    assert rep.delta == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: ...grounding.scorer`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/ontology/grounding/scorer.py
"""Deterministic graders + delta report (design §7).

Grading is type-specific and never calls an LLM: numeric uses relative
tolerance (exact when gold is 0), ranking uses top-1 match, decomposition uses
exact order-qty match. A malformed/missing answer counts as wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from .questions import Question

_QTY_TOL = 1e-6   # decomposition order qty is rounded; tiny float slack


@dataclass(frozen=True)
class QResult:
    id: str
    grader_type: str
    grounded_ok: bool
    rag_ok: bool


@dataclass(frozen=True)
class EvalReport:
    results: list[QResult]
    grounded_accuracy: float
    rag_accuracy: float
    delta: float


def grade(question: Question, answer: dict, gold: dict) -> bool:
    if not isinstance(answer, dict):
        return False
    if question.grader_type == "numeric":
        return _grade_numeric(answer.get("answer_value"), gold["answer_value"], question.tolerance)
    if question.grader_type == "ranking":
        return _grade_ranking(answer.get("top_items"), gold["top_items"])
    if question.grader_type == "decomposition":
        return _grade_qty(answer.get("order_qty"), gold["order_qty"])
    raise KeyError(question.grader_type)


def _grade_numeric(pred, gold: float, tol: float) -> bool:
    if not isinstance(pred, (int, float)):
        return False
    if gold == 0:
        return abs(pred) <= _QTY_TOL
    return abs(pred - gold) / abs(gold) <= tol


def _grade_ranking(pred, gold: list) -> bool:
    if not isinstance(pred, list) or not pred or not gold:
        return False
    return pred[0] == gold[0]            # top-1 match


def _grade_qty(pred, gold: float) -> bool:
    if not isinstance(pred, (int, float)):
        return False
    return abs(pred - gold) <= max(_QTY_TOL, abs(gold) * 1e-3)


def summarize(results: list[QResult]) -> EvalReport:
    n = len(results) or 1
    g = sum(r.grounded_ok for r in results) / n
    r = sum(r.rag_ok for r in results) / n
    return EvalReport(results=results, grounded_accuracy=g, rag_accuracy=r, delta=g - r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding_scorer.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/scorer.py tests/test_grounding_scorer.py
git commit -m "feat: grounding graders + delta report (S3 task 4)"
```

---

### Task 5: arms — grounded / rag-only + tool-loop

**Files:**
- Create: `src/bakery/ontology/grounding/arms.py`
- Create: `tests/test_grounding_arms.py`

**Interfaces:**
- Consumes: `LLMClient`, `Message`, `ToolCall`, `ToolResult`, `LLMResponse` (Task 1); `TOOL_SPECS`, `dispatch` (Task 2); `Question` (Task 3); `BAKERY_ONTOLOGY` (schema.py, for knowledge chunks).
- Produces: `OUTPUT_SCHEMAS: dict[str, dict]` (grader_type → JSON schema for the final answer); `run_grounded(client, question, dataset) -> dict` (parsed answer); `run_rag_only(client, question, dataset) -> dict`.

**tool-loop:** grounded는 `TOOL_SPECS`를 넘기고, 응답에 `tool_calls`가 있으면 `dispatch`로 실행→`Message(role="tool")` 주입→재호출, 없으면 `output_schema`로 최종 답을 강제해 `parsed` 반환. rag-only는 tools 없이 OntologyKnowledge 청크를 시스템 프롬프트에 넣고 바로 `output_schema` 답.

- [ ] **Step 1: Write the failing test (FakeLLMClient, no key)**

```python
# tests/test_grounding_arms.py
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import LLMResponse, ToolCall, Message
from bakery.ontology.grounding.questions import Question
from bakery.ontology.grounding import arms


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


class FakeLLM:
    """Scripted client: first call emits a tool_call, second returns parsed answer."""
    def __init__(self, tool_call, parsed):
        self._tool_call, self._parsed, self.calls = tool_call, parsed, 0

    def generate(self, messages, *, tools=None, output_schema=None):
        self.calls += 1
        if tools and self.calls == 1:
            return LLMResponse(text=None, tool_calls=[self._tool_call], parsed=None)
        return LLMResponse(text=None, tool_calls=[], parsed=self._parsed)


def test_run_grounded_runs_tool_then_answers(dataset):
    store = dataset.daily["store_id"].iloc[0]
    import pandas as pd
    dd = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    tc = ToolCall(id="c1", name="rank_stockout_risk",
                  arguments={"store_id": store, "period": [str(dd.min().date()), str(dd.max().date())], "k": 3})
    fake = FakeLLM(tc, {"top_items": ["X"]})
    q = Question(id="q", text="", grader_type="ranking", source_fn="rank_stockout_risk", fn_kwargs={"k": 3})
    answer = arms.run_grounded(fake, q, dataset)
    assert fake.calls == 2                       # tool turn + answer turn
    assert answer == {"top_items": ["X"]}


def test_run_rag_only_no_tools(dataset):
    fake = FakeLLM(None, {"answer_value": 1.0})
    q = Question(id="q", text="", grader_type="numeric", source_fn="waste_cost")
    answer = arms.run_rag_only(fake, q, dataset)
    assert fake.calls == 1                        # single turn, no tool loop
    assert answer == {"answer_value": 1.0}


def test_output_schemas_cover_grader_types():
    assert set(arms.OUTPUT_SCHEMAS) == {"numeric", "ranking", "decomposition"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_arms.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run_grounded'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/ontology/grounding/arms.py
"""Two eval arms over the same model — grounded (tools) vs rag-only (knowledge).

Fairness §8: identical model/params; the ONLY difference is whether the
OntologyFunction tools are exposed. The grounded arm runs a provider-neutral
tool loop (LLMClient.generate → dispatch → re-ask); rag-only gets the ontology
knowledge chunks as context and must answer without tools.
"""

from __future__ import annotations

from ..schema import BAKERY_ONTOLOGY
from ...data.loader import DailyDataset
from .llm import LLMClient, Message
from .questions import Question
from .tools import TOOL_SPECS, dispatch

MAX_TOOL_TURNS = 6

OUTPUT_SCHEMAS: dict[str, dict] = {
    "numeric": {"type": "object", "properties": {"answer_value": {"type": "number"}},
                "required": ["answer_value"], "additionalProperties": False},
    "ranking": {"type": "object", "properties": {"top_items": {"type": "array", "items": {"type": "string"}}},
                "required": ["top_items"], "additionalProperties": False},
    "decomposition": {"type": "object", "properties": {"order_qty": {"type": "number"}},
                      "required": ["order_qty"], "additionalProperties": False},
}

_GROUNDED_SYS = (
    "You answer bakery ordering questions using ONLY the provided tools. "
    "Call the relevant tool(s), then return the final answer in the required JSON schema. "
    "Never guess a number you can compute with a tool."
)


def _knowledge_text() -> str:
    return "\n".join(f"- {k.name}: {k.content}" for k in BAKERY_ONTOLOGY.knowledge)


_RAG_SYS = (
    "You answer bakery ordering questions using ONLY the domain knowledge below. "
    "You have no data access; give your best estimate in the required JSON schema.\n\n"
    + _knowledge_text()
)


def run_grounded(client: LLMClient, question: Question, dataset: DailyDataset) -> dict:
    schema = OUTPUT_SCHEMAS[question.grader_type]
    messages = [Message(role="system", content=_GROUNDED_SYS),
                Message(role="user", content=question.text)]
    for _ in range(MAX_TOOL_TURNS):
        resp = client.generate(messages, tools=TOOL_SPECS, output_schema=schema)
        if not resp.tool_calls:
            return resp.parsed or {}
        messages.append(Message(role="assistant", tool_calls=resp.tool_calls))
        for call in resp.tool_calls:
            result = dispatch(call, dataset)
            messages.append(Message(role="tool", content=result.content, tool_call_id=result.call_id))
    # tool budget exhausted — force a final answer with no tools
    return (client.generate(messages, output_schema=schema).parsed) or {}


def run_rag_only(client: LLMClient, question: Question, dataset: DailyDataset) -> dict:
    schema = OUTPUT_SCHEMAS[question.grader_type]
    messages = [Message(role="system", content=_RAG_SYS),
                Message(role="user", content=question.text)]
    return client.generate(messages, output_schema=schema).parsed or {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding_arms.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/arms.py tests/test_grounding_arms.py
git commit -m "feat: grounded/rag-only arms + neutral tool-loop (S3 task 5)"
```

---

### Task 6: OpenAIClient 어댑터

**Files:**
- Modify: `src/bakery/ontology/grounding/llm.py` (add `OpenAIClient` + `make_llm_client`)
- Modify: `pyproject.toml` (add `openai` dependency)
- Create: `tests/test_grounding_openai_adapter.py`

**Interfaces:**
- Consumes: neutral types (Task 1).
- Produces: `OpenAIClient(model="gpt-5-mini", api_key=None)` implementing `LLMClient`; `make_llm_client(provider, model) -> LLMClient`. Internal: `_to_openai_tools(tools)`, `_to_response_format(schema)`, `_parse_response(completion) -> LLMResponse` — the conversion seams (unit-testable without a key).

**Note:** Verify `chat.completions.parse(model, messages, tools, response_format)` + `message.tool_calls`/`message.parsed` against context7 `/openai/openai-python` before writing (already confirmed for this plan). `response_format` accepts `{"type": "json_schema", "json_schema": {"name", "schema", "strict": true}}`.

- [ ] **Step 1: Write the failing test (conversion seams only — no key)**

```python
# tests/test_grounding_openai_adapter.py
import os
import pytest
from bakery.ontology.grounding.llm import ToolSpec, make_llm_client
from bakery.ontology.grounding import llm as L


def test_to_openai_tools_shape():
    spec = ToolSpec("f", "desc", {"type": "object", "properties": {"x": {"type": "number"}},
                                  "required": ["x"], "additionalProperties": False})
    out = L._to_openai_tools([spec])
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "f"
    assert out[0]["function"]["parameters"]["properties"]["x"]["type"] == "number"


def test_to_response_format_wraps_json_schema():
    rf = L._to_response_format({"type": "object", "properties": {}, "additionalProperties": False})
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert "schema" in rf["json_schema"]


def test_make_llm_client_openai():
    client = make_llm_client("openai", "gpt-5-mini")
    assert isinstance(client, L.OpenAIClient)


def test_make_llm_client_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        make_llm_client("grok", "x")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_live_structured_answer():
    client = make_llm_client("openai", "gpt-5-mini")
    from bakery.ontology.grounding.llm import Message
    schema = {"type": "object", "properties": {"answer_value": {"type": "number"}},
              "required": ["answer_value"], "additionalProperties": False}
    resp = client.generate([Message(role="user", content="Return answer_value 7.")], output_schema=schema)
    assert resp.parsed["answer_value"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_openai_adapter.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_to_openai_tools'`

- [ ] **Step 3: Add the dependency**

Run: `uv add openai`
Expected: `openai` appears in `pyproject.toml` dependencies.

- [ ] **Step 4: Write the adapter**

```python
# append to src/bakery/ontology/grounding/llm.py
import json
import os


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.parameters, "strict": True}}
            for t in tools]


def _to_response_format(schema: dict) -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": True}}


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append({"role": "assistant", "content": m.content,
                        "tool_calls": [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                       for c in m.tool_calls]})
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _parse_response(completion) -> LLMResponse:
    msg = completion.choices[0].message
    calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
             for tc in (msg.tool_calls or [])]
    parsed = getattr(msg, "parsed", None)
    return LLMResponse(text=msg.content, tool_calls=calls, parsed=parsed)


class OpenAIClient:
    """OpenAI adapter (gpt-5-mini). The only provider-specific code in S3."""

    def __init__(self, model: str = "gpt-5-mini", api_key: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model

    def generate(self, messages, *, tools=None, output_schema=None) -> LLMResponse:
        kwargs: dict = {"model": self._model, "messages": _to_openai_messages(messages)}
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        if output_schema:
            kwargs["response_format"] = _to_response_format(output_schema)
        completion = self._client.chat.completions.parse(**kwargs)
        return _parse_response(completion)


def make_llm_client(provider: str, model: str, **kw) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(model=model, **kw)
    raise ValueError(f"unknown provider: {provider} (anthropic adapter: add when 발급)")
```

Also export from `__init__.py`: add `OpenAIClient, make_llm_client` to the import and `__all__`.

- [ ] **Step 5: Run tests to verify pass (key-free ones)**

Run: `uv run pytest tests/test_grounding_openai_adapter.py -v`
Expected: 4 PASS, 1 SKIP (live test, no key)

- [ ] **Step 6: Commit**

```bash
git add src/bakery/ontology/grounding/llm.py src/bakery/ontology/grounding/__init__.py pyproject.toml uv.lock tests/test_grounding_openai_adapter.py
git commit -m "feat: OpenAI adapter + provider factory (S3 task 6)"
```

---

### Task 7: run.py — 실측 엔트리 + 리포트

**Files:**
- Create: `src/bakery/ontology/grounding/run.py`
- Modify: `src/bakery/cli.py` (add `grounding-eval` 서브커맨드 — 기존 CLI 패턴 따름)

**Interfaces:**
- Consumes: everything above.
- Produces: `run_eval(provider, model, dataset) -> EvalReport`; CLI `bakery grounding-eval --provider openai --model gpt-5-mini`.

- [ ] **Step 1: Write the failing test (FakeLLM end-to-end, no key)**

```python
# tests/test_grounding_run.py
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.run import run_eval_with_client
from bakery.ontology.grounding.llm import LLMResponse


def test_run_eval_produces_report():
    dataset = load_dataset("synthetic")
    # minimal smoke: a fake that always returns empty answers → report still computes
    class Empty:
        def generate(self, messages, *, tools=None, output_schema=None):
            return LLMResponse(text=None, tool_calls=[], parsed={})
    report = run_eval_with_client(Empty(), dataset)
    assert 0.0 <= report.grounded_accuracy <= 1.0
    assert report.delta == report.grounded_accuracy - report.rag_accuracy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding_run.py -v`
Expected: FAIL — `ModuleNotFoundError: ...grounding.run`

- [ ] **Step 3: Write implementation**

```python
# src/bakery/ontology/grounding/run.py
"""Grounding eval entry point. run_eval_with_client is key-free testable;
run_eval wires the real provider client (needs OPENAI_API_KEY)."""

from __future__ import annotations

from ...data.loader import DailyDataset, load_dataset
from .arms import run_grounded, run_rag_only
from .llm import LLMClient, make_llm_client
from .questions import QUESTIONS, build_gold
from .scorer import EvalReport, QResult, grade, summarize


def run_eval_with_client(client: LLMClient, dataset: DailyDataset) -> EvalReport:
    results = []
    for q in QUESTIONS:
        gold = build_gold(q, dataset)
        g_ans = run_grounded(client, q, dataset)
        r_ans = run_rag_only(client, q, dataset)
        results.append(QResult(q.id, q.grader_type,
                               grade(q, g_ans, gold), grade(q, r_ans, gold)))
    return summarize(results)


def run_eval(provider: str = "openai", model: str = "gpt-5-mini",
             source: str = "synthetic") -> EvalReport:
    client = make_llm_client(provider, model)
    return run_eval_with_client(client, load_dataset(source))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding_run.py -v`
Expected: PASS

- [ ] **Step 5: Wire CLI (follow existing `src/bakery/cli.py` subcommand pattern)**

Read `src/bakery/cli.py` first; add a `grounding-eval` subcommand that calls `run_eval(...)` and prints `report.grounded_accuracy`, `report.rag_accuracy`, `report.delta`, plus the `"synthetic 시연 (measured on synthetic, not real data)"` label (fairness §8).

- [ ] **Step 6: Run full suite + commit**

Run: `uv run pytest -q`
Expected: all PASS (live grounding test SKIP without key)

```bash
git add src/bakery/ontology/grounding/run.py src/bakery/cli.py tests/test_grounding_run.py
git commit -m "feat: grounding eval entry + CLI subcommand (S3 task 7)"
```

---

## Manual verification (key required, optional)

With `OPENAI_API_KEY` set: `uv run bakery grounding-eval --provider openai --model gpt-5-mini`.
Expected: grounded_accuracy 높음(함수 진짜값), rag_accuracy 낮음, **delta > 0**. delta가 음수/0이면 프롬프트·도구 노출 점검(공정성 위반 아닌지: 양 arm 동일 모델인지 먼저 확인).

---

## Self-Review (spec coverage)

- §3 2 arm + gold → Task 3(gold), Task 5(arms), Task 7(wiring). ✅
- §4 provider 전환(중립타입/어댑터 격리/팩토리) → Task 1, Task 6. ✅
- §5 Q셋 3유형 + structured output 강제 → Task 3, Task 5(OUTPUT_SCHEMAS). ✅
- §6 arms → Task 5. ✅
- §7 채점 + delta + 시연 라벨 → Task 4, Task 7. ✅
- §8 공정성(동일모델·사전등록·결정론gold) → Task 3/5 설계 + Task 7 라벨. ✅
- §9 키 의존 분리(CI 키없이) → 모든 테스트 키불필요, live만 skipif. ✅
- Open Q(§12): gpt-5-mini SDK 시그니처는 Task 6 note에서 context7 확인 완료; tolerance ±5% = Question 기본값; ranking top-1 = scorer. ✅

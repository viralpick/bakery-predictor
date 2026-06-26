# Closed-Loop Order Recommendation (v7 S5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** grounded 에이전트가 발주를 추천(read-only) → 사람 승인 게이트 → writeback commit → 확정 발주시트로 잇는 하류 closed-loop을 구현한다.

**Architecture:** 옵션 C 하이브리드. LLM은 read 도구만 호출해 구조화된 발주 제안(`list[dict]`)을 emit하고, 결정론 orchestrator(`loop.py`)가 제안을 검증 → `WritebackStore.propose_order`(PENDING) → 주입된 `GatePolicy` 결정 → `approve`/`reject`로 commit한다. LLM은 상태를 직접 mutate하지 않는다.

**Tech Stack:** Python 3.x, pandas, dataclasses, typer(CLI), pytest. 기존 모듈 재사용: `ontology/grounding/arms.py`(tool-loop), `ontology/grounding/llm.py`(LLMClient Protocol), `ontology/writeback.py`(WritebackStore), `ontology/functions.py`(OntologyFunction registry).

## Global Constraints

- **결정론**: 모든 타임스탬프는 caller가 ISO 문자열(`now`)로 주입한다. 어떤 코드도 `datetime.now()`를 호출하지 않는다.
- **LLM read-only**: LLM 도구 surface에 write 함수를 노출하지 않는다. 쓰기는 orchestrator(결정론 코드)만 수행한다.
- **정확도 주장 금지**: closed-loop은 메커니즘 시연이다. 추천 수량에 정확도 주장을 하지 않는다(정확도는 S3 grounding eval 몫). 산출물에 "synthetic 메커니즘 시연" 라벨.
- **게이트 일원화**: closed-loop의 `WritebackStore`는 `require_approval=True` 고정. 승인 동작은 주입된 `GatePolicy` 하나로 제어한다. autonomous(frontier)는 `auto_approve` 정책으로 표현한다.
- **기존 테스트 불변**: `uv run pytest`가 전부 통과해야 한다(현재 217 passed/1 skip). 기존 5개 read OntologyFunctionSpec은 `side` 디폴트로 깨지지 않아야 한다.
- 새 의존성 추가 금지. 커밋 메시지 말미에 Co-Authored-By / Claude-Session 트레일러(기존 커밋 관례).

---

### Task 1: OntologyFunctionSpec `side` 마커 + write 함수 등록

`OntologyFunctionSpec`에 `side: "read" | "write"` 필드를 추가하고, writeback 쓰기 연산을 온톨로지 함수 surface에 등록한다. 목적: surface 완성 + lineage. 이 write 스펙은 orchestrator가 참조하며 **LLM 도구 surface(tools.py의 TOOL_SPECS)에는 들어가지 않는다**(TOOL_SPECS는 수기 리스트라 자동 누출 없음).

**Files:**
- Modify: `src/bakery/ontology/functions.py` (OntologyFunctionSpec 정의 168-176, FUNCTION_REGISTRY 181-197)
- Test: `tests/test_ontology_functions.py`

**Interfaces:**
- Consumes: 기존 `OntologyFunctionSpec`, `FUNCTION_REGISTRY`, `WritebackStore`(writeback.py).
- Produces:
  - `OntologyFunctionSpec(name, description, params, returns, impl, side="read")` — `side` 추가, 디폴트 `"read"`.
  - `FUNCTION_REGISTRY["propose_order"]` → `side="write"`, `impl=WritebackStore.propose_order`.
  - `FUNCTION_REGISTRY["commit_order"]` → `side="write"`, `impl=WritebackStore.approve` (commit = 게이트 승인).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_ontology_functions.py` 끝에 추가:

```python
def test_function_spec_defaults_to_read_side():
    from bakery.ontology.functions import FUNCTION_REGISTRY
    assert FUNCTION_REGISTRY["rank_stockout_risk"].side == "read"
    # 기존 5개 read 함수 전부 read
    read_names = {n for n, s in FUNCTION_REGISTRY.items() if s.side == "read"}
    assert {"rank_stockout_risk", "explain_order", "what_if",
            "waste_cost", "demand_diff_by_condition"} <= read_names


def test_write_functions_registered_with_write_side():
    from bakery.ontology.functions import FUNCTION_REGISTRY
    from bakery.ontology.writeback import WritebackStore
    assert FUNCTION_REGISTRY["propose_order"].side == "write"
    assert FUNCTION_REGISTRY["commit_order"].side == "write"
    # impl이 실제 WritebackStore 메서드를 가리킨다
    assert FUNCTION_REGISTRY["propose_order"].impl is WritebackStore.propose_order
    assert FUNCTION_REGISTRY["commit_order"].impl is WritebackStore.approve
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_ontology_functions.py::test_write_functions_registered_with_write_side -v`
Expected: FAIL — `KeyError: 'propose_order'` 또는 `AttributeError: ... 'side'`.

- [ ] **Step 3: 최소 구현**

`src/bakery/ontology/functions.py` — `OntologyFunctionSpec`에 `side` 필드 추가:

```python
@dataclass(frozen=True)
class OntologyFunctionSpec:
    """Agent-facing metadata for one function (name, params, return, impl)."""

    name: str
    description: str
    params: tuple[str, ...]
    returns: str
    impl: Callable
    side: str = "read"          # "read" | "write" — write는 LLM 도구 surface 제외
```

파일 상단 import 영역에 추가(이미 있으면 생략):

```python
from .writeback import WritebackStore
```

`FUNCTION_REGISTRY` dict 닫는 `}` 직전에 write 스펙 2개 추가:

```python
    "propose_order": OntologyFunctionSpec(
        "propose_order", "Write a PENDING order recommendation (human-approval-gated).",
        ("store_id", "item_id", "date", "proposed_qty"), "OrderRecord",
        WritebackStore.propose_order, side="write"),
    "commit_order": OntologyFunctionSpec(
        "commit_order", "Commit a PENDING order (approve, optionally correcting qty).",
        ("record_id", "approver", "approved_qty"), "OrderRecord",
        WritebackStore.approve, side="write"),
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_ontology_functions.py -v`
Expected: PASS (신규 2개 + 기존 전부).

순환 import 점검 — Run: `uv run python -c "import bakery.ontology.functions; import bakery.ontology.writeback; print('ok')"`
Expected: `ok` (functions→writeback 단방향, writeback는 functions를 import하지 않음).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/functions.py tests/test_ontology_functions.py
git commit -m "feat: OntologyFunctionSpec.side marker + register propose_order/commit_order (write side)"
```

---

### Task 2: 발주 제안·게이트 타입 + 내장 게이트 정책

`loop.py`에 `OrderProposal`, `GateDecision`, 게이트 정책 3종을 만든다. orchestrator(Task 4)와 분리해 먼저 테스트한다.

**Files:**
- Create: `src/bakery/ontology/loop.py`
- Test: `tests/test_closed_loop.py`

**Interfaces:**
- Produces:
  - `APPROVE = "APPROVE"`, `REJECT = "REJECT"` (게이트 action 상수)
  - `OrderProposal(item_id: str, qty: float, rationale: str)` — frozen dataclass.
  - `GateDecision(action: str, approved_qty: float | None, approver: str)` — frozen dataclass.
  - `GatePolicy` = `Callable[[OrderProposal], GateDecision]` (타입 별칭).
  - `auto_approve(proposal) -> GateDecision` — APPROVE, approved_qty=None(제안대로), approver="autonomous".
  - `approve_as_proposed(proposal) -> GateDecision` — APPROVE, approved_qty=None, approver="human".
  - `human_correct(corrections: dict[str, float], approver: str = "human") -> GatePolicy` — 클로저. item_id가 corrections에 있으면 그 수량으로 보정 승인, 없으면 제안대로 승인.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py` 생성:

```python
from bakery.ontology.loop import (
    APPROVE, OrderProposal, GateDecision,
    auto_approve, approve_as_proposed, human_correct,
)


def test_auto_approve_takes_proposal_as_is():
    p = OrderProposal(item_id="P1", qty=10.0, rationale="r")
    d = auto_approve(p)
    assert d.action == APPROVE
    assert d.approved_qty is None          # None = 제안 수량 그대로
    assert d.approver == "autonomous"


def test_approve_as_proposed_is_human_rubber_stamp():
    p = OrderProposal(item_id="P1", qty=10.0, rationale="r")
    d = approve_as_proposed(p)
    assert d.action == APPROVE
    assert d.approved_qty is None
    assert d.approver == "human"


def test_human_correct_overrides_listed_item_only():
    policy = human_correct({"P1": 8.0})
    corrected = policy(OrderProposal("P1", 10.0, "r"))
    untouched = policy(OrderProposal("P2", 5.0, "r"))
    assert corrected.action == APPROVE and corrected.approved_qty == 8.0
    assert untouched.action == APPROVE and untouched.approved_qty is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.ontology.loop'`.

- [ ] **Step 3: 최소 구현**

`src/bakery/ontology/loop.py` 생성:

```python
"""Closed-loop order recommendation (v7 S5).

A grounded agent proposes orders (read-only); this deterministic orchestrator
validates the proposals, writes them as PENDING records, runs them through a
human-approval GatePolicy, and commits the approved ones. The LLM never mutates
state — writes go through WritebackStore behind the gate.

See docs/superpowers/specs/2026-06-25-closed-loop-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

APPROVE = "APPROVE"
REJECT = "REJECT"


@dataclass(frozen=True)
class OrderProposal:
    item_id: str
    qty: float
    rationale: str


@dataclass(frozen=True)
class GateDecision:
    action: str                      # APPROVE | REJECT
    approved_qty: float | None       # None = approve the proposed qty unchanged
    approver: str


GatePolicy = Callable[[OrderProposal], GateDecision]


def auto_approve(proposal: OrderProposal) -> GateDecision:
    """Frontier mode: commit the proposal unchanged, no human."""
    return GateDecision(APPROVE, None, "autonomous")


def approve_as_proposed(proposal: OrderProposal) -> GateDecision:
    """Human rubber-stamp: approve exactly what the agent proposed."""
    return GateDecision(APPROVE, None, "human")


def human_correct(corrections: dict[str, float], approver: str = "human") -> GatePolicy:
    """Human edits specific items' qty; others approved as proposed."""
    def policy(proposal: OrderProposal) -> GateDecision:
        if proposal.item_id in corrections:
            return GateDecision(APPROVE, float(corrections[proposal.item_id]), approver)
        return GateDecision(APPROVE, None, approver)
    return policy
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: PASS (3개).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/loop.py tests/test_closed_loop.py
git commit -m "feat: closed-loop proposal/gate types + 3 built-in gate policies"
```

---

### Task 3: `recommend_orders` arm (grounded 추천 → 구조화 제안)

기존 grounded tool-loop을 재사용해, 에이전트가 read 도구로 위험/비용을 평가하고 발주 제안 배열을 emit하는 arm을 추가한다. 반환은 `list[dict]`(순환 import 회피 — `OrderProposal` 변환은 orchestrator가 함).

**Files:**
- Modify: `src/bakery/ontology/grounding/arms.py`
- Test: `tests/test_closed_loop.py`

**Interfaces:**
- Consumes: `LLMClient`, `Message`(llm.py); `TOOL_SPECS`, `dispatch`(tools.py); `DailyDataset`(loader); `MAX_TOOL_TURNS`(arms.py 기존).
- Produces:
  - `PROPOSAL_SCHEMA: dict` — `{proposals: [{item_id: str, qty: number, rationale: str}]}`.
  - `recommend_orders(client: LLMClient, dataset: DailyDataset, store_id: str, period: tuple[str, str]) -> list[dict]` — 각 dict는 `{"item_id","qty","rationale"}`. 제안 없으면 `[]`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py`에 추가:

```python
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import LLMResponse, ToolCall
from bakery.ontology.grounding import arms


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


class RecommendFakeLLM:
    """1st call (tools present): emits a rank_stockout_risk tool call.
    2nd call: returns the structured proposals."""
    def __init__(self, store_id, period, proposals):
        self._tc = ToolCall(id="c1", name="rank_stockout_risk",
                            arguments={"store_id": store_id, "period": list(period), "k": 3})
        self._proposals = proposals
        self.calls = 0

    def generate(self, messages, *, tools=None, output_schema=None):
        self.calls += 1
        if tools and self.calls == 1:
            return LLMResponse(text=None, tool_calls=[self._tc], parsed=None)
        return LLMResponse(text=None, tool_calls=[], parsed={"proposals": self._proposals})


def test_recommend_orders_returns_proposal_dicts(dataset):
    store = dataset.daily["store_id"].iloc[0]
    period = ("2024-01-01", "2024-01-07")
    proposals = [{"item_id": "P1", "qty": 12.0, "rationale": "high risk"}]
    fake = RecommendFakeLLM(store, period, proposals)
    out = arms.recommend_orders(fake, dataset, store, period)
    assert fake.calls == 2                 # tool turn + proposal turn
    assert out == proposals


def test_recommend_orders_empty_when_no_proposals(dataset):
    store = dataset.daily["store_id"].iloc[0]
    period = ("2024-01-01", "2024-01-07")
    fake = RecommendFakeLLM(store, period, [])
    out = arms.recommend_orders(fake, dataset, store, period)
    assert out == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py::test_recommend_orders_returns_proposal_dicts -v`
Expected: FAIL — `AttributeError: module 'bakery.ontology.grounding.arms' has no attribute 'recommend_orders'`.

- [ ] **Step 3: 최소 구현**

`src/bakery/ontology/grounding/arms.py` 끝에 추가:

```python
PROPOSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "qty": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["item_id", "qty", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}

_RECOMMEND_SYS = (
    "You are a bakery ordering agent. Use ONLY the provided tools to assess each "
    "item's stockout risk and waste cost for the given store and period, then propose "
    "order quantities for the items that need attention. You MAY adjust a tool's "
    "suggested order_qty using judgment about context the model cannot see (events, "
    "promotions, operator experience) — explain any adjustment in the rationale. "
    "Return proposals in the required JSON schema. "
    "도구가 반환한 수치를 근거로 삼되, 모델이 모르는 맥락이 있으면 수량을 조정하고 rationale에 이유를 적어라."
)


def recommend_orders(client: LLMClient, dataset: DailyDataset,
                     store_id: str, period: tuple[str, str]) -> list[dict]:
    """Grounded agent proposes orders (read-only tools). Returns proposal dicts."""
    ctx = (f"분석 대상 — 매장(store_id): {store_id}, 기간: {period[0]} ~ {period[1]}. "
           f"도구를 호출할 때 이 store_id와 period=[{period[0]}, {period[1]}]를 사용하라.")
    messages = [Message(role="system", content=_RECOMMEND_SYS),
                Message(role="user", content=f"{ctx}\n\n발주 추천을 제안하라.")]
    for _ in range(MAX_TOOL_TURNS):
        resp = client.generate(messages, tools=TOOL_SPECS, output_schema=PROPOSAL_SCHEMA)
        if not resp.tool_calls:
            return (resp.parsed or {}).get("proposals", [])
        messages.append(Message(role="assistant", tool_calls=resp.tool_calls))
        for call in resp.tool_calls:
            result = dispatch(call, dataset)
            messages.append(Message(role="tool", content=result.content, tool_call_id=result.call_id))
    return (client.generate(messages, output_schema=PROPOSAL_SCHEMA).parsed or {}).get("proposals", [])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: PASS (Task 2의 3개 + 신규 2개).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/grounding/arms.py tests/test_closed_loop.py
git commit -m "feat: recommend_orders arm — grounded agent emits structured order proposals"
```

---

### Task 4: `run_closed_loop` orchestrator

제안을 검증 → `propose_order`(PENDING) → 게이트 결정 → `approve`/`reject`. 무효 제안은 skip+log.

**Files:**
- Modify: `src/bakery/ontology/loop.py`
- Test: `tests/test_closed_loop.py`

**Interfaces:**
- Consumes: `arms.recommend_orders`(Task 3); `WritebackStore`, `OrderRecord`(writeback.py); `OrderProposal`, `GateDecision`, `GatePolicy`, `APPROVE`(Task 2); `DailyDataset`(loader).
- Produces:
  - `run_closed_loop(client, dataset, store_id: str, period: tuple[str, str], writeback: WritebackStore, gate: GatePolicy, *, now: str) -> list[OrderRecord]` — 생성·확정한 레코드(최종 상태 반영) 순서대로. 무효 제안은 제외. order 대상 date는 `period[0]`.
  - 가드: `writeback.require_approval`이 False면 `ValueError`(closed-loop은 게이트로 승인 제어).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py`에 추가:

```python
from bakery.ontology.loop import run_closed_loop, auto_approve, human_correct
from bakery.ontology.writeback import WritebackStore, APPROVED, REJECTED, PENDING


def _valid_item(dataset, store):
    return dataset.daily.loc[dataset.daily["store_id"] == store, "item_id"].iloc[0]


def test_closed_loop_proposes_and_commits(dataset):
    store = dataset.daily["store_id"].iloc[0]
    item = _valid_item(dataset, store)
    period = ("2024-01-01", "2024-01-07")
    fake = RecommendFakeLLM(store, period,
                            [{"item_id": item, "qty": 12.0, "rationale": "r"}])
    wb = WritebackStore(require_approval=True)
    recs = run_closed_loop(fake, dataset, store, period, wb, auto_approve,
                           now="2024-01-01T09:00:00")
    assert len(recs) == 1
    assert recs[0].status == APPROVED
    assert recs[0].approved_qty == 12.0           # auto_approve → 제안대로
    assert recs[0].date == "2024-01-01"
    assert recs[0].valid_as_of == "2024-01-01T09:00:00"


def test_closed_loop_human_correction_applies(dataset):
    store = dataset.daily["store_id"].iloc[0]
    item = _valid_item(dataset, store)
    period = ("2024-01-01", "2024-01-07")
    fake = RecommendFakeLLM(store, period,
                            [{"item_id": item, "qty": 12.0, "rationale": "r"}])
    wb = WritebackStore(require_approval=True)
    recs = run_closed_loop(fake, dataset, store, period, wb,
                           human_correct({item: 9.0}), now="2024-01-01T09:00:00")
    assert recs[0].approved_qty == 9.0            # 사람 보정
    assert recs[0].override == 9.0 - 12.0


def test_closed_loop_skips_invalid_proposals(dataset):
    store = dataset.daily["store_id"].iloc[0]
    item = _valid_item(dataset, store)
    period = ("2024-01-01", "2024-01-07")
    proposals = [
        {"item_id": item, "qty": 12.0, "rationale": "ok"},
        {"item_id": item, "qty": -3.0, "rationale": "negative"},     # 무효
        {"item_id": "NOT_AN_ITEM", "qty": 5.0, "rationale": "ghost"},  # 무효
        {"item_id": item, "qty": float("nan"), "rationale": "nan"},   # 무효
    ]
    fake = RecommendFakeLLM(store, period, proposals)
    wb = WritebackStore(require_approval=True)
    recs = run_closed_loop(fake, dataset, store, period, wb, auto_approve,
                           now="2024-01-01T09:00:00")
    assert len(recs) == 1                          # 유효 1건만
    assert recs[0].approved_qty == 12.0


def test_closed_loop_rejects_autonomous_store():
    import pytest
    wb = WritebackStore(require_approval=False)
    with pytest.raises(ValueError):
        run_closed_loop(None, None, "S", ("2024-01-01", "2024-01-07"),
                        wb, auto_approve, now="2024-01-01T09:00:00")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py::test_closed_loop_proposes_and_commits -v`
Expected: FAIL — `ImportError: cannot import name 'run_closed_loop'`.

- [ ] **Step 3: 최소 구현**

`src/bakery/ontology/loop.py` 상단 import에 추가:

```python
import logging
import math

from ..data.loader import DailyDataset
from .grounding import arms
from .writeback import OrderRecord, WritebackStore

log = logging.getLogger(__name__)
```

(`from typing import Callable` 아래에 두고, 순환 import 주의 — `arms`는 `loop`을 import하지 않음.)

파일 끝에 추가:

```python
def _is_valid(proposal: OrderProposal, valid_items: set[str]) -> bool:
    if proposal.item_id not in valid_items:
        return False
    if not math.isfinite(proposal.qty) or proposal.qty < 0:
        return False
    return True


def run_closed_loop(client, dataset: DailyDataset, store_id: str,
                    period: tuple[str, str], writeback: WritebackStore,
                    gate: GatePolicy, *, now: str) -> list[OrderRecord]:
    """Recommend → validate → propose(PENDING) → gate → commit. Returns the
    records created this run (invalid proposals skipped)."""
    if not writeback.require_approval:
        raise ValueError(
            "closed-loop drives approval via the GatePolicy; "
            "pass WritebackStore(require_approval=True)")
    target_date = period[0]
    valid_items = set(
        dataset.daily.loc[dataset.daily["store_id"] == store_id, "item_id"])
    raw = arms.recommend_orders(client, dataset, store_id, period)
    out: list[OrderRecord] = []
    for d in raw:
        proposal = OrderProposal(str(d["item_id"]), float(d["qty"]), str(d["rationale"]))
        if not _is_valid(proposal, valid_items):
            log.warning("skipping invalid proposal: %s", d)
            continue
        rec = writeback.propose_order(store_id, proposal.item_id, target_date,
                                      proposal.qty, proposed_at=now)
        decision = gate(proposal)
        if decision.action == APPROVE:
            rec = writeback.approve(rec.record_id, decision.approver,
                                    approved_at=now, approved_qty=decision.approved_qty)
        else:
            rec = writeback.reject(rec.record_id, decision.approver)
        out.append(rec)
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: PASS (전체 9개).

순환 import 점검 — Run: `uv run python -c "import bakery.ontology.loop; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/loop.py tests/test_closed_loop.py
git commit -m "feat: run_closed_loop orchestrator — validate, propose, gate, commit"
```

---

### Task 5: CLI `closed-loop` 명령

end-to-end 실행 진입점. `make_llm_client`로 실 클라이언트를 만들어 `run_closed_loop`을 돌리고, 확정 발주시트를 출력 + 선택적 parquet 저장. CLI는 얇게 — 로직은 Task 4에서 검증됨.

**Files:**
- Modify: `src/bakery/cli.py` (grounding-eval 명령 1161-1182 근처 패턴 따름)
- Test: `tests/test_closed_loop.py` (명령 등록 + 가드 단위 테스트, 키 불필요)

**Interfaces:**
- Consumes: `run_closed_loop`, `auto_approve`, `approve_as_proposed`(loop.py); `make_llm_client`(llm.py); `WritebackStore`(writeback.py); `load_dataset`(loader).
- Produces: typer 명령 `closed-loop` — 옵션 `--store`, `--period`("start,end"), `--policy`("auto"|"human"), `--source`(기본 synthetic), `--model`, `--provider`, `--now`(ISO, 기본 period start의 09:00), `--out`(parquet 경로, 옵션).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py`에 추가:

```python
def test_cli_registers_closed_loop_command():
    from bakery.cli import app
    names = {c.name for c in app.registered_commands}
    assert "closed-loop" in names


def test_select_policy_maps_names():
    from bakery.cli import _select_gate_policy
    from bakery.ontology.loop import auto_approve, approve_as_proposed
    assert _select_gate_policy("auto") is auto_approve
    assert _select_gate_policy("human") is approve_as_proposed
    import pytest
    with pytest.raises(ValueError):
        _select_gate_policy("bogus")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py::test_cli_registers_closed_loop_command -v`
Expected: FAIL — `assert 'closed-loop' in names`.

- [ ] **Step 3: 최소 구현**

`src/bakery/cli.py`의 `grounding-eval` 명령(1182줄) 뒤, `if __name__` 앞에 추가:

```python
def _select_gate_policy(policy: str):
    from .ontology.loop import auto_approve, approve_as_proposed
    if policy == "auto":
        return auto_approve
    if policy == "human":
        return approve_as_proposed
    raise ValueError(f"unknown policy: {policy} (auto|human)")


@app.command("closed-loop")
def cmd_closed_loop(
    store: str,
    period: str,                        # "YYYY-MM-DD,YYYY-MM-DD"
    policy: str = "human",              # auto(frontier) | human(rubber-stamp)
    source: str = "synthetic",
    provider: str = "openai",
    model: str = "gpt-5-mini",
    now: str = "",                      # ISO; 비면 period start의 09:00
    out: str = "",                      # parquet 경로(옵션)
) -> None:
    """v7 하류 closed-loop: grounded 추천 → 사람 게이트 → writeback commit.

    추천은 read 도구만 쓰는 grounded 에이전트가 한다(쓰기는 게이트 통과 후 결정론 코드).
    --source synthetic 이면 시연용. closed-loop은 메커니즘 시연이며 정확도 주장이 아니다.
    """
    from .data.loader import load_dataset
    from .ontology.grounding.llm import make_llm_client
    from .ontology.loop import run_closed_loop
    from .ontology.writeback import WritebackStore

    start, end = (s.strip() for s in period.split(","))
    stamp = now or f"{start}T09:00:00"
    gate = _select_gate_policy(policy)
    client = make_llm_client(provider, model)
    dataset = load_dataset(source)
    wb = WritebackStore(require_approval=True)
    recs = run_closed_loop(client, dataset, store, (start, end), wb, gate, now=stamp)

    console.print(f"[bold]closed-loop[/] store={store} period={start}~{end} policy={policy}")
    for r in recs:
        console.print(f"  {r.item_id}: proposed={r.proposed_qty} → "
                      f"{r.status} qty={r.approved_qty} by={r.approver}")
    if not recs:
        console.print("[yellow]no valid proposals[/]")
    if out:
        wb.to_parquet(out)
        console.print(f"[green]wrote[/] {out} ({len(wb.records)} records)")
    console.print(
        "[yellow]synthetic 메커니즘 시연 (mechanism demo, not accuracy)[/]"
        if source == "synthetic" else f"[cyan]source={source}[/]")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: PASS (전체 11개).

전체 회귀 — Run: `uv run pytest`
Expected: 기존 217 + 신규 = 228 passed, 1 skip(live). 실패 0.

- [ ] **Step 5: 수동 smoke (OPENAI_API_KEY 필요, 선택)**

Run: `uv run bakery closed-loop --store 광교 --period 2024-01-01,2024-01-07 --policy human --source synthetic --out /tmp/cl.parquet`
Expected: 제안 N건 출력 + 각 APPROVED + parquet 기록. 키 없으면 이 단계 skip(테스트는 키 없이 통과).

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/cli.py tests/test_closed_loop.py
git commit -m "feat: closed-loop CLI command — end-to-end recommend→gate→writeback"
```

---

## Self-Review

**Spec coverage:**
- D1(하이브리드, LLM read-only) → Task 3(read 도구만) + Task 4(쓰기는 orchestrator). ✓
- D2(LLM 수량 조정=암묵지) → Task 3 `_RECOMMEND_SYS`(조정 허용+rationale). ✓
- D3(정확도 주장 분리) → Task 5 라벨, plan Global Constraints. ✓
- D4(side 마커 + write 등록) → Task 1. ✓
- D5(게이트 정책 3종) → Task 2. ✓
- 아키텍처(recommend→propose→gate→commit→시트) → Task 4 + Task 5(parquet). ✓
- 에러/엣지(무효 skip, 게이트 거부, 타임스탬프 주입) → Task 4(_is_valid, reject, now). ✓
- 테스트(FakeLLM 결정론, 게이트 3종, 검증 가드, live skip) → Task 2~5. ✓
- 범위 밖(상류 what_if_driver=TODO) → TODO.md 기록 완료(이전 커밋). ✓

**Placeholder scan:** 없음 — 모든 step에 실제 코드/명령/기대출력.

**Type consistency:** `OrderProposal(item_id,qty,rationale)`·`GateDecision(action,approved_qty,approver)`·`run_closed_loop(...,*,now)->list[OrderRecord]`·`recommend_orders(...)->list[dict]`·`_select_gate_policy(str)` — Task 간 시그니처 일치 확인. `WritebackStore.propose_order(store_id,item_id,date,proposed_qty,*,proposed_at)` / `approve(record_id,approver,*,approved_at,approved_qty)` — writeback.py 실제 시그니처와 일치.

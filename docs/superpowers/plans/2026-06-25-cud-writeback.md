# CUD Writeback 골격 (S4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 발주 추천을 사람 승인 게이트(토글) 뒤에서 확정(writeback)하고, 그 확정 시점을 전향적 평가 무결성용 타임스탬프로 보존하는 in-memory store를 만든다.

**Architecture:** `OrderRecord`(frozen dataclass) + `WritebackStore`(append-only in-memory list). store는 우리 모델 추천(`proposed_qty`)과 사람 보정 확정(`approved_qty`)을 한 레코드에 담는 *전향적 정확도 스냅샷* — 학습 데이터가 아니다. 상태전이(PENDING→APPROVED/REJECTED)는 `dataclasses.replace`로 새 인스턴스 교체. LLM 무관, 순수 로직 + 결정론 테스트. 설계 = `docs/superpowers/specs/2026-06-25-cud-writeback-design.md`.

**Tech Stack:** Python, dataclasses, pandas(parquet 직렬화), pytest. `src/bakery/ontology/` 아래 신규 모듈.

## Global Constraints

- 학습 데이터 아님 — store는 우리 모델 시점별 추천(2번)+사람 보정(3번)의 *audit/성능 스냅샷*.
- 시각은 store가 `datetime.now()`를 부르지 않고 **호출자가 명시 전달**(테스트 결정성).
- `valid_as_of`는 leakage 방어가 아니라 **전향적 평가 무결성 + audit** — APPROVED일 때만 채워짐.
- 토글 `require_approval`(기본 True): True=propose는 PENDING(사람 approve 필요), False=propose 즉시 APPROVED(`approver="autonomous"`).
- Open Q 확정: `valid_as_of`/`proposed_at`/`approved_at` = ISO str(YYYY-MM-DDTHH:MM:SS); cutoff 비교는 ISO 사전순(동일 포맷 가정). `record_id` = store 자동 순번 `r{n}`.
- 신규: `src/bakery/ontology/writeback.py`, `tests/test_writeback.py`. `__init__.py`에 export 추가.

---

## File Structure

```
src/bakery/ontology/writeback.py   # OrderRecord + WritebackStore (전부)
tests/test_writeback.py            # 상태전이·토글·confirmed_as_of·override·직렬화
src/bakery/ontology/__init__.py    # OrderRecord, WritebackStore export (modify)
```

---

### Task 1: OrderRecord + WritebackStore 상태전이 + 토글

**Files:**
- Create: `src/bakery/ontology/writeback.py`
- Create: `tests/test_writeback.py`

**Interfaces:**
- Produces:
  - `OrderRecord` frozen dataclass: `record_id, store_id, item_id, date, proposed_qty, proposed_at, status="PENDING", approved_qty=None, approver=None, valid_as_of=None`; property `override -> float | None` (`approved_qty - proposed_qty`, None일 때 None).
  - `WritebackStore(require_approval: bool = True)` with `propose_order(store_id, item_id, date, proposed_qty, *, proposed_at) -> OrderRecord`, `approve(record_id, approver, *, approved_at, approved_qty=None) -> OrderRecord`, `reject(record_id, approver) -> OrderRecord`, and an internal `records` accessor (list).
  - Status constants `PENDING`, `APPROVED`, `REJECTED`, and `AUTONOMOUS_APPROVER = "autonomous"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_writeback.py
import pytest
from bakery.ontology.writeback import (
    OrderRecord, WritebackStore, PENDING, APPROVED, REJECTED, AUTONOMOUS_APPROVER,
)

T0 = "2026-06-12T09:00:00"
T1 = "2026-06-12T15:00:00"


def _store(**kw):
    return WritebackStore(**kw)


def test_propose_creates_pending():
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    assert r.status == PENDING
    assert r.proposed_qty == 100.0
    assert r.approved_qty is None
    assert r.valid_as_of is None
    assert r.override is None
    assert r.record_id == "r1"


def test_approve_without_override():
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    a = s.approve(r.record_id, "alice", approved_at=T1)
    assert a.status == APPROVED
    assert a.approved_qty == 100.0       # defaults to proposed_qty
    assert a.override == 0.0
    assert a.approver == "alice"
    assert a.valid_as_of == T1


def test_approve_with_override():
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    a = s.approve(r.record_id, "alice", approved_at=T1, approved_qty=80.0)
    assert a.approved_qty == 80.0
    assert a.override == -20.0           # human cut 20


def test_reject():
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    j = s.reject(r.record_id, "alice")
    assert j.status == REJECTED
    assert j.valid_as_of is None


def test_reapprove_raises():
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    s.approve(r.record_id, "alice", approved_at=T1)
    with pytest.raises(ValueError):
        s.approve(r.record_id, "bob", approved_at=T1)


def test_unknown_record_raises():
    s = _store()
    with pytest.raises(KeyError):
        s.approve("nope", "alice", approved_at=T1)


def test_toggle_off_auto_approves():
    s = _store(require_approval=False)
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    assert r.status == APPROVED
    assert r.approver == AUTONOMOUS_APPROVER
    assert r.approved_qty == 100.0
    assert r.valid_as_of == T0           # auto-confirmed at propose time
    with pytest.raises(ValueError):      # already APPROVED → cannot re-approve
        s.approve(r.record_id, "alice", approved_at=T1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_writeback.py -v`
Expected: FAIL — `ModuleNotFoundError: bakery.ontology.writeback`

- [ ] **Step 3: Write the implementation**

```python
# src/bakery/ontology/writeback.py
"""CUD writeback skeleton (v7 S4) — human-approval-gated order confirmation.

This store holds our model's per-timestamp order recommendation (proposed_qty,
"발주량 2번") plus the human-corrected confirmed value (approved_qty, "3번").
It is a PROSPECTIVE ACCURACY SNAPSHOT for audit/eval integrity — NOT training
data (the model trains on sales/potential_demand, not order qty). See
docs/superpowers/specs/2026-06-25-cud-writeback-design.md.

Timestamps are caller-supplied ISO strings (the store never calls datetime.now())
so behavior is deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

PENDING = "PENDING"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
AUTONOMOUS_APPROVER = "autonomous"


@dataclass(frozen=True)
class OrderRecord:
    record_id: str
    store_id: str
    item_id: str
    date: str                       # order target date, YYYY-MM-DD
    proposed_qty: float             # 2번 — model output
    proposed_at: str                # ISO timestamp the recommendation was made
    status: str = PENDING
    approved_qty: float | None = None   # 3번 — human-corrected confirmed value
    approver: str | None = None
    valid_as_of: str | None = None      # ISO; set only when APPROVED

    @property
    def override(self) -> float | None:
        """Human correction (approved − proposed); None until approved."""
        if self.approved_qty is None:
            return None
        return self.approved_qty - self.proposed_qty


class WritebackStore:
    """Append-only in-memory store of order recommendations + confirmations."""

    def __init__(self, require_approval: bool = True):
        self.require_approval = require_approval
        self._records: list[OrderRecord] = []
        self._seq = 0

    @property
    def records(self) -> list[OrderRecord]:
        return list(self._records)

    def _next_id(self) -> str:
        self._seq += 1
        return f"r{self._seq}"

    def propose_order(self, store_id: str, item_id: str, date: str,
                      proposed_qty: float, *, proposed_at: str) -> OrderRecord:
        rid = self._next_id()
        if self.require_approval:
            rec = OrderRecord(rid, store_id, item_id, date, float(proposed_qty), proposed_at)
        else:
            rec = OrderRecord(rid, store_id, item_id, date, float(proposed_qty), proposed_at,
                              status=APPROVED, approved_qty=float(proposed_qty),
                              approver=AUTONOMOUS_APPROVER, valid_as_of=proposed_at)
        self._records.append(rec)
        return rec

    def _index(self, record_id: str) -> int:
        for i, r in enumerate(self._records):
            if r.record_id == record_id:
                return i
        raise KeyError(f"unknown record_id: {record_id}")

    def _require_pending(self, rec: OrderRecord) -> None:
        if rec.status != PENDING:
            raise ValueError(f"record {rec.record_id} is {rec.status}, not PENDING")

    def approve(self, record_id: str, approver: str, *, approved_at: str,
                approved_qty: float | None = None) -> OrderRecord:
        i = self._index(record_id)
        rec = self._records[i]
        self._require_pending(rec)
        qty = float(approved_qty) if approved_qty is not None else rec.proposed_qty
        new = replace(rec, status=APPROVED, approved_qty=qty,
                      approver=approver, valid_as_of=approved_at)
        self._records[i] = new
        return new

    def reject(self, record_id: str, approver: str) -> OrderRecord:
        i = self._index(record_id)
        rec = self._records[i]
        self._require_pending(rec)
        new = replace(rec, status=REJECTED, approver=approver)
        self._records[i] = new
        return new
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_writeback.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/writeback.py tests/test_writeback.py
git commit -m "feat: writeback OrderRecord + store state transitions + approval toggle (S4 task 1)"
```

---

### Task 2: confirmed_as_of + to_frame + parquet 직렬화 + export

**Files:**
- Modify: `src/bakery/ontology/writeback.py` (add `confirmed_as_of`, `to_frame`, `to_parquet`, `from_parquet`)
- Modify: `tests/test_writeback.py` (add tests)
- Modify: `src/bakery/ontology/__init__.py` (export `OrderRecord`, `WritebackStore`)

**Interfaces:**
- Consumes: `OrderRecord`, `WritebackStore`, status constants (Task 1).
- Produces:
  - `WritebackStore.confirmed_as_of(cutoff: str) -> list[OrderRecord]` — APPROVED records with `valid_as_of <= cutoff` (ISO lexicographic). Excludes PENDING, REJECTED, and APPROVED with `valid_as_of > cutoff`.
  - `WritebackStore.to_frame() -> pd.DataFrame` — one row per record, all fields + `override` column.
  - `WritebackStore.to_parquet(path) -> None` and classmethod `WritebackStore.from_parquet(path) -> WritebackStore` — round-trip serialization (records only; `require_approval` resets to default True on load).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_writeback.py

def test_confirmed_as_of_filters_by_valid_as_of(tmp_path):
    s = _store()
    # r1: approved at T1 (<= cutoff)
    r1 = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    s.approve(r1.record_id, "alice", approved_at=T1)
    # r2: approved LATER than cutoff
    r2 = s.propose_order("store_A", "item_2", "2026-06-15", 50.0, proposed_at=T0)
    s.approve(r2.record_id, "alice", approved_at="2026-06-20T09:00:00")
    # r3: still PENDING
    s.propose_order("store_A", "item_3", "2026-06-15", 30.0, proposed_at=T0)
    # r4: REJECTED
    r4 = s.propose_order("store_A", "item_4", "2026-06-15", 10.0, proposed_at=T0)
    s.reject(r4.record_id, "alice")

    cutoff = "2026-06-15T00:00:00"
    confirmed = s.confirmed_as_of(cutoff)
    ids = {r.record_id for r in confirmed}
    assert ids == {r1.record_id}        # only r1: APPROVED and valid_as_of <= cutoff


def test_to_frame_has_all_records_and_override():
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    s.approve(r.record_id, "alice", approved_at=T1, approved_qty=80.0)
    df = s.to_frame()
    assert len(df) == 1
    assert "override" in df.columns
    assert df.iloc[0]["override"] == -20.0
    assert df.iloc[0]["status"] == APPROVED


def test_parquet_round_trip(tmp_path):
    s = _store()
    r = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    s.approve(r.record_id, "alice", approved_at=T1, approved_qty=80.0)
    s.propose_order("store_A", "item_2", "2026-06-15", 50.0, proposed_at=T0)  # PENDING
    path = tmp_path / "wb.parquet"
    s.to_parquet(path)
    loaded = WritebackStore.from_parquet(path)
    assert {r.record_id for r in loaded.records} == {r_.record_id for r_ in s.records}
    by_id = {r_.record_id: r_ for r_ in loaded.records}
    assert by_id["r1"].approved_qty == 80.0
    assert by_id["r1"].valid_as_of == T1
    assert by_id["r2"].status == PENDING
    assert by_id["r2"].approved_qty is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_writeback.py -k "confirmed_as_of or to_frame or parquet" -v`
Expected: FAIL — `AttributeError: 'WritebackStore' has no attribute 'confirmed_as_of'`

- [ ] **Step 3: Add the methods to writeback.py**

Add `import pandas as pd` at the top of `src/bakery/ontology/writeback.py` (below `from dataclasses import ...`), and add these methods to `WritebackStore`:

```python
    def confirmed_as_of(self, cutoff: str) -> list[OrderRecord]:
        """APPROVED records confirmed at or before cutoff (ISO lexicographic).

        Reproduces the order state that was confirmed as of a point in time —
        the basis for honest prospective evaluation (no retroactive edits).
        """
        return [r for r in self._records
                if r.status == APPROVED and r.valid_as_of is not None
                and r.valid_as_of <= cutoff]

    def to_frame(self) -> pd.DataFrame:
        cols = ["record_id", "store_id", "item_id", "date", "proposed_qty",
                "proposed_at", "status", "approved_qty", "approver", "valid_as_of"]
        rows = []
        for r in self._records:
            row = {c: getattr(r, c) for c in cols}
            row["override"] = r.override
            rows.append(row)
        return pd.DataFrame(rows, columns=[*cols, "override"])

    def to_parquet(self, path) -> None:
        # override is a derived column; drop before persisting (recomputed on load)
        self.to_frame().drop(columns=["override"]).to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path) -> "WritebackStore":
        df = pd.read_parquet(path)
        store = cls()
        for row in df.itertuples(index=False):
            d = row._asdict()
            aq = d["approved_qty"]
            store._records.append(OrderRecord(
                record_id=str(d["record_id"]), store_id=str(d["store_id"]),
                item_id=str(d["item_id"]), date=str(d["date"]),
                proposed_qty=float(d["proposed_qty"]), proposed_at=str(d["proposed_at"]),
                status=str(d["status"]),
                approved_qty=None if pd.isna(aq) else float(aq),
                approver=None if pd.isna(d["approver"]) else str(d["approver"]),
                valid_as_of=None if pd.isna(d["valid_as_of"]) else str(d["valid_as_of"]),
            ))
        # keep _seq past the loaded ids so new proposes don't collide
        store._seq = len(store._records)
        return store
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_writeback.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 5: Export from `__init__.py`**

In `src/bakery/ontology/__init__.py`, add to the imports and `__all__`:

```python
from .writeback import OrderRecord, WritebackStore
```
Append `"OrderRecord"`, `"WritebackStore"` to the existing `__all__` list (do not remove existing entries).

- [ ] **Step 6: Full suite + commit**

Run: `uv run pytest -q`
Expected: all PASS (only the genuine live OpenAI test skips).

```bash
git add src/bakery/ontology/writeback.py tests/test_writeback.py src/bakery/ontology/__init__.py
git commit -m "feat: confirmed_as_of + to_frame/parquet serialization + export (S4 task 2)"
```

---

## Self-Review (spec coverage)

- §2 OrderRecord 필드(proposed_qty/approved_qty/override/valid_as_of 등) → Task 1. ✅
- §3 propose/approve(보정)/reject + 상태 가드 + 토글 → Task 1. ✅
- §3 confirmed_as_of + to_frame + parquet 헬퍼 → Task 2. ✅
- §1 valid_as_of = 전향적 무결성(APPROVED일 때만, confirmed_as_of로 시점 재현) → Task 1(필드) + Task 2(confirmed_as_of test). ✅
- §3 토글 OFF=자율 즉시 APPROVED → Task 1 test_toggle_off. ✅
- §5 테스트 8항목 → Task 1(6) + Task 2(3, confirmed/frame/parquet) = 9 cases 커버. ✅
- Global: 시각 호출자 주입(proposed_at/approved_at 인자) ✅ / record_id 자동순번 r{n} ✅ / ISO str cutoff 사전순 ✅ / __init__ export ✅
- 범위 밖(OntologyFunction 등록·grounding·실DB·자동발주·아티제 통합) → 어느 task에도 없음(의도적). ✅

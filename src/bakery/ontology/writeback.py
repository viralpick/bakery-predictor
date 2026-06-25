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

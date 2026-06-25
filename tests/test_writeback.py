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

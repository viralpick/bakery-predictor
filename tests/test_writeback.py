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


def test_confirmed_as_of_filters_by_valid_as_of():
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


def test_propose_after_load_no_id_collision(tmp_path):
    s = _store()
    r1 = s.propose_order("store_A", "item_1", "2026-06-15", 100.0, proposed_at=T0)
    s.approve(r1.record_id, "alice", approved_at=T1)
    s.propose_order("store_A", "item_2", "2026-06-15", 50.0, proposed_at=T0)  # r2
    path = tmp_path / "wb.parquet"
    s.to_parquet(path)
    loaded = WritebackStore.from_parquet(path)
    existing = {r.record_id for r in loaded.records}
    new = loaded.propose_order("store_A", "item_3", "2026-06-15", 30.0, proposed_at=T0)
    assert new.record_id not in existing       # no collision
    assert new.record_id == "r3"               # continues the sequence

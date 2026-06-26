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

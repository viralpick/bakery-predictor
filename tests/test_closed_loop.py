import pytest

from bakery.ontology.loop import (
    APPROVE, OrderProposal, GateDecision,
    auto_approve, approve_as_proposed, human_correct,
)
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


# ---------------------------------------------------------------------------
# Task 4: run_closed_loop orchestrator tests
# ---------------------------------------------------------------------------

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

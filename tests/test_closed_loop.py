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

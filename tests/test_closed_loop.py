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


# ---------------------------------------------------------------------------
# Task 5: CLI registration + _select_gate_policy tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 2 (S7): run_scenario_commit orchestrator tests
# ---------------------------------------------------------------------------

from bakery.ontology import scenario as sc
from bakery.ontology.loop import run_scenario_commit, ScenarioCommitResult, auto_approve, human_correct
from bakery.ontology.writeback import WritebackStore, APPROVED, REJECTED
from bakery.ontology.loop import APPROVE, REJECT, GateDecision


class _ScenarioStubModel:
    """before demand=10, after=7 when is_rain flipped to 1 (deterministic)."""
    def predict(self, target):
        import pandas as pd
        rain = float(target["is_rain"].iloc[0]) if "is_rain" in target.columns else 0.0
        return pd.Series([10.0 - 3.0 * rain] * len(target))


def _sc_ctx(dataset):
    import pandas as pd
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    dates = sorted(pd.to_datetime(enriched["date"]).dt.date.unique())
    cutoff = str(dates[-3]); period = (str(dates[-2]), str(dates[-1]))
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    return store, item, period, cutoff


def test_scenario_commit_commits_adjusted_order(dataset, monkeypatch):
    from bakery.decision import apply_policy
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    res = run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, auto_approve,
                              now="2024-01-01T09:00:00", train_cutoff=cutoff)
    assert isinstance(res, ScenarioCommitResult)
    assert res.whatif.before_demand == 10.0 and res.whatif.after_demand == 7.0
    assert res.base_order == apply_policy(item, 10.0)[0]
    scenario_order = apply_policy(item, 7.0)[0]
    assert res.committed.status == APPROVED
    assert res.committed.approved_qty == scenario_order          # auto_approve → 제안대로


def test_scenario_commit_human_correction(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    res = run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb,
                              human_correct({item: 99.0}), now="2024-01-01T09:00:00",
                              train_cutoff=cutoff)
    assert res.committed.approved_qty == 99.0


def test_scenario_commit_reject(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    reject_gate = lambda proposal: GateDecision(REJECT, None, "human")
    res = run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, reject_gate,
                              now="2024-01-01T09:00:00", train_cutoff=cutoff)
    assert res.committed.status == REJECTED


def test_scenario_commit_rejects_autonomous_store(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=False)
    with pytest.raises(ValueError):
        run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, auto_approve,
                            now="2024-01-01T09:00:00", train_cutoff=cutoff)


def test_scenario_commit_rationale_describes_scenario(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, auto_approve,
                        now="2024-01-01T09:00:00", train_cutoff=cutoff)
    # rationale lives on the proposal, not persisted; assert via a capturing gate
    captured = {}
    def cap_gate(p):
        captured["rationale"] = p.rationale
        return GateDecision(APPROVE, None, "human")
    wb2 = WritebackStore(require_approval=True)
    run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb2, cap_gate,
                        now="2024-01-01T09:00:00", train_cutoff=cutoff)
    assert "scenario" in captured["rationale"] and "is_rain" in captured["rationale"]


# ---------------------------------------------------------------------------
# Task 3 (S7): CLI scenario-commit registration + _parse_drivers
# ---------------------------------------------------------------------------


def test_cli_registers_scenario_commit_command():
    from bakery.cli import app
    names = {c.name for c in app.registered_commands}
    assert "scenario-commit" in names


def test_parse_drivers_maps_pairs():
    from bakery.cli import _parse_drivers
    assert _parse_drivers("is_rain=1,is_snow=0") == {"is_rain": 1.0, "is_snow": 0.0}
    assert _parse_drivers(" is_public_holiday = 1 ") == {"is_public_holiday": 1.0}
    import pytest
    with pytest.raises(ValueError):
        _parse_drivers("is_rain")        # no '='
    with pytest.raises(ValueError):
        _parse_drivers("")               # empty

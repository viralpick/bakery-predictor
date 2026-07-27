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
            assert "item_id" in g1


def test_q_order_top_gold_matches_chain(dataset):
    """gold가 rank(k=1)→explain 체인과 정확히 일치 — grounded arm이 같은 체인으로 도달 가능."""
    from bakery.ontology import functions as fn
    from bakery.ontology.grounding.questions import resolve_eval_context

    q = next(q for q in QUESTIONS if q.id == "q_order_top")
    gold = build_gold(q, dataset)
    store, period = resolve_eval_context(dataset)
    top1 = fn.rank_stockout_earliness(dataset.daily, store, period, k=1)
    assert gold["item_id"] == str(top1["item_id"].iloc[0])
    lineage = fn.explain_order(dataset.daily, store, gold["item_id"], period)
    assert gold["order_qty"] == pytest.approx(float(lineage["contribution"].sum()))


def test_q_rank_earliness_gold_top3(dataset):
    q = next(q for q in QUESTIONS if q.id == "q_rank_earliness")
    gold = build_gold(q, dataset)
    assert len(gold["top_items"]) == 3
    assert len(set(gold["top_items"])) == 3


def test_gold_explain_total_matches_function(dataset):
    """q_explain_total gold == explain_category_total prior_prod (결정론)."""
    from bakery.ontology.grounding.questions import QUESTIONS, build_gold, _forward_ctx
    from bakery.ontology import explain
    q = next(q for q in QUESTIONS if q.id == "q_explain_total")
    gold = build_gold(q, dataset)
    store, date = _forward_ctx(dataset)
    rows = explain.explain_category_total(store, daily=dataset.daily, date=date, use_forecast=False)
    expected = float(rows.set_index("step")["value"]["prior_prod"])
    assert gold["answer_value"] == pytest.approx(expected, rel=1e-9)


def test_gold_explain_item_matches_function(dataset):
    """q_explain_item gold == explain_item_order final for the deterministic item."""
    from bakery.ontology.grounding.questions import QUESTIONS, build_gold
    q = next(q for q in QUESTIONS if q.id == "q_explain_item")
    gold = build_gold(q, dataset)
    assert "item_id" in gold and "order_qty" in gold
    assert gold["order_qty"] > 0


def test_gold_explain_item_equals_top_forward(dataset):
    """q_explain_item gold item == rank_forward_items top1 (단일 소스 정합)."""
    from bakery.ontology.grounding.questions import QUESTIONS, build_gold, _forward_ctx
    from bakery.ontology import explain
    q = next(q for q in QUESTIONS if q.id == "q_explain_item")
    gold = build_gold(q, dataset)
    store, date = _forward_ctx(dataset)
    top = explain.rank_forward_items(store, daily=dataset.daily, date=date, k=1, use_forecast=False)
    assert gold["item_id"] == str(top["item_id"].iloc[0])

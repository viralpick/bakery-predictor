# tests/test_grounding_scorer.py
from bakery.ontology.grounding.questions import Question
from bakery.ontology.grounding.scorer import grade, summarize, QResult


def _q(gt):
    return Question(id="x", text="", grader_type=gt, source_fn="f")


def test_numeric_within_tolerance():
    q = Question(id="x", text="", grader_type="numeric", source_fn="f", tolerance=0.05)
    assert grade(q, {"answer_value": 102.0}, {"answer_value": 100.0}) is True
    assert grade(q, {"answer_value": 120.0}, {"answer_value": 100.0}) is False


def test_numeric_zero_gold_exact():
    q = Question(id="x", text="", grader_type="numeric", source_fn="f", tolerance=0.05)
    assert grade(q, {"answer_value": 0.0}, {"answer_value": 0.0}) is True
    assert grade(q, {"answer_value": 1.0}, {"answer_value": 0.0}) is False


def test_ranking_top1_match():
    q = _q("ranking")
    assert grade(q, {"top_items": ["A", "B"]}, {"top_items": ["A", "C"]}) is True   # top-1 match
    assert grade(q, {"top_items": ["B", "A"]}, {"top_items": ["A", "C"]}) is False


def test_decomposition_qty_match():
    q = _q("decomposition")
    assert grade(q, {"order_qty": 23.0}, {"order_qty": 23.0}) is True
    assert grade(q, {"order_qty": 25.0}, {"order_qty": 23.0}) is False


def test_malformed_answer_is_wrong():
    q = _q("numeric")
    assert grade(q, {}, {"answer_value": 5.0}) is False


def test_summarize_computes_delta():
    results = [QResult("a", "numeric", True, False), QResult("b", "ranking", True, True)]
    rep = summarize(results)
    assert rep.grounded_accuracy == 1.0
    assert rep.rag_accuracy == 0.5
    assert rep.delta == 0.5

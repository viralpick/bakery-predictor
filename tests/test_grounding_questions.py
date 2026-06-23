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

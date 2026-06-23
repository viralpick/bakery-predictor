"""Smoke test for grounding eval entry point — no API key required."""

import bakery.ontology.grounding.run as _run_module

from bakery.data.loader import load_dataset
from bakery.ontology.grounding.questions import QUESTIONS
from bakery.ontology.grounding.run import run_eval_with_client
from bakery.ontology.grounding.llm import LLMResponse


class Empty:
    def generate(self, messages, *, tools=None, output_schema=None):
        return LLMResponse(text=None, tool_calls=[], parsed={})


def test_run_eval_produces_report():
    dataset = load_dataset("synthetic")
    # minimal smoke: a fake that always returns empty answers → report still computes
    report = run_eval_with_client(Empty(), dataset)
    assert 0.0 <= report.grounded_accuracy <= 1.0
    assert report.delta == report.grounded_accuracy - report.rag_accuracy


def test_run_eval_skips_failing_question(monkeypatch):
    """Fix B: a question whose build_gold raises must be skipped, not crash the run."""
    dataset = load_dataset("synthetic")

    # Pick the id of the first question to poison
    poison_id = QUESTIONS[0].id

    original_build_gold = _run_module.build_gold

    def patched_build_gold(question, ds):
        if question.id == poison_id:
            raise ValueError("synthetic non-finite gold for test")
        return original_build_gold(question, ds)

    monkeypatch.setattr(_run_module, "build_gold", patched_build_gold)

    report = run_eval_with_client(Empty(), dataset)

    # Run must not crash; skipped question absent from denominator
    assert 0.0 <= report.grounded_accuracy <= 1.0
    assert report.delta == report.grounded_accuracy - report.rag_accuracy
    # The report covers fewer than all questions (poisoned one excluded)
    assert len(report.results) < len(QUESTIONS)

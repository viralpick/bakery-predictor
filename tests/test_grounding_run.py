"""Smoke test for grounding eval entry point — no API key required."""

from bakery.data.loader import load_dataset
from bakery.ontology.grounding.run import run_eval_with_client
from bakery.ontology.grounding.llm import LLMResponse


def test_run_eval_produces_report():
    dataset = load_dataset("synthetic")
    # minimal smoke: a fake that always returns empty answers → report still computes
    class Empty:
        def generate(self, messages, *, tools=None, output_schema=None):
            return LLMResponse(text=None, tool_calls=[], parsed={})
    report = run_eval_with_client(Empty(), dataset)
    assert 0.0 <= report.grounded_accuracy <= 1.0
    assert report.delta == report.grounded_accuracy - report.rag_accuracy

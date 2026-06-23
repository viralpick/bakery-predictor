"""Grounding eval entry point. run_eval_with_client is key-free testable;
run_eval wires the real provider client (needs OPENAI_API_KEY)."""

from __future__ import annotations

from ...data.loader import DailyDataset, load_dataset
from .arms import run_grounded, run_rag_only
from .llm import LLMClient, make_llm_client
from .questions import QUESTIONS, build_gold
from .scorer import EvalReport, QResult, grade, summarize


def run_eval_with_client(client: LLMClient, dataset: DailyDataset) -> EvalReport:
    results = []
    for q in QUESTIONS:
        gold = build_gold(q, dataset)
        g_ans = run_grounded(client, q, dataset)
        r_ans = run_rag_only(client, q, dataset)
        results.append(QResult(q.id, q.grader_type,
                               grade(q, g_ans, gold), grade(q, r_ans, gold)))
    return summarize(results)


def run_eval(provider: str = "openai", model: str = "gpt-5-mini",
             source: str = "synthetic") -> EvalReport:
    client = make_llm_client(provider, model)
    return run_eval_with_client(client, load_dataset(source))

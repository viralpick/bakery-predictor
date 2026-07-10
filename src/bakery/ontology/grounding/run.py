"""Grounding eval entry point. run_eval_with_client is key-free testable;
run_eval wires the real provider client (needs OPENAI_API_KEY or AZURE_OPENAI_*;
provider="auto" picks Azure when AZURE_OPENAI_API_KEY is set)."""

from __future__ import annotations

import dataclasses
import logging

from ...data.loader import DailyDataset, load_dataset
from ...features.category_aggregate import DEFAULT_ALPHA, build_item_adjusted_demand
from .arms import run_grounded, run_rag_only
from .llm import LLMClient, make_llm_client
from .questions import QUESTIONS, build_gold
from .scorer import EvalReport, QResult, grade, summarize

log = logging.getLogger(__name__)


def run_eval_with_client(client: LLMClient, dataset: DailyDataset) -> EvalReport:
    results = []
    for q in QUESTIONS:
        try:
            gold = build_gold(q, dataset)
            g_ans = run_grounded(client, q, dataset)
            r_ans = run_rag_only(client, q, dataset)
            results.append(QResult(q.id, q.grader_type,
                                   grade(q, g_ans, gold), grade(q, r_ans, gold)))
        except Exception as exc:
            log.warning("skipping question %s: %s", q.id, exc)
            continue
    return summarize(results)


def run_eval(provider: str = "auto", model: str = "gpt-5-mini",
             source: str = "synthetic") -> EvalReport:
    client = make_llm_client(provider, model)
    dataset = load_dataset(source)
    if source == "real":
        enriched = build_item_adjusted_demand(dataset.daily, alpha=DEFAULT_ALPHA)
        dataset = dataclasses.replace(dataset, daily=enriched)
    return run_eval_with_client(client, dataset)

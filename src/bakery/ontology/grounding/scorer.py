"""Deterministic graders + delta report (design §7).

Grading is type-specific and never calls an LLM: numeric uses relative
tolerance (exact when gold is 0), ranking uses top-1 match, decomposition uses
exact order-qty match. A malformed/missing answer counts as wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from .questions import Question

_QTY_TOL = 1e-6   # decomposition order qty is rounded; tiny float slack


@dataclass(frozen=True)
class QResult:
    id: str
    grader_type: str
    grounded_ok: bool
    rag_ok: bool


@dataclass(frozen=True)
class EvalReport:
    results: list[QResult]
    grounded_accuracy: float
    rag_accuracy: float
    delta: float


def grade(question: Question, answer: dict, gold: dict) -> bool:
    if not isinstance(answer, dict):
        return False
    if question.grader_type == "numeric":
        return _grade_numeric(answer.get("answer_value"), gold["answer_value"], question.tolerance)
    if question.grader_type == "ranking":
        return _grade_ranking(answer.get("top_items"), gold["top_items"])
    if question.grader_type == "decomposition":
        return _grade_qty(answer.get("order_qty"), gold["order_qty"])
    raise KeyError(question.grader_type)


def _grade_numeric(pred, gold: float, tol: float) -> bool:
    if not isinstance(pred, (int, float)):
        return False
    if gold == 0:
        return abs(pred) <= _QTY_TOL
    return abs(pred - gold) / abs(gold) <= tol


def _grade_ranking(pred, gold: list) -> bool:
    if not isinstance(pred, list) or not pred or not gold:
        return False
    return pred[0] == gold[0]            # top-1 match


def _grade_qty(pred, gold: float) -> bool:
    if not isinstance(pred, (int, float)):
        return False
    return abs(pred - gold) <= _QTY_TOL   # spec: exact match; 1e-6 covers float repr


def summarize(results: list[QResult]) -> EvalReport:
    n = len(results) or 1
    g = sum(r.grounded_ok for r in results) / n
    r = sum(r.rag_ok for r in results) / n
    return EvalReport(results=results, grounded_accuracy=g, rag_accuracy=r, delta=g - r)

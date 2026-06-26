"""Closed-loop order recommendation (v7 S5).

A grounded agent proposes orders (read-only); this deterministic orchestrator
validates the proposals, writes them as PENDING records, runs them through a
human-approval GatePolicy, and commits the approved ones. The LLM never mutates
state — writes go through WritebackStore behind the gate.

See docs/superpowers/specs/2026-06-25-closed-loop-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

APPROVE = "APPROVE"
REJECT = "REJECT"


@dataclass(frozen=True)
class OrderProposal:
    item_id: str
    qty: float
    rationale: str


@dataclass(frozen=True)
class GateDecision:
    action: str                      # APPROVE | REJECT
    approved_qty: float | None       # None = approve the proposed qty unchanged
    approver: str


GatePolicy = Callable[[OrderProposal], GateDecision]


def auto_approve(proposal: OrderProposal) -> GateDecision:
    """Frontier mode: commit the proposal unchanged, no human."""
    return GateDecision(APPROVE, None, "autonomous")


def approve_as_proposed(proposal: OrderProposal) -> GateDecision:
    """Human rubber-stamp: approve exactly what the agent proposed."""
    return GateDecision(APPROVE, None, "human")


def human_correct(corrections: dict[str, float], approver: str = "human") -> GatePolicy:
    """Human edits specific items' qty; others approved as proposed."""
    def policy(proposal: OrderProposal) -> GateDecision:
        if proposal.item_id in corrections:
            return GateDecision(APPROVE, float(corrections[proposal.item_id]), approver)
        return GateDecision(APPROVE, None, approver)
    return policy

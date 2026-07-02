"""Closed-loop order recommendation (v7 S5).

A grounded agent proposes orders (read-only); this deterministic orchestrator
validates the proposals, writes them as PENDING records, runs them through a
human-approval GatePolicy, and commits the approved ones. The LLM never mutates
state — writes go through WritebackStore behind the gate.

See docs/superpowers/specs/2026-06-25-closed-loop-design.md.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

from ..data.loader import DailyDataset
from ..decision import PolicyParams, RiskParams, apply_policy
from . import scenario
from .grounding import arms
from .writeback import OrderRecord, WritebackStore

log = logging.getLogger(__name__)

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


def _is_valid(proposal: OrderProposal, valid_items: set[str]) -> bool:
    if proposal.item_id not in valid_items:
        return False
    if not math.isfinite(proposal.qty) or proposal.qty < 0:
        return False
    return True


def run_closed_loop(client, dataset: DailyDataset, store_id: str,
                    period: tuple[str, str], writeback: WritebackStore,
                    gate: GatePolicy, *, now: str) -> list[OrderRecord]:
    """Recommend → validate → propose(PENDING) → gate → commit. Returns the
    records created this run (invalid proposals skipped)."""
    if not writeback.require_approval:
        raise ValueError(
            "closed-loop drives approval via the GatePolicy; "
            "pass WritebackStore(require_approval=True)")
    target_date = period[0]
    valid_items = set(
        dataset.daily.loc[dataset.daily["store_id"] == store_id, "item_id"])
    raw = arms.recommend_orders(client, dataset, store_id, period)
    out: list[OrderRecord] = []
    for d in raw:
        proposal = OrderProposal(str(d["item_id"]), float(d["qty"]), str(d["rationale"]))
        if not _is_valid(proposal, valid_items):
            log.warning("skipping invalid proposal: %s", d)
            continue
        rec = writeback.propose_order(store_id, proposal.item_id, target_date,
                                      proposal.qty, proposed_at=now)
        decision = gate(proposal)
        if decision.action == APPROVE:
            rec = writeback.approve(rec.record_id, decision.approver,
                                    approved_at=now, approved_qty=decision.approved_qty)
        else:
            rec = writeback.reject(rec.record_id, decision.approver)
        out.append(rec)
    return out


@dataclass(frozen=True)
class ScenarioCommitResult:
    whatif: "scenario.WhatIfDriverResult"
    base_order: float
    committed: OrderRecord


def run_scenario_commit(dataset: DailyDataset, store_id: str, item_id: str,
                        period: tuple[str, str], driver_overrides: dict,
                        writeback: WritebackStore, gate: GatePolicy, *, now: str,
                        train_cutoff: str, policy: PolicyParams = PolicyParams(),
                        risk: RiskParams = RiskParams()) -> ScenarioCommitResult:
    """Upstream scenario → adjusted order → human gate → writeback commit.
    Deterministic (no LLM). Reuses what_if_driver(S6) + apply_policy + writeback(S5)."""
    if not writeback.require_approval:
        raise ValueError(
            "scenario-commit drives approval via the GatePolicy; "
            "pass WritebackStore(require_approval=True)")
    wif = scenario.what_if_driver(
        dataset.daily, dataset.calendar, dataset.weather, store_id, item_id, period,
        driver_overrides, base_order=None, train_cutoff=train_cutoff, risk=risk)
    base_order = apply_policy(item_id, wif.before_demand, policy)[0]
    scenario_order = apply_policy(item_id, wif.after_demand, policy)[0]
    drivers_str = ", ".join(f"{k}={v}" for k, v in driver_overrides.items())
    rationale = (f"scenario [{drivers_str}]: demand {wif.before_demand:.1f}→{wif.after_demand:.1f}, "
                 f"order {base_order:.0f}→{scenario_order:.0f}")
    proposal = OrderProposal(item_id, scenario_order, rationale)
    rec = writeback.propose_order(store_id, item_id, period[0], scenario_order, proposed_at=now)
    decision = gate(proposal)
    if decision.action == APPROVE:
        rec = writeback.approve(rec.record_id, decision.approver,
                                approved_at=now, approved_qty=decision.approved_qty)
    else:
        rec = writeback.reject(rec.record_id, decision.approver)
    return ScenarioCommitResult(whatif=wif, base_order=base_order, committed=rec)

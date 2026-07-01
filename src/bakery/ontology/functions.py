"""OntologyFunction layer — parameterized, reusable operations over the ontology.

Mirrors AOS OntologyFunction: a stable API the agent calls instead of writing
ad-hoc queries. Every function here is a thin wrapper that REUSES the v6 decision
layer (commit 8d13157) — no new modeling logic. The numbers come from the
deterministic engine; the agent's job is only to call + interpret (docs §2,
"수치 = 엔진, 해석 = LLM").

Demand point estimate: this pre-work uses historical `potential_demand` (the
censoring-corrected target) as the point-estimate proxy, since the live forecast
is not wired in yet. Swap `demand_col`/inject a forecast when LayerA lands —
the function signatures don't change. Labeled, not hidden (fairness contract §7).

All functions are *read-only* over the ontology (AOS rule). Writeback lives in
writeback.py (S4) behind a human-approval gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ..decision import (
    PolicyParams,
    RiskParams,
    apply_policy,
    build_recommendation,
    simulate_item_risk,
)
from . import scenario
from .writeback import WritebackStore

DEMAND_PROXY_COL = "potential_demand"
CONDITION_ON, CONDITION_OFF = 1, 0   # 0/1 flag values for demand_diff_by_condition


def _period_slice(daily: pd.DataFrame, store_id: str, start: str, end: str) -> pd.DataFrame:
    """Rows for one store within [start, end] inclusive. Guard clause on emptiness."""
    dates = pd.to_datetime(daily["date"])
    mask = (daily["store_id"] == store_id) & (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sliced = daily.loc[mask]
    if sliced.empty:
        raise ValueError(f"no rows for store={store_id} in [{start}, {end}]")
    return sliced


def _item_demand_points(period: pd.DataFrame, demand_col: str) -> pd.DataFrame:
    """Per-item demand point estimate = mean of the (proxy) demand col over the period."""
    grouped = period.groupby("item_id", observed=True)[demand_col].mean()
    return grouped.reset_index(name="demand_point")


def rank_stockout_risk(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    k: int = 5,
    *,
    demand_col: str = DEMAND_PROXY_COL,
    policy: PolicyParams = PolicyParams(),
    risk: RiskParams = RiskParams(),
) -> pd.DataFrame:
    """Top-k items by P(stockout) for a store over a period (uses risk.py MC)."""
    items = _item_demand_points(_period_slice(daily, store_id, *period), demand_col)
    rec = build_recommendation(items, policy=policy, risk=risk)
    ranked = rec.table.sort_values("p_stockout", ascending=False).head(k)
    return ranked.reset_index(drop=True)


def explain_order(
    daily: pd.DataFrame,
    store_id: str,
    item_id: str,
    period: tuple[str, str],
    *,
    demand_col: str = DEMAND_PROXY_COL,
    policy: PolicyParams = PolicyParams(),
) -> pd.DataFrame:
    """Decision lineage for one item's order: base → safety → floor → rounding."""
    items = _item_demand_points(_period_slice(daily, store_id, *period), demand_col)
    match = items.loc[items["item_id"] == item_id, "demand_point"]
    if match.empty:
        raise ValueError(f"item {item_id} not sold at {store_id} in {period}")
    _, lineage = apply_policy(item_id, float(match.iloc[0]), policy)
    return pd.DataFrame(lineage.to_records(), columns=["item_id", "step", "contribution", "detail"])


@dataclass(frozen=True)
class WhatIfResult:
    """Downstream lever: how risk/cost moves when the order qty is changed."""

    demand_point: float
    base_order: float
    new_order: float
    base_p_stockout: float
    new_p_stockout: float
    base_expected_cost: float
    new_expected_cost: float


def what_if(
    demand_point: float,
    base_order: float,
    delta_order: float,
    *,
    risk: RiskParams = RiskParams(),
) -> WhatIfResult:
    """Downstream what-if: re-score risk/cost for order = base_order + delta_order."""
    new_order = base_order + delta_order
    before = simulate_item_risk(demand_point, base_order, risk)
    after = simulate_item_risk(demand_point, new_order, risk)
    return WhatIfResult(
        demand_point=demand_point, base_order=base_order, new_order=new_order,
        base_p_stockout=before.p_stockout, new_p_stockout=after.p_stockout,
        base_expected_cost=before.expected_cost, new_expected_cost=after.expected_cost,
    )


def waste_cost(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    *,
    unit_cost: float = 1.0,
) -> dict:
    """Aggregate leftover (capacity − sold) cost for a store/period.

    Proxy waste = max(capacity − sold_units, 0); excludes stockout days where
    leftover is structurally zero. Normalized cost unless unit_cost is real KRW.

    Deliberately a *simplified* form of analysis/waste.py CapacityMinusSoldEstimator:
    it omits closing-discount quantities (that frame isn't in the dataset bundle)
    and adds the stockout-day=0 rule. Swap in the full estimator once discount
    data is wired into the ontology's backing frames.
    """
    period_df = _period_slice(daily, store_id, *period)
    leftover = (period_df["capacity"] - period_df["sold_units"]).clip(lower=0)
    # stockout days have structurally zero leftover (item ran out → nothing wasted)
    leftover = leftover.where(~period_df["is_stockout"], 0)
    units = float(leftover.sum())
    return {"store_id": store_id, "leftover_units": units, "waste_cost": units * unit_cost}


def demand_diff_by_condition(
    daily: pd.DataFrame,
    join_frame: pd.DataFrame,
    store_id: str,
    condition_col: str,
) -> dict:
    """Mean daily units when condition_col is on vs off (e.g. is_weekend, is_rain).

    Traverses DailySales → CalendarEvent/Weather via the link's join keys. The
    condition_col must be a 0/1 flag column on join_frame (calendar or weather).
    """
    if condition_col not in join_frame.columns:
        raise ValueError(f"condition_col {condition_col!r} not in join_frame")
    join_keys = ["store_id", "date"] if "store_id" in join_frame.columns else ["date"]
    cols = [*join_keys, condition_col]
    merged = daily.merge(join_frame[cols], on=join_keys, how="inner")
    store_rows = merged.loc[merged["store_id"] == store_id]
    if store_rows.empty:
        raise ValueError(f"no joined rows for store={store_id}")
    daily_units = store_rows.groupby(["date", condition_col], observed=True)["sold_units"].sum().reset_index()
    means = daily_units.groupby(condition_col, observed=True)["sold_units"].mean()
    on, off = float(means.get(CONDITION_ON, float("nan"))), float(means.get(CONDITION_OFF, float("nan")))
    return {"condition": condition_col, "mean_on": on, "mean_off": off, "diff": on - off}


@dataclass(frozen=True)
class OntologyFunctionSpec:
    """Agent-facing metadata for one function (name, params, return, impl)."""

    name: str
    description: str
    params: tuple[str, ...]
    returns: str
    impl: Callable
    side: str = "read"          # "read" | "write" — write는 LLM 도구 surface 제외


# The stable API surface the grounded agent enumerates and calls (S3 consumes this).
FUNCTION_REGISTRY: dict[str, OntologyFunctionSpec] = {
    "rank_stockout_risk": OntologyFunctionSpec(
        "rank_stockout_risk", "Top-k items by stockout probability for a store/period.",
        ("store_id", "period", "k"), "table[item_id, p_stockout, order_qty, ...]", rank_stockout_risk),
    "explain_order": OntologyFunctionSpec(
        "explain_order", "Decision lineage breaking down one item's recommended order.",
        ("store_id", "item_id", "period"), "table[step, contribution, detail]", explain_order),
    "what_if": OntologyFunctionSpec(
        "what_if", "Downstream lever: risk/cost delta when an order qty changes.",
        ("demand_point", "base_order", "delta_order"), "WhatIfResult", what_if),
    "waste_cost": OntologyFunctionSpec(
        "waste_cost", "Aggregate leftover (capacity−sold) cost for a store/period.",
        ("store_id", "period"), "{leftover_units, waste_cost}", waste_cost),
    "demand_diff_by_condition": OntologyFunctionSpec(
        "demand_diff_by_condition", "Mean daily sales when a condition is on vs off.",
        ("store_id", "condition_col"), "{mean_on, mean_off, diff}", demand_diff_by_condition),
    "propose_order": OntologyFunctionSpec(
        "propose_order", "Write a PENDING order recommendation (human-approval-gated).",
        ("store_id", "item_id", "date", "proposed_qty"), "OrderRecord",
        WritebackStore.propose_order, side="write"),
    "commit_order": OntologyFunctionSpec(
        "commit_order", "Commit a PENDING order (approve, optionally correcting qty).",
        ("record_id", "approver", "approved_qty"), "OrderRecord",
        WritebackStore.approve, side="write"),
    "what_if_driver": OntologyFunctionSpec(
        "what_if_driver",
        "Upstream lever: perturb weather/calendar driver(s), re-forecast demand, propagate to stockout risk/cost.",
        ("store_id", "item_id", "period", "driver_overrides", "base_order"),
        "WhatIfDriverResult", scenario.what_if_driver, side="read"),
}

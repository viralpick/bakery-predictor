"""v6 recommendation pipeline — point estimate → order + risk + lineage.

Ties the deterministic policy (decision lineage) and the Monte-Carlo risk shell
into the v6 deliverable: per-item point estimate + recommended order +
P(stockout)/P(waste) + expected cost. The demand point estimate comes from
upstream (category total × item proportion, or any per-item forecast); this
layer is strictly *post-prediction*, so it introduces no feature leakage
(CLAUDE.md absolute rule #1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lineage import DecisionLineage
from .policy import PolicyParams, apply_policy
from .risk import RiskParams, simulate_item_risk

REQUIRED_COLS = ("item_id", "demand_point")
CARRY_COLS = ("store_id", "category_id", "date")   # id columns passed through to output
METRIC_COLS = (
    "item_id", "demand_point", "order_qty", "p_stockout", "p_waste",
    "expected_short", "expected_waste", "expected_cost",
)


@dataclass
class Recommendation:
    table: pd.DataFrame                  # one row per item (the v6 deliverable)
    lineages: list[DecisionLineage]      # decision drilldown, aligned to table rows


def _validate(items: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLS) - set(items.columns)
    if missing:
        raise ValueError(f"items frame missing columns: {sorted(missing)}")
    demand = items["demand_point"]
    if demand.isna().any():
        raise ValueError("demand_point contains NaN")
    if (demand < 0).any():
        raise ValueError("demand_point must be >= 0 (clip upstream)")


def _recommend_one(item_id: str, demand_point: float, policy: PolicyParams,
                   risk: RiskParams, rng: np.random.Generator) -> tuple[dict, DecisionLineage]:
    order, lineage = apply_policy(item_id, demand_point, policy)
    res = simulate_item_risk(demand_point, order, risk, rng)
    row = dict(
        item_id=item_id, demand_point=float(demand_point), order_qty=order,
        p_stockout=res.p_stockout, p_waste=res.p_waste,
        expected_short=res.expected_short, expected_waste=res.expected_waste,
        expected_cost=res.expected_cost,
    )
    return row, lineage


def build_recommendation(
    items: pd.DataFrame,
    policy: PolicyParams = PolicyParams(),
    risk: RiskParams = RiskParams(),
) -> Recommendation:
    """Build the v6 recommendation table + decision lineages from item demand points.

    items: DataFrame with at least [item_id, demand_point]. Extra id columns
    (store_id, category_id, date) are carried through to the output if present.

    Reproducibility: each row gets an INDEPENDENT RNG stream spawned from
    risk.seed (positional), so an item's risk never depends on other items'
    demand values. Same frame (rows + order) → identical output.
    """
    _validate(items)
    seeds = np.random.SeedSequence(risk.seed).spawn(len(items))
    carry = [c for c in CARRY_COLS if c in items.columns]
    rows, lineages = [], []
    for seed, record in zip(seeds, items.itertuples(index=False)):
        item_rng = np.random.default_rng(seed)
        row, lineage = _recommend_one(record.item_id, record.demand_point, policy, risk, item_rng)
        for col in carry:
            row[col] = getattr(record, col)
        rows.append(row)
        lineages.append(lineage)
    table = pd.DataFrame(rows, columns=[*carry, *METRIC_COLS])
    return Recommendation(table=table, lineages=lineages)


def lineage_to_frame(lineages: list[DecisionLineage]) -> pd.DataFrame:
    """Flatten all decision lineages into a long drilldown table."""
    records = [row for lin in lineages for row in lin.to_records()]
    return pd.DataFrame(records, columns=["item_id", "step", "contribution", "detail"])

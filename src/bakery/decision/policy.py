"""Decision policy — demand point estimate → recommended order qty (deterministic).

These are *decision* rules (safety stock, display floor, batch rounding): chosen,
not learned. Demand-pattern knowledge ("비 오면 소시지빵↑") belongs in features
fed to the forecast, NOT here — see docs/kinetic_layer_fit_analysis.md §9.2
(demand=학습 / decision=선언). Keeping only genuine policy here is what makes the
decision lineage meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .lineage import DecisionLineage

DEFAULT_SAFETY_MARGIN = 0.15   # +15% buffer over point demand (newsvendor lean-safe)
DEFAULT_DISPLAY_FLOOR = 0      # minimum units to keep on shelf (0 = no floor)
DEFAULT_ROUND_UNIT = 1         # batch size for rounding (1 = whole units)


@dataclass(frozen=True)
class PolicyParams:
    safety_margin: float = DEFAULT_SAFETY_MARGIN
    display_floor: int = DEFAULT_DISPLAY_FLOOR
    round_unit: int = DEFAULT_ROUND_UNIT


def _round_up_to_unit(qty: float, unit: int) -> float:
    """Round up to the batch unit (conservative); nearest int when unit<=1."""
    if unit <= 1:
        return float(round(qty))
    return float(math.ceil(qty / unit) * unit)


def apply_policy(
    item_id: str,
    demand_point: float,
    params: PolicyParams = PolicyParams(),
) -> tuple[float, DecisionLineage]:
    """Convert a demand point estimate into an order qty, recording each step."""
    lineage = DecisionLineage(item_id=item_id, base=float(demand_point))
    qty = float(demand_point)

    margin_units = demand_point * params.safety_margin
    qty += margin_units
    lineage.add("safety_margin", margin_units, f"+{params.safety_margin:.0%} buffer")

    if qty < params.display_floor:
        floor_units = params.display_floor - qty
        qty = float(params.display_floor)
        lineage.add("display_floor", floor_units, f"floor={params.display_floor}")

    rounded = _round_up_to_unit(qty, params.round_unit)
    lineage.add("rounding", rounded - qty, f"round↑ to {params.round_unit}")
    return rounded, lineage

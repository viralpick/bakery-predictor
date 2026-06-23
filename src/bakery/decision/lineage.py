"""Decision lineage — records how a demand point estimate becomes an order qty.

Each DecisionStep is one deterministic transformation (safety margin, display
floor, batch rounding) with the *signed* contribution it added. The base plus
all step contributions equal the final order qty (conservation) — the same
invariant as the kinetic provenance "보존 법칙", but applied to the
deterministic *decision* pipeline, not the learned forecast.

See docs/kinetic_layer_fit_analysis.md §10.2-B (Decision lineage). The forecast
itself is explained separately by SHAP attribution, not by this lineage (§10.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Float comparisons (rounding leaves tiny residuals); conservation tolerance.
CONSERVATION_TOL = 1e-6


@dataclass(frozen=True)
class DecisionStep:
    name: str             # "safety_margin" | "display_floor" | "rounding"
    contribution: float   # signed units this step added to the order
    detail: str = ""      # human-readable reason


@dataclass
class DecisionLineage:
    """Ordered record of the demand→order transformation for one item."""

    item_id: str
    base: float                                   # starting demand point estimate
    steps: list[DecisionStep] = field(default_factory=list)

    def add(self, name: str, contribution: float, detail: str = "") -> None:
        self.steps.append(DecisionStep(name, float(contribution), detail))

    @property
    def order_qty(self) -> float:
        return self.base + sum(step.contribution for step in self.steps)

    def is_conserved(self, order_qty: float) -> bool:
        """base + Σ contributions must reconstruct the final order qty."""
        return abs(self.order_qty - order_qty) <= CONSERVATION_TOL

    def to_records(self) -> list[dict]:
        """Flatten to rows for CSV/drilldown — base first, then each step."""
        rows = [dict(item_id=self.item_id, step="base", contribution=self.base, detail="demand point estimate")]
        for step in self.steps:
            rows.append(dict(item_id=self.item_id, step=step.name, contribution=step.contribution, detail=step.detail))
        return rows

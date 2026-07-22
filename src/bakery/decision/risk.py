"""Monte-Carlo risk shell — order qty + demand uncertainty → P(stockout)/P(waste).

Keeps the decision pipeline deterministic: uncertainty lives in the *input
distribution* and the sampling loop, not in the rules (docs §8.1). For a given
order qty we sample N demand draws and count how often demand exceeds the order
(stockout) vs falls short of it (waste), plus the expected operational cost.

Simplifications (PoC scope, docs §8.4):
  - demand ~ truncated Normal(point, cv·point); a parametric placeholder until a
    proper predictive distribution (quantile/residual) feeds in.
  - items sampled independently — cannibalization correlation is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_N_SAMPLES = 5000
DEFAULT_DEMAND_CV = 0.30   # demand sigma = cv × point estimate
DEFAULT_UNIT_MARGIN = 1.0  # opportunity loss per unit short (stockout)
DEFAULT_UNIT_COST = 1.0    # loss per leftover unit (waste)
DEFAULT_SEED = 42


@dataclass(frozen=True)
class RiskParams:
    n_samples: int = DEFAULT_N_SAMPLES
    demand_cv: float = DEFAULT_DEMAND_CV
    unit_margin: float = DEFAULT_UNIT_MARGIN
    unit_cost: float = DEFAULT_UNIT_COST
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class RiskResult:
    p_stockout: float       # P(demand > order)
    p_waste: float          # P(order > demand)
    expected_short: float   # mean units short
    expected_waste: float   # mean leftover units
    expected_cost: float    # unit_margin·short + unit_cost·waste; normalized units
                            # unless unit_margin/unit_cost are set to real KRW


def _sample_demand(point: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    sigma = max(point * cv, 1e-9)
    return np.clip(rng.normal(point, sigma, n), 0.0, None)


def _sample_demand_lognormal(
    point: float, sigma_log: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    # median=point → mu=log(point). 비율 스케일 하에서 σ 불변(store-total LogNormal 상속).
    return np.exp(rng.normal(np.log(point), sigma_log, n))


def simulate_item_risk(
    demand_point: float,
    order_qty: float,
    params: RiskParams = RiskParams(),
    rng: np.random.Generator | None = None,
    *,
    demand_sigma_log: float | None = None,
) -> RiskResult:
    """Monte-Carlo P(stockout)/P(waste)/expected cost for one item's order.

    demand_sigma_log가 주어지면(>0, demand_point>0) demand~LogNormal(median=demand_point,
    shape=demand_sigma_log)로 샘플(분포모델 category σ 상속). 그 외에는 기존
    Normal(point, cv·point) placeholder 경로(backward-compat).
    """
    rng = rng if rng is not None else np.random.default_rng(params.seed)
    if demand_sigma_log is not None and demand_sigma_log > 0 and demand_point > 0:
        demand = _sample_demand_lognormal(demand_point, demand_sigma_log, params.n_samples, rng)
    else:
        demand = _sample_demand(demand_point, params.demand_cv, params.n_samples, rng)
    short = np.clip(demand - order_qty, 0.0, None)
    leftover = np.clip(order_qty - demand, 0.0, None)
    cost = params.unit_margin * short + params.unit_cost * leftover
    return RiskResult(
        p_stockout=float((short > 0).mean()),
        p_waste=float((leftover > 0).mean()),
        expected_short=float(short.mean()),
        expected_waste=float(leftover.mean()),
        expected_cost=float(cost.mean()),
    )

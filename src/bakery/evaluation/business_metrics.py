"""Business-impact metrics — turns predictions into KRW figures.

The WAPE family answers "how close is yhat to sold_units?", but for bakery
operations the relevant question is "how much margin did we leave on the
table?". This module exposes:

  - `asymmetric_loss`        single scalar weighted under/over (spec §5 비대칭 비용)
  - `simulate_profit`        per-row revenue / waste / lost-sale split
  - `aggregate_profit`       fold or model level KRW summary

Cost assumptions (default — caller can override per call):
  - margin_rate  0.50  ← 정상 판매 한 단위당 마진 비율 (매출가 대비)
  - cost_rate    0.30  ← 폐기 한 단위당 원재료 손실 비율 (매출가 대비)
  - lost_sale_multiplier 1.7  ← 품절 1개의 비용 = 마진 × 1.7
                                 (cross-sell 손실 + 평판 손상 반영)
  - unit_price   품목별 판매단가 (KRW). 없으면 평균 단가 fallback
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostParams:
    """Per-row cost assumptions used by simulate_profit / asymmetric_loss."""
    margin_rate: float = 0.50           # 매출가 대비 마진 비율
    cost_rate: float = 0.30             # 매출가 대비 원재료 비율 (= 폐기 단위 비용)
    lost_sale_multiplier: float = 1.7   # 품절 단위 비용 = 마진 × multiplier


def asymmetric_loss(
    yhat: np.ndarray | pd.Series,
    sold_units: np.ndarray | pd.Series,
    *,
    params: CostParams | None = None,
) -> float:
    """Asymmetric WAPE variant — weights under-prediction by lost_sale_multiplier.

    asymmetric_loss = Σ(α × under + β × over) / Σ sold_units
      α = lost_sale_multiplier × margin_rate
      β = cost_rate
    """
    params = params or CostParams()
    yhat = np.asarray(yhat, dtype=float)
    sold = np.asarray(sold_units, dtype=float)
    under = np.maximum(sold - yhat, 0.0)
    over = np.maximum(yhat - sold, 0.0)
    alpha = params.lost_sale_multiplier * params.margin_rate
    beta = params.cost_rate
    denom = sold.sum()
    if denom <= 0:
        return float("nan")
    return float((alpha * under.sum() + beta * over.sum()) / denom)


def simulate_profit(
    pred_df: pd.DataFrame,
    *,
    unit_prices: pd.Series | dict | None = None,
    params: CostParams | None = None,
    yhat_col: str = "yhat",
    sold_col: str = "sold_units",
    potential_col: str | None = "potential_demand",
) -> pd.DataFrame:
    """Per-row profit decomposition.

    Per row:
      true_demand     = potential_demand (if available) else sold_units
      sold_realized   = min(yhat, true_demand)   ← 실제로 팔 수 있는 양
      waste_units     = max(yhat - true_demand, 0)
      lost_sale_units = max(true_demand - yhat, 0)
      unit_price      = 매장 단가 (item_id 기반)

      revenue         = sold_realized × unit_price × (1 - cost_rate)   ← 마진 매출
      waste_cost      = waste_units    × unit_price × cost_rate
      lost_margin     = lost_sale_units × unit_price × margin_rate × lost_sale_multiplier
      net_profit      = revenue - waste_cost - lost_margin
    """
    params = params or CostParams()
    df = pred_df.copy()

    # Unit price lookup
    price_map = _unit_price_lookup(unit_prices)
    avg_price = float(np.mean(list(price_map.values()))) if price_map else 3000.0
    df["unit_price"] = df["item_id"].map(price_map).fillna(avg_price).astype(float)

    # True demand: use potential_demand when present (v2/v3 target), else sold_units
    if potential_col and potential_col in df.columns:
        true_demand = df[potential_col].fillna(df[sold_col]).astype(float)
    else:
        true_demand = df[sold_col].astype(float)

    yhat = df[yhat_col].fillna(0.0).astype(float)
    sold_realized = np.minimum(yhat, true_demand)
    waste_units = np.maximum(yhat - true_demand, 0.0)
    lost_sale_units = np.maximum(true_demand - yhat, 0.0)

    df["sold_realized"] = sold_realized
    df["waste_units"] = waste_units
    df["lost_sale_units"] = lost_sale_units
    df["revenue_krw"] = sold_realized * df["unit_price"] * params.margin_rate
    df["waste_cost_krw"] = waste_units * df["unit_price"] * params.cost_rate
    df["lost_margin_krw"] = (
        lost_sale_units
        * df["unit_price"]
        * params.margin_rate
        * params.lost_sale_multiplier
    )
    df["net_profit_krw"] = df["revenue_krw"] - df["waste_cost_krw"] - df["lost_margin_krw"]
    return df


def aggregate_profit(profit_df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    """Sum the krw columns produced by simulate_profit over a grouping."""
    return (
        profit_df.groupby(group_cols, as_index=False)
        .agg(
            revenue_krw=("revenue_krw", "sum"),
            waste_cost_krw=("waste_cost_krw", "sum"),
            lost_margin_krw=("lost_margin_krw", "sum"),
            net_profit_krw=("net_profit_krw", "sum"),
            sold_realized=("sold_realized", "sum"),
            waste_units=("waste_units", "sum"),
            lost_sale_units=("lost_sale_units", "sum"),
        )
        .round(0)
    )


def _unit_price_lookup(unit_prices: pd.Series | dict | None) -> dict:
    if unit_prices is None:
        return {}
    if isinstance(unit_prices, dict):
        return unit_prices
    if isinstance(unit_prices, pd.Series):
        return unit_prices.to_dict()
    raise TypeError(f"unit_prices must be dict, Series, or None — got {type(unit_prices)}")

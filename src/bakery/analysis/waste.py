"""Waste estimation + closing-discount loss quantification.

Provides a swappable WasteEstimator: when 폐기 실측 data arrives, plug in
ActualWaste source — downstream code stays unchanged.

Two complementary measures:
  - revenue_loss_won : direct money lost to closing discounts
                       (lower bound of waste cost — these units were going to be wasted)
  - estimated_waste_qty : capacity − sold − closing_discount_qty
                          (rough upper bound until 입고량 actuals)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

# Cost assumptions for ROI math
MARGIN_RATE = 0.50   # 50% gross margin (PoC default — spec.md)
COST_RATE   = 0.30   # 30% direct cost
# When unit is discounted vs wasted:
#   discounted revenue = paid (already net of discount)
#   wasted             = lost full cost (no revenue)
# So discounting prevents loss of (cost) per unit, recovering (paid − cost).


class WasteEstimator(Protocol):
    """Drop-in replacement when 폐기 실측 data arrives."""
    def per_day_item(self) -> pd.DataFrame:
        """Return: date, item_id, waste_qty, waste_cost_won."""
        ...


@dataclass
class CapacityMinusSoldEstimator:
    """Rough estimator: capacity (production proxy) − sold − closing_discount_qty.

    capacity in bonavi_daily.parquet is itself derived (95-th percentile of
    daily sold + buffer), so this is a *proxy* of waste, not a measurement.
    Will be replaced by ActualWaste when 입고량 or 폐기 수량 arrives.
    """
    daily: pd.DataFrame
    closing_discount: pd.DataFrame  # rows from DiscountSales.closing_discount()

    def per_day_item(self) -> pd.DataFrame:
        cd = self.closing_discount.copy()
        cd_agg = cd.groupby(["date", "item_id"])["qty"].sum().rename("closing_qty").reset_index()
        df = self.daily.merge(cd_agg, on=["date", "item_id"], how="left")
        df["closing_qty"] = df["closing_qty"].fillna(0).astype(int)
        df["estimated_waste_qty"] = (df["capacity"] - df["sold_units"] - df["closing_qty"]).clip(lower=0)
        return df[["date", "item_id", "category_id", "capacity", "sold_units",
                   "closing_qty", "estimated_waste_qty"]]


@dataclass
class ClosingDiscountLoss:
    """Direct measure: revenue we lost by marking down at closing time.

    For each closing-discount line item:
        unit_paid     = paid / qty
        unit_list     = unit_price (정가)
        loss_per_unit = unit_list − unit_paid = discount_amt / qty
        revenue_loss  = discount_amt (직접 손실 매출)

    But there's a deeper number: counterfactual waste cost if the unit had NOT
    been discounted. Roughly = unit_list × COST_RATE per unit not sold. We can
    approximate "recovered cost" as paid − (cost) per unit, but without 입고량
    we can't bound truly-wasted qty.
    """
    discounts: pd.DataFrame  # DiscountSales.closing_discount()

    def daily(self) -> pd.DataFrame:
        d = self.discounts.copy()
        d["unit_paid"]   = d["paid"]   / d["qty"].replace(0, np.nan)
        d["unit_list"]   = d["unit_price"]
        d["loss_per_unit"] = d["unit_list"] - d["unit_paid"]
        return d.groupby("date").agg(
            closing_qty       = ("qty", "sum"),
            revenue_loss_won  = ("discount_amt", "sum"),
            paid_recovered_won = ("paid", "sum"),
            full_value_won    = ("unit_price", lambda s: (s * d.loc[s.index, "qty"]).sum()),
        ).reset_index()

    def by_category(self, item_to_category: pd.Series) -> pd.DataFrame:
        d = self.discounts.copy()
        d["category_id"] = d["item_id"].map(item_to_category)
        return d.groupby("category_id").agg(
            closing_qty       = ("qty", "sum"),
            revenue_loss_won  = ("discount_amt", "sum"),
            paid_recovered_won = ("paid", "sum"),
        ).reset_index().sort_values("revenue_loss_won", ascending=False)

    def by_item(self, item_to_category: pd.Series, item_names: pd.Series) -> pd.DataFrame:
        d = self.discounts.copy()
        d["category_id"] = d["item_id"].map(item_to_category)
        d["item_name"] = d["item_id"].map(item_names)
        return d.groupby(["item_id", "item_name", "category_id"]).agg(
            closing_qty       = ("qty", "sum"),
            revenue_loss_won  = ("discount_amt", "sum"),
            days              = ("date", "nunique"),
        ).reset_index().sort_values("revenue_loss_won", ascending=False)


def business_impact_summary(
    cdl: ClosingDiscountLoss,
    waste: WasteEstimator,
    daily: pd.DataFrame,
) -> dict[str, float]:
    """One-shot business impact numbers for the report."""
    daily_loss = cdl.daily()
    waste_df   = waste.per_day_item()

    n_days = daily["date"].nunique()
    years  = n_days / 365.25

    total_closing_qty   = daily_loss["closing_qty"].sum()
    total_loss_won      = daily_loss["revenue_loss_won"].sum()
    total_paid_recov    = daily_loss["paid_recovered_won"].sum()
    annual_loss_won     = total_loss_won / years
    annual_closing_qty  = total_closing_qty / years

    # estimated waste (rough)
    total_est_waste_qty = waste_df["estimated_waste_qty"].sum()
    annual_est_waste    = total_est_waste_qty / years

    # If we eliminated closing discount entirely (perfect production planning):
    # - we'd not have to give up `total_loss_won` in markdowns
    # - we'd have to ensure we don't add to waste either (counterfactual: those
    #   units would otherwise be 100% wasted = 100% of unit_list cost lost)
    # → preventing the closing discount via better production = save the COST
    #   portion, since revenue would still be foregone (no sale at all if not
    #   produced).
    # So real cost saving = total closing_qty × avg_unit_cost.
    # Without exact cost, use unit_list × COST_RATE.
    full_value = daily_loss["full_value_won"].sum()
    annual_waste_cost_saved_if_eliminated = (full_value * COST_RATE) / years

    return {
        "n_days": n_days,
        "years": years,
        "total_closing_qty": total_closing_qty,
        "annual_closing_qty": annual_closing_qty,
        "total_revenue_loss_won": total_loss_won,
        "annual_revenue_loss_won": annual_loss_won,
        "total_paid_recovered_won": total_paid_recov,
        "annual_avg_daily_revenue_loss_won": annual_loss_won / 365.25,
        "total_estimated_waste_qty": total_est_waste_qty,
        "annual_estimated_waste_qty": annual_est_waste,
        "annual_cost_saved_if_zero_closing_discount_won": annual_waste_cost_saved_if_eliminated,
    }

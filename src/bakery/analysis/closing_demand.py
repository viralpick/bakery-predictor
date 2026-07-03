"""마감할인 실수요 검증 — α 실증 (Phase A: cost-free 3각 식별).

α = B/C: 마감할인 물량 C 중 진짜 수요 B의 비율. 유도분 I=C-B를 인과적으로
분리한다. A1 kink-in-time / A2 depth elasticity / A3 surplus counterfactual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CLOSING_DEPTH_30 = "0077"
CLOSING_DEPTH_20 = "0069"


def build_closing_panel(rows, waste, item_to_category):
    df = rows.copy()
    df["category_id"] = df["item_id"].map(item_to_category)
    df = df[df["category_id"].notna()]
    df["is_closing"] = df["label"] == "closing"
    df["is_c30"] = df["discount_code"] == CLOSING_DEPTH_30
    df["is_c20"] = df["discount_code"] == CLOSING_DEPTH_20
    df["normal_q"] = df["qty"].where(~df["is_closing"], 0)
    df["closing_q"] = df["qty"].where(df["is_closing"], 0)
    df["c30_q"] = df["qty"].where(df["is_closing"] & df["is_c30"], 0)
    df["c20_q"] = df["qty"].where(df["is_closing"] & df["is_c20"], 0)
    agg = df.groupby(["category_id", "date"], observed=True).agg(
        normal_qty=("normal_q", "sum"),
        closing_qty=("closing_q", "sum"),
        closing_qty_30=("c30_q", "sum"),
        closing_qty_20=("c20_q", "sum"),
    ).reset_index()
    w = waste.copy()
    w["category_id"] = w["item_id"].map(item_to_category)
    w = w[w["category_id"].notna()]
    w = w.groupby(["category_id", "date"], observed=True)["waste_qty"].sum().reset_index()
    panel = agg.merge(w, on=["category_id", "date"], how="left")
    panel["waste_qty"] = panel["waste_qty"].fillna(0)
    panel["surplus"] = panel["closing_qty"] + panel["waste_qty"]
    d = pd.to_datetime(panel["date"])
    panel["dow"] = d.dt.dayofweek
    panel["month"] = d.dt.month
    panel["trend"] = (d - d.min()).dt.days
    return panel.sort_values(["category_id", "date"]).reset_index(drop=True)

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


@dataclass(frozen=True)
class DepthResult:
    """Result of depth elasticity estimation."""
    n: int
    slope: float
    se: float | None
    base: float
    alpha: float
    note: str


def _depth_long(panel):
    """Reshape closing_qty_30/closing_qty_20 to long format (depth observation per row)."""
    a = panel[["closing_qty_30", "surplus", "dow", "trend"]].rename(
        columns={"closing_qty_30": "y"}
    )
    a["depth"] = 0.30
    b = panel[["closing_qty_20", "surplus", "dow", "trend"]].rename(
        columns={"closing_qty_20": "y"}
    )
    b["depth"] = 0.20
    return pd.concat([a, b], ignore_index=True)


def fit_depth_elasticity(panel):
    """Estimate depth elasticity via OLS; extrapolate to depth=0 for base demand B.

    Returns α_A2 = B / mean(closing), a lower-bound estimate accounting for endogeneity.
    """
    from ._ols import _ols_hc3

    long = _depth_long(panel)
    long = long[long["y"].notna()]
    if long["depth"].nunique() < 2 or len(long) < 20:
        return DepthResult(
            len(long),
            float("nan"),
            None,
            float("nan"),
            float("nan"),
            "insufficient depth variation",
        )
    y = long["y"].to_numpy(dtype=float)
    # Design: intercept, depth(treat), surplus, trend, dow one-hot(-1)
    dow = pd.get_dummies(long["dow"], prefix="dow", drop_first=True).to_numpy(dtype=float)
    cols = [np.ones(len(long)), long["depth"].to_numpy(dtype=float),
            long["surplus"].to_numpy(dtype=float), long["trend"].to_numpy(dtype=float)]
    cols.append(dow)
    X = np.column_stack(cols)
    # Filter out constant/near-constant columns to avoid singularity
    keep = X.std(axis=0) > 1e-12
    keep[0] = True  # keep intercept
    keep[1] = True  # keep treatment (depth)
    X_filtered = X[:, keep]
    treat_idx_filtered = 1  # depth is at column index 1 after filtering

    out = _ols_hc3(y, X_filtered, treat_idx=treat_idx_filtered)
    if out is None:
        return DepthResult(
            len(long), float("nan"), None, float("nan"), float("nan"), "ill-posed"
        )
    slope, se = out
    # Predict at depth=0: intercept + slope * 0 = intercept
    # Intercept = mean(y) - slope * mean(depth) when centered
    base = float(np.clip(y.mean() - slope * long["depth"].mean(), 0.0, None))
    alpha = (
        float(np.clip(base / y.mean(), 0.0, 1.0))
        if y.mean() > 0
        else float("nan")
    )
    return DepthResult(
        len(long), slope, se, base, alpha, "lower-bound (depth endogeneity)"
    )


def depth_time_overlap(rows):
    """Diagnostic: check if 20% and 30% discounts occur at different times of day.

    Returns dict with median hour for each depth and flag if medians differ ≥ 1 hour.
    """
    c = rows[rows["label"] == "closing"].copy()
    c["tod"] = c["hour"] + c["minute"] / 60.0
    m20 = float(c.loc[c["discount_code"] == CLOSING_DEPTH_20, "tod"].median())
    m30 = float(c.loc[c["discount_code"] == CLOSING_DEPTH_30, "tod"].median())
    return {
        "median_hour_20": round(m20),
        "median_hour_30": round(m30),
        "time_separated": abs(m30 - m20) >= 1.0,
    }

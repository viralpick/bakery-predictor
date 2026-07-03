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

# Depth elasticity estimation
MIN_DEPTH_ROWS = 20
ZERO_VARIANCE_THRESHOLD = 1e-12
TIME_SEPARATION_HOURS = 1.0

# Surplus counterfactual estimation
MIN_SURPLUS_ROWS = 20
HIGH_SURPLUS_QUANTILE = 0.75
SUPPLY_DRIVEN_SLOPE_THRESHOLD = 0.5

# Kink-in-time estimation
DEFAULT_BIN_MIN = 15
PRE_ONSET_START_HOUR = 17


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


@dataclass(frozen=True)
class SurplusResult:
    """Result of surplus counterfactual estimation."""
    n: int
    slope: float
    se: float | None
    clearance_high: float
    note: str


@dataclass(frozen=True)
class KinkResult:
    """Result of kink-in-time (RD at closing onset) estimation."""
    n_days: int
    base: float
    closing_total: float
    alpha: float
    note: str


@dataclass(frozen=True)
class AlphaEstimate:
    """Aggregated α interval from three estimators (A1/A2/A3)."""
    alpha_low: float
    alpha_high: float
    a1: float
    a2: float
    a3_slope: float
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


def _depth_design_matrix(long: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Build design matrix for depth elasticity: [intercept, depth, surplus, trend, dow-dummies].

    Returns (y, X_filtered, treat_idx) where X_filtered has constant/near-constant columns removed.
    """
    y = long["y"].to_numpy(dtype=float)
    dow = pd.get_dummies(long["dow"], prefix="dow", drop_first=True).to_numpy(dtype=float)
    cols = [
        np.ones(len(long)),
        long["depth"].to_numpy(dtype=float),
        long["surplus"].to_numpy(dtype=float),
        long["trend"].to_numpy(dtype=float),
        dow,
    ]
    X = np.column_stack(cols)
    # Filter out constant/near-constant columns to avoid singularity
    keep = X.std(axis=0) > ZERO_VARIANCE_THRESHOLD
    keep[0] = True  # keep intercept
    keep[1] = True  # keep treatment (depth)
    X_filtered = X[:, keep]
    return y, X_filtered, 1  # depth is at column index 1 after filtering


def _nan_depth_result(n: int, note: str) -> DepthResult:
    """Return DepthResult with NaN values for regression failures."""
    return DepthResult(n, float("nan"), None, float("nan"), float("nan"), note)


def fit_depth_elasticity(panel):
    """Estimate depth elasticity via OLS; extrapolate to depth=0 for base demand B.

    Returns α_A2 = B / mean(closing), a lower-bound estimate accounting for endogeneity.
    """
    from ._ols import _ols_hc3

    long = _depth_long(panel)
    long = long[long["y"].notna()]
    if long["depth"].nunique() < 2 or len(long) < MIN_DEPTH_ROWS:
        return _nan_depth_result(len(long), "insufficient depth variation")
    y, X_filtered, treat_idx = _depth_design_matrix(long)
    out = _ols_hc3(y, X_filtered, treat_idx=treat_idx)
    if out is None:
        return _nan_depth_result(len(long), "ill-posed")
    slope, se = out
    base = float(np.clip(y.mean() - slope * long["depth"].mean(), 0.0, None))
    alpha = (
        float(np.clip(base / y.mean(), 0.0, 1.0))
        if y.mean() > 0
        else float("nan")
    )
    return DepthResult(
        len(long), slope, se, base, alpha, "lower-bound (depth endogeneity)"
    )


def fit_surplus_counterfactual(panel):
    """Estimate surplus counterfactual: how closing qty scales with actual end-of-day surplus.

    Slope > 0.5 → supply-driven dumping (low α). Slope ≈ 0 → demand-limited base (higher α).
    Controls for normal_qty, trend, day-of-week.
    """
    from ._ols import _ols_hc3

    p = panel[panel["surplus"] > 0].copy()
    if len(p) < MIN_SURPLUS_ROWS:
        return SurplusResult(len(p), float("nan"), None, float("nan"), "insufficient rows")
    y = p["closing_qty"].to_numpy(dtype=float)
    dow = pd.get_dummies(p["dow"], prefix="dow", drop_first=True).to_numpy(dtype=float)
    cols = [
        np.ones(len(p)),
        p["surplus"].to_numpy(dtype=float),
        p["normal_qty"].to_numpy(dtype=float),
        p["trend"].to_numpy(dtype=float),
        dow,
    ]
    X = np.column_stack(cols)
    keep = X.std(axis=0) > ZERO_VARIANCE_THRESHOLD
    keep[0] = True
    keep[1] = True
    X_filtered = X[:, keep]
    out = _ols_hc3(y, X_filtered, treat_idx=1)
    q75 = p["surplus"].quantile(HIGH_SURPLUS_QUANTILE)
    high = p[p["surplus"] >= q75]
    clearance_high = float((high["closing_qty"] / high["surplus"]).mean()) if len(high) else float("nan")
    if out is None:
        return SurplusResult(len(p), float("nan"), None, clearance_high, "ill-posed")
    slope, se = out
    note = "supply-driven (low α)" if slope > SUPPLY_DRIVEN_SLOPE_THRESHOLD else "demand-limited (higher α)"
    return SurplusResult(len(p), float(slope), se, clearance_high, note)


def depth_time_overlap(rows):
    """Diagnostic: check if 20% and 30% discounts occur at different times of day.

    Returns dict with median hour for each depth and flag if medians differ ≥ TIME_SEPARATION_HOURS.
    If either discount code is absent, time_separated is None (missing data).
    """
    c = rows[rows["label"] == "closing"].copy()
    c["tod"] = c["hour"] + c["minute"] / 60.0
    m20 = float(c.loc[c["discount_code"] == CLOSING_DEPTH_20, "tod"].median())
    m30 = float(c.loc[c["discount_code"] == CLOSING_DEPTH_30, "tod"].median())
    # Guard: if either median is NaN (discount code absent), mark as missing data
    time_separated = (
        None
        if np.isnan(m20) or np.isnan(m30)
        else abs(m30 - m20) >= TIME_SEPARATION_HOURS
    )
    return {
        "median_hour_20": round(m20) if not np.isnan(m20) else None,
        "median_hour_30": round(m30) if not np.isnan(m30) else None,
        "time_separated": time_separated,
    }


def build_intraday_curve(rows, item_to_category, category, bin_min=None):
    """Build intraday sales curve: per-category-date, binned by time, mark closing onset.

    Args:
        rows: raw transaction rows (date, hour, minute, item_id, qty, label)
        item_to_category: Series mapping item_id → category
        category: category name to filter for
        bin_min: bin width in minutes (default DEFAULT_BIN_MIN)

    Returns:
        DataFrame with cols [date, bin, qty, closing, hour] where
        closing=True marks bins with any marked-down (label=="closing") sales.
    """
    if bin_min is None:
        bin_min = DEFAULT_BIN_MIN
    df = rows.copy()
    df["category_id"] = df["item_id"].map(item_to_category)
    df = df[df["category_id"] == category].copy()
    df["tod"] = df["hour"] + df["minute"] / 60.0
    df["bin"] = (df["tod"] // (bin_min / 60.0)).astype(int)
    df["is_closing"] = df["label"] == "closing"
    g = df.groupby(["date", "bin"], observed=True).agg(
        qty=("qty", "sum"), closing=("is_closing", "max"),
        hour=("hour", "min")).reset_index()
    return g


def fit_kink(curve):
    """Estimate α_A1 = B/C via RD at closing onset.

    Uses pre-onset (late afternoon) sales rate, extrapolates into closing window
    to form counterfactual base B. Observed closing window = C. α = B/C is a
    lower bound (evening commute uplift).

    Args:
        curve: output from build_intraday_curve

    Returns:
        KinkResult with n_days, base, closing_total, alpha, note
    """
    days = curve["date"].nunique()
    if days == 0:
        return KinkResult(0, float("nan"), float("nan"), float("nan"), "no data")
    pre = curve[(~curve["closing"].astype(bool)) & (curve["hour"] >= PRE_ONSET_START_HOUR)]
    win = curve[curve["closing"].astype(bool)]
    if len(pre) == 0 or len(win) == 0:
        return KinkResult(days, float("nan"), float("nan"), float("nan"), "no pre/closing bins")
    closing_total = win["qty"].sum()
    if closing_total <= 0:
        return KinkResult(days, float("nan"), float(closing_total), float("nan"),
                          "degenerate: closing total is zero")
    pre_rate = pre["qty"].mean()
    base = pre_rate * len(win)
    alpha = float(np.clip(base / closing_total, 0.0, 1.0))
    return KinkResult(days, float(base), float(closing_total), alpha,
                      "lower-bound (evening commute uplift)")


def aggregate_alpha(kink, depth, surplus):
    """Aggregate three estimators (A1/A2/A3) into a single α interval.

    A1 (kink) and A2 (depth) are lower-bound methods → alpha_low = max(their α).
    A3 (surplus) informs the upper side: if supply-driven (high slope),
    pull the upper bound down; if demand-limited, allow up to 1.0.
    Both bounds are clipped to [0,1] and α_low ≤ α_high is maintained.
    """
    # A1, A2 are lower bounds; take the max (drop NaN)
    lowers = [v for v in (kink.alpha, depth.alpha) if v == v]
    alpha_low = max(lowers) if lowers else float("nan")

    # A3 (surplus) informs upper bound
    alpha_high = 1.0
    if surplus.slope == surplus.slope and surplus.slope > SUPPLY_DRIVEN_SLOPE_THRESHOLD:
        # NaN-safe guard for clearance_high (NaN ≠ NaN, so use x==x check)
        clearance = surplus.clearance_high if surplus.clearance_high == surplus.clearance_high else 0.0
        # Supply-driven (high slope) → pull upper bound down
        alpha_high = float(np.clip(
            1.0 - clearance + alpha_low,
            alpha_low,
            1.0
        ))

    # Ensure invariant: both in [0,1], alpha_low ≤ alpha_high
    alpha_low = float(np.clip(alpha_low, 0.0, 1.0))
    alpha_high = float(np.clip(alpha_high, 0.0, 1.0))
    if not np.isnan(alpha_low):
        alpha_high = max(alpha_high, alpha_low)

    note = f"lower=max(A1,A2) bounds; A3 {surplus.note}"
    return AlphaEstimate(alpha_low, alpha_high, kink.alpha, depth.alpha, surplus.slope, note)


def run_closing_demand(rows, waste, item_to_category, category: str = "bread") -> dict:
    """Orchestrate the A1/A2/A3 estimators for a single category into one α report.

    Builds the closing panel (all categories), filters to `category`, fits
    kink (A1) / depth elasticity (A2) / surplus counterfactual (A3), and
    aggregates them into a single α interval.

    Returns dict with keys: alpha (AlphaEstimate), depth (DepthResult),
    surplus (SurplusResult), kink (KinkResult), panel (category-filtered DataFrame).
    """
    panel = build_closing_panel(rows, waste, item_to_category)
    cat_panel = panel[panel["category_id"] == category].reset_index(drop=True)
    curve = build_intraday_curve(rows, item_to_category, category)
    kink = fit_kink(curve)
    depth = fit_depth_elasticity(cat_panel)
    surplus = fit_surplus_counterfactual(cat_panel)
    alpha = aggregate_alpha(kink, depth, surplus)
    return {"alpha": alpha, "depth": depth, "surplus": surplus, "kink": kink, "panel": cat_panel}

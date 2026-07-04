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

# Evening-traffic floor diagnostic (A1 counterfactual validity)
AFTERNOON_END_HOUR = 19
EVENING_START_HOUR = 20
EVENING_END_HOUR = 21
FOOTFALL_STABLE_RATIO = 0.9  # evening/afternoon total-footfall ratio treated as "stable"

# a1_bias: honest signal for whether A1's floor assumption is established.
A1_BIAS_INDETERMINATE = "indeterminate"  # footfall stable -> cannibalization, direction unclear
A1_BIAS_LIKELY_OVERSTATES = "likely_overstates"  # footfall itself declines -> A1 overstates alpha
A1_BIAS_UNKNOWN = "unknown"  # degenerate: footfall ratio itself unavailable
A1_EXCLUDING_BIASES = frozenset({A1_BIAS_INDETERMINATE, A1_BIAS_LIKELY_OVERSTATES})


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


def _degenerate_negative_base_result(n: int, slope: float, se: float | None, raw_base: float) -> DepthResult:
    """Steep positive slope extrapolated to a negative base at depth=0.

    That's the extrapolation itself being degenerate (depth endogeneity), not a
    valid "0% real demand" estimate -- must surface as NaN, not clip to 0.0.
    """
    return DepthResult(
        n, slope, se, raw_base, float("nan"),
        "degenerate: negative extrapolation (depth endogeneity) → uninformative",
    )


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
    raw_base = float(y.mean() - slope * long["depth"].mean())
    if raw_base < 0.0:
        return _degenerate_negative_base_result(len(long), slope, se, raw_base)
    alpha = (
        float(np.clip(raw_base / y.mean(), 0.0, 1.0))
        if y.mean() > 0
        else float("nan")
    )
    return DepthResult(
        len(long), slope, se, raw_base, alpha, "lower-bound (depth endogeneity)"
    )


def _with_clearance_marker(note: str, clearance_high: float) -> str:
    """Flag clearance_high > 1.0 (can occur when waste `out` is negative -> data-quality issue)."""
    if clearance_high == clearance_high and clearance_high > 1.0:
        return note + " [clearance>1: check waste data]"
    return note


def _surplus_design_matrix(p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build design matrix for surplus counterfactual: [intercept, surplus, normal_qty, trend, dow-dummies].

    Returns (y, X_filtered) with constant/near-constant columns removed.
    """
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
    keep[0] = True  # keep intercept
    keep[1] = True  # keep treatment (surplus)
    return y, X[:, keep]


def fit_surplus_counterfactual(panel):
    """Estimate surplus counterfactual: how closing qty scales with actual end-of-day surplus.

    Slope > 0.5 → supply-driven dumping (low α). Slope ≈ 0 → demand-limited base (higher α).
    Controls for normal_qty, trend, day-of-week.
    """
    from ._ols import _ols_hc3

    p = panel[panel["surplus"] > 0].copy()
    if len(p) < MIN_SURPLUS_ROWS:
        return SurplusResult(len(p), float("nan"), None, float("nan"), "insufficient rows")
    y, X_filtered = _surplus_design_matrix(p)
    out = _ols_hc3(y, X_filtered, treat_idx=1)
    q75 = p["surplus"].quantile(HIGH_SURPLUS_QUANTILE)
    high = p[p["surplus"] >= q75]
    clearance_high = float((high["closing_qty"] / high["surplus"]).mean()) if len(high) else float("nan")
    if out is None:
        note = _with_clearance_marker("ill-posed", clearance_high)
        return SurplusResult(len(p), float("nan"), None, clearance_high, note)
    slope, se = out
    note = "supply-driven (low α)" if slope > SUPPLY_DRIVEN_SLOPE_THRESHOLD else "demand-limited (higher α)"
    note = _with_clearance_marker(note, clearance_high)
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


def _window_rate(df: pd.DataFrame, start_hour: int, end_hour: int, value_col: str, how: str) -> float:
    """Mean per-hour rate of `value_col` in [start_hour, end_hour], across days then averaged.

    `how="sum"` sums value_col per hour (e.g. qty); `how="nunique"` counts unique
    values per hour (e.g. receipt_id, for footfall). NaN if the window has no rows.
    """
    win = df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)]
    if win.empty:
        return float("nan")
    n_days = win["date"].nunique()
    if n_days == 0:
        return float("nan")
    grouped = win.groupby("hour")[value_col]
    per_hour = (grouped.sum() if how == "sum" else grouped.nunique()) / n_days
    return float(per_hour.mean())


def _ratio(numerator: float, denominator: float) -> float:
    """Safe ratio: NaN if denominator is NaN/<=0 (NaN numerator propagates naturally)."""
    if denominator != denominator or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _footfall_ratios(cat_df: pd.DataFrame) -> tuple[float, float]:
    """Discount-agnostic total footfall ratios (evening / afternoon), over ALL rows.

    Unlike the non-closing rate A1 uses as its counterfactual, these count every
    visit regardless of discount -- so they can tell genuine traffic decline
    apart from cannibalization to the discounted item.
    """
    afternoon_receipts = _window_rate(cat_df, PRE_ONSET_START_HOUR, AFTERNOON_END_HOUR, "receipt_id", "nunique")
    evening_receipts = _window_rate(cat_df, EVENING_START_HOUR, EVENING_END_HOUR, "receipt_id", "nunique")
    afternoon_qty = _window_rate(cat_df, PRE_ONSET_START_HOUR, AFTERNOON_END_HOUR, "qty", "sum")
    evening_qty = _window_rate(cat_df, EVENING_START_HOUR, EVENING_END_HOUR, "qty", "sum")
    return _ratio(evening_receipts, afternoon_receipts), _ratio(evening_qty, afternoon_qty)


def _a1_bias(traffic_stable: bool | None) -> str:
    """Classify A1's floor-assumption bias direction from the footfall discriminator."""
    if traffic_stable is None:
        return A1_BIAS_UNKNOWN
    return A1_BIAS_INDETERMINATE if traffic_stable else A1_BIAS_LIKELY_OVERSTATES


def evening_traffic_check(rows, item_to_category, category: str) -> dict:
    """Diagnostic: does evening (20-21h) traffic hold up vs afternoon (17-19h)?

    A low non-closing evening rate alone is ambiguous: it can mean evening
    traffic genuinely declines (A1 overstates α), or it can mean total footfall
    is stable and customers just switch to the discounted item within the same
    visit (cannibalization -- A1 bias indeterminate). `footfall_*_ratio` /
    `traffic_stable` (ALL rows, discount-agnostic) discriminate between the two;
    `a1_bias` is the honest resulting signal consumed by `aggregate_alpha`.
    """
    df = rows.copy()
    df["category_id"] = df["item_id"].map(item_to_category)
    cat_df = df[df["category_id"] == category]
    non_closing = cat_df[cat_df["label"] != "closing"]

    afternoon_rate = _window_rate(non_closing, PRE_ONSET_START_HOUR, AFTERNOON_END_HOUR, "qty", "sum")
    evening_rate = _window_rate(non_closing, EVENING_START_HOUR, EVENING_END_HOUR, "qty", "sum")
    ratio = _ratio(evening_rate, afternoon_rate)
    a1_floor_valid = None if ratio != ratio else ratio >= 1.0

    receipts_ratio, qty_ratio = _footfall_ratios(cat_df)
    traffic_stable = None if receipts_ratio != receipts_ratio else receipts_ratio >= FOOTFALL_STABLE_RATIO

    return {
        "afternoon_rate": afternoon_rate,
        "evening_rate": evening_rate,
        "evening_to_afternoon_ratio": ratio,
        "a1_floor_valid": a1_floor_valid,
        "footfall_receipts_ratio": receipts_ratio,
        "footfall_qty_ratio": qty_ratio,
        "traffic_stable": traffic_stable,
        "a1_bias": _a1_bias(traffic_stable),
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
    note = (
        "degenerate: base exceeds closing (no discount lift)"
        if base > closing_total
        else "lower-bound (evening commute uplift)"
    )
    return KinkResult(days, float(base), float(closing_total), alpha, note)


def _reconcile_lower_bound(
    kink_alpha: float, depth_alpha: float, a1_bias: str | None
) -> tuple[list[float], bool]:
    """Drop A1 from the lower-bound candidates when its floor status isn't established.

    `a1_bias` comes from `evening_traffic_check`: None (not checked -- backward-
    compatible default) or "unknown" (degenerate) → A1 kept. "indeterminate"
    (footfall stable → cannibalization, direction unclear) or "likely_overstates"
    (footfall itself declines) → A1 dropped; it can't anchor the lower bound.

    Returns (valid_lowers, a1_excluded) with NaN candidates dropped.
    """
    a1_excluded = a1_bias in A1_EXCLUDING_BIASES
    candidates = (depth_alpha,) if a1_excluded else (kink_alpha, depth_alpha)
    return [v for v in candidates if v == v], a1_excluded


def _lower_bound_note(a1_excluded: bool, lowers: list[float], a1_bias: str | None) -> str:
    """Describe how alpha_low was derived, honestly reflecting A1's exclusion reason."""
    if not a1_excluded:
        return "lower=max(A1,A2) bounds"
    reason = (
        "evening full-price low but footfall stable → cannibalization, bias indeterminate"
        if a1_bias == A1_BIAS_INDETERMINATE
        else "evening traffic genuinely declines (A1 overstates α)"
    )
    if lowers:
        return f"lower=A2 only (A1 floor unverified: {reason})"
    return f"no validated lower bound: A1 floor unverified ({reason}), A2 degenerate"


def aggregate_alpha(kink, depth, surplus, a1_bias=None):
    """Aggregate A1/A2/A3 into a single α interval.

    alpha_low = max of valid lower-bound methods (A1 kink, A2 depth). A1 only
    counts if `a1_bias` (from evening_traffic_check) is not "indeterminate" or
    "likely_overstates" -- either means A1's floor assumption is not established,
    so it can't anchor alpha_low on a value whose bias direction is unknown.
    A3 pulls alpha_high down when supply-driven, else allows up to 1.0.
    Bounds are clipped to [0,1] (NaN preserved) with alpha_low ≤ alpha_high.
    """
    lowers, a1_excluded = _reconcile_lower_bound(kink.alpha, depth.alpha, a1_bias)
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

    # Ensure invariant: both in [0,1] (NaN preserved by np.clip), alpha_low ≤ alpha_high
    alpha_low = float(np.clip(alpha_low, 0.0, 1.0))
    alpha_high = float(np.clip(alpha_high, 0.0, 1.0))
    if not np.isnan(alpha_low):
        alpha_high = max(alpha_high, alpha_low)

    note = f"{_lower_bound_note(a1_excluded, lowers, a1_bias)}; A3 {surplus.note}"
    return AlphaEstimate(alpha_low, alpha_high, kink.alpha, depth.alpha, surplus.slope, note)


def run_closing_demand(rows, waste, item_to_category, category: str = "bread") -> dict:
    """Orchestrate the A1/A2/A3 estimators for a single category into one α report.

    Builds the closing panel (all categories), filters to `category`, fits
    kink (A1) / depth elasticity (A2) / surplus counterfactual (A3), checks
    whether A1's floor assumption holds (evening_traffic_check), and
    aggregates the three estimators into a single, reconciled α interval.

    Returns dict with keys: alpha (AlphaEstimate), depth (DepthResult),
    surplus (SurplusResult), kink (KinkResult), panel (category-filtered DataFrame).
    """
    panel = build_closing_panel(rows, waste, item_to_category)
    cat_panel = panel[panel["category_id"] == category].reset_index(drop=True)
    curve = build_intraday_curve(rows, item_to_category, category)
    kink = fit_kink(curve)
    depth = fit_depth_elasticity(cat_panel)
    surplus = fit_surplus_counterfactual(cat_panel)
    evening_check = evening_traffic_check(rows, item_to_category, category)
    alpha = aggregate_alpha(kink, depth, surplus, a1_bias=evening_check["a1_bias"])
    return {"alpha": alpha, "depth": depth, "surplus": surplus, "kink": kink, "panel": cat_panel}

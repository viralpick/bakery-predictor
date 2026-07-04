"""Discount-depth regime-shift re-validation (다중시각 재검증, angle ①).

On 2025-01-17 the closing discount depth was cut company-wide 30% → 20% (codes
0077 → 0069, confirmed in the panel: effective depth 0.30 → 0.20, all four
stores switching simultaneously). Because every store switched at once and
never-discounted items are ~1% of volume, there is NO clean control group and
total-demand identification is swamped by secular swings. The one robust,
internally controlled signal is the *composition*:

    closing_share = closing_qty / (normal_qty + closing_qty)

A store-wide demand shock moves normal and closing proportionally, but a change
in discount attractiveness specifically moves the split. If cutting the discount
from 30% to 20% made the closing channel less attractive to bargain-hunters,
closing_share should drop at the cut. We test that with an item-fixed-effects
OLS (post_cut + linear trend + month FE + item FE, HC3 SE) plus a placebo
break-date distribution. A null (CI spanning 0, placebo-comparable) means the
closing channel is depth-inelastic → supply-driven surplus, not price-sensitive
bargain demand → leans toward high α, though α is NOT point-identified here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._ols import _ols_hc3

DEFAULT_CUT_DATE = "2025-01-17"
# The month straddling the cut has partial (blended) depth — drop it so the
# pre/post contrast is between clean 30% and clean 20% regimes.
TRANSITION_MONTH = "2025-01"
Z_95 = 1.959963984540054


@dataclass
class RegimeResult:
    """Estimated jump in an outcome at a (real or placebo) cut date."""

    beta: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    n_params: int
    cut_date: str
    ill_posed: bool


def build_regime_panel(rows, item_to_category, category, *, cut_date=DEFAULT_CUT_DATE,
                       exclude_transition=True):
    """Item-day panel for one category with closing_share, closing_intensity, post_cut.

    Parameters
    ----------
    rows : pd.DataFrame
        waste_alpha item-day rows (date, item_id, normal_qty, closing_qty, made, out).
    item_to_category : pd.Series
        Mapping item_id -> category.
    category : str
        Category to keep (e.g. "bread").
    """
    df = rows.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["category_id"] = df["item_id"].astype(str).map(item_to_category)
    df = df[df["category_id"] == category].copy()

    if exclude_transition:
        df = df[df["date"].dt.strftime("%Y-%m") != TRANSITION_MONTH]

    sold = df["normal_qty"] + df["closing_qty"]
    df = df[sold > 0].copy()
    df["closing_share"] = df["closing_qty"] / (df["normal_qty"] + df["closing_qty"])
    made = df["made"].where(df["made"] > 0, np.nan)
    df["closing_intensity"] = df["closing_qty"] / made

    cut_ts = pd.Timestamp(cut_date)
    df["post_cut"] = (df["date"] >= cut_ts).astype(int)
    df["trend"] = (df["date"] - df["date"].min()).dt.days.astype(float)
    df["month"] = df["date"].dt.month
    return df.sort_values(["item_id", "date"]).reset_index(drop=True)


def _demean_within(values, groups):
    """Subtract each group's mean (within transformation to absorb item FE)."""
    return values - pd.Series(values).groupby(groups).transform("mean").to_numpy()


def _regime_design_matrix(panel, outcome_col):
    """Within-item-demeaned Y and X = [post_cut, trend, month dummies].

    Item fixed effects are absorbed by within-item demeaning (Frisch-Waugh-Lovell)
    rather than a dummy block, which for 200+ items would trip the condition-number
    guard on scale/sparsity alone. This is algebraically identical for post_cut but
    numerically stable. Trend is scaled to unit std (harmless: rescaling one
    regressor leaves the others' coefficients unchanged). Treatment is index 0.
    """
    valid = panel[outcome_col].notna()
    p = panel[valid]
    y = p[outcome_col].to_numpy(dtype=float)
    n = len(y)
    trend = p["trend"].to_numpy(float)
    trend_sd = trend.std()
    trend = trend / trend_sd if trend_sd > 1e-12 else trend
    month = pd.get_dummies(p["month"], drop_first=True).to_numpy(float)
    if month.size == 0:
        month = np.empty((n, 0))
    raw = np.hstack([p["post_cut"].to_numpy(float).reshape(-1, 1),
                     trend.reshape(-1, 1), month])
    item = p["item_id"].to_numpy()
    y_dem = _demean_within(y, item)
    x_dem = np.column_stack([_demean_within(raw[:, j], item) for j in range(raw.shape[1])])
    keep = x_dem.std(axis=0) > 1e-12
    keep[0] = True                       # keep post_cut → caller's guard fires if degenerate
    return y_dem, x_dem[:, keep], 0


def fit_regime_shift(panel, outcome_col="closing_share"):
    """Fit the item-FE regime-shift OLS and return the post_cut jump with HC3 CI."""
    y, X, treat_idx = _regime_design_matrix(panel, outcome_col)
    # post_cut with no variation (single regime) is unidentified → ill-posed.
    if X[:, treat_idx].std() <= 1e-12:
        return RegimeResult(np.nan, np.nan, np.nan, np.nan, len(y), X.shape[1],
                            "", ill_posed=True)
    fit = _ols_hc3(y, X, treat_idx)
    if fit is None:
        return RegimeResult(np.nan, np.nan, np.nan, np.nan, len(y), X.shape[1],
                            "", ill_posed=True)
    beta, se = fit
    return RegimeResult(beta, se, beta - Z_95 * se, beta + Z_95 * se,
                        len(y), X.shape[1], "", ill_posed=False)


def placebo_shifts(rows, item_to_category, category, placebo_cut_dates,
                   *, real_cut_date=DEFAULT_CUT_DATE, outcome_col="closing_share"):
    """Refit the same model at fake cut dates within the pre-real-cut period.

    Restricting to strictly-pre-cut data means every observed jump is spurious;
    the spread of these placebo betas is the empirical null the real estimate is
    judged against.
    """
    pre = rows.copy()
    pre["date"] = pd.to_datetime(pre["date"])
    pre = pre[pre["date"] < pd.Timestamp(real_cut_date)]
    results = []
    for fake_cut in placebo_cut_dates:
        panel = build_regime_panel(pre, item_to_category, category,
                                   cut_date=fake_cut, exclude_transition=False)
        res = fit_regime_shift(panel, outcome_col)
        res.cut_date = fake_cut
        results.append(res)
    return results


def _verdict(real, placebo):
    """'shift' only if the real jump is significant AND exceeds placebo spread."""
    if real.ill_posed:
        return "ill_posed"
    if real.ci_low <= 0 <= real.ci_high:
        return "depth_invariant"
    placebo_betas = [p.beta for p in placebo if not p.ill_posed]
    if placebo_betas and abs(real.beta) <= max(abs(b) for b in placebo_betas):
        return "depth_invariant"          # real jump within placebo noise
    return "shift"


def run_discount_regime(rows, item_to_category, category, *,
                        cut_date=DEFAULT_CUT_DATE, placebo_cut_dates=None):
    """Orchestrate the regime-shift test for one category into one report dict."""
    panel = build_regime_panel(rows, item_to_category, category, cut_date=cut_date)
    share = fit_regime_shift(panel, "closing_share")
    share.cut_date = cut_date
    intensity = fit_regime_shift(panel, "closing_intensity")
    intensity.cut_date = cut_date
    placebo = placebo_shifts(rows, item_to_category, category,
                             placebo_cut_dates or [], real_cut_date=cut_date)
    return {
        "category": category,
        "cut_date": cut_date,
        "n": int(len(panel)),
        "closing_share": share,
        "closing_intensity": intensity,
        "placebo": placebo,
        "verdict": _verdict(share, placebo),
    }

"""카테고리 총량 수요이전 흡수 검증 (W0 게이트).

leave-one-out 총량보존 계수 β: 카테고리 내 품목 조기품절(품절강도 T)이
같은 카테고리 총 sold(Y)를 떨어뜨리는가. β≈0 = 흡수(총량 보존), β<0 = walk-away.
confound(고수요일=품절많은날)는 OtherCatSold(그날 전반 traffic) + cat_baseline
(c의 최근 4주 동일요일 평균, lag)로 이중 통제. 타깃은 raw sold_units(순환 회피).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_CLOSE_HOUR = 22
BASELINE_WEEKS = 4
PANEL_COLUMNS = [
    "store_id", "category_id", "date", "cat_sold", "stockout_hours",
    "other_cat_sold", "cat_baseline", "dow", "month", "trend",
]
EQUIV_FRAC = 0.05            # δ = 5% of mean category sold, mapped via T IQR
MIN_PANEL_ROWS = 30
MAX_CONDITION_NUMBER = 1e10


@dataclass(frozen=True)
class AbsorptionResult:
    store_id: str
    category_id: str
    n: int
    beta: float              # effect of 1 stockout-hour on category total sold
    se: float
    ci_low: float            # 90% CI
    ci_high: float
    delta: float             # equivalence bound (in sold units per stockout-hour)
    verdict: str             # "absorb" | "walkaway" | "inconclusive"


def _stockout_hours(sub: pd.DataFrame, close_hour: int) -> float:
    """Category-day stockout intensity = Σ max(close_hour − stockout time-of-day, 0)."""
    so = pd.to_datetime(sub["stockout_time"])
    tod = so.dt.hour + so.dt.minute / 60.0
    return float((close_hour - tod).clip(lower=0.0).fillna(0.0).sum())


def _category_day_frame(daily: pd.DataFrame, close_hour: int) -> pd.DataFrame:
    """Aggregate item×day → (store, category, date) with cat_sold + stockout_hours."""
    grp = daily.groupby(["store_id", "category_id", "date"], observed=True)
    agg = grp.agg(cat_sold=("sold_units", "sum")).reset_index()
    hours = (grp.apply(lambda s: _stockout_hours(s, close_hour), include_groups=False)
             .rename("stockout_hours").reset_index())
    return agg.merge(hours, on=["store_id", "category_id", "date"])


def _add_other_cat_sold(cat_day: pd.DataFrame) -> pd.DataFrame:
    """OtherCatSold = same store-day total sold across all OTHER categories."""
    store_day = (cat_day.groupby(["store_id", "date"], observed=True)["cat_sold"]
                 .sum().rename("store_total").reset_index())
    out = cat_day.merge(store_day, on=["store_id", "date"])
    out["other_cat_sold"] = out["store_total"] - out["cat_sold"]
    return out.drop(columns=["store_total"])


def _add_leakage_safe_baseline(cat_day: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """cat_baseline = mean of same (store,category,dow) cat_sold over the prior
    `weeks` occurrences, strictly before the row's date (no leakage)."""
    df = cat_day.sort_values("date").copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    def _roll(g: pd.DataFrame) -> pd.Series:
        return g["cat_sold"].shift(1).rolling(weeks, min_periods=weeks).mean()
    df["cat_baseline"] = (df.groupby(["store_id", "category_id", "dow"], observed=True,
                                     group_keys=False).apply(_roll, include_groups=False))
    return df


def build_absorption_panel(daily: pd.DataFrame, *, close_hour: int = DEFAULT_CLOSE_HOUR,
                           baseline_weeks: int = BASELINE_WEEKS) -> pd.DataFrame:
    """Build the (store, category, date) regression panel. Rows without a full
    baseline window are dropped. Target/controls are all raw sold_units."""
    cat_day = _category_day_frame(daily, close_hour)
    cat_day = _add_other_cat_sold(cat_day)
    cat_day = _add_leakage_safe_baseline(cat_day, baseline_weeks)
    cat_day = cat_day.dropna(subset=["cat_baseline"]).copy()
    dt = pd.to_datetime(cat_day["date"])
    cat_day["month"] = dt.dt.month
    cat_day["trend"] = (dt - dt.min()).dt.days.astype(float)
    return cat_day[PANEL_COLUMNS].reset_index(drop=True)


def _design_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Y and X=[const, T, other_cat_sold, cat_baseline, dow dummies, month dummies, trend]."""
    y = panel["cat_sold"].to_numpy(dtype=float)
    n = len(y)
    const = np.ones((n, 1))
    cols = [const,
            panel["stockout_hours"].to_numpy(float).reshape(-1, 1),      # T = index 1
            panel["other_cat_sold"].to_numpy(float).reshape(-1, 1),
            panel["cat_baseline"].to_numpy(float).reshape(-1, 1),
            pd.get_dummies(panel["dow"], drop_first=True).to_numpy(float),
            pd.get_dummies(panel["month"], drop_first=True).to_numpy(float),
            panel["trend"].to_numpy(float).reshape(-1, 1)]
    X = np.hstack(cols)
    keep = X.std(axis=0) > 1e-12
    keep[0] = True                       # keep constant
    keep[1] = True                       # keep treatment even if degenerate → caller guards
    return y, X[:, keep], 1              # treatment is column index 1 after keep (const stays 0)


def _ols_hc3(y: np.ndarray, X: np.ndarray, treat_idx: int) -> tuple[float, float] | None:
    """OLS β and HC3 robust SE for the treatment column. numpy only. None if ill-posed."""
    n, k = X.shape
    if n - k < 5:
        return None
    XtX = X.T @ X
    try:
        cond = np.linalg.cond(XtX)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
        return None
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverages
    denom = np.clip((1.0 - h) ** 2, 1e-8, None)
    meat = X.T @ (X * (resid ** 2 / denom)[:, None])     # HC3 sandwich meat
    cov = XtX_inv @ meat @ XtX_inv
    treat_var = cov[treat_idx, treat_idx]
    if not np.isfinite(treat_var) or treat_var < 0:
        return None
    se = float(np.sqrt(treat_var))
    if not np.isfinite(se) or se <= 0:
        return None
    return float(beta[treat_idx]), se


def fit_absorption(panel: pd.DataFrame, store_id: str, category_id: str, *,
                   equiv_frac: float = EQUIV_FRAC) -> AbsorptionResult | None:
    """Regress category total sold on stockout intensity (dual-controlled) and
    judge absorption via TOST. Returns None on an unusable panel."""
    from scipy.stats import norm
    sub = panel[(panel["store_id"] == store_id)
                & (panel["category_id"] == category_id)]
    if len(sub) < MIN_PANEL_ROWS:
        return None
    y, X, treat_idx = _design_matrix(sub)
    fit = _ols_hc3(y, X, treat_idx)
    if fit is None:
        return None
    beta, se = fit
    z = norm.ppf(0.95)                                   # 90% CI (two-sided)
    ci_low, ci_high = beta - z * se, beta + z * se
    t_iqr = np.subtract(*np.percentile(sub["stockout_hours"], [75, 25]))
    mean_y = float(sub["cat_sold"].mean())
    delta = (equiv_frac * mean_y / t_iqr) if t_iqr > 1e-9 else float("inf")
    if ci_low > -delta and ci_high < delta:
        verdict = "absorb"
    elif ci_high < 0:
        verdict = "walkaway"
    else:
        verdict = "inconclusive"
    return AbsorptionResult(store_id, category_id, len(sub), beta, se,
                            ci_low, ci_high, delta, verdict)

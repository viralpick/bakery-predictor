"""Substitution effect (v3) — Difference-in-Differences with fixed effects.

Replaces the day-aggregate RD estimator in `substitution.py` with a
receipt-level DiD that separates the substitution signal from confounds.

## v3 design (user-confirmed 2026-05-21)

The naive day-level RD's main weaknesses, and the fixes applied here:

  - **j-side censoring (simultaneity)** — j itself may stockout on day t.
    Its post-window sales are then truncated and "j is selling slower" is
    misread as "i's substitution effect is small". **Fix A**: limit post
    window to `min(close, j_stockout_hour_t)` per day. Drop the day if j
    stocked out before i's cutoff.

  - **Store traffic / atypical control days** — when nearly every day has
    *some* popular item stocking out (광교: 92% of days), the "no stockout"
    control population is small *and* atypical (lower-traffic days). **Fix
    a**: include `log(store_total_sales_t)` as a covariate so each day's
    traffic level is absorbed.

  - **j's own baseline drift** — j may be a new menu item, on promo, or
    in/out of season. **Fix b**: include `log1p(j_rolling_mean_7d_t)` as
    covariate so j's recent baseline is held fixed.

  - **Day-of-week + month + year + holiday baselines** — absorbed by FE
    dummies. The remaining `β_iearly` coefficient is the substitution
    effect net of all controls above.

  - **Time-of-day peak structure** — instead of normalizing by raw hours
    (peak-direct vs peak-after cutoff would be noisy), we normalize by
    the store's measured cumulative-sales share for the pre and post
    windows. Peak hours carry their natural weight.

## Per-pair OLS specification

For each (i, j) pair:

    y_t = log1p( post_intensity_t / pre_intensity_t )
    X_t = [ i_early_t,
            month_FE, dow_FE, year_FE, is_holiday,
            log(store_total_t), log1p(j_rolling7_t) ]
    OLS:  β_iearly = effect of "i was unavailable" on j's pre→post rate change
                     after traffic + j baseline + calendar controls.

Where (with h_i = i's representative stockout hour, h_j_t = j's stockout
hour on day t, h_close fallback if j didn't stock out):

    post_end_t       = min(h_close, h_j_t)
    pre_sales_t      = j sales in [open, h_i)        on day t
    post_sales_t     = j sales in [h_i, post_end_t)  on day t
    expected_pre     = cum_profile[h_i - 1]
    expected_post_t  = cum_profile[post_end_t - 1] - cum_profile[h_i - 1]
    pre_intensity_t  = pre_sales_t  / (expected_pre    × j_total_t)
    post_intensity_t = post_sales_t / (expected_post_t × j_total_t)

## Validity filters

  - `min_cutoff ≤ i_stockout_hour < max_cutoff`: per-store cumulative-sales
    window (default 15%–75%). Too-early/too-late stockouts have insufficient
    pre/post.
  - `post_end_t > h_i + 1`: j must have ≥1h of post window available.
  - per-day `pre_sales_j ≥ 1`, `post_sales_j ≥ 1`, `j_total ≥ 3`.
  - per-pair `n_treated ≥ 10` and `n_control ≥ 10`.

## Limitations (still PoC)

  - Single representative cutoff `h_i` per item (median of treated stockout
    hours). Real per-day stockout timing varies — using each day's actual
    stockout hour for cutoff would be more accurate but breaks the standard
    DiD shape.
  - Customer panel data (membership ID) is not available, so "visit-level"
    substitution still has to be inferred from item-level intensity shifts.
  - 광교 92% i_early rate: control day baseline still small (~8% of days).
    Multi-store data is the real fix for population-level identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Defaults
TOP_ITEMS_PER_CATEGORY = 20
MIN_TREATED_DAYS = 10
MIN_CONTROL_DAYS = 10
DEFAULT_CUTOFF_MAX_THRESHOLD = 0.75
DEFAULT_CUTOFF_MIN_THRESHOLD = 0.15


@dataclass
class DidResult:
    coefficients: pd.DataFrame    # cols: from_item, to_item, beta_did, se, p_value, ...
    outflow_ratio: pd.Series      # index=item_id, Σ_j positive β_did (uncapped)
    cutoffs: dict[str, tuple[int, int]] = field(default_factory=dict)  # store_id → (min, max)


def compute_did_substitution(
    daily: pd.DataFrame,
    receipts: pd.DataFrame,
    hour_profiles: dict[str, np.ndarray],
    *,
    cutoff_max_threshold: float = DEFAULT_CUTOFF_MAX_THRESHOLD,
    cutoff_min_threshold: float = DEFAULT_CUTOFF_MIN_THRESHOLD,
    holidays: pd.Series | None = None,
    include_inter_category: bool = True,
) -> DidResult:
    """DiD-based substitution from receipt-level hourly data.

    `daily`     DAILY_COLUMNS frame (sold_units, is_stockout, stockout_time)
    `receipts`  long-form with [receipt_id, date, item_id, hour] (hh granularity)
    `hour_profiles`  store_id → length-24 sales distribution
    `holidays`  optional Series of pd.Timestamp dates (overrides if provided)
    """
    if "hour" not in receipts.columns:
        raise ValueError("receipts must include 'hour' column (DiD needs hh granularity)")

    receipts = receipts.copy()
    receipts["date"] = pd.to_datetime(receipts["date"]).dt.normalize()
    receipts["item_id"] = receipts["item_id"].astype(str)
    receipts["hour"] = receipts["hour"].astype(int).clip(0, 23)

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["item_id"] = daily["item_id"].astype(str)

    cutoffs = _store_cutoff_window(daily, hour_profiles, cutoff_min_threshold, cutoff_max_threshold)
    top_items = _select_top_items(daily, TOP_ITEMS_PER_CATEGORY)
    item_cat = daily.drop_duplicates("item_id").set_index("item_id")["category_id"].to_dict()

    # Hourly sales pivot: (date, item_id, hour) → units sold
    hourly = (
        receipts.groupby(["date", "item_id", "hour"]).size()
        .unstack("hour", fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )

    # i's representative cutoff: median stockout hour over its valid treated days
    representative_cutoffs = _representative_cutoffs(daily, top_items, cutoffs)

    # Pre-compute fix a: store traffic per day (Σ sold_units over all items)
    store_total = daily.groupby("date")["sold_units"].sum()
    # Pre-compute fix b: j's rolling 7d mean (shift 1 to avoid leakage)
    j_rolling = (
        daily.sort_values(["item_id", "date"])
        .groupby("item_id", group_keys=False, observed=True)["sold_units"]
        .apply(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    )
    j_rolling.index = daily.sort_values(["item_id", "date"]).set_index(
        ["item_id", "date"]
    ).index
    j_rolling_map = j_rolling.unstack("item_id")  # date × item_id
    # Per-item stockout-hour map (for Fix A: j-side censoring)
    j_stockout_hour_map = (
        daily.assign(h=pd.to_datetime(daily["stockout_time"]).dt.hour)
        .pivot(index="date", columns="item_id", values="h")
    )

    rows = []
    for i in top_items:
        cat_i = item_cat.get(i)
        if not cat_i or i not in representative_cutoffs:
            continue
        h_i = representative_cutoffs[i]
        if include_inter_category:
            peers = [x for x in top_items if x != i and item_cat.get(x)]
        else:
            peers = [x for x in top_items if item_cat.get(x) == cat_i and x != i]

        # Pre-compute per-day i_early flag + j daily totals around h_i
        i_daily = daily[daily["item_id"] == i].set_index("date")
        if i_daily.empty:
            continue
        # Store profile for cumulative shares
        store_id = str(i_daily["store_id"].iloc[0])
        profile = hour_profiles.get(store_id)
        if profile is None:
            continue
        cum = np.cumsum(profile) / profile.sum() if profile.sum() > 0 else None
        if cum is None or cum[h_i] <= 0 or cum[h_i] >= 1:
            continue
        expected_pre = float(cum[h_i - 1]) if h_i > 0 else 0.0
        if expected_pre < 0.05:
            continue
        # close_hour fallback for j_stockout-derived post_end
        close_hour = int(np.where(cum >= 0.999)[0][0] + 1) if (cum >= 0.999).any() else 23

        # i_early flag per date
        cutoff_min, cutoff_max = cutoffs.get(store_id, (10, 18))
        stockout_hours = pd.to_datetime(i_daily["stockout_time"]).dt.hour
        valid_treated = (
            i_daily["is_stockout"]
            & stockout_hours.between(cutoff_min, cutoff_max - 1)
        )
        i_early_flag = valid_treated.astype(int)

        # Build OLS panel for each (i, j) pair
        for j in peers:
            if j not in hourly.index.get_level_values("item_id"):
                continue
            j_hourly = hourly.xs(j, level="item_id", drop_level=True).reindex(
                i_daily.index, fill_value=0
            )

            # Fix A: per-day post_end = min(close, j_stockout_hour)
            j_stockout_h = j_stockout_hour_map[j].reindex(i_daily.index) if j in j_stockout_hour_map.columns else pd.Series(np.nan, index=i_daily.index)
            post_end = j_stockout_h.fillna(close_hour).astype(int).clip(0, 24)
            post_end = post_end.where(post_end > h_i, other=h_i)  # j 일찍 품절 → post window 0

            pre_sales = j_hourly.iloc[:, :h_i].sum(axis=1) if h_i > 0 else pd.Series(0, index=i_daily.index)

            # Per-day post sales: integrate from h_i to post_end_t
            post_sales = pd.Series(0, index=i_daily.index, dtype=float)
            for d, pe in post_end.items():
                if pe > h_i:
                    post_sales[d] = j_hourly.loc[d, h_i:pe-1].sum()

            j_total = pre_sales + post_sales
            # Per-day expected_post (varies because post_end varies)
            expected_post_t = pd.Series(
                [(cum[int(pe) - 1] - expected_pre) if pe > h_i else 0.0 for pe in post_end.values],
                index=post_end.index,
            )

            valid = (
                (pre_sales >= 1)
                & (post_sales >= 1)
                & (j_total >= 3)
                & (expected_post_t >= 0.05)
                & (post_end > h_i)
            )
            if valid.sum() < MIN_TREATED_DAYS + MIN_CONTROL_DAYS:
                continue

            pre_intensity = pre_sales / (expected_pre * j_total.replace(0, 1))
            post_intensity = post_sales / (expected_post_t.replace(0, np.nan) * j_total.replace(0, 1))
            ratio = post_intensity / pre_intensity.replace(0, np.nan)
            y = np.log1p(ratio.where(valid))

            # Fix a: store traffic, Fix b: j rolling baseline (both as covariates)
            log_store_total = np.log(store_total.reindex(y.index).clip(lower=1))
            j_lag = j_rolling_map[j].reindex(y.index) if j in j_rolling_map.columns else pd.Series(np.nan, index=y.index)
            log_j_lag = np.log1p(j_lag.clip(lower=0))

            df_panel = pd.DataFrame({
                "y": y,
                "i_early": i_early_flag.reindex(y.index, fill_value=0),
                "log_store_total": log_store_total,
                "log_j_lag": log_j_lag,
            }).dropna()
            df_panel["month"] = df_panel.index.month
            df_panel["dow"] = df_panel.index.dayofweek
            df_panel["year"] = df_panel.index.year
            if holidays is not None:
                holiday_set = set(pd.to_datetime(holidays).dt.normalize())
                df_panel["holiday"] = df_panel.index.normalize().isin(holiday_set).astype(int)
            else:
                df_panel["holiday"] = 0

            n_treat = int(df_panel["i_early"].sum())
            n_ctrl = len(df_panel) - n_treat
            if n_treat < MIN_TREATED_DAYS or n_ctrl < MIN_CONTROL_DAYS:
                continue

            fit = _ols_fe(df_panel)
            if fit is None:
                continue
            beta, se, p = fit
            rows.append({
                "category_id": cat_i,
                "to_category_id": item_cat[j],
                "from_item": i, "to_item": j,
                "same_category": cat_i == item_cat[j],
                "beta_did": float(beta),
                "se": float(se),
                "p_value": float(p),
                "n_treated": n_treat,
                "n_control": n_ctrl,
                "h_i": h_i,
            })

    coefficients = pd.DataFrame(rows)
    # outflow: sum of positive β over j's (significant or not)
    if len(coefficients) > 0:
        coefficients["effect"] = coefficients["beta_did"].clip(lower=0)
        outflow = coefficients.groupby("from_item")["effect"].sum()
    else:
        outflow = pd.Series(dtype=float)
    outflow = outflow.reindex(top_items, fill_value=0.0)
    outflow.name = "outflow_ratio"
    return DidResult(coefficients=coefficients, outflow_ratio=outflow, cutoffs=cutoffs)


def _store_cutoff_window(
    daily: pd.DataFrame,
    hour_profiles: dict[str, np.ndarray],
    min_threshold: float,
    max_threshold: float,
) -> dict[str, tuple[int, int]]:
    """Per store: (min_hour, max_hour) of valid stockout window from profile."""
    out: dict[str, tuple[int, int]] = {}
    for sid in daily["store_id"].astype(str).unique():
        prof = hour_profiles.get(sid)
        if prof is None or prof.sum() <= 0:
            out[sid] = (10, 18)  # safe fallback
            continue
        cum = np.cumsum(prof) / prof.sum()
        min_h = int(np.where(cum >= min_threshold)[0][0]) if (cum >= min_threshold).any() else 10
        max_h = int(np.where(cum >= max_threshold)[0][0]) if (cum >= max_threshold).any() else 18
        out[sid] = (min_h, max_h)
    return out


def _select_top_items(daily: pd.DataFrame, n: int) -> list[str]:
    ranked = (
        daily.groupby(["category_id", "item_id"], observed=True)["sold_units"]
        .sum().reset_index().sort_values(["category_id", "sold_units"], ascending=[True, False])
    )
    return ranked.groupby("category_id", observed=True).head(n)["item_id"].astype(str).tolist()


def _representative_cutoffs(
    daily: pd.DataFrame,
    items: list[str],
    cutoffs: dict[str, tuple[int, int]],
) -> dict[str, int]:
    """Per item, the median stockout hour among that item's valid-window stockouts."""
    out: dict[str, int] = {}
    for i in items:
        sub = daily[daily["item_id"] == i]
        if sub.empty:
            continue
        sid = str(sub["store_id"].iloc[0])
        cmin, cmax = cutoffs.get(sid, (10, 18))
        so = pd.to_datetime(sub["stockout_time"]).dt.hour
        valid = sub["is_stockout"] & so.between(cmin, cmax - 1)
        if valid.sum() == 0:
            continue
        out[i] = int(so[valid].median())
    return out


MAX_CONDITION_NUMBER = 1e8     # X'X stability threshold
MAX_REASONABLE_BETA = 5.0       # log1p scale: |β| > 5 is almost certainly a fit artifact


def _ols_fe(df: pd.DataFrame) -> tuple[float, float, float] | None:
    """OLS with month/dow/year/holiday fixed effects; returns (β_iearly, SE, p).

    Hand-rolled to avoid statsmodels dependency. Uses numpy linalg only.
    Returns None on singular X'X, ill-conditioning, or unreasonable β
    magnitude (suggests rank deficiency we missed).
    """
    from scipy.stats import t as student_t

    y = df["y"].to_numpy(dtype=float)
    n = len(y)
    if n < 30:
        return None

    treat = df["i_early"].to_numpy(dtype=float).reshape(-1, 1)
    month_d = pd.get_dummies(df["month"], drop_first=True).to_numpy(dtype=float)
    dow_d = pd.get_dummies(df["dow"], drop_first=True).to_numpy(dtype=float)
    year_d = pd.get_dummies(df["year"], drop_first=True).to_numpy(dtype=float)
    hol = df["holiday"].to_numpy(dtype=float).reshape(-1, 1)
    const = np.ones((n, 1))

    blocks = [const, treat, month_d, dow_d, year_d]
    if hol.sum() > 0:  # holiday is not all-zero
        blocks.append(hol)
    # Fix a: log store traffic, Fix b: log j rolling baseline
    if "log_store_total" in df.columns:
        blocks.append(df["log_store_total"].to_numpy(dtype=float).reshape(-1, 1))
    if "log_j_lag" in df.columns:
        blocks.append(df["log_j_lag"].to_numpy(dtype=float).reshape(-1, 1))
    X = np.hstack(blocks)
    # Drop zero-variance columns (e.g. a month dummy with no obs after filtering)
    nonzero_cols = X.std(axis=0) > 1e-12
    nonzero_cols[0] = True  # keep constant
    X = X[:, nonzero_cols]
    k = X.shape[1]
    if n - k < 5:
        return None

    XtX = X.T @ X
    # Condition number check — if XtX is ill-conditioned, inv blows up
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
    sigma2 = float(resid @ resid) / (n - k)
    if not np.isfinite(sigma2) or sigma2 < 0:
        return None
    se_diag = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    # treat coefficient is at index 1
    b = float(beta[1])
    se_b = float(se_diag[1])
    if se_b <= 0 or not np.isfinite(b) or not np.isfinite(se_b):
        return None
    if abs(b) > MAX_REASONABLE_BETA:
        return None  # fit artifact — rank deficiency we didn't catch
    t_stat = b / se_b
    p = 2 * (1 - student_t.cdf(abs(t_stat), df=n - k))
    return b, se_b, p

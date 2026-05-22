"""Substitution effect (v2 partial) — quantifying item substitution from RD.

When item i goes out of stock, undecided customers may pick another item
instead. Our default `potential_demand` corrects each item independently,
ignoring this flow, which over-estimates lost demand.

## v2 partial changes applied (2026-05-21, user feedback)

  (1) **Store-aware cutoff** — `STOCKOUT_EARLY_HOUR` is no longer hardcoded;
      derived per store from each store's measured hour-of-day sales profile
      (cumulative-sales threshold, default 75%). Falls back to 18 if no
      measured profile available. 광교 매장 measured profile에서도 18시가 도출됨.

  (2) **Co-occurrence penalty removed** — `(1 - co_occ)` was meant to
      down-weight complement pairs, but in a single-domain bakery the
      complement concept is weak (bread ↔ cake aren't complements, they're
      different purposes). Co-occ also misclassified variety seekers
      (e.g. customers buying three different sandwiches) as complements,
      shrinking real substitution signal. We now use β_RD directly.

## v2 changes NOT applied (deferred until DiD lands)

  - **OUTFLOW_CAP=0.7 retained.** User probed whether the cap was necessary,
    so we tested removing it: raw Σ_j β_RD inflated to 15–55 per item.
    Diagnosis: current RD estimator confounds "i was unavailable" with
    "this was a busy day" (all j_overall rises on stockout days from store
    traffic). Until DiD-with-hh:mm separates traffic from substitution
    (task #61), the cap remains the only thing keeping outflow interpretable.
  - **Normalize stays `/ j_overall`** (v1 form). Switching to `/ i_overall`
    only matters once the cap is removed (and even then DiD is the real fix).

## Output

    SubstitutionMatrix.coefficients : DataFrame[from_item, to_item, beta_rd, sub_rate, ...]
    SubstitutionMatrix.outflow_ratio: Series[item_id → outflow] (capped at OUTFLOW_CAP)
    SubstitutionMatrix.cutoffs      : dict[store_id → derived early-stockout hour]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Tuning knobs
TOP_ITEMS_PER_CATEGORY = 20    # only model the most-sold N per category
MIN_DAYS = 30                  # need this many days per group for RD
BETA_CAP = 1.5                 # per-pair β_RD noise filter
OUTFLOW_CAP = 0.7              # 단일 품목 outflow 최대 — 100% 가정 비현실적
                               # (deferred removal — see module docstring)
DEFAULT_CUTOFF_THRESHOLD = 0.75  # store cutoff = first hour where cumulative sales ≥ this

FALLBACK_CUTOFF_HOUR = 18      # used if no measured profile available


@dataclass
class SubstitutionMatrix:
    coefficients: pd.DataFrame    # cols: category_id, from_item, to_item, beta_rd, sub_rate, same_category
    outflow_ratio: pd.Series      # index=item_id, value=Σ_j sub_rate (uncapped raw)
    cutoffs: dict[str, int] = field(default_factory=dict)  # store_id → derived cutoff hour


def derive_store_cutoff(
    hour_profile: np.ndarray,
    *,
    threshold: float = DEFAULT_CUTOFF_THRESHOLD,
) -> int:
    """First hour where cumulative sales of the day reach `threshold`.

    The intuition: once a store has booked `threshold` of the day's
    sales, the remaining sales window is small enough that an early
    stockout effectively removes most of the day's selling opportunity.
    """
    if hour_profile.shape != (24,):
        raise ValueError(f"hour_profile must be length-24; got {hour_profile.shape}")
    total = hour_profile.sum()
    if total <= 0:
        return FALLBACK_CUTOFF_HOUR
    cum = np.cumsum(hour_profile) / total
    above = np.where(cum >= threshold)[0]
    return int(above[0]) if len(above) else 23


def compute_substitution_matrix(
    daily: pd.DataFrame,
    receipts: pd.DataFrame,
    *,
    include_inter_category: bool = True,
    hour_profiles: dict[str, np.ndarray] | None = None,
    cutoff_threshold: float = DEFAULT_CUTOFF_THRESHOLD,
) -> SubstitutionMatrix:
    """End-to-end substitution estimation from daily + receipt data.

    `hour_profiles`: per-store length-24 sales distribution (typically from
        `bonavi_loader.measure_hour_profile`). If provided, each store's
        early-stockout cutoff is derived as the first hour reaching
        `cutoff_threshold` of cumulative daily sales. If None, falls back
        to `FALLBACK_CUTOFF_HOUR`.

    `include_inter_category=True` (default): RD treats every item pair, not
        just within-category. Cross-category coefficients tend to be smaller
        but capture "wanted bread, settled for a pastry" flows.

    `receipts` is unused by the new logic but kept in the signature for
        backward compatibility — co_occ is no longer computed.
    """
    del receipts  # co_occ removed in v2

    cutoffs = _store_cutoffs(daily, hour_profiles, cutoff_threshold)
    top_items = _select_top_items(daily, n=TOP_ITEMS_PER_CATEGORY)
    daily_top = daily[daily["item_id"].isin(top_items)].copy()

    rd_coefs = _rd_pairs(daily_top, cutoffs=cutoffs, include_inter_category=include_inter_category)

    # sub_rate = β_RD clipped to [0, BETA_CAP], no co_occ penalty
    rd_coefs["sub_rate"] = rd_coefs["beta_rd"].clip(0, BETA_CAP).clip(0, 1)

    # Outflow: Σ_j sub_rate(i→j), capped at OUTFLOW_CAP (domain safety net,
    # see module docstring — removal deferred until DiD task #61).
    out = rd_coefs.groupby("from_item")["sub_rate"].sum().clip(0, OUTFLOW_CAP)
    outflow = out.reindex(top_items, fill_value=0.0)
    outflow.name = "outflow_ratio"
    return SubstitutionMatrix(coefficients=rd_coefs, outflow_ratio=outflow, cutoffs=cutoffs)


def _store_cutoffs(
    daily: pd.DataFrame,
    hour_profiles: dict[str, np.ndarray] | None,
    threshold: float,
) -> dict[str, int]:
    cutoffs: dict[str, int] = {}
    stores = daily["store_id"].unique()
    for sid in stores:
        sid_str = str(sid)
        if hour_profiles and sid_str in hour_profiles:
            cutoffs[sid_str] = derive_store_cutoff(hour_profiles[sid_str], threshold=threshold)
        else:
            cutoffs[sid_str] = FALLBACK_CUTOFF_HOUR
    return cutoffs


def _select_top_items(daily: pd.DataFrame, *, n: int) -> list[str]:
    """Top N by total sold_units within each category."""
    ranked = (
        daily.groupby(["category_id", "item_id"], observed=True)["sold_units"]
        .sum()
        .reset_index()
        .sort_values(["category_id", "sold_units"], ascending=[True, False])
    )
    return ranked.groupby("category_id", observed=True).head(n)["item_id"].astype(str).tolist()


def _rd_pairs(
    daily: pd.DataFrame,
    *,
    cutoffs: dict[str, int],
    include_inter_category: bool = True,
) -> pd.DataFrame:
    """Regression-discontinuity on daily aggregate (per-store cutoff).

    Per item pair (from_item i, to_item j):
      β_RD(i → j) = (mean(j.sold_units | i stocked out before store's cutoff)
                   - mean(j.sold_units | i did not stock out early))
                  / mean(j.sold_units)

    Normalize-by-j keeps each pair on a "j 매출 변화율" scale. The aggregate
    interpretation depends on `OUTFLOW_CAP` (without it Σ_j is dominated by
    store-traffic confound — see module docstring).
    """
    d = daily.copy()
    d["stockout_hour"] = d["stockout_time"].dt.hour
    d["_cutoff"] = d["store_id"].astype(str).map(cutoffs).fillna(FALLBACK_CUTOFF_HOUR).astype(int)
    d["i_early"] = (d["is_stockout"] & (d["stockout_hour"] < d["_cutoff"])).astype(int)

    sold_wide = d.pivot_table(
        index=["date"], columns="item_id", values="sold_units", aggfunc="sum", fill_value=0
    )
    early_wide = d.pivot_table(
        index=["date"], columns="item_id", values="i_early", aggfunc="max", fill_value=0
    )

    item_cat = d.drop_duplicates("item_id").set_index("item_id")["category_id"].to_dict()
    items = list(sold_wide.columns)
    rows = []
    for i in items:
        cat = item_cat.get(i)
        if not cat:
            continue
        if include_inter_category:
            peers = [x for x in items if x != i and item_cat.get(x)]
        else:
            peers = [x for x in items if item_cat.get(x) == cat and x != i]
        if not peers:
            continue
        treated_days = early_wide[i] == 1
        if treated_days.sum() < MIN_DAYS or (~treated_days).sum() < MIN_DAYS:
            continue
        for j in peers:
            j_mean_treated = sold_wide.loc[treated_days, j].mean()
            j_mean_control = sold_wide.loc[~treated_days, j].mean()
            j_overall = sold_wide[j].mean()
            if j_overall <= 0:
                continue
            beta = (j_mean_treated - j_mean_control) / j_overall
            rows.append({
                "category_id": cat,
                "to_category_id": item_cat[j],
                "from_item": i,
                "to_item": j,
                "same_category": cat == item_cat[j],
                "beta_rd": beta,
                "treated_days": int(treated_days.sum()),
                "j_mean_treated": j_mean_treated,
                "j_mean_control": j_mean_control,
            })
    return pd.DataFrame(rows)


def adjust_lost_units(
    lost: pd.DataFrame,
    outflow_ratio: pd.Series,
    *,
    cap: float | None = None,
) -> pd.DataFrame:
    """Subtract substitution outflow from per-row lost_units estimates.

    lost: estimated_lost_demand output with columns [store_id, item_id, date, lost_units]
    outflow_ratio: per-item fraction of lost_units that flowed to other items
    cap: optional ceiling for outflow_ratio (e.g. 0.7 for v1-style safety net,
        None to use raw RD signal). None is the new v2 default.

    adjusted_lost = lost_units × (1 − min(outflow_ratio, cap))
    """
    df = lost.copy()
    df["item_id"] = df["item_id"].astype(str)
    raw = df["item_id"].map(outflow_ratio.to_dict()).fillna(0.0)
    if cap is not None:
        df["outflow_ratio"] = raw.clip(0.0, cap)
    else:
        df["outflow_ratio"] = raw.clip(lower=0.0)
    df["lost_units_adjusted"] = df["lost_units"] * (1 - df["outflow_ratio"])
    return df


def sensitivity_summary(outflow_ratio: pd.Series) -> pd.DataFrame:
    """Show how the outflow distribution shifts under different caps.

    Useful for the user to see how much the cap choice changes the
    interpretation (e.g. lost-revenue estimate).
    """
    caps = [None, 0.7, 0.5, 0.3, 0.0]
    rows = []
    for c in caps:
        if c is None:
            shrunk = outflow_ratio.clip(lower=0.0)
            label = "no_cap (raw)"
        else:
            shrunk = outflow_ratio.clip(0.0, c)
            label = f"cap={c}"
        rows.append({
            "cap": label,
            "n_items": len(shrunk),
            "mean_outflow": shrunk.mean(),
            "median_outflow": shrunk.median(),
            "pct_above_0_5": (shrunk > 0.5).mean(),
            "pct_above_0_7": (shrunk > 0.7).mean(),
            "max_outflow": shrunk.max(),
            "implied_mean_remain": 1 - shrunk.mean(),  # 떠난 unit 비율
        })
    return pd.DataFrame(rows)

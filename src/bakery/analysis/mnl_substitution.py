"""Multinomial Logit Choice Model for substitution matrix estimation.

Treats each single-item receipt as one customer's discrete choice over the
items that were actually available at that time. We learn item-level utilities
α_i via per-category conditional logit:

    P(choose i | available set A_t)  =  exp(α_i) / Σ_{k ∈ A_t} exp(α_k)

Once utilities are fit, substitution coefficients come from counterfactual
predictions — "what if item i had been out at this moment?":

    s(i → j)  =  P(choose j | A_t \\ {i})  −  P(choose j | A_t)

Aggregated across all receipts where i was originally chosen, this gives a
per-pair flow estimate that's grounded in microeconomic theory (McFadden, 1974).
Compared to the daily-aggregate RD module, MNL:

  + uses microscopic receipt-level signal (24만 영수증)
  + IIA-grounded substitution (transferable across stores)
  − stronger IIA assumption (may over-state substitution within homogenous bundles)
  − single-item receipts only (52% of all receipts)

PoC: caps outflow at OUTFLOW_CAP=0.7 to mirror the RD module's assumption
that some lost customers leave the store entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

OUTFLOW_CAP = 0.7
TOP_ITEMS_PER_CATEGORY = 20
MIN_RECEIPTS = 100  # need this many per item to identify utility


@dataclass
class MnlResult:
    utilities: pd.DataFrame             # cols: category_id, item_id, utility
    substitution: pd.DataFrame          # cols: category_id, from_item, to_item, s_share, s_raw
    outflow_ratio: pd.Series            # index=item_id, value=outflow under MNL (IIA cap)


def fit_mnl_per_category(
    receipts: pd.DataFrame,
    daily: pd.DataFrame,
) -> MnlResult:
    """Fit conditional logit per category on single-item receipts.

    receipts: long-form (receipt_id, date, item_id) — one row per line item.
    daily: DAILY_COLUMNS frame for the same store, used to construct each day's
           availability set (item is available before its stockout_time).
    """
    # Single-item receipts only — cleanest "this is what the customer wanted"
    single = receipts.groupby("receipt_id").filter(lambda g: g["item_id"].nunique() == 1)
    single = single.drop_duplicates(subset=["receipt_id"])  # one row per receipt

    # Item ↔ category lookup
    item_cat = daily.drop_duplicates("item_id").set_index("item_id")["category_id"].to_dict()
    single["category_id"] = single["item_id"].map(item_cat)
    single = single.dropna(subset=["category_id"])

    # Per-day availability — `is_stockout=False` is the loose proxy for "in stock
    # all day"; we then OR-in any (date, item) that appears in a receipt, since a
    # sale is hard evidence that the item was available at that moment.
    avail = daily[["store_id", "date", "item_id", "is_stockout"]].copy()
    avail["available"] = (~avail["is_stockout"]).astype(int)
    avail_lookup = (
        avail.pivot_table(
            index="date", columns="item_id", values="available", fill_value=0, aggfunc="max"
        )
    ).astype(int)
    sold_dates = single.groupby("item_id")["date"].apply(set).to_dict()
    for it, dates in sold_dates.items():
        if it in avail_lookup.columns:
            for dt in dates:
                if dt in avail_lookup.index:
                    avail_lookup.at[dt, it] = 1

    util_rows: list[dict] = []
    sub_rows: list[dict] = []
    outflow_partial: dict[str, float] = {}

    for cat, cat_choices in single.groupby("category_id"):
        items = _top_items_in_category(daily, cat, n=TOP_ITEMS_PER_CATEGORY)
        if len(items) < 3:
            continue
        item_idx = {it: i for i, it in enumerate(items)}
        cat_choices = cat_choices[cat_choices["item_id"].isin(items)]
        counts = cat_choices["item_id"].value_counts()
        items = [it for it in items if counts.get(it, 0) >= MIN_RECEIPTS]
        if len(items) < 3:
            continue
        item_idx = {it: i for i, it in enumerate(items)}
        cat_choices = cat_choices[cat_choices["item_id"].isin(items)]
        n_items = len(items)

        # Build choice array + availability masks
        choice_idx = cat_choices["item_id"].map(item_idx).to_numpy()
        avail_for_cat = avail_lookup[items].reindex(cat_choices["date"]).fillna(0).to_numpy()

        # Optimize utilities (fix α_0 = 0 for identification)
        def neg_ll(u_free: np.ndarray) -> float:
            u = np.concatenate([[0.0], u_free])
            u_avail = np.where(avail_for_cat > 0, u, -np.inf)
            ls = logsumexp(u_avail, axis=1)
            return -(u[choice_idx] - ls).sum()

        x0 = np.zeros(n_items - 1)
        res = minimize(neg_ll, x0, method="L-BFGS-B")
        u = np.concatenate([[0.0], res.x])

        for item, alpha in zip(items, u):
            util_rows.append({"category_id": cat, "item_id": item, "utility": float(alpha)})

        # Substitution: P_minus_i(j) - P_full(j) averaged across receipts where i was available.
        # `s_raw` is the absolute point change in j's selection probability.
        # `s_share` normalizes by P_full(i) — "of customers who wanted i, what
        # fraction goes to j?" This is what's directly comparable to the RD
        # module's per-pair sub_rate. IIA implies Σ_j s_share = 1.0.
        exp_u = np.exp(u)  # (n_items,)
        for i_idx, i in enumerate(items):
            mask_i_available = avail_for_cat[:, i_idx] > 0
            if not mask_i_available.any():
                continue
            relevant_avail = avail_for_cat[mask_i_available]
            denom_full = (relevant_avail * exp_u).sum(axis=1, keepdims=True)
            denom_full = np.where(denom_full > 0, denom_full, 1.0)
            p_full = (relevant_avail * exp_u) / denom_full
            avail_minus = relevant_avail.copy()
            avail_minus[:, i_idx] = 0
            denom_minus = (avail_minus * exp_u).sum(axis=1, keepdims=True)
            denom_minus = np.where(denom_minus > 0, denom_minus, 1.0)
            p_minus = (avail_minus * exp_u) / denom_minus
            diff = (p_minus - p_full).mean(axis=0)
            pi_avg = float(p_full[:, i_idx].mean())  # average P(i) over receipts where i avail
            for j_idx, j in enumerate(items):
                if j == i:
                    continue
                share = float(diff[j_idx] / pi_avg) if pi_avg > 1e-9 else 0.0
                sub_rows.append({
                    "category_id": cat,
                    "from_item": i, "to_item": j,
                    "s_raw": float(diff[j_idx]),
                    "s_share": share,
                })
            # MNL/IIA has no "no purchase" option, so theoretically outflow = 1.0.
            # We cap at OUTFLOW_CAP=0.7 to mirror the RD module's assumption that
            # some customers leave the store. Uniform across items — MNL is
            # informative for substitution *direction*, not the leave-or-stay split.
            outflow_partial[i] = OUTFLOW_CAP

    utilities = pd.DataFrame(util_rows)
    substitution = pd.DataFrame(sub_rows)
    outflow_ratio = pd.Series(outflow_partial, name="outflow_ratio").astype(float)
    return MnlResult(utilities=utilities, substitution=substitution, outflow_ratio=outflow_ratio)


def _top_items_in_category(daily: pd.DataFrame, cat: str, *, n: int) -> list[str]:
    cat_daily = daily[daily["category_id"] == cat]
    ranking = (
        cat_daily.groupby("item_id")["sold_units"].sum().sort_values(ascending=False)
    )
    return ranking.head(n).index.astype(str).tolist()

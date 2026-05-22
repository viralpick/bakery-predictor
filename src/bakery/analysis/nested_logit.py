"""Nested Logit choice model — relaxes IIA across nests (= categories).

Standard MNL forces a customer who loses item *i* to redistribute to every
other item in strict proportion to its utility. In practice we expect a
broken-bread shopper to fall back on **another bread** before reaching for a
cake. The nested logit captures that by giving each nest *g* a dissimilarity
parameter `λ_g ∈ (0, 1]`:

    P(i)           = P(g(i)) · P(i | g(i))
    P(g)           = exp(λ_g · IV_g) / Σ_h exp(λ_h · IV_h)
    P(i | g)       = exp(α_i / λ_g) / Σ_{j∈g} exp(α_j / λ_g)
    IV_g           = log Σ_{j∈g} exp(α_j / λ_g)         (inclusive value)

λ_g → 1 collapses to MNL. λ_g → 0 means perfect within-nest substitution.

Differences vs `mnl_substitution.py`:
  + cross-category fit (one joint log-likelihood over all single-item receipts)
  + within-nest substitution stronger than cross-nest (controlled by λ_g)
  + can produce inter-category substitution shares too, not just within
  − more parameters (n_items + n_categories) and a constrained optimizer
  − λ_g ∈ (0, 1] needs to be enforced (we reparameterize through softplus)

PoC: same single-item filter, same daily availability proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUTFLOW_CAP = 0.7
TOP_ITEMS_PER_CATEGORY = 20
MIN_RECEIPTS = 100


@dataclass
class NestedLogitResult:
    utilities: pd.DataFrame             # cols: category_id, item_id, utility (α_i)
    lambdas: pd.Series                  # index=category_id, value=λ_g
    substitution: pd.DataFrame          # cols: from_item, from_cat, to_item, to_cat,
                                        #       s_raw, s_share, same_nest
    outflow_ratio: pd.Series            # index=item_id, value=OUTFLOW_CAP


def fit_nested_logit(
    receipts: pd.DataFrame,
    daily: pd.DataFrame,
) -> NestedLogitResult:
    """Joint cross-category nested logit fit.

    Returns the same shape as MnlResult plus a per-nest λ series. We use
    softplus(x) + ε to keep λ strictly positive, and we keep λ ≤ 1 implicit
    by penalizing values above 1 in the optimizer (a soft barrier — for PoC
    we'd otherwise need an SLSQP constrained solver).
    """
    # Single-item receipts; map item → category
    single = receipts.groupby("receipt_id").filter(lambda g: g["item_id"].nunique() == 1)
    single = single.drop_duplicates(subset=["receipt_id"])
    item_cat = daily.drop_duplicates("item_id").set_index("item_id")["category_id"].to_dict()
    single["category_id"] = single["item_id"].map(item_cat)
    single = single.dropna(subset=["category_id"])

    # Choice universe: top N per category, filtered to those with enough receipts
    items: list[str] = []
    cats: list[str] = []
    for cat, sub in single.groupby("category_id"):
        ranking = (
            daily[daily["category_id"] == cat]
            .groupby("item_id")["sold_units"].sum().sort_values(ascending=False)
        )
        candidates = ranking.head(TOP_ITEMS_PER_CATEGORY).index.astype(str).tolist()
        counts = sub["item_id"].value_counts()
        kept = [it for it in candidates if counts.get(it, 0) >= MIN_RECEIPTS]
        items.extend(kept)
        cats.extend([cat] * len(kept))

    if len(items) < 5:
        raise ValueError("Not enough items survived MIN_RECEIPTS for nested logit.")

    item_idx = {it: i for i, it in enumerate(items)}
    cat_codes = sorted(set(cats))
    cat_idx = {c: i for i, c in enumerate(cat_codes)}
    item_to_nest = np.array([cat_idx[c] for c in cats])
    n_items = len(items)
    n_nests = len(cat_codes)

    # Filter receipts to items in the choice universe
    use = single[single["item_id"].isin(items)].copy()
    choice_idx = use["item_id"].map(item_idx).to_numpy()

    # Availability matrix per receipt
    avail = daily[["date", "item_id", "is_stockout"]].copy()
    avail["available"] = (~avail["is_stockout"]).astype(int)
    avail_lookup = avail.pivot_table(
        index="date", columns="item_id", values="available", fill_value=0, aggfunc="max"
    ).astype(int)
    # OR-in sold (date, item) — sale is hard evidence of availability
    for it, dts in single.groupby("item_id")["date"].apply(set).items():
        if it in avail_lookup.columns:
            for dt in dts:
                if dt in avail_lookup.index:
                    avail_lookup.at[dt, it] = 1
    missing = [it for it in items if it not in avail_lookup.columns]
    if missing:
        for it in missing:
            avail_lookup[it] = 0
    avail_mat = avail_lookup[items].reindex(use["date"]).fillna(0).to_numpy().astype(int)

    # Param vector: u_free (n_items - 1, with first item α=0) + lam_raw (n_nests)
    # λ_g = sigmoid(lam_raw[g]) so λ ∈ (0, 1). We add a tiny floor to avoid 1/λ → ∞.
    LAM_FLOOR = 0.05

    def _unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.concatenate([[0.0], theta[: n_items - 1]])
        lam_raw = theta[n_items - 1 :]
        lam = LAM_FLOOR + (1.0 - LAM_FLOOR) / (1.0 + np.exp(-lam_raw))
        return u, lam

    def neg_log_lik(theta: np.ndarray) -> float:
        u, lam = _unpack(theta)
        lam_per_item = lam[item_to_nest]                   # (n_items,)
        scaled_u = u / lam_per_item                         # (n_items,)
        # Per-receipt: mask unavailable items with -inf in scaled_u
        # Then for each nest g, IV_g = log Σ_{j∈g, avail} exp(scaled_u_j)
        scaled_avail = np.where(avail_mat > 0, scaled_u, -np.inf)   # (N, n_items)
        # Compute IV per nest per receipt
        iv = np.full((scaled_avail.shape[0], n_nests), -np.inf)
        for g in range(n_nests):
            mask = item_to_nest == g
            iv[:, g] = _logsumexp_safe(scaled_avail[:, mask], axis=1)
        # Denominator: log Σ_g exp(λ_g · IV_g) — but IV may be -inf for empty nests
        weighted = lam * iv                                # (N, n_nests)
        log_denom = _logsumexp_safe(weighted, axis=1)      # (N,)
        # Numerator for the chosen item i, in nest g=g(i):
        #   log P(i) = α_i - λ_g · log(λ_g) · ... — easier form:
        #     log P(i) = (α_i / λ_g) + (λ_g - 1) · IV_{g(i)} - log_denom
        chosen_lam = lam[item_to_nest[choice_idx]]
        chosen_iv = iv[np.arange(len(choice_idx)), item_to_nest[choice_idx]]
        log_pi = (
            u[choice_idx] / chosen_lam
            + (chosen_lam - 1.0) * chosen_iv
            - log_denom
        )
        return -float(log_pi.sum())

    x0 = np.zeros(n_items - 1 + n_nests)
    # warm-start: λ ≈ 0.7 (logit ≈ 0.85)
    x0[n_items - 1 :] = 0.85
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 500})
    u, lam = _unpack(res.x)

    # Substitution matrix (counterfactual i removed)
    sub_rows: list[dict] = []
    lam_per_item = lam[item_to_nest]
    scaled_u = u / lam_per_item
    for i_idx, src in enumerate(items):
        mask_src_avail = avail_mat[:, i_idx] > 0
        if not mask_src_avail.any():
            continue
        relevant_avail = avail_mat[mask_src_avail]
        # Original probabilities
        p_full = _nested_probs(u, lam, item_to_nest, relevant_avail, scaled_u)
        # Counterfactual: src unavailable
        avail_minus = relevant_avail.copy()
        avail_minus[:, i_idx] = 0
        p_minus = _nested_probs(u, lam, item_to_nest, avail_minus, scaled_u)
        diff = (p_minus - p_full).mean(axis=0)
        pi_avg = float(p_full[:, i_idx].mean())
        src_nest = item_to_nest[i_idx]
        for j_idx, tgt in enumerate(items):
            if j_idx == i_idx:
                continue
            share = float(diff[j_idx] / pi_avg) if pi_avg > 1e-9 else 0.0
            sub_rows.append({
                "from_item": src, "from_cat": cat_codes[src_nest],
                "to_item": tgt, "to_cat": cat_codes[item_to_nest[j_idx]],
                "s_raw": float(diff[j_idx]),
                "s_share": share,
                "same_nest": bool(item_to_nest[j_idx] == src_nest),
            })

    utilities = pd.DataFrame({
        "category_id": [cat_codes[item_to_nest[i]] for i in range(n_items)],
        "item_id": items,
        "utility": u,
    })
    lambdas = pd.Series(lam, index=cat_codes, name="lambda_g")
    substitution = pd.DataFrame(sub_rows)
    outflow_ratio = pd.Series({it: OUTFLOW_CAP for it in items}, name="outflow_ratio")
    return NestedLogitResult(
        utilities=utilities, lambdas=lambdas,
        substitution=substitution, outflow_ratio=outflow_ratio,
    )


def _logsumexp_safe(arr: np.ndarray, axis: int) -> np.ndarray:
    """logsumexp that returns -inf when all entries along axis are -inf."""
    m = np.max(arr, axis=axis, keepdims=True)
    finite = np.isfinite(m)
    safe_m = np.where(finite, m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sum_term = np.sum(np.exp(arr - safe_m), axis=axis, keepdims=True)
        log_sum = np.log(np.where(sum_term > 0, sum_term, 1.0))
    out = (safe_m + log_sum).squeeze(axis)
    bad = ~finite.squeeze(axis)
    return np.where(bad, -np.inf, out)


def _nested_probs(
    u: np.ndarray,
    lam: np.ndarray,
    item_to_nest: np.ndarray,
    avail_mat: np.ndarray,
    scaled_u: np.ndarray,
) -> np.ndarray:
    """Return P(i) for each receipt × item under the fitted nested logit.

    Vectorized over receipts. Items with avail_mat == 0 get probability 0.
    """
    n_receipts = avail_mat.shape[0]
    n_items = avail_mat.shape[1]
    n_nests = lam.shape[0]
    scaled_avail = np.where(avail_mat > 0, scaled_u, -np.inf)
    iv = np.full((n_receipts, n_nests), -np.inf)
    for g in range(n_nests):
        mask = item_to_nest == g
        iv[:, g] = _logsumexp_safe(scaled_avail[:, mask], axis=1)
    weighted = lam * iv
    log_denom = _logsumexp_safe(weighted, axis=1)
    # log P(i) = scaled_u_i + (λ_{g(i)} - 1) · IV_{g(i)} - log_denom
    lam_per_item = lam[item_to_nest]
    iv_per_item = iv[:, item_to_nest]                          # (N, n_items)
    log_p = scaled_avail + (lam_per_item - 1.0) * iv_per_item - log_denom[:, None]
    p = np.exp(log_p)
    p = np.where(avail_mat > 0, p, 0.0)
    # Guard against tiny numerical drift
    row_sums = p.sum(axis=1, keepdims=True)
    p = np.where(row_sums > 0, p / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    return p

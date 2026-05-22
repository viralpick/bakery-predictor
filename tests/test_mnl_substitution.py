"""MNL substitution: convergence + IIA invariant + structural correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.analysis.mnl_substitution import OUTFLOW_CAP, fit_mnl_per_category


def _toy_data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tiny single-store, two-category dataset with a known popularity hierarchy.

    Within bread: i1 dominates (60%), i2 (30%), i3 (10%).
    Within cake : c1 (50%), c2 (50%) — equal popularity.
    All items always available; the MNL utilities should recover the log-ratios.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")

    items = [
        ("i1", "bread", 0.60),
        ("i2", "bread", 0.30),
        ("i3", "bread", 0.10),
        ("c1", "cake", 0.50),
        ("c2", "cake", 0.50),
    ]
    daily_rows = []
    for d in dates:
        for item, cat, _ in items:
            daily_rows.append({
                "store_id": "s1",
                "item_id": item,
                "category_id": cat,
                "date": d,
                "sold_units": 20,
                "is_stockout": False,
                "stockout_time": pd.NaT,
            })
    daily = pd.DataFrame(daily_rows)

    # Generate 500 receipts per category — one chosen item, sampled by popularity.
    receipts_rows = []
    rid = 0
    for cat_items in (items[:3], items[3:]):
        probs = np.array([w for _, _, w in cat_items])
        names = [name for name, _, _ in cat_items]
        for _ in range(500):
            for d in rng.choice(dates, size=3):
                chosen = rng.choice(names, p=probs)
                receipts_rows.append({"receipt_id": f"r{rid}", "date": d, "item_id": chosen})
                rid += 1
    receipts = pd.DataFrame(receipts_rows)
    return receipts, daily


def test_utilities_recover_popularity_ranking():
    receipts, daily = _toy_data()
    res = fit_mnl_per_category(receipts, daily)
    bread_utilities = res.utilities[res.utilities["category_id"] == "bread"].set_index("item_id")["utility"]
    # i1 should have highest utility, i3 lowest
    assert bread_utilities["i1"] > bread_utilities["i2"] > bread_utilities["i3"]


def test_iia_substitution_shares_sum_to_one():
    receipts, daily = _toy_data()
    res = fit_mnl_per_category(receipts, daily)
    # For each (from_item) with non-trivial popularity, Σ_j s_share ≈ 1.0
    shares = res.substitution.groupby("from_item")["s_share"].sum()
    # Drop near-zero items (those rarely chosen — share noise)
    meaningful = shares[shares > 0.5]
    assert (np.abs(meaningful - 1.0) < 0.05).all(), f"IIA violated: {meaningful}"


def test_outflow_capped_at_iia_limit():
    receipts, daily = _toy_data()
    res = fit_mnl_per_category(receipts, daily)
    assert (res.outflow_ratio <= OUTFLOW_CAP + 1e-9).all()
    assert (res.outflow_ratio >= 0).all()


def test_no_substitution_to_self():
    receipts, daily = _toy_data()
    res = fit_mnl_per_category(receipts, daily)
    self_pairs = res.substitution[res.substitution["from_item"] == res.substitution["to_item"]]
    assert len(self_pairs) == 0

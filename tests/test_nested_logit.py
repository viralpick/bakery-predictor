"""Nested logit: IIA-relaxation invariants on synthetic ground-truth data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.analysis.nested_logit import OUTFLOW_CAP, fit_nested_logit


def _toy_two_nest(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two nests, three items each. Within-nest substitution should outweigh cross-nest.

    Nest 'bread' : b1 (60%), b2 (30%), b3 (10%) — share Σ = 1 within nest
    Nest 'cake'  : c1 (50%), c2 (30%), c3 (20%)
    Two nests overall 50/50 weight.

    All items available all days. Single-item receipts only.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    items = [
        ("b1", "bread", 0.30), ("b2", "bread", 0.15), ("b3", "bread", 0.05),
        ("c1", "cake", 0.25),  ("c2", "cake", 0.15),  ("c3", "cake", 0.10),
    ]
    daily_rows = [
        {"store_id": "s1", "item_id": it, "category_id": cat, "date": d,
         "sold_units": 20, "is_stockout": False, "stockout_time": pd.NaT}
        for it, cat, _ in items for d in dates
    ]
    daily = pd.DataFrame(daily_rows)

    probs = np.array([w for _, _, w in items])
    probs = probs / probs.sum()
    names = [it for it, _, _ in items]
    receipts_rows = []
    for rid in range(3000):
        chosen = rng.choice(names, p=probs)
        d = rng.choice(dates)
        receipts_rows.append({"receipt_id": f"r{rid}", "date": d, "item_id": chosen})
    receipts = pd.DataFrame(receipts_rows)
    return receipts, daily


def test_lambdas_in_unit_interval():
    receipts, daily = _toy_two_nest()
    res = fit_nested_logit(receipts, daily)
    assert (res.lambdas > 0).all()
    assert (res.lambdas <= 1.0 + 1e-9).all()


def test_iia_relaxed_substitution_sums_to_one():
    receipts, daily = _toy_two_nest()
    res = fit_nested_logit(receipts, daily)
    shares = res.substitution.groupby("from_item")["s_share"].sum()
    assert (np.abs(shares - 1.0) < 0.02).all(), f"IIA total broken: {shares}"


def test_within_nest_share_at_least_matches_cross_nest():
    """Within-nest substitution should be ≥ cross-nest (because the DGP is unstructured,
    we use ≥ rather than >; structural inequality only emerges with non-uniform data)."""
    receipts, daily = _toy_two_nest()
    res = fit_nested_logit(receipts, daily)
    within = res.substitution[res.substitution["same_nest"]]["s_share"].mean()
    cross = res.substitution[~res.substitution["same_nest"]]["s_share"].mean()
    assert within >= cross * 0.9, f"within {within:.4f} < cross {cross:.4f}"


def test_outflow_uniform_cap():
    receipts, daily = _toy_two_nest()
    res = fit_nested_logit(receipts, daily)
    assert (res.outflow_ratio == OUTFLOW_CAP).all()

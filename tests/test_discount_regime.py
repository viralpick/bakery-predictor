"""Tests for the discount-depth regime-shift re-validation (다중시각 재검증 ①).

The 2025-01-17 company-wide depth cut (30%→20%) is not a clean natural
experiment (no control store, secular demand swings). The one internally
controlled signal is the *composition* closing_share = closing/(normal+closing):
a store-wide demand shock moves normal and closing proportionally, but a change
in discount attractiveness moves the split. These tests pin the regime-shift
estimator on synthetic panels with a known jump / known null.
"""

import numpy as np
import pandas as pd

from bakery.analysis.discount_regime import (
    DEFAULT_CUT_DATE,
    build_regime_panel,
    fit_regime_shift,
    placebo_shifts,
    run_discount_regime,
)

CATEGORY = "bread"


def _make_rows(share_pre, share_post, *, n_items=20, start="2023-01-01",
               end="2025-06-30", cut=DEFAULT_CUT_DATE, total=100, noise_sd=0.0,
               seed=0):
    """Build a waste_alpha-shaped item-day panel realizing a target closing_share.

    closing_share = share_pre before `cut`, share_post on/after. Per-item and
    monthly deterministic offsets are added (absorbed by item/month FE); optional
    Gaussian noise gives nonzero residual variance so HC3 SE is well-defined.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    cut_ts = pd.Timestamp(cut)
    recs = []
    for i in range(n_items):
        item_offset = 0.02 * (i - n_items / 2) / n_items  # small per-item level
        for d in dates:
            base = share_post if d >= cut_ts else share_pre
            month_offset = 0.01 * np.sin(d.month)
            share = base + item_offset + month_offset
            if noise_sd:
                share += rng.normal(0, noise_sd)
            share = min(max(share, 0.0), 0.95)
            closing = round(share * total)
            recs.append({
                "date": d, "item_id": f"I{i:03d}",
                "normal_qty": float(total - closing), "closing_qty": float(closing),
                "made": float(total + 10), "out": 10.0,
            })
    return pd.DataFrame(recs)


def _cat_map(rows):
    return pd.Series({iid: CATEGORY for iid in rows["item_id"].unique()})


def test_build_regime_panel_computes_closing_share_and_post_flag():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2025-03-01"]),
        "item_id": ["I000", "I000"],
        "normal_qty": [80.0, 60.0],
        "closing_qty": [20.0, 40.0],
        "made": [110.0, 110.0],
        "out": [10.0, 10.0],
    })
    panel = build_regime_panel(rows, _cat_map(rows), CATEGORY)
    assert list(panel["closing_share"]) == [0.2, 0.4]
    assert list(panel["post_cut"]) == [0, 1]


def test_build_regime_panel_excludes_transition_month():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2025-01-20", "2025-03-01"]),
        "item_id": ["I000", "I000", "I000"],
        "normal_qty": [80.0, 50.0, 60.0],
        "closing_qty": [20.0, 50.0, 40.0],
        "made": [110.0, 110.0, 110.0],
        "out": [10.0, 10.0, 10.0],
    })
    panel = build_regime_panel(rows, _cat_map(rows), CATEGORY)
    # the 2025-01 transition row (partial depth) is dropped
    assert pd.Timestamp("2025-01-20") not in set(panel["date"])
    assert len(panel) == 2


def test_fit_regime_shift_detects_real_jump():
    rows = _make_rows(share_pre=0.10, share_post=0.30, noise_sd=0.01, seed=1)
    panel = build_regime_panel(rows, _cat_map(rows), CATEGORY)
    res = fit_regime_shift(panel, "closing_share")
    assert not res.ill_posed
    # jump of ~0.20 in closing_share; CI must exclude 0
    assert 0.15 < res.beta < 0.25
    assert res.ci_low > 0


def test_fit_regime_shift_null_when_flat():
    rows = _make_rows(share_pre=0.12, share_post=0.12, noise_sd=0.02, seed=2)
    panel = build_regime_panel(rows, _cat_map(rows), CATEGORY)
    res = fit_regime_shift(panel, "closing_share")
    assert not res.ill_posed
    # no true jump; CI must include 0
    assert res.ci_low < 0 < res.ci_high


def test_placebo_shifts_are_null_on_flat_data():
    rows = _make_rows(share_pre=0.12, share_post=0.12, noise_sd=0.02, seed=3)
    placebo = placebo_shifts(
        rows, _cat_map(rows), CATEGORY,
        placebo_cut_dates=["2023-07-01", "2024-01-17", "2024-07-01"],
    )
    assert len(placebo) == 3
    # every placebo break on flat data must be a null (CI spans 0)
    for res in placebo:
        assert res.ci_low < 0 < res.ci_high


def test_fit_regime_shift_stable_with_many_items():
    """Regression: 200+ items must not blow up the design matrix (item FE via
    within-demeaning, not a 200-column dummy block that trips the condition guard)."""
    # multi-year span so every calendar month appears in both regimes (else
    # post-only months are collinear with post_cut — a data-shape issue, not the
    # design-matrix blow-up this regression guards against).
    rows = _make_rows(share_pre=0.10, share_post=0.30, n_items=220,
                      start="2023-06-01", end="2025-04-30", noise_sd=0.01, seed=9)
    panel = build_regime_panel(rows, _cat_map(rows), CATEGORY)
    res = fit_regime_shift(panel, "closing_share")
    assert not res.ill_posed
    assert 0.15 < res.beta < 0.25


def test_run_discount_regime_verdict_depth_invariant_on_flat():
    rows = _make_rows(share_pre=0.12, share_post=0.12, noise_sd=0.02, seed=4)
    out = run_discount_regime(
        rows, _cat_map(rows), CATEGORY,
        placebo_cut_dates=["2023-07-01", "2024-01-17"],
    )
    assert out["verdict"] == "depth_invariant"
    assert out["closing_share"].ci_low < 0 < out["closing_share"].ci_high

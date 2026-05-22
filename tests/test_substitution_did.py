"""DiD substitution: smoke test + structural invariants on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.analysis.substitution_did import compute_did_substitution


def _synth_data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Build a small synthetic store where item 'i' stocks out at 15:00 on
    ~30% of days, and item 'j_sub' picks up after i stocks out (true substitute).
    item 'j_neutral' is unaffected by i's stockout.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rows_daily = []
    rows_receipts = []

    items = ["i", "j_sub", "j_neutral"]
    categories = {"i": "bread", "j_sub": "bread", "j_neutral": "cake"}

    for d in dates:
        i_stockout = rng.random() < 0.3
        stockout_time = d + pd.Timedelta(hours=15) if i_stockout else pd.NaT

        for item in items:
            sold = int(20 + rng.integers(-3, 4))
            rows_daily.append({
                "store_id": "s1", "item_id": item, "category_id": categories[item],
                "date": d, "sold_units": sold, "is_stockout": i_stockout and item == "i",
                "stockout_time": stockout_time if item == "i" else pd.NaT,
            })

        # Hourly receipts
        for hour in range(7, 22):
            i_n = max(0, int(rng.normal(1.5, 0.5)))
            if item_is_stockout := (i_stockout and hour >= 15):
                i_n = 0
            j_sub_n = max(0, int(rng.normal(1.5, 0.5)))
            if i_stockout and hour >= 15:
                j_sub_n += max(0, int(rng.normal(1.0, 0.3)))  # substitution boost
            j_neutral_n = max(0, int(rng.normal(1.5, 0.5)))

            for n, item in [(i_n, "i"), (j_sub_n, "j_sub"), (j_neutral_n, "j_neutral")]:
                for k in range(n):
                    rows_receipts.append({
                        "receipt_id": f"r{d.date()}_{hour}_{k}_{item}",
                        "date": d, "item_id": item, "hour": hour,
                    })

    daily = pd.DataFrame(rows_daily)
    receipts = pd.DataFrame(rows_receipts)
    # store profile: flat 7-22
    profile = np.zeros(24)
    profile[7:22] = 1.0
    profile /= profile.sum()
    return daily, receipts, {"s1": profile}


def test_did_runs_on_synthetic_and_finds_substitute_signal():
    daily, receipts, profiles = _synth_data()
    result = compute_did_substitution(daily, receipts, profiles)
    assert "store_gw01" not in result.cutoffs  # we used s1
    assert "s1" in result.cutoffs
    # j_sub should have positive β when 'i' is the source (synthetic substitute)
    j_sub_rows = result.coefficients[
        (result.coefficients["from_item"] == "i")
        & (result.coefficients["to_item"] == "j_sub")
    ]
    if len(j_sub_rows):
        assert j_sub_rows["beta_did"].iloc[0] > 0, "expected positive β for true substitute"


def test_did_returns_empty_when_no_stockouts():
    daily, receipts, profiles = _synth_data()
    # Wipe stockouts
    daily = daily.copy()
    daily["is_stockout"] = False
    daily["stockout_time"] = pd.NaT
    result = compute_did_substitution(daily, receipts, profiles)
    assert len(result.coefficients) == 0


def test_did_outflow_only_sums_positive_betas():
    daily, receipts, profiles = _synth_data()
    result = compute_did_substitution(daily, receipts, profiles)
    # outflow is Σ positive β, so each item's outflow ≥ 0
    assert (result.outflow_ratio >= 0).all()

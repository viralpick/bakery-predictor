"""수요이전 흡수 검증 (W0 게이트) — 패널 빌더 + 회귀/TOST 테스트.

합성 fixture로 완전흡수(β≈0)와 walk-away(β<0) 시나리오를 심어 회귀가
부호를 회복하는지 검증한다. leakage-safe baseline은 미래 미참조를 확인."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.analysis.demand_absorption import (
    DEFAULT_CLOSE_HOUR, BASELINE_WEEKS, build_absorption_panel,
    AbsorptionResult, EQUIV_FRAC, fit_absorption,
)


def _daily_two_items_one_cat(n_weeks: int = 12, seed: int = 0) -> pd.DataFrame:
    """1 store, 1 category, 2 items, daily rows over n_weeks. No stockouts (base)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks * 7, freq="D")
    rows = []
    for d in dates:
        for item in ("i1", "i2"):
            rows.append({
                "store_id": "s1", "item_id": item, "category_id": "bread",
                "date": d, "sold_units": float(rng.integers(8, 12)),
                "is_stockout": False, "stockout_time": pd.NaT,
            })
    return pd.DataFrame(rows)


def test_panel_has_expected_columns_and_grain():
    panel = build_absorption_panel(_daily_two_items_one_cat())
    assert list(panel.columns) == [
        "store_id", "category_id", "date", "cat_sold", "stockout_hours",
        "other_cat_sold", "cat_baseline", "dow", "month", "trend"]
    # one row per (store, category, date) that survives baseline warmup
    assert (panel.groupby(["store_id", "category_id", "date"]).size() == 1).all()


def test_panel_stockout_hours_from_earliest():
    daily = _daily_two_items_one_cat(n_weeks=12)
    # inject a stockout: i1 sells out at 14:00 on one date
    d0 = daily["date"].max()
    mask = (daily["item_id"] == "i1") & (daily["date"] == d0)
    daily.loc[mask, "is_stockout"] = True
    daily.loc[mask, "stockout_time"] = pd.Timestamp(f"{d0.date()} 14:00")
    panel = build_absorption_panel(daily)
    row = panel[(panel["category_id"] == "bread") & (panel["date"] == d0)].iloc[0]
    assert row["stockout_hours"] == pytest.approx(DEFAULT_CLOSE_HOUR - 14.0)  # 8.0


def test_panel_baseline_is_leakage_safe():
    """cat_baseline for date d must use only same-dow rows strictly before d."""
    daily = _daily_two_items_one_cat(n_weeks=12)
    panel = build_absorption_panel(daily).sort_values("date")
    # first BASELINE_WEEKS of each dow are dropped (no prior window)
    first_date = daily["date"].min()
    assert (panel["date"] > first_date).all()
    # baseline of a row equals mean of same-dow cat_sold on strictly-earlier dates
    r = panel.iloc[-1]
    same_dow_earlier = panel[(panel["dow"] == r["dow"]) & (panel["date"] < r["date"])]
    expected = same_dow_earlier["cat_sold"].tail(BASELINE_WEEKS).mean()
    assert r["cat_baseline"] == pytest.approx(expected)


def test_panel_other_cat_sold_excludes_own_category():
    """other_cat_sold must equal sum of OTHER categories' cat_sold on same store+date."""
    # Build bread category (2 items, 12 weeks)
    bread_daily = _daily_two_items_one_cat(n_weeks=12, seed=0)
    assert bread_daily["category_id"].unique().tolist() == ["bread"]

    # Build pastry category (2 items, 12 weeks) with different seeds for variety
    pastry_daily = _daily_two_items_one_cat(n_weeks=12, seed=42)
    pastry_daily["category_id"] = "pastry"
    # Rename item_ids to avoid collision with bread items
    pastry_daily["item_id"] = pastry_daily["item_id"].map({"i1": "i3", "i2": "i4"})
    assert pastry_daily["category_id"].unique().tolist() == ["pastry"]
    assert pastry_daily["item_id"].unique().tolist() == ["i3", "i4"]

    # Concat both categories (same store, same dates, different items & categories)
    daily = pd.concat([bread_daily, pastry_daily], ignore_index=True)
    assert daily["category_id"].unique().tolist() == ["bread", "pastry"]

    # Build panel
    panel = build_absorption_panel(daily)

    # Pick a date to test (any row after baseline warmup)
    test_date = panel["date"].iloc[0]
    store_id = panel["store_id"].iloc[0]

    # Get both category rows for the same store+date
    bread_row = panel[
        (panel["store_id"] == store_id)
        & (panel["category_id"] == "bread")
        & (panel["date"] == test_date)
    ].iloc[0]
    pastry_row = panel[
        (panel["store_id"] == store_id)
        & (panel["category_id"] == "pastry")
        & (panel["date"] == test_date)
    ].iloc[0]

    # For bread row: other_cat_sold should equal pastry's cat_sold on same store+date
    assert bread_row["other_cat_sold"] == pytest.approx(pastry_row["cat_sold"])
    # For pastry row: other_cat_sold should equal bread's cat_sold on same store+date
    assert pastry_row["other_cat_sold"] == pytest.approx(bread_row["cat_sold"])

    # Both other_cat_sold values should be > 0 (proving multi-category path is real)
    assert bread_row["other_cat_sold"] > 0.0
    assert pastry_row["other_cat_sold"] > 0.0


def _panel_with_effect(beta_true: float, n_weeks: int = 40, seed: int = 1):
    """Synthetic (store,cat,date) panel: cat_sold = base + beta_true*T + traffic + noise.
    beta_true=0 → absorption; beta_true<0 → walk-away. T correlated with traffic to
    stress the confound control. traffic is a near-perfect proxy for demand_level
    (tiny noise) so the confound is (near-)fully absorbed by the other_cat_sold
    control — this makes beta_true=0.0 a genuine β≈0 fixture rather than one that
    depends on errors-in-variables luck at a particular seed."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks * 7, freq="D")
    demand_level = rng.normal(100, 15, len(dates))          # daily category demand
    traffic = demand_level + rng.normal(0, 1, len(dates))   # near-perfect other-cat proxy
    # high-demand days → more stockout hours (the confound)
    stockout_hours = np.clip((demand_level - 100) * 0.3 + rng.normal(0, 1, len(dates)), 0, None)
    cat_sold = demand_level + beta_true * stockout_hours + rng.normal(0, 3, len(dates))
    daily = pd.DataFrame({
        "store_id": "s1", "category_id": "bread", "date": dates,
        "cat_sold": cat_sold, "stockout_hours": stockout_hours,
        "other_cat_sold": traffic,
        "dow": dates.dayofweek, "month": dates.month,
        "trend": (dates - dates.min()).days.astype(float),
    })
    # leakage-safe baseline on this pre-aggregated frame
    daily = daily.sort_values("date")
    daily["cat_baseline"] = (daily.groupby("dow")["cat_sold"]
                             .shift(1).rolling(4, min_periods=4).mean())
    return daily.dropna(subset=["cat_baseline"]).reset_index(drop=True)


def test_fit_recovers_absorption_zero_beta():
    """β_true=0.0 with a near-perfect traffic proxy must yield 'absorb' robustly —
    not just at one lucky seed. Loop over seeds to guard against errors-in-variables
    residual confound flipping the verdict."""
    for seed in range(5):
        panel = _panel_with_effect(beta_true=0.0, seed=seed)
        res = fit_absorption(panel, "s1", "bread")
        assert res is not None, f"seed={seed}: fit returned None"
        assert res.verdict == "absorb", (
            f"seed={seed}: verdict={res.verdict} beta={res.beta} delta={res.delta}")
        assert abs(res.beta) < 1.0, f"seed={seed}: beta={res.beta} not near 0"
        assert abs(res.beta) < res.delta            # inside equivalence band


def test_fit_recovers_walkaway_negative_beta():
    # strong negative: each stockout-hour loses ~4 units, no absorption
    panel = _panel_with_effect(beta_true=-4.0)
    res = fit_absorption(panel, "s1", "bread")
    assert res.beta < 0
    assert res.verdict == "walkaway"


def test_fit_returns_none_on_tiny_panel():
    panel = _panel_with_effect(beta_true=0.0).head(10)
    assert fit_absorption(panel, "s1", "bread") is None

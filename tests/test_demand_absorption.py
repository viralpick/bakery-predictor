"""수요이전 흡수 검증 (W0 게이트) — 패널 빌더 + 회귀/TOST 테스트.

합성 fixture로 완전흡수(β≈0)와 walk-away(β<0) 시나리오를 심어 회귀가
부호를 회복하는지 검증한다. leakage-safe baseline은 미래 미참조를 확인."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.analysis.demand_absorption import (
    DEFAULT_CLOSE_HOUR, BASELINE_WEEKS, build_absorption_panel,
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

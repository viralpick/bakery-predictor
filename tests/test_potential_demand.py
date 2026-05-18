"""Censored-demand correction: profile blending, monotonicity, edge clips."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.features.potential_demand import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_MULTIPLIER,
    DEFAULT_MIN_CUM_WEIGHT,
    StoreHours,
    attach_potential_demand,
    bakery_hour_profile,
    compute_row_potential,
    cumulative_weight_at,
)


def test_profile_sums_to_one_over_open_window():
    p = bakery_hour_profile(9, 22, alpha=0.5)
    assert pytest.approx(p.sum(), abs=1e-6) == 1.0
    # zero outside open hours
    assert (p[:9] == 0).all()
    assert (p[22:] == 0).all()


def test_alpha_zero_is_uniform():
    p = bakery_hour_profile(9, 22, alpha=0.0)
    open_mask = slice(9, 22)
    inside = p[open_mask]
    assert np.allclose(inside, inside[0])  # all equal


def test_alpha_one_has_morning_peak():
    p = bakery_hour_profile(9, 22, alpha=1.0)
    # 9 AM peak is the largest single-hour slot
    assert p[9] > p[12]
    assert p[9] > p[20]


def test_cumulative_is_monotonic():
    p = bakery_hour_profile(9, 22, alpha=0.5)
    cums = [cumulative_weight_at(p, h) for h in range(0, 25)]
    for a, b in zip(cums[:-1], cums[1:], strict=True):
        assert b >= a - 1e-9  # allow fp noise at the cap


def test_potential_demand_no_stockout_returns_sold_units():
    out = compute_row_potential(
        sold_units=50.0, stockout_time=None, date=pd.Timestamp("2024-01-01"),
        open_hour=9, close_hour=22,
    )
    assert out == 50.0


def test_potential_demand_late_stockout_minimal_correction():
    out = compute_row_potential(
        sold_units=50.0,
        stockout_time=pd.Timestamp("2024-01-01 21:00"),
        date=pd.Timestamp("2024-01-01"),
        open_hour=9, close_hour=22,
    )
    # 21시 품절은 영업 거의 끝까지 진행 → 보정 작아야 함
    assert 50.0 <= out <= 60.0


def test_potential_demand_early_stockout_hits_clip():
    out = compute_row_potential(
        sold_units=50.0,
        stockout_time=pd.Timestamp("2024-01-01 10:00"),
        date=pd.Timestamp("2024-01-01"),
        open_hour=9, close_hour=22,
    )
    # 10시 품절은 너무 일러서 max_multiplier (3x) clip 발동
    assert out == pytest.approx(50.0 * DEFAULT_MAX_MULTIPLIER)


def test_potential_demand_floor_caps_multiplier():
    # min_cum_weight=0.15 → multiplier ≤ 1/0.15 ≈ 6.67. Then max_multiplier=3 clips further.
    out = compute_row_potential(
        sold_units=20.0,
        stockout_time=pd.Timestamp("2024-01-01 10:30"),
        date=pd.Timestamp("2024-01-01"),
        open_hour=9, close_hour=22,
    )
    assert out <= 20.0 * DEFAULT_MAX_MULTIPLIER + 1e-6


def test_stockout_at_open_hour_treated_as_pre_open():
    # h <= open_hour branch
    out = compute_row_potential(
        sold_units=10.0,
        stockout_time=pd.Timestamp("2024-01-01 09:00"),
        date=pd.Timestamp("2024-01-01"),
        open_hour=9, close_hour=22,
    )
    assert out == pytest.approx(10.0 * DEFAULT_MAX_MULTIPLIER)


def test_alpha_blending_midday_lies_between_extremes():
    # 13시 품절 시 alpha=0 (uniform) vs alpha=1 (curve) vs alpha=0.5 (중간)
    args = dict(
        sold_units=100.0,
        stockout_time=pd.Timestamp("2024-01-01 13:00"),
        date=pd.Timestamp("2024-01-01"),
        open_hour=9, close_hour=22,
    )
    uni = compute_row_potential(**args, alpha=0.0)
    curve = compute_row_potential(**args, alpha=1.0)
    mid = compute_row_potential(**args, alpha=0.5)
    lo, hi = sorted([uni, curve])
    assert lo - 1e-6 <= mid <= hi + 1e-6


def test_attach_potential_demand_matches_compute_row():
    daily = pd.DataFrame(
        {
            "store_id": ["s1", "s1", "s2"],
            "item_id": ["i1", "i1", "i1"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "sold_units": [80, 120, 50],
            "is_stockout": [True, False, True],
            "stockout_time": pd.to_datetime(["2024-01-01 14:00", pd.NaT, "2024-01-01 11:00"]),
        }
    )
    stores = [StoreHours("s1", 9, 22), StoreHours("s2", 8, 20)]
    out = attach_potential_demand(daily, stores)
    assert "potential_demand" in out.columns
    # row 0: stockout at 14:00 → corrected up
    assert out.iloc[0]["potential_demand"] > 80
    # row 1: no stockout → unchanged
    assert out.iloc[1]["potential_demand"] == pytest.approx(120.0)
    # row 2: early stockout → likely clipped
    assert out.iloc[2]["potential_demand"] <= 50 * DEFAULT_MAX_MULTIPLIER + 1e-3


def test_attach_with_unknown_store_falls_back_to_sold_units():
    daily = pd.DataFrame(
        {
            "store_id": ["unknown"],
            "item_id": ["i1"],
            "date": pd.to_datetime(["2024-01-01"]),
            "sold_units": [40],
            "is_stockout": [True],
            "stockout_time": pd.to_datetime(["2024-01-01 12:00"]),
        }
    )
    stores = [StoreHours("s1", 9, 22)]
    out = attach_potential_demand(daily, stores)
    assert out.iloc[0]["potential_demand"] == pytest.approx(40.0)


def test_defaults_are_documented_constants():
    assert DEFAULT_ALPHA == 0.5
    assert DEFAULT_MAX_MULTIPLIER == 3.0
    assert DEFAULT_MIN_CUM_WEIGHT == 0.15

"""OntologyFunction smoke tests — each function returns real numbers off the
synthetic dataset. Verifies the wrappers wire the v6 decision layer correctly;
the decision math itself is covered by test_decision_layer.py."""

from __future__ import annotations

import pandas as pd
import pytest

from bakery.data.loader import load_dataset
from bakery.decision import PolicyParams, apply_policy
from bakery.ontology.functions import (
    FUNCTION_REGISTRY,
    demand_diff_by_condition,
    explain_order,
    rank_stockout_risk,
    waste_cost,
    what_if,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


@pytest.fixture(scope="module")
def store_period(dataset):
    daily = dataset.daily
    store_id = daily["store_id"].iloc[0]
    dates = pd.to_datetime(daily.loc[daily["store_id"] == store_id, "date"])
    return store_id, (str(dates.min().date()), str(dates.max().date()))


def test_rank_stockout_risk_returns_topk(dataset, store_period):
    store_id, period = store_period
    ranked = rank_stockout_risk(dataset.daily, store_id, period, k=5)
    assert len(ranked) <= 5
    assert {"item_id", "p_stockout", "order_qty"} <= set(ranked.columns)
    # sorted descending by stockout probability
    assert ranked["p_stockout"].is_monotonic_decreasing
    assert (ranked["p_stockout"].between(0.0, 1.0)).all()


def test_explain_order_lineage_conserved(dataset, store_period):
    store_id, period = store_period
    item_id = dataset.daily.loc[dataset.daily["store_id"] == store_id, "item_id"].iloc[0]
    lineage = explain_order(dataset.daily, store_id, item_id, period)
    assert list(lineage["step"])[0] == "base"
    assert "safety_margin" in set(lineage["step"])
    # base + Σ step contributions must reconstruct the order apply_policy returns
    # (보존 법칙) — verified against the decision layer directly, not self-referentially
    base = float(lineage.loc[lineage["step"] == "base", "contribution"].iloc[0])
    expected_order, _ = apply_policy(item_id, base, PolicyParams())
    assert lineage["contribution"].sum() == pytest.approx(expected_order)


def test_what_if_more_order_lowers_stockout_risk():
    """Downstream lever sanity: ordering more cannot raise P(stockout)."""
    result = what_if(demand_point=30.0, base_order=30.0, delta_order=10.0)
    assert result.new_order == 40.0
    assert result.new_p_stockout <= result.base_p_stockout


def test_waste_cost_nonnegative(dataset, store_period):
    store_id, period = store_period
    out = waste_cost(dataset.daily, store_id, period)
    assert out["leftover_units"] >= 0
    assert out["waste_cost"] >= 0


def test_demand_diff_by_condition_weekend(dataset, store_period):
    store_id, _ = store_period
    out = demand_diff_by_condition(dataset.daily, dataset.calendar, store_id, "is_weekend")
    assert out["condition"] == "is_weekend"
    # independently recompute mean weekend daily units to verify the join+agg
    store_daily = dataset.daily[dataset.daily["store_id"] == store_id]
    merged = store_daily.merge(dataset.calendar[["date", "is_weekend"]], on="date")
    weekend_daily = merged[merged["is_weekend"] == 1].groupby("date")["sold_units"].sum()
    assert out["mean_on"] == pytest.approx(weekend_daily.mean())


def test_demand_diff_by_condition_rejects_missing_column(dataset, store_period):
    store_id, _ = store_period
    with pytest.raises(ValueError, match="not in join_frame"):
        demand_diff_by_condition(dataset.daily, dataset.calendar, store_id, "nonexistent")


def test_function_registry_impls_match_module():
    """Registry must point at the real callables (agent enumerates this)."""
    assert FUNCTION_REGISTRY["rank_stockout_risk"].impl is rank_stockout_risk
    assert FUNCTION_REGISTRY["what_if"].impl is what_if
    assert set(FUNCTION_REGISTRY) == {
        "rank_stockout_risk", "explain_order", "what_if", "waste_cost", "demand_diff_by_condition",
    }


def test_empty_period_raises(dataset):
    store_id = dataset.daily["store_id"].iloc[0]
    with pytest.raises(ValueError, match="no rows"):
        rank_stockout_risk(dataset.daily, store_id, ("1900-01-01", "1900-01-02"))

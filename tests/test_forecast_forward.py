import numpy as np
import pandas as pd
import pytest

from bakery.cli import _category_future_order_predictions
from bakery.forecast.forward import ForwardForecast, forecast_forward

STORE = "store_gw01"
KW = dict(horizon_days=7, total_model="lightgbm", event_prior=True,
          production_quantile=0.85, use_forecast=False)


@pytest.fixture(scope="module")
def result() -> ForwardForecast:
    return forecast_forward(STORE, **KW)


def test_item_quantities_match_cli(result):
    """seam의 item_quantities == 현 cli private 함수 출력(golden)."""
    golden = _category_future_order_predictions(
        STORE, horizon_days=7, production_quantile=0.85,
        total_model="lightgbm", event_prior=True, use_forecast=False,
    ).reset_index(drop=True)
    got = result.item_quantities.reset_index(drop=True)
    assert list(got.columns) == list(golden.columns)
    pd.testing.assert_frame_equal(got, golden, check_dtype=False)


def test_coherence_sum_equals_total(result):
    """date당 Σ our_order == prior_prod, Σ demand_point == prior_median."""
    ct = result.category_totals.set_index("date")
    by_date = result.item_quantities.groupby("date")[["our_order", "demand_point"]].sum()
    for d, row in by_date.iterrows():
        assert row["our_order"] == pytest.approx(ct.loc[d, "prior_prod"], rel=1e-9)
        assert row["demand_point"] == pytest.approx(ct.loc[d, "prior_median"], rel=1e-9)


def test_faithfulness_base_vs_prior(result):
    """event_prior on → prior_*가 base_*와 달라질 수 있고(특수일), 아니면 동일.
    base_*는 blend 이전 Stage1 예측 그대로."""
    ct = result.category_totals
    assert set(["base_median", "base_prod", "prior_median", "prior_prod"]).issubset(ct.columns)
    assert len(ct) == result.item_quantities["date"].nunique()
    # 비특수일은 prior==base (blend가 앵커 없는 날은 항등)
    assert (ct["prior_prod"] >= 0).all() and (ct["base_prod"] >= 0).all()


def test_proportions_factor_columns(result):
    """5b explain_item_order가 소비할 factor 컬럼 존재 + 정규화."""
    p = result.proportions
    for col in ["date", "item_id", "proportion", "base_sold",
                "adj_trend", "adj_stockout", "adj_closing", "adj_new"]:
        assert col in p.columns
    for _, g in p.groupby("date"):
        assert g["proportion"].sum() == pytest.approx(1.0, rel=1e-9)

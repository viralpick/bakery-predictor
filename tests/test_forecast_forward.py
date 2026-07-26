import numpy as np
import pandas as pd
import pytest

from bakery.cli import _category_future_order_predictions
from bakery.forecast.forward import ForwardForecast, forecast_forward
from bakery.forecast.loaders import load_real_daily

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
    pd.testing.assert_frame_equal(got, golden, check_dtype=False, check_exact=True)


def test_coherence_sum_equals_total(result):
    """date당 Σ our_order == prior_prod, Σ demand_point == prior_median."""
    ct = result.category_totals.set_index("date")
    by_date = result.item_quantities.groupby("date")[["our_order", "demand_point"]].sum()
    for d, row in by_date.iterrows():
        assert row["our_order"] == pytest.approx(ct.loc[d, "prior_prod"], rel=1e-9)
        assert row["demand_point"] == pytest.approx(ct.loc[d, "prior_median"], rel=1e-9)


def test_faithfulness_base_vs_prior(result):
    """base_*는 event_prior blend 이전 Stage1 스냅샷이어야 한다 — 값으로 검증.

    event_prior=True/False 두 호출의 base_median/base_prod가 정확히 같아야
    (blend는 base 산출 이후에만 개입) 한다. event_prior=False에서는 blend가
    아예 없으므로 prior_*==base_*가 항등이어야 한다. 누군가 pre_* 스냅샷을
    blend 이후 값으로 잘못 넣는 회귀가 나면 이 항등성이 깨져 잡힌다.
    """
    ct = result.category_totals
    assert set(["base_median", "base_prod", "prior_median", "prior_prod"]).issubset(ct.columns)
    assert len(ct) == result.item_quantities["date"].nunique()

    no_prior_kw = dict(KW)
    no_prior_kw["event_prior"] = False
    result_no_prior = forecast_forward(STORE, **no_prior_kw)
    ct_no_prior = result_no_prior.category_totals

    # base_*는 event_prior 플래그와 무관 — pre-blend 스냅샷이므로 동일해야 한다.
    np.testing.assert_array_equal(ct["base_median"].to_numpy(), ct_no_prior["base_median"].to_numpy())
    np.testing.assert_array_equal(ct["base_prod"].to_numpy(), ct_no_prior["base_prod"].to_numpy())

    # event_prior=False면 blend가 없으므로 prior_*==base_* 항등.
    np.testing.assert_array_equal(ct_no_prior["prior_median"].to_numpy(), ct_no_prior["base_median"].to_numpy())
    np.testing.assert_array_equal(ct_no_prior["prior_prod"].to_numpy(), ct_no_prior["base_prod"].to_numpy())


def test_proportions_factor_columns(result):
    """5b explain_item_order가 소비할 factor 컬럼 존재 + 정규화."""
    p = result.proportions
    for col in ["date", "item_id", "proportion", "base_sold",
                "adj_trend", "adj_stockout", "adj_closing", "adj_new"]:
        assert col in p.columns
    for _, g in p.groupby("date"):
        assert g["proportion"].sum() == pytest.approx(1.0, rel=1e-9)


def test_daily_injection_branch():
    """daily=<주입 프레임>이면 build_category_daily(daily_raw=...) 분기를 탄다.

    daily=None 경로(golden, module fixture)와 값이 같아야 하는지는 강제하지
    않는다 — 광교 단일매장이라 우연히 같을 수 있으나 그건 별개 주장.
    여기서는 분기가 정상 동작해 결과가 비어있지 않고 coherence가 성립함만 본다.
    """
    daily = load_real_daily(STORE)
    result_injected = forecast_forward(
        STORE, daily=daily, horizon_days=7, total_model="lightgbm",
        event_prior=True, production_quantile=0.85, use_forecast=False,
    )
    assert not result_injected.item_quantities.empty

    ct = result_injected.category_totals.set_index("date")
    by_date = result_injected.item_quantities.groupby("date")["our_order"].sum()
    for d, total in by_date.items():
        assert total == pytest.approx(ct.loc[d, "prior_prod"], rel=1e-9)

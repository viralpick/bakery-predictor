import pandas as pd
import pytest

from bakery.forecast.forward import forecast_forward
from bakery.forecast.loaders import load_real_daily
from bakery.ontology.explain import (
    BATCH_ROUND_UNIT,
    explain_category_total,
    explain_item_order,
)

STORE = "store_gw01"


@pytest.fixture(scope="module")
def daily():
    return load_real_daily(STORE)


@pytest.fixture(scope="module")
def ff(daily):
    return forecast_forward(STORE, daily=daily, horizon_days=7, use_forecast=False)


@pytest.fixture(scope="module")
def target_date(ff):
    return str(pd.to_datetime(ff.item_quantities["date"]).min().date())


def test_category_total_reconciles_with_seam(daily, ff, target_date):
    """explain_category_total의 prior_prod 단계 == seam category_totals.prior_prod."""
    rows = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False)
    ct = ff.category_totals[pd.to_datetime(ff.category_totals["date"]) == pd.Timestamp(target_date)].iloc[0]
    got = rows.set_index("step")["value"]
    assert got["base_median"] == pytest.approx(ct["base_median"], rel=1e-9)
    assert got["prior_median"] == pytest.approx(ct["prior_median"], rel=1e-9)
    assert got["prior_prod"] == pytest.approx(ct["prior_prod"], rel=1e-9)
    # event_prior 기여 = prior_median − base_median (실제 blend 차이, "룰" 아님)
    assert got["event_prior"] == pytest.approx(ct["prior_median"] - ct["base_median"], rel=1e-9)


def test_category_total_conservation(daily, target_date):
    """base_median + event_prior = prior_median; prior_median + buffer = prior_prod."""
    got = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False).set_index("step")["value"]
    assert got["base_median"] + got["event_prior"] == pytest.approx(got["prior_median"], rel=1e-9)
    assert got["prior_median"] + got["quantile_buffer"] == pytest.approx(got["prior_prod"], rel=1e-9)


def test_item_order_reconciles_with_seam(daily, ff, target_date):
    """explain_item_order의 item_order 단계(라운딩 전) == seam our_order."""
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    seam_order = float(iq_d[iq_d["item_id"].astype(str) == item]["our_order"].iloc[0])
    rows = explain_item_order(STORE, item, daily=daily, date=target_date, use_forecast=False).set_index("step")["value"]
    assert rows["item_order"] == pytest.approx(seam_order, rel=1e-9)


def test_item_order_conservation(daily, ff, target_date):
    """category_total × proportion = item_order; ceil_3(item_order) = final."""
    import math
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    got = explain_item_order(STORE, item, daily=daily, date=target_date, round_unit=3, use_forecast=False).set_index("step")["value"]
    assert got["category_total"] * got["proportion"] == pytest.approx(got["item_order"], rel=1e-9)
    assert got["final"] == pytest.approx(math.ceil(got["item_order"] / 3) * 3, rel=1e-9)


def test_item_order_exposes_proportion_factors(daily, ff, target_date):
    """비중 인자 5종이 step으로 노출되고, 값이 seam proportions와 정확히 일치한다."""
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    props = ff.proportions
    seam = props[(pd.to_datetime(props["date"]) == pd.Timestamp(target_date))
                 & (props["item_id"].astype(str) == item)].iloc[0]

    rows = explain_item_order(STORE, item, daily=daily, date=target_date, use_forecast=False)
    got = rows.set_index("step")["value"]
    for factor in ("base_sold", "adj_trend", "adj_stockout", "adj_closing", "adj_new"):
        assert got[factor] == pytest.approx(float(seam[factor]), rel=1e-9)


def test_item_order_exposes_raw_evidence(daily, ff, target_date):
    """인자 행은 evidence 컬럼에 원시 근거값(매진시각·추세·마감률·경과일)을 함께 싣는다."""
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    props = ff.proportions
    seam = props[(pd.to_datetime(props["date"]) == pd.Timestamp(target_date))
                 & (props["item_id"].astype(str) == item)].iloc[0]

    rows = explain_item_order(STORE, item, daily=daily, date=target_date, use_forecast=False)
    evidence = rows.set_index("step")["evidence"]
    assert evidence["adj_stockout"] == pytest.approx(float(seam["avg_stockout_h"]), rel=1e-9)
    assert evidence["adj_trend"] == pytest.approx(float(seam["trend_pct"]), rel=1e-9)
    assert evidence["adj_closing"] == pytest.approx(float(seam["closing_rate"]), rel=1e-9)
    assert evidence["adj_new"] == pytest.approx(float(seam["days_since_first"]), rel=1e-9)


def test_item_order_factor_product_reconstructs_proportion(daily, ff, target_date):
    """★faithfulness: raw_weight = base_sold×인자4, proportion = raw_weight / weight_sum.

    인자를 곱해 비중이 정확히 복원되지 않으면 설명이 계산을 반영하지 않는다는 뜻이다.
    """
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    got = explain_item_order(STORE, item, daily=daily, date=target_date,
                             use_forecast=False).set_index("step")["value"]

    product = (got["base_sold"] * got["adj_trend"] * got["adj_stockout"]
               * got["adj_closing"] * got["adj_new"])
    assert got["raw_weight"] == pytest.approx(product, rel=1e-9)
    assert got["raw_weight"] / got["weight_sum"] == pytest.approx(got["proportion"], rel=1e-9)


def test_item_order_weight_sum_equals_seam_total(daily, ff, target_date):
    """weight_sum == 해당 날짜 전 품목 raw_weight 합 (정규화 분모 = seam과 동일 집합)."""
    props = ff.proportions
    day = props[pd.to_datetime(props["date"]) == pd.Timestamp(target_date)]
    expected_sum = float((day["base_sold"] * day["adj_trend"] * day["adj_stockout"]
                          * day["adj_closing"] * day["adj_new"]).sum())
    item = str(day.sort_values("proportion", ascending=False).iloc[0]["item_id"])
    got = explain_item_order(STORE, item, daily=daily, date=target_date,
                             use_forecast=False).set_index("step")["value"]
    assert got["weight_sum"] == pytest.approx(expected_sum, rel=1e-9)


def test_item_order_step_names_are_unique(daily, ff, target_date):
    """step은 set_index 조회 키다 — 중복되면 소비처(gold·tool)가 Series를 받아 깨진다."""
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)]
    item = str(iq_d.sort_values("our_order", ascending=False).iloc[0]["item_id"])
    rows = explain_item_order(STORE, item, daily=daily, date=target_date, use_forecast=False)
    assert rows["step"].is_unique


def test_batch_round_unit_default():
    assert BATCH_ROUND_UNIT == 3


def test_rank_forward_items_matches_seam(daily, ff, target_date):
    """rank_forward_items top-k == seam item_quantities our_order 내림차순."""
    from bakery.ontology.explain import rank_forward_items
    ranked = rank_forward_items(STORE, daily=daily, date=target_date, k=3, use_forecast=False)
    iq = ff.item_quantities
    iq_d = iq[pd.to_datetime(iq["date"]) == pd.Timestamp(target_date)].copy()
    iq_d["item_id"] = iq_d["item_id"].astype(str)
    expected = list(iq_d.sort_values(["our_order", "item_id"], ascending=[False, True])
                    .head(3)["item_id"])
    assert list(ranked["item_id"].astype(str)) == expected
    assert len(ranked) == 3

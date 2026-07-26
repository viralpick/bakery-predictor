import pandas as pd
import pytest

from bakery.forecast.forward import forecast_forward
from bakery.forecast.loaders import load_real_daily
from bakery.ontology.explain import (
    BATCH_ROUND_UNIT, explain_category_total, explain_item_order,
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


def test_batch_round_unit_default():
    assert BATCH_ROUND_UNIT == 3

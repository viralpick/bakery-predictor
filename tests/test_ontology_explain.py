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


# ---------------------------------------------------------------------------
# 피처 기여도 분해 (b) — SHAP 가법 분해를 설명 행으로 노출
# ---------------------------------------------------------------------------

def test_seam_contributions_are_additive(ff):
    """★가법성 앵커: base_value + Σ기여 == raw_prediction (트리 SHAP 성질)."""
    import numpy as np
    contrib = ff.contributions
    assert contrib is not None
    feats = [c for c in contrib.columns if c not in ("date", "base_value", "raw_prediction")]
    total = contrib["base_value"] + contrib[feats].sum(axis=1)
    assert np.allclose(total.to_numpy(), contrib["raw_prediction"].to_numpy(), rtol=1e-12)


def test_seam_raw_prediction_matches_base_median(ff):
    """clip이 걸리지 않는 정상 구간에서 raw_prediction == base_median."""
    import numpy as np
    assert (ff.category_totals["base_median"] > 0).all()
    assert np.allclose(
        ff.contributions["raw_prediction"].to_numpy(),
        ff.category_totals["base_median"].to_numpy(), rtol=1e-12,
    )


def test_grouping_is_exhaustive_and_disjoint(ff):
    """그룹 합 == 전체 피처 기여 합. 누락/중복이 있으면 가법 분해가 깨진다."""
    from bakery.ontology.explain import group_contributions
    row = ff.contributions.iloc[0]
    feats = [c for c in ff.contributions.columns
             if c not in ("date", "base_value", "raw_prediction")]
    grouped = group_contributions(row, target_col="adjusted_demand_unit")
    assert sum(grouped.values()) == pytest.approx(float(row[feats].sum()), rel=1e-12)


def test_cyclic_features_collapse_into_one_axis():
    """dow_sin/cos가 하나의 '요일' 축으로 합쳐진다 — 사람이 읽을 수 있게."""
    from bakery.ontology.explain import _contrib_group
    for feature in ("dow", "dow_sin", "dow_cos", "is_weekend"):
        assert _contrib_group(feature, "adjusted_demand_unit") == "contrib_dow"
    for feature in ("month_sin", "month_cos", "dom_sin"):
        assert _contrib_group(feature, "adjusted_demand_unit") == "contrib_month"


def test_autoregressive_features_group_for_both_engines():
    """대상일 기준(windowed)과 원점 기준(panel) AR 피처 모두 '최근 수요'로."""
    from bakery.ontology.explain import _contrib_group
    for feature in ("adjusted_demand_unit_lag7", "adjusted_demand_unit_rmean28",
                    "y_origin_lag0", "y_same_dow_latest"):
        assert _contrib_group(feature, "adjusted_demand_unit") == "contrib_recent_demand"


def test_explain_rows_reconstruct_base_median(daily, target_date):
    """★설명 행만으로 base_median이 복원된다 — 설명이 계산을 반영한다는 계약."""
    rows = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False)
    got = rows.set_index("step")["value"]
    groups = [s for s in rows["step"]
              if s.startswith("contrib_") and s != "contrib_base_value"]
    assert groups, "기여 그룹 행이 없다"
    total = got["contrib_base_value"] + sum(got[g] for g in groups)
    assert total == pytest.approx(got["base_median"], rel=1e-9)


def test_contribution_rows_sorted_by_magnitude(daily, target_date):
    """기여 큰 축이 위에 온다(읽는 순서 = 중요한 순서)."""
    rows = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False)
    groups = rows[rows["step"].str.startswith("contrib_")
                  & (rows["step"] != "contrib_base_value")]
    magnitudes = groups["value"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_stage_rows_unchanged_by_contribution_addition(daily, ff, target_date):
    """기여 행을 추가해도 기존 단계 행(계약)은 그대로다 — 소비처 gold가 이걸 읽는다."""
    rows = explain_category_total(STORE, daily=daily, date=target_date, use_forecast=False)
    got = rows.set_index("step")["value"]
    ct = ff.category_totals[
        pd.to_datetime(ff.category_totals["date"]) == pd.Timestamp(target_date)
    ].iloc[0]
    assert got["base_median"] == pytest.approx(ct["base_median"], rel=1e-9)
    assert got["prior_prod"] == pytest.approx(ct["prior_prod"], rel=1e-9)
    assert rows["step"].is_unique

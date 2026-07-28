import pandas as pd
import pytest

from bakery.analysis.lab.handlers.stockout import (
    LOST_SHARE_THRESHOLD,
    POPULARITY_CORR_THRESHOLD,
    lost_demand_summary,
    popularity_verdict,
    stockout_revenue,
    stockout_revenue_verdict,
)
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _daily():
    """b1이 3일 중 1일 19시 완판. 총 sold 60."""
    rows = [
        ("2025-01-01", "b1", "bread", 20, True, "2025-01-01 19:00"),
        ("2025-01-02", "b1", "bread", 20, False, None),
        ("2025-01-03", "b1", "bread", 20, False, None),
    ]
    return pd.DataFrame([{"store_id": "store_gw01", "item_id": i, "category_id": c,
                          "date": pd.Timestamp(d), "sold_units": q,
                          "is_stockout": so, "open_hours": 13.0, "capacity": q,
                          "stockout_time": pd.Timestamp(t) if t else pd.NaT}
                         for d, i, c, q, so, t in rows])


def test_threshold_constants():
    assert LOST_SHARE_THRESHOLD == 0.02
    assert POPULARITY_CORR_THRESHOLD == 0.8


def test_lost_demand_summary_counts_and_share():
    summary = lost_demand_summary(_daily()).iloc[0]
    assert summary["store_id"] == "store_gw01"
    assert summary["n_stockout_days"] == 1
    assert summary["est_lost_units"] > 0.0
    # 손실 추정치는 sold 총합 60에 대한 비율로 표현된다
    assert summary["lost_share_of_sold"] == pytest.approx(
        summary["est_lost_units"] / 60.0)


def test_stockout_revenue_verdict_no_impact_below_threshold():
    summary = pd.DataFrame([{"store_id": "store_gw01", "n_stockout_days": 1,
                             "est_lost_units": 1.0, "lost_share_of_sold": 0.01}])
    assert stockout_revenue_verdict(summary) == (
        "지지(무영향) — 매장 1곳 전부 추정 손실 비중 2% 미만 (최대 1.0%)")


def test_stockout_revenue_verdict_flags_material_store():
    summary = pd.DataFrame([
        {"store_id": "store_gw01", "n_stockout_days": 1, "est_lost_units": 1.0,
         "lost_share_of_sold": 0.01},
        {"store_id": "store_mp01", "n_stockout_days": 5, "est_lost_units": 30.0,
         "lost_share_of_sold": 0.05},
    ])
    assert stockout_revenue_verdict(summary) == (
        "부분 기각 — ['store_mp01'] 매장에서 추정 손실 비중 2% 이상 (최대 5.0%)")


def test_popularity_verdict_reports_rank_stability():
    corr = pd.DataFrame([{"pair": "raw_vs_stockout_boosted", "spearman": 0.95, "n": 100}])
    assert popularity_verdict(corr) == (
        "매진 부스트가 배분 순위를 거의 바꾸지 않음 — spearman 0.950 (n=100), "
        "부스트 기여 작음")


def test_popularity_verdict_flags_reordering():
    corr = pd.DataFrame([{"pair": "raw_vs_stockout_boosted", "spearman": 0.55, "n": 100}])
    assert popularity_verdict(corr) == (
        "매진 부스트가 배분 순위를 크게 재배열 — spearman 0.550 (n=100), "
        "임계 0.8 미만이므로 부스트 강도 검토 필요")


def test_stockout_revenue_handler_shape(stub_inputs):
    result = stockout_revenue(stub_inputs(daily=_daily()))
    assert result.kind == KIND_HYPOTHESIS
    assert [label for label, _ in result.tables] == [
        "summary", "top_self_fulfilling", "hour_distribution"]
    hours = dict(result.tables)["hour_distribution"]
    assert hours.columns.tolist() == ["store_id", "item_id", "dow",
                                      "stockout_hour_mean", "stockout_hour_std", "n_weeks"]

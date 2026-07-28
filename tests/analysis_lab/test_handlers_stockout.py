import pandas as pd
import pytest

from bakery.analysis.lab.handlers.stockout import (
    LOST_SHARE_THRESHOLD,
    POPULARITY_CORR_THRESHOLD,
    lost_demand_summary,
    lost_demand_verdict,
    popularity_boost_correlation,
    popularity_verdict,
    stockout_lost_demand,
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


def _proportions_history():
    """4품목 40일 — base_sold 차등 + 매진시각 차등. b1은 매진 없음(NaN 경로 포함).

    스피어만 상수-입력 함정 방지: base_sold도 avg_stockout_h도 품목마다 다르게 설계.
    """
    rows = []
    start = pd.Timestamp("2024-01-01")
    items = [
        ("b1", 30, None),   # 매진 없음 → avg_stockout_h NaN
        ("b2", 20, 20),     # 매일 20시 매진
        ("b3", 10, 14),     # 매일 14시 매진
        ("b4", 5, 10),      # 매일 10시 매진(가장 이름)
    ]
    for day in range(40):
        date = start + pd.Timedelta(days=day)
        for item_id, qty, stockout_hour in items:
            is_stockout = stockout_hour is not None
            rows.append({
                "date": date, "item_id": item_id, "category_id": "bread",
                "sold_units": qty, "is_stockout": is_stockout,
                "stockout_time": (date + pd.Timedelta(hours=stockout_hour)
                                  if is_stockout else pd.NaT),
            })
    return pd.DataFrame(rows)


def test_threshold_constants():
    assert LOST_SHARE_THRESHOLD == 0.02
    assert POPULARITY_CORR_THRESHOLD == 0.8


def test_lost_demand_summary_counts_and_share():
    summary = lost_demand_summary(_daily()).iloc[0]
    assert summary["store_id"] == "store_gw01"
    assert summary["n_stockout_item_days"] == 1
    assert summary["est_lost_units"] > 0.0
    # 손실 추정치는 sold 총합 60에 대한 비율로 표현된다
    assert summary["lost_share_of_sold"] == pytest.approx(
        summary["est_lost_units"] / 60.0)


def test_lost_demand_verdict_reports_magnitude_single_store():
    summary = pd.DataFrame([{"store_id": "store_gw01", "n_stockout_item_days": 1,
                             "est_lost_units": 1.0, "lost_share_of_sold": 0.01}])
    assert lost_demand_verdict(summary) == (
        "추정 손실 비중(하한) 최대 store_gw01 1.0%, 매장 1곳 중 0곳이 보고 임계 2% 이상")


def test_lost_demand_verdict_reports_magnitude_multi_store():
    summary = pd.DataFrame([
        {"store_id": "store_gw01", "n_stockout_item_days": 1, "est_lost_units": 1.0,
         "lost_share_of_sold": 0.01},
        {"store_id": "store_mp01", "n_stockout_item_days": 5, "est_lost_units": 30.0,
         "lost_share_of_sold": 0.05},
    ])
    assert lost_demand_verdict(summary) == (
        "추정 손실 비중(하한) 최대 store_mp01 5.0%, 매장 2곳 중 1곳이 보고 임계 2% 이상")


def test_popularity_verdict_reports_rank_stability():
    corr = pd.DataFrame([{"pair": "proportion_vs_no_stockout_boost", "spearman": 0.95,
                         "n": 100, "n_top_changed": 0, "max_abs_share_delta": 0.0012}])
    assert popularity_verdict(corr) == (
        "매진 부스트가 배분 순위를 거의 바꾸지 않음 — ablation spearman 0.950 (n=100), "
        "top10 교체 0건, 최대 비중 변화 0.0012")


def test_popularity_verdict_flags_reordering():
    corr = pd.DataFrame([{"pair": "proportion_vs_no_stockout_boost", "spearman": 0.55,
                         "n": 100, "n_top_changed": 4, "max_abs_share_delta": 0.0532}])
    assert popularity_verdict(corr) == (
        "매진 부스트가 배분 순위를 크게 재배열 — ablation spearman 0.550 (n=100), "
        "top10 교체 4건, 최대 비중 변화 0.0532")


def test_popularity_boost_correlation_not_vacuous():
    """빈 프레임/상수입력이면 spearman이 NaN이거나 n=0으로 무증상 통과할 수 있다.

    4품목·비상수 base_sold·비상수 avg_stockout_h 픽스처로 그 경로를 막고, 실측값을
    직접 재현(compute_proportions으로 반사실 배분을 손계산)해 정확값으로 고정한다.
    """
    history = _proportions_history()
    target_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=40)
    closing = pd.DataFrame(columns=["item_id", "date", "qty"])
    corr = popularity_boost_correlation(history, closing, target_date=target_date).iloc[0]
    assert corr["pair"] == "proportion_vs_no_stockout_boost"
    assert corr["n"] == 4  # 크기 단언 — 빈 프레임이면 여기서 바로 드러난다
    assert corr["spearman"] == pytest.approx(1.0)
    assert corr["n_top_changed"] == 0  # 품목 4개 < top10 → 멤버십은 항상 동일 집합
    assert corr["max_abs_share_delta"] == pytest.approx(0.009277155005798177)
    assert corr["adj_stockout_min"] == pytest.approx(1.0)
    assert corr["adj_stockout_max"] == pytest.approx(1.0 + 0.2 * 2 / 3)
    assert corr["adj_stockout_std"] > 0.0  # 상수였다면 0 — 산포가 있어야 부스트가 검증 가능했다는 뜻


def test_stockout_lost_demand_handler_shape(stub_inputs):
    result = stockout_lost_demand(stub_inputs(daily=_daily()))
    assert result.kind == KIND_HYPOTHESIS
    assert result.name == "stockout_lost_demand"
    assert result.title == "매진 추정 손실 규모(하한)"
    assert [label for label, _ in result.tables] == [
        "summary", "top_self_fulfilling", "hour_distribution"]
    hours = dict(result.tables)["hour_distribution"]
    assert hours.columns.tolist() == ["store_id", "item_id", "dow",
                                      "stockout_hour_mean", "stockout_hour_std", "n_weeks"]

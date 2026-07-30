"""order_cost.py 검증 — 품목유형 분류 정답 대조 + 비용 산식 정확값."""
import numpy as np
import pandas as pd
import pytest

from bakery.config import PROJECT_ROOT
from bakery.evaluation.order_cost import (
    classify_item_kind,
    cost_rate_for,
    order_cost,
    summarize_order_kpi,
)

GROUND_TRUTH_XLSX = (
    PROJECT_ROOT / "data" / "etc" / "보나비_베이커리_생산량관리_Claude용"
    / "converted" / "수원광교점- 브레드 품목.xlsx"
)
KIND_KOREAN_TO_ENGLISH = {"매장생산": "store_produced", "완제품": "finished_goods"}


@pytest.mark.skipif(not GROUND_TRUTH_XLSX.exists(), reason="브레드 품목 정답 xlsx 로컬 전용")
def test_classify_item_kind_matches_ground_truth_xlsx():
    """41품목 정답(품목타입 컬럼)과 오분류 0건 — 이 모듈의 근거 자체."""
    df = pd.read_excel(GROUND_TRUTH_XLSX, sheet_name="Sheet1", header=1)
    df = df.dropna(subset=["품목코드"]).reset_index(drop=True)
    df["item_id"] = df["품목코드"].apply(lambda code: str(int(code)))
    expected = df["품목타입"].map(KIND_KOREAN_TO_ENGLISH)
    assert expected.isna().sum() == 0  # 매핑 누락 없음(41품목 전체 커버)
    assert len(df) == 41

    actual = classify_item_kind(df["item_id"]).reset_index(drop=True)
    mismatches = int((actual != expected).sum())
    assert mismatches == 0


def test_classify_item_kind_raises_on_unknown_prefix():
    """미등록 접두어는 조용히 기본값을 쓰지 않고 ValueError로 fails-loud."""
    with pytest.raises(ValueError):
        classify_item_kind(["999900000001"])


def test_cost_rate_for_maps_known_prefixes_exactly():
    rates = cost_rate_for(pd.Series(["151100000241", "151300000700"]))
    assert rates.tolist() == [0.40, 0.60]


def test_order_cost_formula_exact_values():
    """waste_cost = r×p×waste, lost_margin = (1-r)×p×short×k 를 정확값으로."""
    rows = pd.DataFrame({
        "item_id": ["151100000001", "151300000001"],  # store_produced, finished_goods
        "order": [10.0, 5.0],
        "demand": [7.0, 8.0],
        "price": [1000.0, 2000.0],
    })
    out = order_cost(
        rows, order_col="order", demand_col="demand", price_col="price", absorption_k=1.0
    )
    assert out["waste_units"].tolist() == pytest.approx([3.0, 0.0], rel=1e-12)
    assert out["short_units"].tolist() == pytest.approx([0.0, 3.0], rel=1e-12)
    # waste_cost = cost_rate * price * waste_units
    assert out["waste_cost_krw"].tolist() == pytest.approx([0.40 * 1000 * 3, 0.60 * 2000 * 0], rel=1e-12)
    # lost_margin = (1-cost_rate) * price * short_units * k(=1.0)
    assert out["lost_margin_krw"].tolist() == pytest.approx([0.0, (1 - 0.60) * 2000 * 3], rel=1e-12)
    assert out["total_cost_krw"].tolist() == pytest.approx([1200.0, 2400.0], rel=1e-12)


def test_absorption_k_affects_only_lost_margin():
    """absorption_k는 lost_margin_krw에만 영향, waste_cost_krw는 불변."""
    rows = pd.DataFrame({
        "item_id": ["151100000001", "151300000001"],
        "order": [10.0, 5.0],
        "demand": [7.0, 8.0],
        "price": [1000.0, 2000.0],
    })
    full = order_cost(rows, order_col="order", demand_col="demand", price_col="price", absorption_k=1.0)
    half = order_cost(rows, order_col="order", demand_col="demand", price_col="price", absorption_k=0.5)

    assert half["waste_cost_krw"].tolist() == pytest.approx(full["waste_cost_krw"].tolist(), rel=1e-12)
    assert half["lost_margin_krw"].tolist() == pytest.approx(
        (full["lost_margin_krw"] * 0.5).tolist(), rel=1e-12
    )
    assert half["lost_margin_krw"].tolist() == pytest.approx([0.0, 1200.0], rel=1e-12)


def _synthetic_costed_for_shortfall() -> pd.DataFrame:
    n = 4
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "short_units": [0.0, 4.0, 0.0, 6.0],
        "waste_units": [0.0] * n,
        "waste_cost_krw": [0.0] * n,
        "lost_margin_krw": [0.0] * n,
        "total_cost_krw": [0.0] * n,
        "is_stockout": [False, True, False, True],
        "soldout_hour": [np.nan, 19.5, np.nan, 20.5],
    })


def test_shortfall_mean_on_stockout_days_differs_from_all_days():
    costed = _synthetic_costed_for_shortfall()
    result = summarize_order_kpi(costed, early_hour=20)
    # 매진 발생 2행만: mean(4, 6) = 5.0
    assert result["shortfall_mean_on_stockout_days"] == pytest.approx(5.0, rel=1e-12)
    # 전체 4행: mean(0, 4, 0, 6) = 2.5
    assert result["shortfall_mean_all_days"] == pytest.approx(2.5, rel=1e-12)
    assert result["shortfall_mean_on_stockout_days"] != result["shortfall_mean_all_days"]


def test_early_stockout_rate_exact():
    costed = _synthetic_costed_for_shortfall()
    result = summarize_order_kpi(costed, early_hour=20)
    # 매진행 2개 중 soldout_hour<20 인 것은 19.5 하나 → 1/2
    assert result["early_stockout_rate"] == pytest.approx(0.5, rel=1e-12)
    # 전체 4행 중 조기매진은 1개 → 1/4
    assert result["early_stockout_rate_all"] == pytest.approx(0.25, rel=1e-12)
    assert result["stockout_rate"] == pytest.approx(0.5, rel=1e-12)


def test_soldout_hour_median_mean_differs_from_pooled_median():
    """날짜별 median의 평균 != 전체 풀링 median — 두 정의가 다름을 합성데이터로 증명."""
    n = 4
    costed = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-01", "2024-02-02"]),
        "short_units": [0.0] * n,
        "waste_units": [0.0] * n,
        "waste_cost_krw": [0.0] * n,
        "lost_margin_krw": [0.0] * n,
        "total_cost_krw": [0.0] * n,
        "is_stockout": [True, True, False, True],
        "soldout_hour": [18.0, 22.0, np.nan, 10.0],
    })
    result = summarize_order_kpi(costed, early_hour=20)
    # 2/1: median(18, 22) = 20.0.  2/2: median(10) = 10.0.  평균 = 15.0
    assert result["soldout_hour_median_mean"] == pytest.approx(15.0, rel=1e-12)
    # 전체 풀링 median([18, 22, 10]) = 18.0 — 위 15.0과 다름을 확인
    pooled_median = float(np.median([18.0, 22.0, 10.0]))
    assert pooled_median == pytest.approx(18.0, rel=1e-12)
    assert result["soldout_hour_median_mean"] != pooled_median


# ---------------------------------------------------------------------------
# 전체매진(카테고리) 비용 — 흡수 가정(k=0)의 정확한 귀결
# ---------------------------------------------------------------------------

def test_default_absorption_k_is_zero():
    """★프로젝트 확정 가정: 품목 단위 품절엔 마진 손실이 없다(흡수).

    근거 = 측정 헌장 매진 2관점(SKU품절은 critical 아님) + 흡수 실측 +
    운영 KPI 경로가 waste_krw만 센다. 이 기본값이 1.0으로 되돌아가면
    "품절을 전액 손실로 본다"는 다른 프로젝트가 된다.
    """
    from bakery.evaluation.order_cost import DEFAULT_ABSORPTION_K

    assert DEFAULT_ABSORPTION_K == 0.0


def _two_item_day(order_a, order_b, demand=10.0, price=1000.0, date="2025-01-01"):
    """매장생산(1511)·완제품(1513) 각 1품목인 하루."""
    return pd.DataFrame({
        "date": pd.to_datetime([date, date]),
        "item_id": ["1511000000001", "1513000000001"],
        "order": [float(order_a), float(order_b)],
        "demand": [demand, demand],
        "unit_price": [price, price],
    })


def test_item_shortfall_costs_nothing_by_default():
    """품목 부족은 기본(k=0)에서 비용 0 — 폐기만 비용이다."""
    from bakery.evaluation.order_cost import order_cost

    costed = order_cost(_two_item_day(12, 8), order_col="order",
                        demand_col="demand", price_col="unit_price")
    assert costed["lost_margin_krw"].sum() == 0.0
    # 폐기 2개 × 원가율 0.40 × 1000 = 800
    assert costed["waste_cost_krw"].sum() == pytest.approx(800.0, rel=1e-12)


def test_category_not_stockout_when_total_is_met():
    """★핵심: 품목별 과·부족이 있어도 카테고리 총량이 충족되면 전체매진이 아니다."""
    from bakery.evaluation.order_cost import category_stockout_cost, order_cost

    costed = order_cost(_two_item_day(12, 8), order_col="order",
                        demand_col="demand", price_col="unit_price")
    cat = category_stockout_cost(costed, order_col="order", demand_col="demand")
    assert len(cat) == 1
    assert cat["category_short_units"].iloc[0] == 0.0
    assert bool(cat["is_category_stockout"].iloc[0]) is False
    assert cat["category_lost_margin_krw"].iloc[0] == 0.0


def test_category_stockout_uses_demand_weighted_margin():
    """전체매진 손실 = 부족량 × 실수요가중 (1−원가율)×단가.

    매장생산(r=0.40 → 마진 600) + 완제품(r=0.60 → 마진 400), 실수요 동일 →
    가중 마진 500원/개. 부족 10개 → 5,000원.
    """
    from bakery.evaluation.order_cost import category_stockout_cost, order_cost

    costed = order_cost(_two_item_day(5, 5), order_col="order",
                        demand_col="demand", price_col="unit_price")
    cat = category_stockout_cost(costed, order_col="order", demand_col="demand")
    assert cat["category_short_units"].iloc[0] == pytest.approx(10.0, rel=1e-12)
    assert cat["category_lost_margin_krw"].iloc[0] == pytest.approx(5000.0, rel=1e-12)


def test_category_margin_shifts_with_mix():
    """구성이 완제품 쪽으로 쏠리면 가중 마진이 내려간다(단일 상수면 못 잡는 효과)."""
    from bakery.evaluation.order_cost import category_stockout_cost, order_cost

    rows = _two_item_day(0, 0)
    rows.loc[rows.item_id.str.startswith("1513"), "demand"] = 30.0   # 완제품 비중↑
    costed = order_cost(rows, order_col="order", demand_col="demand", price_col="unit_price")
    cat = category_stockout_cost(costed, order_col="order", demand_col="demand")
    # 마진 = (10×600 + 30×400)/40 = 450 → 부족 40개 → 18,000
    assert cat["category_lost_margin_krw"].iloc[0] == pytest.approx(18000.0, rel=1e-12)


def test_summary_total_cost_with_category():
    """★k=0 총비용 = 품목 폐기비용 + 전체매진 마진손실."""
    from bakery.evaluation.order_cost import (
        category_stockout_cost,
        order_cost,
        summarize_order_kpi,
    )

    costed = order_cost(_two_item_day(5, 5), order_col="order",
                        demand_col="demand", price_col="unit_price")
    costed["is_stockout"] = True
    costed["soldout_hour"] = [18.0, 19.0]
    cat = category_stockout_cost(costed, order_col="order", demand_col="demand")
    s = summarize_order_kpi(costed, early_hour=20, category=cat)
    assert s["waste_cost_krw"] == 0.0                       # 전량 부족이라 폐기 없음
    assert s["category_short_units"] == pytest.approx(10.0, rel=1e-12)
    assert s["category_stockout_days"] == 1
    assert s["total_cost_with_category_krw"] == pytest.approx(5000.0, rel=1e-12)


def test_summary_without_category_omits_keys():
    """category=None이면 전체매진 항이 없다(계약 명확성)."""
    from bakery.evaluation.order_cost import order_cost, summarize_order_kpi

    costed = order_cost(_two_item_day(12, 8), order_col="order",
                        demand_col="demand", price_col="unit_price")
    costed["is_stockout"] = False
    costed["soldout_hour"] = [float("nan"), float("nan")]
    s = summarize_order_kpi(costed, early_hour=20)
    assert "total_cost_with_category_krw" not in s
    assert "category_short_units" not in s

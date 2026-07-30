"""발주 KPI 단일 기준 — A/B basis 병기 + 정의 5축 확정."""
import numpy as np
import pandas as pd
import pytest

from bakery.evaluation.order_cost import order_cost
from bakery.evaluation.order_kpi import (
    BASIS_ACTUAL,
    BASIS_SIM,
    CATEGORY_SELLOUT_WASTE_TOL,
    basis_actual,
    basis_sim,
    compare_to_actual,
    kpi_table,
    sku_soldout_rate_rowwise,
    waste_negative_diagnostics,
    waste_reduction_pct,
)

STORE_ITEM = "1511000000001"      # 매장생산 r=0.40
FINISHED_ITEM = "1513000000001"   # 완제품    r=0.60
PRICE = 1000.0


def _rows(spec: list[tuple]) -> pd.DataFrame:
    """(date, item_id, order, demand, waste, is_stockout) 목록 → 프레임."""
    frame = pd.DataFrame(spec, columns=[
        "date", "item_id", "order_qty", "adjusted_demand", "waste_qty", "is_stockout",
    ])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["unit_price"] = PRICE
    return frame


def _costed(rows: pd.DataFrame) -> pd.DataFrame:
    return order_cost(rows, order_col="order_qty", demand_col="adjusted_demand",
                      price_col="unit_price")


# ---------------------------------------------------------------------------
# 원가율 — 품목별이어야 한다
# ---------------------------------------------------------------------------

def test_actual_waste_cost_uses_per_item_cost_rate():
    """★전역 0.30이 아니라 품목별(0.40/0.60)이다."""
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 0, 0, 10.0, False),      # 10 × 1000 × 0.40 = 4000
        ("2025-01-01", FINISHED_ITEM, 0, 0, 10.0, False),   # 10 × 1000 × 0.60 = 6000
    ])
    assert basis_actual(rows)["waste_cost_krw"] == pytest.approx(10000.0, rel=1e-12)


# ---------------------------------------------------------------------------
# ★SKU 품절율 = 날별 비율의 평균 (행 단위 평균과 다르다)
# ---------------------------------------------------------------------------

def test_sku_soldout_rate_is_mean_of_daily_rates():
    """날마다 품목 수가 다르면 날별 평균 != 행 단위 평균. 헌장은 날별 평균이다.

    1/1: 품목 1개 중 1개 품절 → 1.0
    1/2: 품목 3개 중 1개 품절 → 1/3
    날별 평균 = (1.0 + 1/3)/2 = 2/3 ≈ 0.6667
    행 단위    = 2/4 = 0.5              ← 다르다
    """
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 0, 0, 0.0, True),
        ("2025-01-02", STORE_ITEM, 0, 0, 0.0, True),
        ("2025-01-02", FINISHED_ITEM, 0, 0, 0.0, False),
        ("2025-01-02", "1511000000002", 0, 0, 0.0, False),
    ])
    assert basis_actual(rows)["sku_soldout_rate"] == pytest.approx(2 / 3, rel=1e-12)
    assert sku_soldout_rate_rowwise(rows, "is_stockout") == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# A basis 전체매진 = 그날 총폐기 ≤ tolerance
# ---------------------------------------------------------------------------

def test_actual_category_stockout_uses_waste_tolerance():
    """폐기가 tolerance 이하인 날만 전체매진(입력오류 감안)."""
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 0, 0, CATEGORY_SELLOUT_WASTE_TOL, False),      # 경계 = 전체매진
        ("2025-01-02", STORE_ITEM, 0, 0, CATEGORY_SELLOUT_WASTE_TOL + 1, False),  # 초과 = 아님
    ])
    assert basis_actual(rows)["category_stockout_day_rate"] == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# ★음수 폐기 — clip 여부를 병기한다
# ---------------------------------------------------------------------------

def test_negative_waste_clip_changes_actual_total():
    """clip on/off가 A basis 폐기 총량을 바꾼다 — 숨기면 절감률 분모가 자의적이 된다."""
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 0, 0, 10.0, False),
        ("2025-01-01", STORE_ITEM, 0, 0, -4.0, False),
    ])
    clipped = basis_actual(rows, waste_clip_negative=True)["waste_units"]
    raw = basis_actual(rows, waste_clip_negative=False)["waste_units"]
    assert clipped == pytest.approx(10.0, rel=1e-12)
    assert raw == pytest.approx(6.0, rel=1e-12)


def test_negative_waste_diagnostics_exact():
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 0, 0, 10.0, False),
        ("2025-01-01", STORE_ITEM, 0, 0, -4.0, False),
        ("2025-01-02", STORE_ITEM, 0, 0, 2.0, False),
    ])
    diag = waste_negative_diagnostics(rows)
    assert diag["negative_rows"] == 1
    assert diag["n_rows"] == 3
    assert diag["negative_units"] == pytest.approx(-4.0, rel=1e-12)
    assert diag["waste_units_raw"] == pytest.approx(8.0, rel=1e-12)
    assert diag["waste_units_clipped"] == pytest.approx(12.0, rel=1e-12)
    assert diag["clip_effect_pct"] == pytest.approx(50.0, rel=1e-12)


# ---------------------------------------------------------------------------
# B basis
# ---------------------------------------------------------------------------

def test_sim_total_cost_is_waste_plus_category_loss():
    """★k=0 총비용 = 품목 폐기비용 + 전체매진 마진손실.

    1/1: 발주 12+8=20 == 실수요 20 → 전체매진 아님. 품목 폐기 = (12−10)=2개(r=0.40) → 800원
    1/2: 발주 5+5=10 < 실수요 20 → 전체매진 10개. 가중마진 (600+400)/2=500 → 5000원
    """
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 12, 10, 0.0, False),
        ("2025-01-01", FINISHED_ITEM, 8, 10, 0.0, False),
        ("2025-01-02", STORE_ITEM, 5, 10, 0.0, False),
        ("2025-01-02", FINISHED_ITEM, 5, 10, 0.0, False),
    ])
    out = basis_sim(_costed(rows), order_col="order_qty", demand_col="adjusted_demand")
    assert out["basis"] == BASIS_SIM
    assert out["waste_cost_krw"] == pytest.approx(800.0, rel=1e-12)
    assert out["category_short_units"] == pytest.approx(10.0, rel=1e-12)
    assert out["category_lost_margin_krw"] == pytest.approx(5000.0, rel=1e-12)
    assert out["category_stockout_days"] == 1
    assert out["total_cost_krw"] == pytest.approx(5800.0, rel=1e-12)


def test_sim_omits_soldout_hour_keys_without_timing():
    """매진시각 컬럼이 없으면 관련 키를 만들지 않는다(가짜 nan 금지)."""
    rows = _rows([("2025-01-01", STORE_ITEM, 5, 10, 0.0, False)])
    out = basis_sim(_costed(rows), order_col="order_qty", demand_col="adjusted_demand")
    assert "soldout_hour_median" not in out
    assert "early_stockout_rate" not in out


def test_sim_reports_both_soldout_hour_views():
    """★매진시각 2관점 병기 — 전체 median과 날별 median의 평균은 다르다.

    1/1: 매진 2건 시각 10, 20 → 날 median 15
    1/2: 매진 1건 시각 18     → 날 median 18
    전체 median = median(10,20,18) = 18 / 날별 median의 평균 = (15+18)/2 = 16.5
    """
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 5, 10, 0.0, True),
        ("2025-01-01", FINISHED_ITEM, 5, 10, 0.0, True),
        ("2025-01-02", STORE_ITEM, 5, 10, 0.0, True),
    ])
    costed = _costed(rows)
    costed["soldout_hour"] = [10.0, 20.0, 18.0]
    out = basis_sim(costed, order_col="order_qty", demand_col="adjusted_demand", early_hour=19)
    assert out["soldout_hour_median"] == pytest.approx(18.0, rel=1e-12)
    assert out["soldout_hour_median_mean"] == pytest.approx(16.5, rel=1e-12)
    assert out["early_stockout_rate"] == pytest.approx(2 / 3, rel=1e-12)   # 10, 18 < 19


# ---------------------------------------------------------------------------
# ★아띠제 대비 절감률 — 잣대를 라벨링한다
# ---------------------------------------------------------------------------

def test_waste_reduction_pct_sign_and_guard():
    assert waste_reduction_pct(60.0, 100.0) == pytest.approx(-40.0, rel=1e-12)
    assert waste_reduction_pct(150.0, 100.0) == pytest.approx(50.0, rel=1e-12)
    assert np.isnan(waste_reduction_pct(50.0, 0.0))


def test_compare_to_actual_separates_censoring_gap():
    """★핵심: 모델(B) vs 아띠제(A)는 잣대가 다르다. 동일-basis 비교를 함께 낸다.

    A 실측 폐기 100 / 아띠제를 B로 시뮬 80 / 모델 B 60
      vs_actual      = 60/100−1 = −40%   (잣대 다름 — 절감 + censoring 섞임)
      vs_actual_sim  = 60/80−1  = −25%   (공정 비교)
      censoring_gap  = 80/100−1 = −20%   (잣대 효과)
    """
    out = compare_to_actual(
        {"waste_cost_krw": 60.0}, {"waste_cost_krw": 100.0},
        actual_sim={"waste_cost_krw": 80.0},
    )
    assert out["vs_actual_pct"] == pytest.approx(-40.0, rel=1e-12)
    assert out["vs_actual_sim_pct"] == pytest.approx(-25.0, rel=1e-12)
    assert out["censoring_gap_pct"] == pytest.approx(-20.0, rel=1e-12)


def test_compare_without_actual_sim_omits_fair_comparison():
    """actual_sim이 없으면 공정 비교 키를 만들지 않는다(있는 척 금지)."""
    out = compare_to_actual({"waste_cost_krw": 60.0}, {"waste_cost_krw": 100.0})
    assert "vs_actual_sim_pct" not in out
    assert "censoring_gap_pct" not in out


def test_kpi_table_keeps_basis_label_first():
    """basis 라벨이 표 앞에 온다 — 어느 잣대인지 모르고 인용하는 것을 막는다."""
    table = kpi_table([
        {"policy": "artisee", "basis": BASIS_ACTUAL, "waste_cost_krw": 100.0},
        {"policy": "ours", "basis": BASIS_SIM, "waste_cost_krw": 60.0},
    ])
    assert list(table.columns)[:2] == ["policy", "basis"]
    assert len(table) == 2


def test_kpi_table_empty_is_empty_frame():
    assert kpi_table([]).empty


def test_actual_sim_makes_censoring_explicit_on_real_shape():
    """★actual_sim 배선의 계약: 같은 정책을 A/B로 재면 차이가 잣대 효과다.

    아띠제 실생산을 그대로 발주로 쓰면(order == production_qty) B 잣대 폐기는
    max(생산−실수요, 0)이고, A 잣대 폐기는 실측 QT_OUT이다. 둘의 차이가 censoring.
    """
    rows = _rows([
        ("2025-01-01", STORE_ITEM, 12, 10, 1.0, False),   # 생산 12, 실수요 10 → B폐기 2 / A폐기 1
    ])
    actual = basis_actual(rows)
    sim = basis_sim(_costed(rows), order_col="order_qty", demand_col="adjusted_demand")
    assert actual["waste_units"] == pytest.approx(1.0, rel=1e-12)   # 실측
    assert sim["waste_units"] == pytest.approx(2.0, rel=1e-12)      # 시뮬
    # 모델이 아띠제와 동일 발주라면 vs_actual_sim은 0이어야 한다(자기 자신 비교)
    out = compare_to_actual(sim, actual, actual_sim=sim)
    assert out["vs_actual_sim_pct"] == pytest.approx(0.0, rel=1e-12)
    # censoring gap은 A→B 잣대 차이만 반영: 800원 → 1600원 = +100%
    assert out["censoring_gap_pct"] == pytest.approx(100.0, rel=1e-12)

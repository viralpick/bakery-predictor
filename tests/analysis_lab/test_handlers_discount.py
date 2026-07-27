import pandas as pd

from bakery.analysis.discount import DiscountSales
from bakery.analysis.lab.handlers.discount import (
    CLOSING_CATEGORY_DEFAULT,
    alpha_verdict,
    closing_waste_frame,
    discount_hour_table,
    other_discounts,
    regime_verdict,
)
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _rows():
    return pd.DataFrame({
        "receipt_id": ["r1", "r2", "r3", "r4"],
        "date": pd.to_datetime(["2025-01-01"] * 4),
        "hour": [11, 20, 20, 15],
        "minute": [0, 30, 45, 0],
        "item_id": ["b1", "b1", "p1", "p1"],
        "qty": [2, 3, 1, 4],
        "unit_price": [3000.0, 3000.0, 5000.0, 5000.0],
        "paid": [6000.0, 6300.0, 3500.0, 20000.0],
        "discount_amt": [0.0, 2700.0, 1500.0, 0.0],
        "discount_code": ["", "0069", "0069", ""],
        "label": ["none", "closing", "closing", "none"],
        "is_set": [False, False, False, False],
    })


def _waste():
    return pd.DataFrame({
        "cd": ["1000000047"] * 2, "store": ["광교"] * 2,
        "item_id": ["b1", "p1"], "date": pd.to_datetime(["2025-01-01"] * 2),
        "production_qty": [10, 8], "waste_qty": [2, 1],
        "normal_qty": [2.0, 4.0], "closing_qty": [3.0, 1.0],
        "unit_price": [3000, 5000], "waste_cost": [6000, 5000],
        "identity_diff": [0.0, 0.0], "sold_total": [5.0, 5.0],
    })


def test_closing_category_default():
    assert CLOSING_CATEGORY_DEFAULT == "bread"


def test_closing_waste_frame_matches_primitive_contract():
    """run_closing_demand는 (item_id, date, waste_qty) 컬럼을 요구한다."""
    frame = closing_waste_frame(_waste())
    assert frame.columns.tolist() == ["item_id", "date", "waste_qty"]
    assert frame["waste_qty"].tolist() == [2, 1]


def test_discount_hour_table_counts_closing_by_hour():
    ds = DiscountSales(rows=_rows())
    table = discount_hour_table(ds, pd.Series({"b1": "bread", "p1": "pastry"}))
    bread_20 = table[(table["category_id"] == "bread") & (table["hour"] == 20)].iloc[0]
    assert bread_20["qty"] == 3


def test_alpha_verdict_reports_interval():
    from bakery.analysis.closing_demand import AlphaEstimate

    alpha = AlphaEstimate(alpha_low=0.6, alpha_high=0.9, a1=0.55, a2=0.8,
                          a3_slope=0.7, note="A1 제외(저녁 상시할인)")
    assert alpha_verdict(alpha) == (
        "구간 추정 α ∈ [0.600, 0.900] (A1 0.550 / A2 0.800 / A3 0.700) "
        "— A1 제외(저녁 상시할인)")


def test_alpha_verdict_handles_missing_estimators():
    from bakery.analysis.closing_demand import AlphaEstimate

    alpha = AlphaEstimate(alpha_low=0.5, alpha_high=1.0, a1=None, a2=None,
                          a3_slope=None, note="식별 불가")
    assert alpha_verdict(alpha) == (
        "구간 추정 α ∈ [0.500, 1.000] (A1 없음 / A2 없음 / A3 없음) — 식별 불가")


def test_regime_verdict_uses_ci_and_placebo():
    from bakery.analysis.discount_regime import RegimeResult

    share = RegimeResult(beta=-0.05, se=0.01, ci_low=-0.07, ci_high=-0.03,
                         n=500, n_params=4, cut_date=pd.Timestamp("2024-01-01"),
                         ill_posed=False)
    report = {"category": "bread", "cut_date": pd.Timestamp("2024-01-01"), "n": 500,
              "closing_share": share, "closing_intensity": share,
              "placebo": [], "verdict": "shift"}
    assert regime_verdict(report) == (
        "레짐 전환 shift — closing_share β=-0.0500 CI90[-0.0700,-0.0300], "
        "placebo 0건, n=500 (cut=2024-01-01)")


def test_other_discounts_handler_shape(stub_inputs):
    inputs = stub_inputs(discount_rows=_rows(), waste=_waste(),
                         item_to_category=pd.Series({"b1": "bread", "p1": "pastry"}))
    result = other_discounts(inputs)
    assert result.kind == KIND_HYPOTHESIS
    assert [label for label, _ in result.tables] == ["by_code", "by_label", "by_hour"]
    # 마감(0069)은 제외되고 마감 외 코드만 남는다 — 이 fixture엔 마감 외 할인이 없다
    assert result.verdict == "마감 외 할인 0건 — 이 매장은 마감할인이 전부다"
    # 빈 결과에서도 라벨별 스키마가 유지돼야 한다(오라벨 방지)
    tables = dict(result.tables)
    assert tables["by_code"].columns.tolist() == [
        "discount_code", "label", "rows", "qty_total", "amt_total", "avg_amt",
        "peak_hour", "share_at_pm8"]
    assert tables["by_label"].columns.tolist() == [
        "label", "rows", "qty_total", "amt_total", "share_at_pm8"]
    assert tables["by_hour"].columns.tolist() == ["discount_code", "hour", "qty"]
    assert len(tables["by_code"]) == 0

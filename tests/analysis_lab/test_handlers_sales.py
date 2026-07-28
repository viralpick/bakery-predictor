import pandas as pd
import pytest

from bakery.analysis.lab.handlers.sales import (
    category_mix,
    category_share,
    median_unit_price,
    monthly_share_stability,
)
from bakery.analysis.lab.result import KIND_DATA


def _daily():
    """광교 2품목 3일. bread 60 / pastry 30 (총 90) — 손계산 가능한 fixture."""
    rows = [
        ("2025-01-01", "b1", "bread", 10), ("2025-01-01", "p1", "pastry", 5),
        ("2025-01-02", "b1", "bread", 30), ("2025-01-02", "p1", "pastry", 5),
        ("2025-02-01", "b1", "bread", 20), ("2025-02-01", "p1", "pastry", 20),
    ]
    return pd.DataFrame([{"store_id": "store_gw01", "item_id": i, "category_id": c,
                          "date": pd.Timestamp(d), "sold_units": q,
                          "is_stockout": False, "stockout_time": pd.NaT}
                         for d, i, c, q in rows])


def _waste():
    """단가만 쓰는 fixture — b1=3000, p1=5000 (중앙값 계산 대상으로 중복 행 포함)."""
    return pd.DataFrame({
        "item_id": ["b1", "b1", "p1"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
        "unit_price": [3000, 3000, 5000],
        "production_qty": [12, 32, 6], "waste_qty": [2, 2, 1],
    })


def test_median_unit_price_is_per_item_median():
    prices = median_unit_price(_waste())
    assert prices["b1"] == 3000.0
    assert prices["p1"] == 5000.0


def test_category_share_units_and_revenue_exact():
    share = category_share(_daily(), median_unit_price(_waste()))
    bread = share[share["category_id"] == "bread"].iloc[0]
    pastry = share[share["category_id"] == "pastry"].iloc[0]
    assert bread["sold_units"] == 60
    assert pastry["sold_units"] == 30
    assert bread["share"] == pytest.approx(60 / 90)
    assert pastry["share"] == pytest.approx(30 / 90)
    # revenue: bread 60×3000=180000, pastry 30×5000=150000 → 총 330000
    assert bread["revenue"] == 180000.0
    assert pastry["revenue"] == 150000.0
    assert bread["revenue_share"] == pytest.approx(180000 / 330000)


def test_category_share_sums_to_one_per_store():
    share = category_share(_daily(), median_unit_price(_waste()))
    assert share.groupby("store_id")["share"].sum().iloc[0] == pytest.approx(1.0)
    assert share.groupby("store_id")["revenue_share"].sum().iloc[0] == pytest.approx(1.0)


def test_monthly_share_stability_exact():
    # 1월: bread 40/50=0.8, pastry 10/50=0.2 | 2월: 20/40=0.5, 20/40=0.5
    # ddof=0 모집단 std → |0.8-0.5|/2 = 0.15
    stability = monthly_share_stability(_daily()).set_index("category_id")
    assert stability.loc["bread", "n_months"] == 2
    assert stability.loc["bread", "share_std"] == pytest.approx(0.15)
    assert stability.loc["bread", "share_min"] == pytest.approx(0.5)
    assert stability.loc["bread", "share_max"] == pytest.approx(0.8)
    assert stability.loc["pastry", "share_std"] == pytest.approx(0.15)


def test_monthly_share_stability_groups_by_store():
    daily = _daily()
    other = daily.copy()
    other["store_id"] = "store_ss01"
    stability = monthly_share_stability(pd.concat([daily, other], ignore_index=True))
    assert sorted(stability["store_id"].unique()) == ["store_gw01", "store_ss01"]
    assert len(stability) == 4          # 2매장 × 2카테고리


def test_handler_returns_data_kind_without_verdict(stub_inputs):
    result = category_mix(stub_inputs(daily=_daily(), waste=_waste()))
    assert result.name == "category_mix"
    assert result.kind == KIND_DATA
    assert result.verdict is None       # 데이터 분석은 판정 없음
    assert [label for label, _ in result.tables] == ["share", "monthly_stability"]
    assert len(result.figures) == 2


def test_handler_notes_price_coverage(stub_inputs):
    daily = _daily()
    daily.loc[len(daily)] = {"store_id": "store_gw01", "item_id": "unknown",
                             "category_id": "bread", "date": pd.Timestamp("2025-02-02"),
                             "sold_units": 10, "is_stockout": False, "stockout_time": pd.NaT}
    result = category_mix(stub_inputs(daily=daily, waste=_waste()))
    # 단가 미매핑 10개 / 총 100개 → coverage 0.9. 은폐 방지로 note에 남긴다.
    assert result.notes == ["단가 매핑 커버리지 0.900 — 미매핑 품목의 revenue는 0으로 계산됨"]


def test_daily_totals_aggregates_per_store_day():
    from bakery.analysis.lab.handlers.sales import daily_totals

    totals = daily_totals(_daily(), median_unit_price(_waste()))
    assert len(totals) == 3                                  # 3일
    first = totals[totals["date"] == pd.Timestamp("2025-01-01")].iloc[0]
    assert first["sold_units"] == 15                         # bread 10 + pastry 5
    assert first["revenue"] == 55000.0                       # 10×3000 + 5×5000
    assert first["n_items_active"] == 2


def test_distribution_summary_exact():
    from bakery.analysis.lab.handlers.sales import daily_totals, distribution_summary

    totals = daily_totals(_daily(), median_unit_price(_waste()))
    # 일별 수량 = [15, 35, 40] → mean 30, median 35
    summary = distribution_summary(totals).iloc[0]
    assert summary["n_days"] == 3
    assert summary["mean"] == pytest.approx(30.0)
    assert summary["median"] == pytest.approx(35.0)
    assert summary["std"] == pytest.approx(13.228756555322953)   # ddof=1
    assert summary["cv"] == pytest.approx(13.228756555322953 / 30.0)


def test_sales_distribution_handler_shape(stub_inputs):
    from bakery.analysis.lab.handlers.sales import sales_distribution

    result = sales_distribution(stub_inputs(daily=_daily(), waste=_waste()))
    assert result.name == "sales_distribution"
    assert result.verdict is None
    assert [label for label, _ in result.tables] == ["daily_totals", "summary"]
    assert len(result.figures) == 2

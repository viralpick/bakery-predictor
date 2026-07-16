import numpy as np
import pandas as pd
from bakery.data.calendar import build_calendar_daily
from bakery.data.weather import build_synthetic_weather
from bakery.features.calendar_features import add_calendar_features
from bakery.features.potential_demand import StoreHours
from bakery.features.weather_features import add_weather_features
from bakery.evaluation.prospective import (
    build_arrival_profile, simulate_item_day_kpis, compare_policies,
)
from bakery.evaluation.business_metrics import CostParams
from bakery.evaluation.diagnostics import decoupling_score
import pytest
from bakery.cli import _assemble_real_rows, _fill_our_order, _quantile_backtest_predictions, _decoupling_by_category, _receipts_profile_frame, _apply_category_margin
from bakery.models.conformal_order import ConformalOrderCalibrator


def test_end_to_end_our_beats_worse_baseline():
    # 합성: item a, 2일. 복원수요 100/100. 우리 발주=수요근접, baseline=과발주.
    receipts = pd.DataFrame({
        "item_id": ["a"]*4, "date": ["2025-01-01"]*2 + ["2025-01-02"]*2,
        "hour": [9, 14, 9, 14], "qty": [50, 50, 50, 50],
    })
    prof = build_arrival_profile(receipts, group_cols=["item_id"])
    rows = pd.DataFrame({
        "item_id": ["a", "a"], "date": ["2025-01-01", "2025-01-02"],
        "potential_demand": [100.0, 100.0],
        "our_order": [105.0, 105.0], "base_order": [140.0, 140.0],
    })
    sh = StoreHours("gwangyo", 8, 22)
    our = simulate_item_day_kpis(rows, prof, order_col="our_order",
                                 store_hours=sh, group_cols=["item_id"],
                                 params=CostParams(), unit_prices={"a": 1000.0})
    base = simulate_item_day_kpis(rows, prof, order_col="base_order",
                                  store_hours=sh, group_cols=["item_id"],
                                  params=CostParams(), unit_prices={"a": 1000.0})
    cmp = compare_policies(our, base).set_index("policy")
    # 우리 발주가 baseline보다 과발주 덜 함 → 폐기비용 Δ < 0
    assert cmp.loc["delta", "waste_cost_krw"] < 0.0


def _real_shaped_daily() -> pd.DataFrame:
    """bonavi_daily.parquet 서브셋 형태 (store/category 필터 + build_item_adjusted_demand
    적용 이후 상태 — 실제 호출 경로(_real_prospective_inputs)는 이 함수 이전에
    adjusted_demand를 주입하므로, 여기서도 컬럼을 미리 채워 시뮬레이션한다.
    무마감할인 단순화이므로 build_item_adjusted_demand 계약대로 adjusted_demand는
    sold_units와 동일(α항 소멸), potential_demand와는 다르다."""
    return pd.DataFrame({
        "item_id": ["101", "101", "102", "999"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-01"]),
        "category_id": ["bread", "bread", "pastry", "bread"],
        "sold_units": [10, 12, 5, 3],
        "is_stockout": [False, True, False, False],
        "potential_demand": [12.0, 15.0, 5.0, 3.0],
        "adjusted_demand": [10.0, 12.0, 5.0, 3.0],
    })


def _real_shaped_inventory() -> pd.DataFrame:
    """load_inventory 반환 형태 — date는 YYYYMMDD 문자열(daily의 datetime과 다른 표현)."""
    return pd.DataFrame({
        "date": ["20210101", "20210102", "20210101"],
        "item_id": ["101", "101", "102"],
        "production_qty": [15, 18, 6],
        "waste_qty": [3, 2, 1],
    })


def test_assemble_real_rows_base_order_matches_production_qty():
    result = _assemble_real_rows(_real_shaped_daily(), _real_shaped_inventory())

    assert list(result.columns) == [
        "item_id", "date", "category_id", "potential_demand", "adjusted_demand",
        "sold_units", "is_stockout", "base_order", "waste_qty",
    ]

    row = result[
        (result["item_id"] == "101") & (result["date"] == pd.Timestamp("2021-01-02"))
    ].iloc[0]
    assert row["base_order"] == 18
    assert row["waste_qty"] == 2
    assert row["potential_demand"] == 15.0
    assert row["adjusted_demand"] == 12.0
    assert row["category_id"] == "bread"
    assert bool(row["is_stockout"]) is True

    row2 = result[result["item_id"] == "102"].iloc[0]
    assert row2["base_order"] == 6
    assert row2["waste_qty"] == 1
    assert row2["category_id"] == "pastry"


def test_fill_our_order_restricts_rows_to_scored_window():
    """Task B rows(전체 기간) 중 our_order 예측이 있는 item-day만 남기고,
    나머지(2024-01-01 "a", 전체 "c")는 제외되며 our_order 값이 정확히 붙는다."""
    rows = pd.DataFrame({
        "item_id": ["a", "a", "b", "c"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]),
        "potential_demand": [10.0, 11.0, 5.0, 3.0],
    })
    predictions = pd.DataFrame({
        "item_id": ["a", "b"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
        "our_order": [12.5, 6.0],
    })

    result = _fill_our_order(rows, predictions)

    assert len(result) == 2
    assert set(zip(result["item_id"], result["date"].dt.strftime("%Y-%m-%d"))) == {
        ("a", "2024-01-02"), ("b", "2024-01-01"),
    }
    row_a = result[result["item_id"] == "a"].iloc[0]
    assert row_a["our_order"] == 12.5
    row_b = result[result["item_id"] == "b"].iloc[0]
    assert row_b["our_order"] == 6.0


def _enriched_v2_toy(n_days: int = 110, seed: int = 11) -> pd.DataFrame:
    """v2 LightGBM 학습에 필요한 최소 enrich 프레임 — store 1개, item 2개, 카테고리
    공유(cannibalization 계산 대상). potential_demand=adjusted_demand=sold_units
    (무품절·무마감할인 단순화 — _quantile_backtest_predictions 기본 target_col은
    adjusted_demand)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for item in ("i1", "i2"):
        for i, d in enumerate(dates):
            sold = int(20 + (i % 7) * 3 + rng.integers(0, 5))
            rows.append({
                "store_id": "s1", "item_id": item, "category_id": "bread",
                "date": d, "sold_units": sold, "is_stockout": False,
                "potential_demand": float(sold), "adjusted_demand": float(sold),
            })
    df = pd.DataFrame(rows)
    cal = build_calendar_daily(dates.min(), dates.max())
    weather = build_synthetic_weather(dates.min(), dates.max(), store_ids=["s1"], seed=seed)
    df = add_calendar_features(df, cal)
    df = add_weather_features(df, weather)
    return df


def test_quantile_backtest_predictions_structural_properties():
    """작은 합성 v2 프레임 → our_order 예측이 (a) 비음수 (b) NaN 없음 (c) val 기간
    item-day 수와 정확히 일치. 실제 모델 출력값 자체는 model-dependent이라 exact-value
    단언 대신 구조 속성만 검증한다."""
    daily = _enriched_v2_toy()

    preds, windows = _quantile_backtest_predictions(
        daily, val_weeks=1, production_quantile=0.85
    )

    assert len(windows) == 1
    window = windows[0]
    val_days = pd.date_range(window.val_start, window.val_end, freq="D")
    n_items = daily["item_id"].nunique()
    assert len(preds) == len(val_days) * n_items
    assert preds["our_order"].notna().all()
    assert (preds["our_order"] >= 0.0).all()
    assert set(preds["fold"].unique()) == {0}


def test_assemble_real_rows_drops_item_days_without_inventory_match():
    result = _assemble_real_rows(_real_shaped_daily(), _real_shaped_inventory())

    # item "999" (재고정보 매칭 없음) 은 base_order 미정이라 평가셋에서 제외된다
    assert set(result["item_id"]) == {"101", "102"}
    assert len(result) == 3


def test_assemble_real_rows_subtracts_bulk_from_production():
    """헌장 §1: base_order(=생산량)에서 해당 item-day 예약(bulk) 판매량을 차감한다."""
    bulk_qty = pd.DataFrame({
        "item_id": ["101", "102"],
        "date": pd.to_datetime(["2021-01-02", "2021-01-01"]),
        "bulk_qty": [5.0, 100.0],  # 101: 정상 차감 / 102: 생산(6) 초과 → clip(0)
    })
    result = _assemble_real_rows(
        _real_shaped_daily(), _real_shaped_inventory(), bulk_qty=bulk_qty
    )

    # 101 2021-01-02: 생산 18 − bulk 5 = 13
    row = result[
        (result["item_id"] == "101") & (result["date"] == pd.Timestamp("2021-01-02"))
    ].iloc[0]
    assert row["base_order"] == 13
    assert row["waste_qty"] == 2  # 폐기 불변

    # 102 2021-01-01: 생산 6 − bulk 100 = -94 → clip 0
    row2 = result[result["item_id"] == "102"].iloc[0]
    assert row2["base_order"] == 0

    # bulk 없는 item-day(101 2021-01-01)는 생산 15 그대로
    row3 = result[
        (result["item_id"] == "101") & (result["date"] == pd.Timestamp("2021-01-01"))
    ].iloc[0]
    assert row3["base_order"] == 15


def test_decoupling_by_category_with_two_categories():
    """ρ_DS 카테고리별 산출 — category='bread'는 강한 양의 상관,
    category='pastry'는 constant stockout(분산 0) → 결과 0.0."""
    rows = pd.DataFrame({
        "category_id": ["bread", "bread", "bread", "bread", "pastry", "pastry", "pastry"],
        "item_id": ["i1", "i1", "i2", "i2", "i3", "i3", "i4"],
        "date": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01"],
        # adjusted_demand는 _decoupling_by_category의 실제 잣대(Task 5). potential_demand는
        # 진단·비교용으로 값만 동일하게 유지(fixture 단순화 목적, 실제 α 보정과 무관).
        "potential_demand": [10.0, 20.0, 15.0, 25.0, 5.0, 5.0, 5.0],
        "adjusted_demand": [10.0, 20.0, 15.0, 25.0, 5.0, 5.0, 5.0],
        "is_stockout": [0, 1, 0, 1, 0, 0, 0],  # bread: demand↑⟹stockout↑ (strong positive)
    })  # pastry: is_stockout constant → var=0 → score=0.0

    scores = _decoupling_by_category(rows)

    # bread: demand=[10, 20, 15, 25], stockout=[0, 1, 0, 1] → exact value per numpy calculation
    bread_score = scores["bread"]
    expected_bread = decoupling_score(
        np.array([10.0, 20.0, 15.0, 25.0]),
        np.array([0.0, 1.0, 0.0, 1.0])
    )
    assert abs(bread_score - expected_bread) < 1e-9, f"bread score {bread_score}, expected {expected_bread}"

    # pastry: is_stockout constant → var=0 → score=0.0 (per decoupling_score contract)
    pastry_score = scores["pastry"]
    assert pastry_score == 0.0, f"pastry score {pastry_score}, expected 0.0"

    # only 2 categories
    assert len(scores) == 2


def test_receipts_profile_frame_qty_weighted_and_bulk_excluded():
    # a: 정상 3개(qty 2,4) + bulk 1개(qty 50) / b: 정상 1개. bulk는 profile에서 제외,
    # 수량은 1.0이 아니라 판매수량 그대로 유지돼야 한다.
    receipts = pd.DataFrame({
        "item_id": ["a", "a", "a", "b"],
        "date": pd.to_datetime(["2025-01-01"] * 4),
        "hour": [9, 14, 15, 10],
        "qty": [2.0, 4.0, 50.0, 3.0],
        "is_bulk": [False, False, True, False],
    })
    out = _receipts_profile_frame(receipts, {"a", "b"})
    # bulk 라인(qty=50) 제외 → 3행
    assert len(out) == 3
    assert 50.0 not in out["qty"].to_numpy()
    # a는 정상 2행(qty 2,4) 유지, 합 6 (footfall 1-count였다면 2였을 것)
    assert out[out["item_id"] == "a"]["qty"].sum() == 6.0
    assert out[out["item_id"] == "b"]["qty"].sum() == 3.0
    assert list(out.columns) == ["item_id", "date", "hour", "qty"]


def _cat_totals():
    """카테고리 총량 fold 예측 fixture (2 fold: fold0=이른=cal, fold1=늦은=test)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
                                "2025-02-01", "2025-02-02"]),
        "fold": [0, 0, 0, 0, 1, 1],
        "base_median": [100.0, 100.0, 100.0, 100.0, 200.0, 200.0],
        "base_prod": [130.0, 130.0, 130.0, 130.0, 260.0, 260.0],
        "actual": [110.0, 120.0, 130.0, 140.0, 0.0, 0.0],
    })


def test_apply_category_margin_quantile_is_base_prod():
    totals = _cat_totals()
    out = _apply_category_margin(totals, "quantile")
    # quantile = production-quantile 직접, 전 fold 유지
    assert out["total_order"].to_numpy().tolist() == [130.0, 130.0, 130.0, 130.0, 260.0, 260.0]


def test_apply_category_margin_nk_is_mult_add_on_base_median():
    # nk 버퍼는 base_median(100/200) 기준(base_prod 130/260 아님) — conformal과 동일 base.
    totals = _cat_totals()
    add = _apply_category_margin(totals, "nk", nk_mult=1.0, nk_add=15.0)
    assert add["total_order"].to_numpy().tolist() == [115.0]*4 + [215.0]*2
    mult = _apply_category_margin(totals, "nk", nk_mult=1.2, nk_add=0.0)
    assert mult["total_order"].to_numpy() == pytest.approx([120.0]*4 + [240.0]*2)


def test_apply_category_margin_conformal_uses_median_base_and_level_scale():
    totals = _cat_totals()
    # cal(fold0) scores E=(actual−base_median)/base_median = [0.1,0.2,0.3,0.4]
    q_s = ConformalOrderCalibrator().fit(np.array([0.1, 0.2, 0.3, 0.4]), 0.75).q_s
    out = _apply_category_margin(totals, "conformal", service_level=0.75, cal_fold_frac=0.5)
    # cal fold(0) 드롭, test fold(1)만. order = base_median + q_s×base_median = 200×(1+q_s)
    assert out["fold"].unique().tolist() == [1]
    assert out["total_order"].to_numpy() == pytest.approx([200.0 * (1 + q_s)] * 2)


def test_receipts_profile_frame_legacy_parquet_falls_back_to_footfall():
    # qty·is_bulk 컬럼 없는 legacy parquet → qty=1.0(footfall), 전 행 보존.
    receipts = pd.DataFrame({
        "item_id": ["a", "a", "c"],
        "date": pd.to_datetime(["2025-01-01"] * 3),
        "hour": [9, 14, 10],
    })
    out = _receipts_profile_frame(receipts, {"a"})
    assert len(out) == 2  # item filter만
    assert out["qty"].to_numpy().tolist() == [1.0, 1.0]

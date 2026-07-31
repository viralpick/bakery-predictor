"""order_level='item' 배선 — 배분은 **가법**이고 총량 경로를 건드리지 않는다."""
import numpy as np
import pandas as pd
import pytest

from bakery.harness.backtest_core import windowed_backtest

TARGET = "y"
N_FOLDS = 3
HORIZON = 7
WINDOW_DAYS = 120


def _category_frame(n_days: int = 400) -> pd.DataFrame:
    """단조 추세 + 요일 효과가 있는 합성 카테고리 일별 프레임."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    dow_lift = np.where(dates.dayofweek >= 5, 1.3, 1.0)
    y = 100.0 + np.arange(n_days) * 0.1
    return pd.DataFrame({"date": dates, TARGET: y * dow_lift, "feat": np.arange(n_days) % 13})


def _item_history(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """a:b = 1:3 인 2품목 history (배분 비율 0.25 / 0.75)."""
    rows = []
    for day in dates:
        rows.append([day, "a", "bread", 10.0, False, pd.NaT])
        rows.append([day, "b", "bread", 30.0, False, pd.NaT])
    return pd.DataFrame(rows, columns=["date", "item_id", "category_id",
                                       "sold_units", "is_stockout", "stockout_time"])


def _run(order_level: str, *, item_history=None, lead_days: int = 0):
    df = _category_frame()
    return windowed_backtest(
        df, window_days=WINDOW_DAYS, target_col=TARGET, n_folds=N_FOLDS,
        horizon_days=HORIZON, order_level=order_level, item_history=item_history,
        lead_days=lead_days,
    )


@pytest.fixture(scope="module")
def runs():
    df = _category_frame()
    hist = _item_history(pd.DatetimeIndex(df["date"]))
    return {
        "category": _run("category"),
        "item": _run("item", item_history=hist),
    }


def test_item_orders_absent_for_category_level(runs):
    """기본(category)에서는 item_orders가 None — 배분 비용을 안 낸다."""
    assert runs["category"].item_orders is None


def test_distribution_is_additive(runs):
    """★핵심: 배분을 켜도 총량 예측/발주/실측이 **정확히** 같다.

    엔진 동등성 hard gate(rtol=1e-9)가 카테고리 경로를 지키는 것과 같은 취지를
    합성 데이터로 빠르게 고정한다. 여기가 깨지면 배분이 총량을 오염시킨 것이다.
    """
    cat, item = runs["category"].predictions, runs["item"].predictions
    assert len(cat) == len(item)
    for col in ("expected", "production", "actual"):
        np.testing.assert_allclose(item[col].to_numpy(), cat[col].to_numpy(), rtol=1e-12)


def test_fold_metrics_unchanged_by_distribution(runs):
    """fold 지표도 불변이어야 한다(리포트가 이걸 소비한다)."""
    cat, item = runs["category"].folds, runs["item"].folds
    np.testing.assert_allclose(item["wape"].to_numpy(), cat["wape"].to_numpy(), rtol=1e-12)


def test_item_orders_preserve_category_total(runs):
    """★배분은 재분배다 — 날짜별 품목 발주 합 == 카테고리 총 발주."""
    orders, preds = runs["item"].item_orders, runs["item"].predictions
    by_date = orders.groupby("date")["order_qty"].sum()
    total = preds.set_index("date")["production"]
    joined = pd.DataFrame({"dist": by_date, "total": total}).dropna()
    assert len(joined) == N_FOLDS * HORIZON
    np.testing.assert_allclose(joined["dist"].to_numpy(), joined["total"].to_numpy(), rtol=1e-12)


def test_item_orders_shape_and_columns(runs):
    """계약: [date, item_id, fold, order_qty], fold는 총량 fold와 같은 집합."""
    orders = runs["item"].item_orders
    assert list(orders.columns) == ["date", "item_id", "fold", "order_qty"]
    assert sorted(orders["fold"].unique().tolist()) == list(range(N_FOLDS))
    assert sorted(orders["item_id"].unique().tolist()) == ["a", "b"]


def test_item_orders_follow_history_proportions(runs):
    """a:b = 1:3 history → 발주도 0.25 : 0.75 (배분 비율이 실제로 흐르는지)."""
    orders = runs["item"].item_orders
    share = orders.groupby("item_id")["order_qty"].sum()
    ratio = share["a"] / (share["a"] + share["b"])
    # popularity/매진/추세 보정이 없는 평탄한 합성 history라 base 비율이 그대로 나온다
    assert ratio == pytest.approx(0.25, abs=0.02)


def test_item_level_requires_history():
    """order_level='item' + history 없음 → fails-loud."""
    with pytest.raises(ValueError, match="item_history"):
        _run("item")


def test_unknown_order_level_rejected():
    with pytest.raises(ValueError, match="order_level"):
        _run("sku")


def test_lead_days_changes_item_orders_not_shape():
    """★리드타임이 배분에 실제로 전달된다 — 수량은 달라지고 계약은 유지된다.

    PR#74에서 막은 축이 harness 경로로 흐르는지 확인한다. 이 단언이 깨지면
    lead_days가 배분까지 전달되지 않는다는 뜻이다(= 파이프라인 부분 leaky).
    """
    df = _category_frame()
    hist = _item_history(pd.DatetimeIndex(df["date"]))
    # 리드 구간에만 a를 폭증시킨다 — lead_days>0이면 이 실적은 배분에 반영되면 안 된다
    spike_dates = pd.DatetimeIndex(df["date"]).sort_values()[-HORIZON * N_FOLDS - 3:]
    spike = pd.DataFrame({
        "date": spike_dates, "item_id": "a", "category_id": "bread",
        "sold_units": 5000.0, "is_stockout": False, "stockout_time": pd.NaT,
    })
    leaky_hist = pd.concat([hist, spike], ignore_index=True)
    lead0 = _run("item", item_history=leaky_hist, lead_days=0).item_orders
    lead9 = _run("item", item_history=leaky_hist, lead_days=9).item_orders
    a0 = lead0.groupby("item_id")["order_qty"].sum()["a"]
    a9 = lead9.groupby("item_id")["order_qty"].sum()["a"]
    assert a0 != pytest.approx(a9, rel=1e-6)          # 리드타임이 배분을 실제로 바꾼다
    assert list(lead9.columns) == ["date", "item_id", "fold", "order_qty"]


# ---------------------------------------------------------------------------
# kpi 플래그 게이트 — 층위가 안 맞으면 실행 전에 막는다
# ---------------------------------------------------------------------------

def test_kpi_requires_item_order_level():
    """★kpi=true + order_level=category → SpecError.

    폐기·매진시각은 품목 층위가 아니면 정의되지 않는다. 조용히 빈 KPI를 내는 대신
    스펙 로드 시점에 막는다.
    """
    from bakery.harness.config import DataSpec, ExperimentSpec, SpecError, _enforce

    data = DataSpec(source="real", store="store_gw01")
    with pytest.raises(SpecError, match="order_level"):
        _enforce(ExperimentSpec(name="x", data=data, kpi=True))


def test_kpi_with_item_order_level_allowed():
    from bakery.harness.config import DataSpec, ExperimentSpec, _enforce

    data = DataSpec(source="real", store="store_gw01")
    _enforce(ExperimentSpec(name="x", data=data, kpi=True, order_level="item"))


def test_kpi_defaults_off():
    """기본은 off — 헤드라인 실험이 무거운 KPI 경로를 타지 않는다."""
    from bakery.harness.config import DataSpec, ExperimentSpec

    spec = ExperimentSpec(name="x", data=DataSpec(source="real", store="store_gw01"))
    assert spec.kpi is False
    assert spec.order_level == "category"


# ---------------------------------------------------------------------------
# ★영업 종료 시각 — 시트값을 쓰면 안 된다
# ---------------------------------------------------------------------------

def test_kpi_close_hour_is_22():
    """★광교 영업 종료 = 22시. 21로 되돌아가면 매진시각 지표가 조용히 틀린다.

    광교(**점포코드 1000000047**) `영업시간` 시트 median **21.93시**, 2026 상반기 21.83시.
    영수증 마지막 판매 median 21.82시와 날짜별 **76%가 15분 내 일치**(상관 0.495)하고,
    시트값이 오르면 마지막 판매도 오른다(20시대→20.87 / 21시대→21.70 / 22시+→22.12).
    반올림하면 22시이고 architect 확인과 일치한다.
    레거시 `scripts/unified_policy_kpi.py` 도 8,22를 쓴다(그쪽이 맞았다).

    ⚠️**매장 코드 혼동 주의**: 처음 이 상수를 21로 잡은 이유가 **삼성타운(1000000009)**
    시트값(median 21.42)을 광교 영수증과 비교한 교차매장 오류였다. 광교는 1000000047이다.
    """
    from bakery.harness.runner import KPI_CLOSE_HOUR, KPI_OPEN_HOUR

    assert KPI_CLOSE_HOUR == 22
    assert KPI_OPEN_HOUR == 8


def test_kpi_hours_match_legacy_script():
    """백본과 레거시 KPI 경로의 영업시간이 같아야 한다 — 다르면 두 수치가 비교 불가."""
    import re
    from pathlib import Path

    from bakery.harness.runner import KPI_CLOSE_HOUR, KPI_OPEN_HOUR

    src = Path("scripts/unified_policy_kpi.py").read_text(encoding="utf-8")
    found = re.search(r"StoreHours\(STORE,\s*(\d+),\s*(\d+)\)", src)
    assert found is not None, "레거시 스크립트에서 StoreHours 호출을 찾지 못했다"
    assert (int(found.group(1)), int(found.group(2))) == (KPI_OPEN_HOUR, KPI_CLOSE_HOUR)

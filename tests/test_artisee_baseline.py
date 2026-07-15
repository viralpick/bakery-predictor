from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bakery.cli import _artisee_baseline_order
from bakery.evaluation.prospective import compare_policies, simulate_item_day_kpis
from bakery.features.potential_demand import StoreHours
from bakery.models.artisee_baseline import (
    ArtiseeBaseline,
    applied_quantity,
    build_item_residual_curve,
    dow_group,
    dow_scaling,
    round_order,
    soldout_multiplier,
)


def _daily(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_applied_quantity_weekday_weekend_means():
    # 2026-06-01(월)~06-21(일) = 3주. 주중 sold=10, 주말 sold=20, 휴일 없음.
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-21"):
        sold = 20 if d.dayofweek >= 5 else 10
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": sold, "is_stockout": False,
                     "stockout_time": pd.NaT, "is_holiday": False})
    out = applied_quantity(_daily(rows), weeks=3)
    got = out.set_index("dow_group")["base_qty"].to_dict()
    assert got["weekday"] == pytest.approx(10.0)
    assert got["weekend"] == pytest.approx(20.0)


def test_applied_quantity_excludes_holiday():
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-21"):
        holiday = d.date() == pd.Timestamp("2026-06-03").date()
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": 100 if holiday else 10, "is_stockout": False,
                     "stockout_time": pd.NaT, "is_holiday": holiday})
    out = applied_quantity(_daily(rows), weeks=3)
    # 06-03(수) 휴일 제외 → 주중 평균은 100에 오염되지 않고 10.
    assert out.set_index("dow_group").loc["weekday", "base_qty"] == pytest.approx(10.0)


def test_applied_quantity_caps_spike():
    rows = []
    for i, d in enumerate(pd.date_range("2026-06-01", "2026-06-19")):  # 주중만 관심
        sold = 10
        if d.date() == pd.Timestamp("2026-06-10").date():
            sold = 100  # 스파이크
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": sold, "is_stockout": False,
                     "stockout_time": pd.NaT, "is_holiday": False})
    out = applied_quantity(_daily(rows), weeks=3, spike_ratio=1.3)
    wk = out.set_index("dow_group").loc["weekday", "base_qty"]
    # median=10 → cap=13. 스파이크(100) 적용 시 capped sum = 14×10 + 13 = 153; mean = 153/15 = 10.2.
    assert wk == pytest.approx(10.2)


def test_residual_curve_shape_and_values():
    # 하루: 07시 6개, 12시 4개(누적10). 다른 날도 동일 분포.
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-10"):
        rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 7, "qty": 6.0})
        rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 12, "qty": 4.0})
    hourly = pd.DataFrame(rows)
    hourly["date"] = pd.to_datetime(hourly["date"])
    curves = build_item_residual_curve(hourly, months=3)
    curve = curves["A"]
    assert curve.shape == (24,)
    # 07시 직후 잔여 = 1 - 6/10 = 0.4; 12시 직후 = 1 - 10/10 = 0.0.
    assert curve[7] == pytest.approx(0.4)
    assert curve[12] == pytest.approx(0.0)
    # 07시 이전(예: 06시)은 아직 아무것도 안 팔림 → 잔여 1.0.
    assert curve[6] == pytest.approx(1.0)


def test_residual_curve_skips_zero_demand_day():
    # 정상일 1개(07시 10개) + 전일 qty=0인 날 1개 → 0일은 제외되고 곡선 불변.
    rows = [
        {"store_id": "S", "item_id": "A", "date": pd.Timestamp("2026-06-01"), "hour": 7, "qty": 10.0},
        {"store_id": "S", "item_id": "A", "date": pd.Timestamp("2026-06-02"), "hour": 7, "qty": 0.0},
    ]
    hourly = pd.DataFrame(rows)
    curves = build_item_residual_curve(hourly, months=3)
    # 07시 직후 잔여 = 1 - 10/10 = 0.0 (0일 미포함, 정상일만 평균).
    assert curves["A"][7] == pytest.approx(0.0)
    assert curves["A"][6] == pytest.approx(1.0)


def test_soldout_multiplier_reads_curve_at_stockout_hour():
    # 곡선: 12시 잔여 0.4. 주중 3일 매진(12시), 나머지 매진無.
    curves = {"A": np.array([1.0]*12 + [0.4] + [0.0]*11)}
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-05"):  # 월~금
        so = d.day <= 3
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": 10, "is_stockout": so, "is_holiday": False,
                     "stockout_time": (d + pd.Timedelta(hours=12)) if so else pd.NaT})
    daily = pd.DataFrame(rows); daily["date"] = pd.to_datetime(daily["date"])
    out = soldout_multiplier(daily, curves, weeks=3)
    wk = out.set_index("dow_group").loc["weekday", "multiplier"]
    # 놓친% = [0.4, 0.4, 0.4, 0, 0] 평균 = 0.24 → 1.24.
    assert wk == pytest.approx(1.24)


def test_soldout_multiplier_no_stockout_is_one():
    curves = {"A": np.array([1.0]*24)}
    rows = [{"store_id": "S", "item_id": "A", "date": d, "sold_units": 10,
             "is_stockout": False, "is_holiday": False, "stockout_time": pd.NaT}
            for d in pd.date_range("2026-06-01", "2026-06-05")]
    daily = pd.DataFrame(rows); daily["date"] = pd.to_datetime(daily["date"])
    out = soldout_multiplier(daily, curves, weeks=3)
    assert out.set_index("dow_group").loc["weekday", "multiplier"] == pytest.approx(1.0)


def test_dow_scaling_ratio_within_group():
    # 주중: 월=20, 화~금=10. 주중평균 = (20+10+10+10+10)/5 = 12.
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-19"):  # 3주 주중 15일
        if d.dayofweek >= 5:
            continue
        sold = 20 if d.dayofweek == 0 else 10
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": sold, "is_holiday": False, "is_stockout": False,
                     "stockout_time": pd.NaT})
    daily = pd.DataFrame(rows); daily["date"] = pd.to_datetime(daily["date"])
    out = dow_scaling(daily, weeks=3)
    w = out.set_index("dow")["weight"].to_dict()
    assert w[0] == pytest.approx(20.0 / 12.0)  # 월
    assert w[1] == pytest.approx(10.0 / 12.0)  # 화


def _make_history():
    daily_rows, hourly_rows = [], []
    for d in pd.date_range("2026-06-01", "2026-06-21"):
        sold = 20 if d.dayofweek >= 5 else 10
        so = (d.dayofweek < 5) and (d.day % 5 == 0)
        daily_rows.append({"store_id": "S", "item_id": "A", "date": d,
                           "sold_units": sold, "is_stockout": so, "is_holiday": False,
                           "stockout_time": (d + pd.Timedelta(hours=12)) if so else pd.NaT})
        hourly_rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 7, "qty": 6.0})
        hourly_rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 12, "qty": 4.0})
    daily = pd.DataFrame(daily_rows)
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.DataFrame(hourly_rows)
    hourly["date"] = pd.to_datetime(hourly["date"])
    return daily, hourly


def test_round_order_generic_and_multiple():
    raw = pd.Series([12.4, 12.6, 13.0])
    items = pd.Series(["A", "A", "B"])
    generic = round_order(raw, items, rounding="generic")
    assert list(generic) == [12.0, 13.0, 13.0]
    mult = round_order(raw, items, rounding="multiple", multiple_map={"A": 3, "B": 6})
    # A=3배수 floor: 12.4→12, 12.6→12; B=6배수 floor: 13→12.
    assert list(mult) == [12.0, 12.0, 12.0]


def test_artisee_baseline_predict_positive_order():
    daily, hourly = _make_history()
    model = ArtiseeBaseline(weeks=3, curve_months=3).fit(daily, hourly)
    target = pd.DataFrame({"store_id": ["S", "S"], "item_id": ["A", "A"],
                           "date": pd.to_datetime(["2026-06-22", "2026-06-27"])})  # 월, 토
    pred = model.predict(target)
    assert (pred.to_numpy() > 0).all()
    assert pred.index.equals(target.index)
    # 주말(토) 제시량 > 주중(월) — weekend base 20 > weekday base 10.
    assert pred.iloc[1] > pred.iloc[0]


def test_predict_ignores_out_of_window_history():
    daily, hourly = _make_history()
    target = pd.DataFrame({"store_id": ["S"], "item_id": ["A"],
                           "date": pd.to_datetime(["2026-06-22"])})
    base = ArtiseeBaseline().fit(daily, hourly).predict(target)
    # window 밖 과거를 검증: fit은 max(date) 기준 3주(weeks=3)만 봄.
    old = daily.copy()
    old_extra = daily.head(1).copy()
    old_extra["date"] = pd.to_datetime(["2026-01-01"]); old_extra["sold_units"] = 9999
    old = pd.concat([old_extra, old], ignore_index=True)
    with_old = ArtiseeBaseline().fit(old, hourly).predict(target)
    # 2026-01-01은 3주 창(06-01 이전) 밖 → 예측 불변.
    assert with_old.iloc[0] == base.iloc[0]


def test_artisee_order_feeds_prospective_compare():
    daily, hourly = _make_history()
    model = ArtiseeBaseline().fit(daily, hourly)
    rows = pd.DataFrame({
        "store_id": ["S", "S"], "item_id": ["A", "A"],
        "date": pd.to_datetime(["2026-06-22", "2026-06-23"]),
        "potential_demand": [10.0, 11.0], "our_order": [12.0, 12.0],
    })
    rows["artisee_order"] = model.predict(rows).to_numpy()
    hours = StoreHours(store_id="S", open_hour=7, close_hour=22)
    profiles: dict[tuple, np.ndarray] = {}
    ours = simulate_item_day_kpis(rows, profiles, order_col="our_order",
                                  store_hours=hours, group_cols=["item_id"],
                                  demand_col="potential_demand")
    theirs = simulate_item_day_kpis(rows, profiles, order_col="artisee_order",
                                    store_hours=hours, group_cols=["item_id"],
                                    demand_col="potential_demand")
    cmp = compare_policies(ours, theirs)
    assert set(cmp["policy"]) == {"our", "baseline", "delta"}
    assert "waste_cost_krw" in cmp.columns


def test_predict_holiday_target_uses_weekend_treatment():
    """M1: is_holiday=True인 target date는 dow_group=weekend + 대표요일(토=5)로 취급.

    _make_history(): 주중 sold=10, 주말 sold=20, 매진시각=정오(=잔여수요 0 → 증산배수
    1.0, 주중/주말 동일)라서 두 그룹의 차이는 순수 base_qty(10 vs 20)뿐 — 정확값 검증 가능.
    """
    daily, hourly = _make_history()
    model = ArtiseeBaseline(weeks=3, curve_months=3).fit(daily, hourly)
    target = pd.DataFrame({"store_id": ["S"], "item_id": ["A"],
                           "date": pd.to_datetime(["2026-06-22"])})  # 월(weekday)
    weekday_order = model.predict(target)
    assert weekday_order.iloc[0] == pytest.approx(10.0)

    holiday_target = target.copy()
    holiday_target["is_holiday"] = [True]
    holiday_order = model.predict(holiday_target)
    assert holiday_order.iloc[0] == pytest.approx(20.0)
    assert holiday_order.iloc[0] > weekday_order.iloc[0]

    # is_holiday 컬럼 부재 시 기존 동작 완전 불변(하위호환 가드).
    unchanged = model.predict(target)
    assert unchanged.iloc[0] == weekday_order.iloc[0]


def _calendar_stub():
    dates = pd.date_range("2026-01-01", "2026-12-31")
    return pd.DataFrame({
        "date": dates, "is_public_holiday": False,
        "off_streak_length": 0, "off_position_in_streak": 0,
        "is_white_day": False,
    })


def _patch_real_loaders(monkeypatch, daily, hourly):
    """_artisee_baseline_order가 부르는 3개 real 로더를 고정 프레임으로 대체."""
    monkeypatch.setattr("bakery.cli._load_dataset",
                        lambda source, data_dir: SimpleNamespace(calendar=_calendar_stub()))
    monkeypatch.setattr("bakery.cli._load_real_daily", lambda store_id: daily.copy())
    monkeypatch.setattr("bakery.cli._load_real_receipts", lambda item_ids: hourly.copy())


def test_artisee_baseline_order_cli_ignores_future_history(monkeypatch):
    """_artisee_baseline_order의 cutoff=rows['date'].min() 트렁케이션 가드.

    모델 자체(ArtiseeBaseline.fit)는 받은 daily의 max(date) 기준으로 창을 잡으므로,
    leakage 방지는 전적으로 CLI 헬퍼가 rows 평가기간 이전으로 daily/hourly를
    잘라내는 데 달려 있다(cli.py의 `daily[daily["date"] < cutoff]` /
    `hourly[hourly["date"] < cutoff]`). 이 테스트는 그 트렁케이션이 없으면
    실패해야 한다: cutoff 당일·이후에 극단값 행을 추가해도 잘려나가 예측이
    바뀌지 않아야 정상이다.
    """
    daily, hourly = _make_history()  # 2026-06-01 ~ 2026-06-21, store S / item A
    rows = pd.DataFrame({"item_id": ["A"], "date": pd.to_datetime(["2026-06-22"])})

    _patch_real_loaders(monkeypatch, daily, hourly)
    base = _artisee_baseline_order("S", rows)

    future_daily = daily.tail(1).copy()
    future_daily["date"] = pd.to_datetime(["2026-06-22"])  # == cutoff
    future_daily["sold_units"] = 9999
    future_daily["is_stockout"] = False
    future_daily["stockout_time"] = pd.NaT
    leaked_daily_row = daily.tail(1).copy()
    leaked_daily_row["date"] = pd.to_datetime(["2026-06-25"])  # cutoff 이후
    leaked_daily_row["sold_units"] = 9999
    leaked_daily = pd.concat([daily, future_daily, leaked_daily_row], ignore_index=True)

    future_hourly = hourly.tail(1).copy()
    future_hourly["date"] = pd.to_datetime(["2026-06-22"])
    future_hourly["qty"] = 9999.0
    leaked_hourly_row = hourly.tail(1).copy()
    leaked_hourly_row["date"] = pd.to_datetime(["2026-06-25"])
    leaked_hourly_row["qty"] = 9999.0
    leaked_hourly = pd.concat([hourly, future_hourly, leaked_hourly_row], ignore_index=True)

    _patch_real_loaders(monkeypatch, leaked_daily, leaked_hourly)
    leaked = _artisee_baseline_order("S", rows)

    assert leaked.iloc[0] == base.iloc[0]



def test_artisee_baseline_order_per_fold_refit_uses_fold_recency(monkeypatch):
    """I1: rows에 fold 컬럼이 있으면 fold별 cutoff=그 fold 타깃 최소일로 재fit한다.

    2026-06-01~07-12(6주) 리짐 전환: 처음 3주(06-01~06-21) 주중 sold=10, 이후 3주
    (06-22~07-12) 주중 sold=100. fold=1(이른 target 2026-06-22)의 cutoff 이전 3주엔
    옛 값(10)만 존재 → base_qty=10. fold=0(늦은 target 2026-07-13, our_order 관례상
    fold 0=최신)의 cutoff 이전 3주엔 새 값(100)이 존재 → base_qty=100. 단일 정적
    cutoff(옛 구현)였다면 두 fold 모두 옛 base(10)에 묶여 late_fold_order도 10이
    됐을 것 — 이 회귀를 잡는다.
    """
    dates = pd.date_range("2026-06-01", "2026-07-12")
    daily_rows, hourly_rows = [], []
    for d in dates:
        is_weekday = d.dayofweek < 5
        if is_weekday:
            sold = 100 if d >= pd.Timestamp("2026-06-22") else 10
        else:
            sold = 20
        daily_rows.append({"store_id": "S", "item_id": "A", "date": d,
                           "sold_units": sold, "is_stockout": False,
                           "stockout_time": pd.NaT, "is_holiday": False})
        hourly_rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 7, "qty": 6.0})
        hourly_rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 12, "qty": 4.0})
    daily = pd.DataFrame(daily_rows)
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.DataFrame(hourly_rows)
    hourly["date"] = pd.to_datetime(hourly["date"])

    _patch_real_loaders(monkeypatch, daily, hourly)
    rows = pd.DataFrame({
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2026-06-22", "2026-07-13"]),  # 둘 다 월요일
        "fold": [1, 0],
    })
    order = _artisee_baseline_order("S", rows)
    assert order.iloc[0] == pytest.approx(10.0)   # fold=1, 이른 target → 옛 레짐
    assert order.iloc[1] == pytest.approx(100.0)  # fold=0, 늦은 target → 새 레짐

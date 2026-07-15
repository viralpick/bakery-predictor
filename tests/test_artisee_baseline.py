import numpy as np
import pandas as pd
import pytest

from bakery.models.artisee_baseline import applied_quantity, build_item_residual_curve, dow_group


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

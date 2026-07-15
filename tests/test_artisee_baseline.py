import numpy as np
import pandas as pd
import pytest

from bakery.models.artisee_baseline import applied_quantity, dow_group


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
    # median≈10 → 캡 13. 100이 13으로 눌려 평균이 10 근처(spike 미적용 시 훨씬 큼).
    assert wk < 15.0

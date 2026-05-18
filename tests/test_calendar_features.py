"""Pin behavior of the calendar daily frame and the merge helper.

These features are functions of `date` alone, so the regression here is
mostly: streak/adjacency logic is right, event flags hit the right dates,
and merging is leakage-safe by virtue of being a date→features lookup.
"""

from __future__ import annotations

import pandas as pd

from bakery.data.calendar import build_calendar_daily, validate_calendar
from bakery.features.calendar_features import CALENDAR_FEATURE_COLUMNS, add_calendar_features


def _cal_2024() -> pd.DataFrame:
    return build_calendar_daily("2024-01-01", "2024-12-31")


def test_calendar_frame_validates():
    cal = _cal_2024()
    validate_calendar(cal)
    assert len(cal) == 366  # 2024 is a leap year


def test_xmas_and_event_flags():
    cal = _cal_2024()
    xmas = cal[cal["date"] == pd.Timestamp("2024-12-25")].iloc[0]
    assert xmas["is_xmas"] == 1
    assert xmas["is_public_holiday"] == 1  # 성탄절은 한국 공휴일
    eve = cal[cal["date"] == pd.Timestamp("2024-12-24")].iloc[0]
    assert eve["is_xmas_eve"] == 1
    assert eve["is_xmas"] == 0
    valentine = cal[cal["date"] == pd.Timestamp("2024-02-14")].iloc[0]
    assert valentine["is_valentine"] == 1
    pepero = cal[cal["date"] == pd.Timestamp("2024-11-11")].iloc[0]
    assert pepero["is_pepero"] == 1


def test_weekend_is_off_day_but_not_public_holiday():
    cal = _cal_2024()
    sat = cal[cal["date"] == pd.Timestamp("2024-01-06")].iloc[0]  # 토요일
    assert sat["is_weekend"] == 1
    assert sat["is_off_day"] == 1
    # 1월 6일은 공휴일 아님
    assert sat["is_public_holiday"] == 0


def test_seollal_streak_length_2024():
    # 2024년 설날 연휴: 2/9(금) ~ 2/12(월, 대체공휴일). 2/10이 토요일.
    # 사실상 2/9~2/12가 4일 연속 off (금:설연휴 전날 ↘ 토:weekend+설연휴, 일:설연휴, 월:대체)
    cal = _cal_2024()
    seollal = cal[
        (cal["date"] >= pd.Timestamp("2024-02-09"))
        & (cal["date"] <= pd.Timestamp("2024-02-12"))
    ]
    # 4일 모두 off
    assert (seollal["is_off_day"] == 1).all()
    # streak 길이 4 (앞뒤가 평일이라 끊긴다 — 2/8은 화요일 working day)
    assert (seollal["off_streak_length"] == 4).all()
    # position 1,2,3,4
    assert list(seollal["off_position_in_streak"]) == [1, 2, 3, 4]


def test_day_before_and_after_off_flags():
    cal = _cal_2024()
    # 2/8(목)은 설연휴 직전 working day → is_day_before_off=1
    thu = cal[cal["date"] == pd.Timestamp("2024-02-08")].iloc[0]
    assert thu["is_off_day"] == 0
    assert thu["is_day_before_off"] == 1
    assert thu["is_day_after_off"] == 0
    # 2/13(화)은 설연휴 직후 working day → is_day_after_off=1
    tue = cal[cal["date"] == pd.Timestamp("2024-02-13")].iloc[0]
    assert tue["is_off_day"] == 0
    assert tue["is_day_after_off"] == 1
    assert tue["is_day_before_off"] == 0


def test_substitute_holiday_detected():
    cal = _cal_2024()
    # 2/12 월요일은 설연휴 대체공휴일
    sub = cal[cal["date"] == pd.Timestamp("2024-02-12")].iloc[0]
    assert sub["is_substitute_holiday"] == 1
    assert sub["is_public_holiday"] == 1


def test_add_calendar_features_merge_shape():
    cal = build_calendar_daily("2024-01-01", "2024-02-29")
    sales = pd.DataFrame(
        {
            "store_id": ["s1"] * 10,
            "item_id": ["i1"] * 10,
            "date": pd.date_range("2024-02-05", periods=10, freq="D"),
            "sold_units": list(range(10)),
        }
    )
    merged = add_calendar_features(sales, cal)
    for col in CALENDAR_FEATURE_COLUMNS:
        assert col in merged.columns
    assert len(merged) == 10


def test_add_calendar_features_no_future_leakage():
    """Merging is by date; mutating a future calendar row cannot change past rows."""
    cal = build_calendar_daily("2024-01-01", "2024-12-31")
    sales = pd.DataFrame(
        {
            "store_id": ["s1"] * 30,
            "item_id": ["i1"] * 30,
            "date": pd.date_range("2024-02-01", periods=30, freq="D"),
            "sold_units": [10] * 30,
        }
    )
    merged_a = add_calendar_features(sales, cal)
    cal_mut = cal.copy()
    pivot = pd.Timestamp("2024-03-01")
    cal_mut.loc[cal_mut["date"] >= pivot, "off_streak_length"] = 99
    merged_b = add_calendar_features(sales, cal_mut)
    past_a = merged_a[merged_a["date"] < pivot][CALENDAR_FEATURE_COLUMNS]
    past_b = merged_b[merged_b["date"] < pivot][CALENDAR_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(past_a, past_b)

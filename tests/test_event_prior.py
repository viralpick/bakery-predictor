import numpy as np
import pandas as pd
import pytest

from bakery.models.event_prior import EventLevelPrior

TARGET = "adjusted_demand_unit"


def _lunar_daily():
    # 추석 date-map (실제 LUNAR_EVENT_DATES 형식)으로 합성 시리즈
    chuseok = {2021: "2021-09-21", 2022: "2022-09-10", 2023: "2023-09-29", 2024: "2024-09-17"}
    dates = pd.date_range("2021-06-01", periods=1400, freq="D")
    df = pd.DataFrame({"date": dates, TARGET: 100.0})
    for yr, lvl in {2021: 200.0, 2022: 210.0, 2023: 220.0}.items():
        df.loc[df["date"] == pd.Timestamp(chuseok[yr]), TARGET] = lvl
    return df, {"chuseok": chuseok}


def _daily(start="2021-06-01", periods=1800, value=100.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    df = pd.DataFrame({"date": dates, TARGET: float(value)})
    # xmas 마다 뚜렷한 레벨 심기: 2021=300, 2022=310, 2023=320, 2024=330
    for yr, lvl in {2021: 300.0, 2022: 310.0, 2023: 320.0, 2024: 330.0}.items():
        df.loc[df["date"] == pd.Timestamp(yr, 12, 25), TARGET] = lvl
    return df


def test_is_event_day():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    assert p.is_event_day(pd.Timestamp(2023, 12, 25)) is True
    assert p.is_event_day(pd.Timestamp(2023, 12, 24)) is False


def test_level_for_averages_strictly_past_events():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    # 2024-12-25 예측 → 과거 3개(300,310,320) 평균 = 310, n_past=3
    level, n_past = p.level_for(pd.Timestamp(2024, 12, 25))
    assert n_past == 3
    assert level == pytest.approx(310.0)


def test_level_for_uses_median_not_mean():
    # 과거 4개 이벤트에 outlier 심어 median≠mean 구분
    df = _daily()
    # 2025 xmas 예측 시 과거 2021..2024 = {300,310,320,330} + outlier 하나로 교체
    df.loc[df["date"] == pd.Timestamp(2024, 12, 25), TARGET] = 900.0  # outlier
    p = EventLevelPrior().fit(df, target_col=TARGET)
    level, n_past = p.level_for(pd.Timestamp(2025, 12, 25))
    assert n_past == 4
    # median of [300,310,320,900] = (310+320)/2 = 315  (mean would be 457.5)
    assert level == pytest.approx(315.0)


def test_level_for_first_occurrence_returns_none():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    level, n_past = p.level_for(pd.Timestamp(2021, 12, 25))
    assert level is None
    assert n_past == 0


def test_blend_corrects_only_event_day():
    p = EventLevelPrior(k=1.5).fit(_daily(), target_col=TARGET)
    dates = [pd.Timestamp(2024, 12, 24), pd.Timestamp(2024, 12, 25)]  # non-event, event
    base_exp = np.array([200.0, 240.0])
    base_prod = np.array([230.0, 276.0])  # buffer 1.15x
    exp2, prod2 = p.blend(dates, base_exp, base_prod)
    # non-event day unchanged
    assert exp2[0] == pytest.approx(200.0)
    assert prod2[0] == pytest.approx(230.0)
    # event day: prior=310 (n_past=3), shrink=3/4.5=0.6667
    shrink = 3 / (3 + 1.5)
    expected_exp = shrink * 310.0 + (1 - shrink) * 240.0
    assert exp2[1] == pytest.approx(expected_exp)
    # production keeps buffer ratio: correction = expected_exp/240
    correction = expected_exp / 240.0
    assert prod2[1] == pytest.approx(276.0 * correction)


def test_blend_first_occurrence_unchanged():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    dates = [pd.Timestamp(2021, 12, 25)]  # n_past=0
    exp2, prod2 = p.blend(dates, np.array([250.0]), np.array([280.0]))
    assert exp2[0] == pytest.approx(250.0)
    assert prod2[0] == pytest.approx(280.0)


def test_blend_skips_when_below_min_events():
    # 2022 xmas 예측: 과거 1개(2021=300)뿐 → n_past=1 < min_events=2 → base 유지
    p = EventLevelPrior(min_events=2).fit(_daily(), target_col=TARGET)
    dates = [pd.Timestamp(2022, 12, 25)]
    exp2, prod2 = p.blend(dates, np.array([250.0]), np.array([280.0]))
    assert exp2[0] == pytest.approx(250.0)   # 단일샘플이라 미보정
    assert prod2[0] == pytest.approx(280.0)


def test_blend_applies_when_at_min_events():
    # 2023 xmas 예측: 과거 2개(300,310) median=305, n_past=2 == min_events → 보정
    p = EventLevelPrior(min_events=2).fit(_daily(), target_col=TARGET)
    exp2, _ = p.blend([pd.Timestamp(2023, 12, 25)], np.array([200.0]), np.array([230.0]))
    shrink = 2 / (2 + 1.5)
    assert exp2[0] == pytest.approx(shrink * 305.0 + (1 - shrink) * 200.0)


def test_is_event_day_matches_lunar_date():
    _, lunar = _lunar_daily()
    p = EventLevelPrior(lunar_events=lunar)
    assert p.is_event_day(pd.Timestamp(2023, 9, 29)) is True   # 추석 당일
    assert p.is_event_day(pd.Timestamp(2023, 9, 28)) is False  # 전날


def test_level_for_lunar_uses_past_lunar_actuals():
    df, lunar = _lunar_daily()
    p = EventLevelPrior(lunar_events=lunar).fit(df, target_col=TARGET)
    # 2024 추석(2024-09-17) 예측 → 과거 3개(200,210,220) median=210
    level, n_past = p.level_for(pd.Timestamp(2024, 9, 17))
    assert n_past == 3
    assert level == pytest.approx(210.0)

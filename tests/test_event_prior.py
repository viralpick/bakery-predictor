import numpy as np
import pandas as pd
import pytest

from bakery.models.event_prior import EventLevelPrior

TARGET = "adjusted_demand_unit"


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


def test_level_for_first_occurrence_returns_none():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    level, n_past = p.level_for(pd.Timestamp(2021, 12, 25))
    assert level is None
    assert n_past == 0

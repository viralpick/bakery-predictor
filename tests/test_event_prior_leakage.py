import numpy as np
import pandas as pd
import pytest

from bakery.models.event_prior import EventLevelPrior

TARGET = "adjusted_demand_unit"


def _daily_with_future(future_xmas_value: float) -> pd.DataFrame:
    dates = pd.date_range("2021-06-01", periods=1400, freq="D")
    df = pd.DataFrame({"date": dates, TARGET: 100.0})
    for yr, lvl in {2021: 300.0, 2022: 310.0, 2023: 320.0}.items():
        df.loc[df["date"] == pd.Timestamp(yr, 12, 25), TARGET] = lvl
    # 미래 이벤트(2024-12-25 이후는 데이터 없음이지만, 만약 오염되면 값이 바뀌게)
    df.loc[df["date"] == pd.Timestamp(2023, 12, 25), TARGET] = future_xmas_value
    return df


def test_level_for_ignores_future_events():
    # 예측 date 2023-12-25: 과거는 2021(300),2022(310)만 써야 함 → 2023 자기값 무관
    p_a = EventLevelPrior().fit(_daily_with_future(320.0), target_col=TARGET)
    p_b = EventLevelPrior().fit(_daily_with_future(999.0), target_col=TARGET)
    lvl_a, n_a = p_a.level_for(pd.Timestamp(2023, 12, 25))
    lvl_b, n_b = p_b.level_for(pd.Timestamp(2023, 12, 25))
    assert n_a == 2 and n_b == 2
    assert lvl_a == pytest.approx(305.0)  # (300+310)/2
    assert lvl_b == pytest.approx(305.0)  # 미래 자기값 오염 없음


def test_blend_at_date_unaffected_by_same_or_future_data():
    p = EventLevelPrior().fit(_daily_with_future(320.0), target_col=TARGET)
    exp2, _ = p.blend([pd.Timestamp(2023, 12, 25)], np.array([200.0]), np.array([230.0]))
    # prior=305, n_past=2, shrink=2/3.5
    shrink = 2 / (2 + 1.5)
    assert exp2[0] == pytest.approx(shrink * 305.0 + (1 - shrink) * 200.0)

"""운영 예측 패널 프리미티브 — leakage 회귀 + 출처 날짜 검증.

패널의 존재 이유는 "모든 feature가 원점 시점에 가용"이다. 그래서 테스트는 값이 아니라
**출처 날짜**를 검증한다 — 어떤 feature도 원점 이후를 출처로 가질 수 없다.
"""
import numpy as np
import pandas as pd
import pytest

from bakery.features.category_aggregate import CategoryDaily, build_category_daily
from bakery.features.forecast_panel import (
    DEFAULT_OFFSETS,
    ORIGIN_LAGS,
    SAME_DOW_COUNT,
    build_forecast_panel,
    origin_feature_columns,
    same_dow_back_days,
)

TARGET = "adjusted_demand_unit"


@pytest.fixture(scope="module")
def base():
    return build_category_daily().df


@pytest.fixture(scope="module")
def panel(base):
    return build_forecast_panel(CategoryDaily(df=base, alpha=0.8), target_col=TARGET)


def test_same_dow_back_days_exact():
    """대상일 요일은 offset으로 정해지므로 되돌아갈 일수도 offset만의 함수다."""
    assert {h: same_dow_back_days(h) for h in DEFAULT_OFFSETS} == {
        6: 1, 7: 0, 8: 6, 9: 5, 10: 4, 11: 3, 12: 2,
    }


def test_horizon_offset_is_target_minus_origin(panel):
    delta = (panel["target_date"] - panel["origin_date"]).dt.days
    assert list(delta.unique()) == sorted(panel["horizon_offset"].unique())
    assert (delta == panel["horizon_offset"]).all()


def test_actual_matches_base(panel, base):
    """대상일 실측이 원본과 일치(inner join 정합)."""
    truth = base.set_index("date")[TARGET]
    got = panel.set_index("target_date")[TARGET]
    assert np.allclose(got.to_numpy(), truth.reindex(got.index).to_numpy(), rtol=1e-12)


def test_origin_lag_source_dates(panel, base):
    """y_origin_lag{k} 의 출처는 정확히 origin − k 일이다(원점 이후 아님)."""
    truth = base.set_index("date")[TARGET]
    sample = panel[panel["horizon_offset"] == 6].tail(200)
    for lag in ORIGIN_LAGS:
        src = sample["origin_date"] - pd.Timedelta(days=lag)
        assert (src <= sample["origin_date"]).all()
        assert np.allclose(
            sample[f"y_origin_lag{lag}"].to_numpy(),
            truth.reindex(src).to_numpy(), rtol=1e-12, equal_nan=True,
        )


def test_same_dow_latest_source_is_same_weekday_at_or_before_origin(panel, base):
    """★핵심: 출처가 (a) 대상일과 같은 요일 (b) 원점 이하 — 두 조건을 동시에 만족."""
    truth = base.set_index("date")[TARGET]
    sample = panel.tail(700)
    back = sample["horizon_offset"].map(same_dow_back_days)
    src = sample["origin_date"] - pd.to_timedelta(back, unit="D")
    assert (src <= sample["origin_date"]).all()
    assert (src.dt.dayofweek.to_numpy() == sample["target_date"].dt.dayofweek.to_numpy()).all()
    assert np.allclose(
        sample["y_same_dow_latest"].to_numpy(),
        truth.reindex(src).to_numpy(), rtol=1e-12, equal_nan=True,
    )


def test_same_dow_mean4_is_mean_of_last_four_same_weekdays(panel, base):
    truth = base.set_index("date")[TARGET]
    row = panel[panel["horizon_offset"] == 12].tail(1).iloc[0]
    back = same_dow_back_days(12)
    srcs = [row["origin_date"] - pd.Timedelta(days=back + 7 * k) for k in range(SAME_DOW_COUNT)]
    assert all(s <= row["origin_date"] for s in srcs)
    expected = float(np.mean([truth.loc[s] for s in srcs]))
    assert row["y_same_dow_mean4"] == pytest.approx(expected, rel=1e-12)


def test_rolling_window_ends_at_origin_not_after(panel, base):
    """rolling은 원점 당일을 포함하고 그 이후는 보지 않는다(shift(1) 없음이 의도)."""
    truth = base.set_index("date")[TARGET]
    row = panel[panel["horizon_offset"] == 6].tail(1).iloc[0]
    window = pd.date_range(row["origin_date"] - pd.Timedelta(days=6), row["origin_date"])
    assert window.max() == row["origin_date"]
    assert row["y_origin_rmean7"] == pytest.approx(
        float(truth.reindex(window).mean()), rel=1e-12,
    )


def test_no_ar_feature_sources_future(panel):
    """AR feature 12종이 모두 존재하고, 원점 기준 컬럼만 골라진다."""
    cols = origin_feature_columns(panel)
    assert len(cols) == len(ORIGIN_LAGS) + 4 + 2 + 2   # lag4 + rmean/rstd 2쌍 + ewma2 + same_dow2
    assert all(c.startswith(("y_origin_", "y_same_dow_")) for c in cols)


def test_gapped_input_fails_loud(base):
    """날짜 gap 있는 입력은 조용히 위치 shift 되지 않고 에러."""
    gapped = base[base["date"] != base["date"].iloc[10]]
    with pytest.raises(ValueError, match="gap"):
        build_forecast_panel(CategoryDaily(df=gapped, alpha=0.8), target_col=TARGET)


def test_target_date_calendar_features_present(panel):
    """대상일 기준 캘린더 feature가 붙는다(사전에 알 수 있는 축)."""
    for col in ("dow_sin", "is_weekend", "is_public_holiday", "days_to_xmas"):
        assert col in panel.columns

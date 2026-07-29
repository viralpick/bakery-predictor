"""패널 백테스트 — fold가 원점 기준인지 + 학습이 원점 이후를 보지 않는지.

leakage 차단은 fold 정의 한 줄이 전부다: 평가 원점 O에 대해 `target_date <= O` 인 행만
학습한다. 그래서 테스트도 그 한 줄을 직접 겨눈다(train 프레임을 spy로 캡처).
"""
import pandas as pd
import pytest

from bakery.features.category_aggregate import CategoryDaily, build_category_daily
from bakery.features.forecast_panel import DEFAULT_OFFSETS, build_forecast_panel
from bakery.harness.forecasters import CategoryTotalForecaster
from bakery.harness.panel_backtest import TUESDAY, horizon_diagnostics, panel_backtest

TARGET = "adjusted_demand_unit"
N_FOLDS = 3


class _SpyForecaster:
    """fit에 넘어온 train 프레임의 date 최대값을 캡처한다."""

    name = "spy_panel"

    def __init__(self) -> None:
        self.train_max_dates: list[pd.Timestamp] = []
        self._inner = CategoryTotalForecaster()

    def fit(self, train, *, target_col, alpha, production_q):
        self.train_max_dates.append(pd.Timestamp(train["date"].max()))
        return self._inner.fit(
            train, target_col=target_col, alpha=alpha, production_q=production_q,
        )


@pytest.fixture(scope="module")
def panel():
    return build_forecast_panel(
        CategoryDaily(df=build_category_daily().df, alpha=0.8), target_col=TARGET,
    )


@pytest.fixture(scope="module")
def run(panel):
    spy = _SpyForecaster()
    result = panel_backtest(
        panel, window_days=730, n_folds=N_FOLDS, target_col=TARGET,
        production_q=0.85, alpha=0.8, forecaster=spy,
    )
    return result, spy


def test_eval_origins_are_all_tuesday(run):
    result, _ = run
    dows = pd.DatetimeIndex(result.folds["origin_date"]).dayofweek
    assert list(dows) == [TUESDAY] * len(result.folds)


def test_each_fold_covers_full_offset_block(run):
    """fold 하나 = 원점 하나 = offset 전부(월~일 7일)."""
    result, _ = run
    assert list(result.folds["n_test"]) == [len(DEFAULT_OFFSETS)] * len(result.folds)
    assert len(result.predictions) == N_FOLDS * len(DEFAULT_OFFSETS)


def test_fold_targets_are_next_week_monday_to_sunday(run):
    """대상일 블록이 원점+6 ~ 원점+12 (다음주 월~일)."""
    result, _ = run
    for _, row in result.folds.iterrows():
        origin = pd.Timestamp(row["origin_date"])
        assert pd.Timestamp(row["test_start"]) == origin + pd.Timedelta(days=6)
        assert pd.Timestamp(row["test_end"]) == origin + pd.Timedelta(days=12)
        assert pd.Timestamp(row["test_start"]).dayofweek == 0      # 월요일


def test_train_never_uses_targets_after_origin(run):
    """★leakage 회귀: 학습 행의 target_date 최대값 <= 평가 원점."""
    result, spy = run
    origins = [pd.Timestamp(o) for o in result.folds["origin_date"]]
    assert len(spy.train_max_dates) == len(origins)
    assert [m <= o for m, o in zip(spy.train_max_dates, origins, strict=True)] == [True] * len(origins)


def test_predictions_carry_horizon_offset(run):
    result, _ = run
    assert sorted(result.predictions["horizon_offset"].unique()) == list(DEFAULT_OFFSETS)


def test_horizon_diagnostics_one_row_per_offset(run):
    """offset별 진단 — horizon을 feature로 준 것만으로 편향이 자동 해소되진 않으므로 필수."""
    result, _ = run
    diag = horizon_diagnostics(result.predictions)
    assert list(diag["horizon_offset"]) == list(DEFAULT_OFFSETS)
    assert list(diag["n"]) == [N_FOLDS] * len(DEFAULT_OFFSETS)


def test_no_eval_origin_raises(panel):
    """평가 원점이 없으면 조용히 빈 결과를 내지 않고 에러."""
    with pytest.raises(ValueError, match="평가 원점"):
        panel_backtest(panel, window_days=730, n_folds=N_FOLDS, target_col=TARGET,
                       origin_dow=99)

"""운영 리드타임/요일 앵커 opt-in 회귀 테스트.

핵심 계약 3가지:
  1. lead_days=0, anchor_dow=None 은 인자 없이 호출한 것과 정확 일치(헤드라인 불변).
  2. lead_days=d 면 train 최대 날짜 == test_start − (d+1)일 (원점 이후 데이터 미사용).
  3. event_prior history도 동일 cutoff — prior 경로로 leakage가 새지 않는다.
  4. anchor_dow=0 이면 모든 fold가 월요일 시작 + 7일 블록.

실데이터 fold는 비싸므로 n_folds=3으로 작게 유지하고, feature 프레임과 lead 실행 결과는
모듈 스코프에서 1회만 만들어 공유한다.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.backtest_core import _fold_starts_by_dow, windowed_backtest
from bakery.harness.event_priors import resolve_event_priors
from bakery.harness.forecasters import CategoryTotalForecaster
from bakery.models.event_prior import EventLevelPrior

N_FOLDS = 3
HORIZON_DAYS = 7
LEAD_DAYS = 5
MONDAY = 0
# lead_days=5 → cutoff = test_start−5, train은 cutoff 미만 → 최대 날짜 = test_start−6
TRAIN_GAP_DAYS = LEAD_DAYS + 1


class _SpyForecaster:
    """CategoryTotalForecaster를 감싸 fit에 넘어온 train_df의 최대 날짜를 캡처한다."""

    name = "spy_category_total"

    def __init__(self) -> None:
        self.train_max_dates: list[pd.Timestamp] = []
        self._inner = CategoryTotalForecaster()

    def fit(self, train, *, target_col, alpha, production_q):
        self.train_max_dates.append(pd.Timestamp(train["date"].max()))
        return self._inner.fit(
            train, target_col=target_col, alpha=alpha, production_q=production_q,
        )


@pytest.fixture(scope="module")
def feat():
    return build_features(
        build_category_daily(alpha=0.8), target_col="adjusted_demand_unit",
    )


@pytest.fixture(scope="module")
def priors():
    return resolve_event_priors("gwangyo")


@pytest.fixture(scope="module")
def lead_run(feat, priors):
    """lead_days=5 실행 1회 — train 최대일과 prior history 최대일을 함께 캡처."""
    events, lunar = priors
    spy = _SpyForecaster()
    hist_max_dates: list[pd.Timestamp] = []
    original_fit = EventLevelPrior.fit

    def _spy_fit(self, history, *args, **kwargs):
        hist_max_dates.append(pd.Timestamp(history["date"].max()))
        return original_fit(self, history, *args, **kwargs)

    with patch.object(EventLevelPrior, "fit", _spy_fit):
        result = windowed_backtest(
            feat, window_days=730, n_folds=N_FOLDS, horizon_days=HORIZON_DAYS,
            production_q=0.85, alpha=0.8, events=events, lunar_events=lunar,
            forecaster=spy, lead_days=LEAD_DAYS,
        )
    return result, spy.train_max_dates, hist_max_dates


def test_defaults_match_explicit_zero_lead(feat, priors):
    """lead_days=0 / anchor_dow=None 은 기존 호출과 fold·prediction 값이 정확히 같다."""
    events, lunar = priors
    kw = dict(
        window_days=730, n_folds=N_FOLDS, horizon_days=HORIZON_DAYS,
        production_q=0.85, alpha=0.8, events=events, lunar_events=lunar,
    )
    baseline = windowed_backtest(feat, **kw)
    explicit = windowed_backtest(feat, lead_days=0, anchor_dow=None, **kw)
    pd.testing.assert_frame_equal(explicit.folds, baseline.folds, rtol=1e-9)
    pd.testing.assert_frame_equal(explicit.predictions, baseline.predictions, rtol=1e-9)


def test_lead_days_shifts_train_cutoff(lead_run):
    """각 fold의 train 최대 날짜 == test_start − 6일 (lead_days=5)."""
    result, train_max_dates, _ = lead_run
    assert len(train_max_dates) == N_FOLDS
    expected = [
        pd.Timestamp(ts) - pd.Timedelta(days=TRAIN_GAP_DAYS)
        for ts in result.folds.sort_values("fold")["test_start"]
    ]
    assert train_max_dates == expected


def test_event_prior_history_respects_lead_cutoff(lead_run):
    """leakage 회귀: prior fit history 최대 날짜도 test_start − 6일 이하."""
    result, _, hist_max_dates = lead_run
    assert len(hist_max_dates) == N_FOLDS
    limits = [
        pd.Timestamp(ts) - pd.Timedelta(days=TRAIN_GAP_DAYS)
        for ts in result.folds.sort_values("fold")["test_start"]
    ]
    assert [h <= limit for h, limit in zip(hist_max_dates, limits, strict=True)] == [True] * N_FOLDS


def test_anchor_dow_blocks_start_on_monday(feat, priors):
    """anchor_dow=0 이면 모든 fold가 월요일 시작이고 test_end = test_start + 6일."""
    events, lunar = priors
    result = windowed_backtest(
        feat, window_days=730, n_folds=N_FOLDS, horizon_days=HORIZON_DAYS,
        production_q=0.85, alpha=0.8, events=events, lunar_events=lunar,
        lead_days=LEAD_DAYS, anchor_dow=MONDAY,
    )
    folds = result.folds
    assert len(folds) == N_FOLDS
    assert list(pd.DatetimeIndex(folds["test_start"]).dayofweek) == [MONDAY] * N_FOLDS
    span = pd.Timedelta(days=HORIZON_DAYS - 1)
    assert list(folds["test_end"]) == [pd.Timestamp(ts) + span for ts in folds["test_start"]]


def test_fold_starts_by_dow_are_weekly_and_in_range():
    """헬퍼 단독: 마지막 완전 블록에서 7일씩 뒤로, 블록이 데이터 끝을 넘지 않는다."""
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")   # 2025-12-31 = 수요일
    df = pd.DataFrame({"date": dates})
    starts = _fold_starts_by_dow(df, MONDAY, HORIZON_DAYS, 3)
    assert starts == [
        pd.Timestamp("2025-12-22"), pd.Timestamp("2025-12-15"), pd.Timestamp("2025-12-08"),
    ]
    assert starts[0] + pd.Timedelta(days=HORIZON_DAYS - 1) == pd.Timestamp("2025-12-28")

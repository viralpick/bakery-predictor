"""운영 feature 가용성 정렬(align_features) 회귀 테스트.

lead_days는 학습 시점만 옮기므로 test 행의 자기회귀 feature는 여전히 원점 이후를
본다. ar_history를 주면 그 경로를 끊는다. 계약 4가지:
  1. gapless history로 재계산한 AR == 엔진이 원래 만든 AR (의미론 보존 앵커).
  2. ar_history=None 이면 결과가 인자 없이 호출한 것과 정확 일치(기존 경로 불변).
  3. 원점 이후를 출처로 하는 lag는 NaN이 된다(leakage 차단). 타깃은 보존된다.
  4. 날짜 gap 있는 history는 fails-loud — 위치 shift로 조용히 틀리는 것을 막는다.
"""
import numpy as np
import pandas as pd
import pytest

from bakery.features.category_aggregate import (
    add_lag_rolling_ewma,
    build_category_daily,
    build_features,
)
from bakery.harness.backtest_core import (
    _ar_columns,
    _blind_ar_features,
    _require_gapless,
    windowed_backtest,
)
from bakery.harness.event_priors import resolve_event_priors

TARGET = "adjusted_demand_unit"
N_FOLDS = 3
LEAD_DAYS = 5
MONDAY = 0


@pytest.fixture(scope="module")
def gapless():
    """dropna 이전 프레임 — 엔진이 AR feature를 만든 바로 그 프레임(날짜 연속)."""
    return build_features(build_category_daily(alpha=0.8), target_col=TARGET)


@pytest.fixture(scope="module")
def feat(gapless):
    """backtest가 실제로 받는 프레임 — dropna로 gap이 생겨 있다."""
    return gapless.dropna().reset_index(drop=True)


@pytest.fixture(scope="module")
def priors():
    return resolve_event_priors("gwangyo")


def test_gapless_recompute_reproduces_engine_ar(gapless, feat):
    """★의미론 앵커: gapless history 재계산 == 엔진 원본 AR(같은 날짜에서 정확 일치).

    이게 깨지면 blinding이 leakage를 막는 게 아니라 lag 정의를 바꿔버린 것이다.
    """
    cols = _ar_columns(feat, TARGET)
    recomputed = add_lag_rolling_ewma(gapless[["date", TARGET]].copy(), TARGET)
    merged = feat[["date"]].merge(recomputed[["date", *cols]], on="date", how="left")
    assert np.allclose(
        feat[cols].to_numpy(), merged[cols].to_numpy(), rtol=1e-12, equal_nan=True,
    )


def test_backtest_frame_has_date_gaps(feat):
    """전제 확인: backtest가 받는 프레임엔 gap이 있다 → 여기서 재계산하면 안 된다."""
    gaps = feat["date"].diff().dt.days.dropna()
    assert int((gaps != 1).sum()) > 0


def test_none_ar_history_is_identical(feat, priors):
    """ar_history=None 이면 기존 호출과 fold·prediction 값이 정확히 같다."""
    events, lunar = priors
    kw = dict(window_days=730, n_folds=N_FOLDS, horizon_days=7, production_q=0.85,
              alpha=0.8, events=events, lunar_events=lunar, lead_days=LEAD_DAYS)
    baseline = windowed_backtest(feat, **kw)
    explicit = windowed_backtest(feat, ar_history=None, **kw)
    pd.testing.assert_frame_equal(explicit.folds, baseline.folds, rtol=1e-9)
    pd.testing.assert_frame_equal(explicit.predictions, baseline.predictions, rtol=1e-9)


def test_blind_nulls_post_origin_lags_and_keeps_target(gapless, feat):
    """원점 이후를 출처로 하는 lag는 NaN, 타깃은 보존."""
    test_start = pd.Timestamp("2025-12-22")          # 월요일
    cutoff = test_start - pd.Timedelta(days=LEAD_DAYS)
    block = feat[(feat["date"] >= test_start)
                 & (feat["date"] <= test_start + pd.Timedelta(days=6))].copy()
    blinded = _blind_ar_features(
        block, ar_history=gapless[["date", TARGET]], cutoff=cutoff, target_col=TARGET,
    )
    # lag1의 출처일(= date − 1)이 cutoff 이후인 행은 전부 NaN이어야 한다
    src_post_origin = (blinded["date"] - pd.Timedelta(days=1)) >= cutoff
    assert blinded.loc[src_post_origin, f"{TARGET}_lag1"].isna().all()
    # 반대로 cutoff 이전을 출처로 하는 lag7(첫 이틀)은 살아 있어야 한다
    assert blinded[f"{TARGET}_lag7"].notna().iloc[0]
    # 타깃은 평가에 쓰이므로 절대 변하지 않는다
    assert list(blinded[TARGET]) == list(block[TARGET])


def test_blind_changes_only_ar_columns(gapless, feat):
    """AR 컬럼 외(캘린더·날씨·경쟁점)는 손대지 않는다 — 타깃과 무관해 fold-invariant."""
    test_start = pd.Timestamp("2025-12-22")
    block = feat[(feat["date"] >= test_start)
                 & (feat["date"] <= test_start + pd.Timedelta(days=6))].copy()
    blinded = _blind_ar_features(
        block, ar_history=gapless[["date", TARGET]],
        cutoff=test_start - pd.Timedelta(days=LEAD_DAYS), target_col=TARGET,
    )
    ar = set(_ar_columns(block, TARGET))
    others = [c for c in block.columns if c not in ar]
    pd.testing.assert_frame_equal(blinded[others], block[others])


def test_gapped_history_fails_loud():
    """gap 있는 history는 조용히 위치 shift 되지 않고 에러를 낸다."""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04"])   # 03 누락
    with pytest.raises(ValueError, match="gap"):
        _require_gapless(pd.DataFrame({"date": dates, TARGET: [1.0, 2.0, 3.0]}))


def test_align_features_degrades_wape_vs_leaky(feat, gapless, priors):
    """AR를 가리면 정보가 줄어 WAPE가 나빠진다(= 상한). 방향만 본다."""
    events, lunar = priors
    kw = dict(window_days=730, n_folds=N_FOLDS, horizon_days=7, production_q=0.85,
              alpha=0.8, events=events, lunar_events=lunar,
              lead_days=LEAD_DAYS, anchor_dow=MONDAY)
    leaky = windowed_backtest(feat, **kw)
    aligned = windowed_backtest(feat, ar_history=gapless[["date", TARGET]], **kw)
    leaky_wape = float(np.abs(leaky.predictions["actual"] - leaky.predictions["expected"]).sum()
                       / leaky.predictions["actual"].sum())
    aligned_wape = float(np.abs(aligned.predictions["actual"] - aligned.predictions["expected"]).sum()
                         / aligned.predictions["actual"].sum())
    assert aligned_wape > leaky_wape

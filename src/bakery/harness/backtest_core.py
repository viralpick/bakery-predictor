"""windowed_backtest 코어 — forecaster-general expanding backtest + event_prior.

forecaster 인자로 category_total(LightGBM, 기본) / distributional_total(NGBoost) 등을
태운다(Forecaster 어댑터). default=None→CategoryTotalForecaster라 category 경로는
scripts/store_predictive_power.py 원본과 바이트 동등(엔진 동등성 게이트로 회귀 방지).

leakage-safe: event_prior는 cutoff 이전 전체 history로 fit(train window보다 김), 예측 이후
post-model 블렌드라 forecaster 무관하게 균일 적용. 원본 모듈 상수(ALPHA=0.8, PROD_Q=0.85,
HORIZON=7, MIN_TRAIN_ROWS=60)는 함수 인자 기본값으로 흡수했다(값 동일).

운영 정렬 opt-in(lead_days / anchor_dow): 발주 리드타임과 요일 앵커링을 표현한다.
train과 event_prior history 둘 다 동일한 cutoff(= test_start − lead_days)를 쓰므로
원점 이후 데이터는 어느 경로로도 새지 않는다. 기본값(0 / None)은 헤드라인 동작과
완전히 동일하다 — cutoff == test_start_date, fold 경계는 기존 인덱스 산술 그대로.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.models.category_total import BacktestResult
from bakery.models.event_prior import EventLevelPrior
from bakery.harness.forecasters import CategoryTotalForecaster, Forecaster

MIN_TRAIN_ROWS = 60
DAYS_PER_WEEK = 7


def _fold_starts_by_dow(
    df: pd.DataFrame, anchor_dow: int, horizon_days: int, n_folds: int,
) -> list[pd.Timestamp]:
    """요일 앵커 fold 시작일 목록(최신 → 과거, horizon_days 간격).

    데이터 마지막 날짜 이하에서 horizon 블록이 완전히 들어가는 마지막 anchor_dow
    시작일을 찾고, 거기서 horizon_days씩 과거로 n_folds개를 만든다. 데이터 범위를
    벗어난 시작일은 test_df가 비어 호출부에서 skip된다.
    """
    span = pd.Timedelta(days=horizon_days - 1)
    latest = pd.Timestamp(df["date"].max()) - span
    back = (int(latest.dayofweek) - anchor_dow) % DAYS_PER_WEEK
    latest -= pd.Timedelta(days=back)
    step = pd.Timedelta(days=horizon_days)
    return [latest - k * step for k in range(n_folds)]


def _iter_test_folds(
    df: pd.DataFrame, *, n_folds: int, horizon_days: int, anchor_dow: int | None,
):
    """fold별 (index, test_df)를 최신→과거 순으로 낸다.

    anchor_dow=None이면 기존 인덱스 기반 연속 블록(헤드라인 경로, 산술 불변).
    anchor_dow가 주어지면 날짜 기반 요일 앵커 블록.
    """
    if anchor_dow is None:
        total = len(df)
        test_size = horizon_days
        for k in range(n_folds):
            test_end = total - k * test_size
            test_start = test_end - test_size
            yield k, df.iloc[test_start:test_end]
        return
    span = pd.Timedelta(days=horizon_days - 1)
    starts = _fold_starts_by_dow(df, anchor_dow, horizon_days, n_folds)
    for k, start in enumerate(starts):
        yield k, df[(df["date"] >= start) & (df["date"] <= start + span)]


def windowed_backtest(
    df: pd.DataFrame, *, window_days: int,
    target_col: str = "adjusted_demand_unit", n_folds: int = 52,
    horizon_days: int = 7, production_q: float = 0.85, alpha: float = 0.8,
    events: dict | None = None, lunar_events: dict | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    forecaster: Forecaster | None = None,
    lead_days: int = 0, anchor_dow: int | None = None,
) -> BacktestResult:
    fc = forecaster if forecaster is not None else CategoryTotalForecaster()
    df = df.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).copy()
    df = df.dropna().reset_index(drop=True)
    total = len(df)
    test_size = horizon_days
    if total <= n_folds * test_size + min_train_rows:
        raise ValueError(f"Not enough data: total={total}, folds={n_folds}")

    window = pd.Timedelta(days=window_days)
    folds, preds = [], []
    for k, test_df in _iter_test_folds(
        df, n_folds=n_folds, horizon_days=horizon_days, anchor_dow=anchor_dow,
    ):
        if len(test_df) == 0:      # 요일 앵커 fold가 데이터 범위를 벗어난 경우
            continue
        test_start_date = test_df["date"].iloc[0]
        # 운영 리드타임: 원점 이후 데이터는 train/prior 어디에도 넣지 않는다(leakage 차단).
        # lead_days=0이면 cutoff == test_start_date → 기존 산술과 완전 동일.
        cutoff = test_start_date - pd.Timedelta(days=lead_days)
        # === 유일한 변경점(원본 대비): train slice 를 날짜 기반 rolling window 로 ===
        train_df = df[(df["date"] < cutoff) & (df["date"] >= cutoff - window)]
        if len(train_df) < min_train_rows:
            continue
        model = fc.fit(
            train_df, target_col=target_col, alpha=alpha, production_q=production_q,
        )
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        # 특수일 레벨-앵커 prior: cutoff 이전 전체 history로 fit (train window보다 길게, leakage-safe)
        hist = df[df["date"] < cutoff]
        if events or lunar_events:   # event_prior 명시적으로 있을 때만 적용 (None/{}=비활성, EventLevelPrior의 xmas default fallback 방지)
            prior = EventLevelPrior(events=events, lunar_events=lunar_events).fit(hist, target_col=target_col)
            exp_pred, prod_pred = prior.blend(test_df["date"].values, exp_pred, prod_pred)
        actual = test_df[target_col].values
        wape = np.abs(actual - exp_pred).sum() / max(np.abs(actual).sum(), 1)
        folds.append(dict(
            fold=k, n_train=len(train_df), n_test=len(test_df),
            test_start=test_start_date, test_end=test_df["date"].iloc[-1],
            wape=wape,
            wpe=(exp_pred - actual).sum() / max(actual.sum(), 1),
            prod_pct_under=(prod_pred < actual).mean(),
        ))
        preds.append(pd.DataFrame({
            "date": test_df["date"].values, "fold": k,
            "actual": actual, "expected": exp_pred, "production": prod_pred,
        }))
    return BacktestResult(
        folds=pd.DataFrame(folds).sort_values("fold").reset_index(drop=True),
        predictions=pd.concat(preds, ignore_index=True),
    )


def metrics_from_preds(p: pd.DataFrame) -> dict:
    actual, expected, prod = p["actual"], p["expected"], p["production"]
    surplus = (prod - actual).clip(lower=0)
    return {
        "n_test": int(len(p)),
        "wape": float(np.abs(actual - expected).sum() / max(np.abs(actual).sum(), 1)),
        "wpe": float((expected - actual).sum() / max(actual.sum(), 1)),
        "stockout_risk": float((prod < actual).mean()),
        "surplus_mean_units": float(surplus.mean()),
        "surplus_rate": float(surplus.sum() / max(actual.sum(), 1)),
    }

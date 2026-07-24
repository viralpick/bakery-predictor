"""windowed_backtest 코어 — 카테고리 총량 + event_prior (발행 헤드라인 엔진).

scripts/store_predictive_power.py에서 추출한 단일 출처. leakage-safe:
prior는 pre-test 전체 history로 fit (train window보다 김).

원본 모듈 상수(ALPHA=0.8, PROD_Q=0.85, HORIZON=7, MIN_TRAIN_ROWS=60)는 함수 인자
기본값으로 흡수했다 — 값 동일이라 로직 불변(엔진 동등성 게이트).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.models.category_total import BacktestResult, fit_category_total
from bakery.models.event_prior import EventLevelPrior

MIN_TRAIN_ROWS = 60


def windowed_backtest(
    df: pd.DataFrame, *, window_days: int,
    target_col: str = "adjusted_demand_unit", n_folds: int = 52,
    horizon_days: int = 7, production_q: float = 0.85, alpha: float = 0.8,
    events: dict | None = None, lunar_events: dict | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> BacktestResult:
    df = df.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).copy()
    df = df.dropna().reset_index(drop=True)
    total = len(df)
    test_size = horizon_days
    if total <= n_folds * test_size + min_train_rows:
        raise ValueError(f"Not enough data: total={total}, folds={n_folds}")

    window = pd.Timedelta(days=window_days)
    folds, preds = [], []
    for k in range(n_folds):
        test_end = total - k * test_size
        test_start = test_end - test_size
        test_df = df.iloc[test_start:test_end]
        test_start_date = test_df["date"].iloc[0]
        # === 유일한 변경점(원본 대비): train slice 를 날짜 기반 rolling window 로 ===
        train_df = df[(df["date"] < test_start_date) & (df["date"] >= test_start_date - window)]
        if len(train_df) < min_train_rows:
            continue
        model = fit_category_total(
            train_df, target_col=target_col,
            alpha_demand=alpha, production_q=production_q,
        )
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        # 특수일 레벨-앵커 prior: pre-test 전체 history로 fit (train window보다 길게, leakage-safe)
        hist = df[df["date"] < test_start_date]
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

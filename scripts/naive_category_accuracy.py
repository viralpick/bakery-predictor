"""naive/MA baseline의 '빵 판매 총량(카테고리 합)' 정확도 — 우리 모델과 동일 grain·fold.

우리 모델(+prior) 카테고리-총량 WAPE는 store_predictive_power.py가 이미 산출
(reports/store_predictive_power_summary.json). 이 스크립트는 3-way 비교에서 빠져 있는
naive 수요예측 baseline(SeasonalNaive 4주 동일요일 / MovingAverage 28일)을 **동일
series(date→adjusted_demand_unit 총량)·동일 fold(52주, 목요일 cadence, 730d rolling
train)**로 재측정한다. windowed_backtest의 fold 슬라이싱(iloc)을 그대로 미러링한다.

실행: PYTHONPATH=scripts uv run --with matplotlib python scripts/naive_category_accuracy.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# store_predictive_power에서 build/상수 재사용 (동일 grain·fold 보장)
from store_predictive_power import (
    STORES, TARGET, MAIN_FOLDS, HORIZON, MIN_TRAIN_ROWS, DEFAULT_WINDOW_DAYS,
    build_store_data,
)
from bakery.models.seasonal_naive import SeasonalNaive
from bakery.models.moving_average import MovingAverage

GROUP = dict(store_id="store_gw01", item_id="TOTAL")   # 단일 그룹으로 baseline 모델 재사용


def _series(feat: pd.DataFrame) -> pd.DataFrame:
    """feat(카테고리 총량 1행/일) → baseline 모델 입력(store_id/item_id/date/dow/TARGET)."""
    s = feat[["date", TARGET]].dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)
    s["store_id"] = GROUP["store_id"]
    s["item_id"] = GROUP["item_id"]
    s["dow"] = pd.to_datetime(s["date"]).dt.dayofweek
    return s


def _backtest_baseline(series: pd.DataFrame, model_factory, *,
                       n_folds: int = MAIN_FOLDS, horizon: int = HORIZON,
                       window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """windowed_backtest와 동일한 iloc fold 슬라이싱으로 baseline WAPE/WPE 집계."""
    df = series.sort_values("date").reset_index(drop=True)
    total = len(df)
    preds = []
    for k in range(n_folds):
        test_end = total - k * horizon
        test_start = test_end - horizon
        if test_start <= 0:
            break
        test_df = df.iloc[test_start:test_end]
        test_start_date = test_df["date"].iloc[0]
        window = pd.Timedelta(days=window_days)
        train_df = df[(df["date"] < test_start_date) & (df["date"] >= test_start_date - window)]
        if len(train_df) < MIN_TRAIN_ROWS:
            continue
        model = model_factory().fit(train_df)
        yhat = np.asarray(model.predict(test_df), dtype=float)
        actual = test_df[TARGET].to_numpy(dtype=float)
        preds.append(pd.DataFrame({"actual": actual, "yhat": yhat}))
    p = pd.concat(preds, ignore_index=True)
    p = p.dropna()
    actual, yhat = p["actual"].to_numpy(), p["yhat"].to_numpy()
    denom = max(np.abs(actual).sum(), 1)
    return {
        "n_test": int(len(p)),
        "wape": float(np.abs(actual - yhat).sum() / denom),
        "wpe": float((yhat - actual).sum() / denom),
    }


def main() -> None:
    rows = []
    for cd_code, store_id, label, _ in STORES:
        sd = build_store_data(cd_code, store_id, label)
        series = _series(sd.feat)
        naive = _backtest_baseline(series, lambda: SeasonalNaive(n_weeks=4, y_col=TARGET))
        ma = _backtest_baseline(series, lambda: MovingAverage(window=28, y_col=TARGET))
        rows.append(dict(store=label, model="seasonal_naive(4wk)", **naive))
        rows.append(dict(store=label, model="moving_average(28d)", **ma))
        print(f"[{label}] n={naive['n_test']}  "
              f"naive WAPE={naive['wape']*100:.2f}% WPE={naive['wpe']*100:+.2f}%  |  "
              f"MA WAPE={ma['wape']*100:.2f}% WPE={ma['wpe']*100:+.2f}%")
    out = pd.DataFrame(rows)
    out.to_csv("reports/naive_category_accuracy.csv", index=False)
    print("\nwrote reports/naive_category_accuracy.csv")


if __name__ == "__main__":
    main()

"""운영 호환 multi-horizon backtest.

운영 시나리오:
  D = 목요일 (cutoff)
  → 다음주 월~일 발주 예측
  → horizon offset: D+4 (월), D+5 (화), ..., D+10 (일)

각 horizon별 별도 LightGBM 모델 학습:
  - features = D-1까지 lag/rolling/ewma + D+offset 캘린더/날씨/특수일
  - target   = sold(D+offset)

비교:
  - 기존 1-step backtest: WAPE 8.40% (낙관적)
  - 운영 호환 backtest:   각 horizon WAPE → 평균

D+4가 가장 정확 (3일 갭), D+10이 가장 부정확 (10일 갭).
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import lightgbm as lgb

from bakery.features.category_aggregate import (
    build_category_daily, build_features,
)


TARGET_COL = "adjusted_demand_unit"
HORIZONS = [4, 5, 6, 7, 8, 9, 10]
ALPHA_DEMAND = 0.6
PRODUCTION_Q = 0.90


def select_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    """LEAK_COLS 제거 + 자기상관 lag1, lag7 (둘 다 D-1 기준이므로 OK) + 모든 외부 features."""
    LEAK = (
        "sold_total_unit", "sold_total_revenue",
        "sold_normal_unit", "sold_normal_revenue",
        "sold_closing", "sold_closing_revenue",
        "adjusted_demand_unit", "adjusted_demand_revenue",
        "n_stockout_items", "n_early_stockout", "n_items_active",
    )
    return [c for c in df.columns if c not in ("date", target_col, *LEAK)]


def _add_target_horizon_features(df: pd.DataFrame, h: int, holiday_dates: set, target_col: str = TARGET_COL) -> pd.DataFrame:
    """target date (D+h) 기준 features — 캘린더 + 같은 dow lag (강한 신호 ★)."""
    d = df.copy()
    target_date = d["date"] + pd.Timedelta(days=h)
    target_dow = target_date.dt.dayofweek
    target_month = target_date.dt.month
    target_dom = target_date.dt.day
    target_dim = target_date.dt.days_in_month

    # 캘린더
    d["tgt_dow_sin"]    = np.sin(2 * np.pi * target_dow / 7)
    d["tgt_dow_cos"]    = np.cos(2 * np.pi * target_dow / 7)
    d["tgt_month_sin"]  = np.sin(2 * np.pi * target_month / 12)
    d["tgt_month_cos"]  = np.cos(2 * np.pi * target_month / 12)
    d["tgt_dom_sin"]    = np.sin(2 * np.pi * (target_dom - 1) / target_dim)
    d["tgt_dom_cos"]    = np.cos(2 * np.pi * (target_dom - 1) / target_dim)
    d["tgt_is_weekend"] = (target_dow >= 5).astype(int)
    d["tgt_is_holiday"] = target_date.isin(holiday_dates).astype(int)

    # ★ target_date 기준 같은 dow, k주 전 매출 (lag features의 진짜 강한 신호)
    # shift(7k - h) = sold(row_d + h - 7k) = sold(target_d - 7k)
    same_dow_shifts = []
    for k in [1, 2, 3, 4]:
        shift_offset = 7 * k - h
        if shift_offset > 0:  # row d 기준 과거 = D 시점에 알려진 값
            d[f"tgt_dow_lag{k}w"] = d[target_col].shift(shift_offset)
            same_dow_shifts.append(shift_offset)

    # target dow의 최근 평균 (alignment-aware)
    if same_dow_shifts:
        lag_cols = [d[target_col].shift(s) for s in same_dow_shifts]
        d["tgt_dow_recent_mean"] = pd.concat(lag_cols, axis=1).mean(axis=1)
        d["tgt_dow_recent_std"]  = pd.concat(lag_cols, axis=1).std(axis=1)

    return d


def _load_holiday_set():
    cal = pd.read_parquet("data/external/calendar_raw.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return set(cal.loc[cal["is_holiday"] == True, "date"])


def fit_multi_horizon(train: pd.DataFrame, target_col: str, horizons: list[int]):
    """horizon별 LGBM 학습. target horizon features 추가로 dow/holiday 인식 개선."""
    holiday_dates = _load_holiday_set()
    feat_cols_base = select_feature_cols(train, target_col)
    models = {}
    for h in horizons:
        train_h = train.copy()
        train_h["future_target"] = train_h[target_col].shift(-h)
        # target horizon features 추가 (캘린더 + same-dow lag features)
        train_h = _add_target_horizon_features(train_h, h, holiday_dates, target_col)
        train_h = train_h.dropna(subset=["future_target"])

        # 추가된 features 추출 (동적)
        added_cols = [c for c in train_h.columns if c.startswith("tgt_")]
        feat_cols = feat_cols_base + added_cols
        # NaN 있는 row 추가 drop (tgt_dow_lag*)
        train_h_clean = train_h.dropna(subset=feat_cols + ["future_target"])
        X = train_h_clean[feat_cols]
        y = train_h_clean["future_target"]
        common = dict(n_estimators=400, learning_rate=0.05, max_depth=6,
                      num_leaves=31, random_state=42, verbosity=-1)
        models[h] = {
            "expected": lgb.LGBMRegressor(objective="regression_l1", **common).fit(X, y),
            "quantile": lgb.LGBMRegressor(objective="quantile", alpha=PRODUCTION_Q, **common).fit(X, y),
            "feat_cols": feat_cols,
        }
    return models, holiday_dates


def operational_backtest(
    df: pd.DataFrame,
    horizons: list[int] = HORIZONS,
    n_thursdays: int = 16,            # 마지막 16개 목요일
    min_train_days: int = 365,
    target_col: str = TARGET_COL,
):
    """매 목요일마다 D 시점 cutoff → D+horizons 예측 → 평가."""
    df = df.sort_values("date").reset_index(drop=True).dropna().reset_index(drop=True)
    df["dow"] = df["date"].dt.dayofweek

    thursdays = df[df["dow"] == 3]["date"].tolist()  # dow=3 = 목요일
    thursdays = [t for t in thursdays if (t - df["date"].min()).days >= min_train_days]
    test_thursdays = thursdays[-n_thursdays:]
    print(f'  test Thursdays: {len(test_thursdays)}, 기간 {test_thursdays[0].date()} ~ {test_thursdays[-1].date()}')

    results = []
    for D in test_thursdays:
        train = df[df["date"] <= D].copy()
        models, holiday_dates = fit_multi_horizon(train, target_col, horizons)
        for h in horizons:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df["date"] == test_date]
            if len(test_row) == 0:
                continue
            cutoff_row = df[df["date"] == D].copy()
            if len(cutoff_row) == 0:
                continue
            # cutoff row에 target horizon features 추가
            cutoff_row = _add_target_horizon_features(cutoff_row, h, holiday_dates)
            X = cutoff_row[models[h]["feat_cols"]]
            exp_pred = models[h]["expected"].predict(X)[0]
            prod_pred = models[h]["quantile"].predict(X)[0]
            actual = test_row[target_col].values[0]
            results.append({
                "cutoff_D": D,
                "horizon_h": h,
                "test_date": test_date,
                "test_dow": test_date.dayofweek,
                "expected": exp_pred,
                "production": prod_pred,
                "actual": actual,
                "abs_err": abs(actual - exp_pred),
            })
    return pd.DataFrame(results)


def main():
    print('=== 운영 호환 multi-horizon backtest (α=0.6, q=0.90) ===\n')
    print('운영 시나리오: D=목요일 → D+4~D+10 (다음주 월~일) 예측\n')

    cd = build_category_daily(alpha=ALPHA_DEMAND)
    df = build_features(cd, target_col=TARGET_COL)

    print('Running ... (16 Thursdays × 7 horizons = 112 predictions)')
    res = operational_backtest(df, n_thursdays=16)

    # horizon별 WAPE
    print('\n=== horizon별 정확도 ===')
    print(f'{"horizon":>9s} {"test_dow":>10s} {"n":>4s} {"WAPE":>7s} {"MAE":>7s} {"avg_pred":>9s} {"avg_actual":>10s}')
    for h in HORIZONS:
        sub = res[res["horizon_h"]==h]
        n = len(sub)
        if n == 0:
            continue
        wape = sub["abs_err"].sum() / sub["actual"].sum()
        mae = sub["abs_err"].mean()
        dow_name = {3:'목', 4:'금', 5:'토', 6:'일', 0:'월', 1:'화', 2:'수'}[(3+h)%7]
        print(f'{f"D+{h}":>9s} {dow_name:>10s} {n:>4d} {wape*100:>5.2f}% {mae:>6.1f} {sub["expected"].mean():>7.1f}  {sub["actual"].mean():>8.1f}')

    # 전체 WAPE (전 horizon 합산)
    overall_wape = res["abs_err"].sum() / res["actual"].sum()
    overall_mae = res["abs_err"].mean()
    print(f'\n=== 전체 (운영 호환 backtest, D+4~D+10 합산) ===')
    print(f'  WAPE: {overall_wape*100:.2f}%')
    print(f'  MAE:  {overall_mae:.1f}')
    print(f'  N predictions: {len(res)}')

    # 비교
    print(f'\n=== vs 기존 1-step backtest ===')
    print(f'  기존 1-step: WAPE 8.40%')
    print(f'  운영 호환 (D+4~D+10): WAPE {overall_wape*100:.2f}%')
    print(f'  차이: {(overall_wape - 0.084)*100:+.2f}pp ({"운영 시 더 부정확" if overall_wape > 0.084 else "비슷"})')

    # 매진 risk (production < actual) per horizon
    print(f'\n=== 매진 risk per horizon (production < actual) ===')
    print(f'{"horizon":>9s} {"매진_pct":>9s} {"평균_over":>10s}')
    for h in HORIZONS:
        sub = res[res["horizon_h"]==h]
        if len(sub) == 0:
            continue
        under = (sub["production"] < sub["actual"]).mean()
        avg_over = (sub["production"] - sub["actual"]).mean()
        print(f'{f"D+{h}":>9s} {under*100:>7.1f}%  {avg_over:>+8.1f}')

    res.to_csv('reports/v4_operational_backtest.csv', index=False)
    print('\nsaved: reports/v4_operational_backtest.csv')

if __name__ == "__main__":
    main()

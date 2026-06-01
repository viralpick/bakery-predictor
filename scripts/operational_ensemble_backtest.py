"""운영 호환 Ensemble Backtest.

Approach:
  1. baseline = target_dow 최근 4주 평균 (= naive recent_4w)
  2. LGBM이 residual (actual - baseline) 학습
  3. final_pred = baseline + LGBM_residual_pred
  4. quantile production = baseline + quantile_residual_pred (매진 risk 보호)

비교:
  - naive_recent_4w (point estimate only)
  - LGBM-only (이전 운영 backtest)
  - **ensemble (baseline + LGBM residual)** ← 새

운영 시나리오: D=목요일 → D+4~D+10 (7 horizons).
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import lightgbm as lgb

from bakery.features.category_aggregate import build_category_daily, build_features


TARGET_COL    = "adjusted_demand_unit"
HORIZONS      = [4, 5, 6, 7, 8, 9, 10]
ALPHA_DEMAND  = 0.6
PRODUCTION_Q  = 0.90


def select_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    LEAK = (
        "sold_total_unit", "sold_total_revenue",
        "sold_normal_unit", "sold_normal_revenue",
        "sold_closing", "sold_closing_revenue",
        "adjusted_demand_unit", "adjusted_demand_revenue",
        "n_stockout_items", "n_early_stockout", "n_items_active",
    )
    return [c for c in df.columns if c not in ("date", target_col, *LEAK)]


def compute_baseline(df: pd.DataFrame, h: int, target_col: str = TARGET_COL) -> pd.Series:
    """target_dow 최근 N주 평균 baseline.

    target_date = row_date + h
    target_date - 7k 의 매출 = row_date + h - 7k
    row 기준 shift = 7k - h (양수일 때 알려진 과거)
    """
    shifts = [7*k - h for k in [1, 2, 3, 4] if 7*k - h > 0]
    if not shifts:
        return pd.Series(np.nan, index=df.index)
    lag_cols = pd.concat([df[target_col].shift(s) for s in shifts], axis=1)
    return lag_cols.mean(axis=1)


def fit_ensemble(train: pd.DataFrame, target_col: str, h: int):
    """baseline + LGBM residual 학습."""
    train_h = train.copy()
    train_h["baseline"] = compute_baseline(train_h, h, target_col)
    train_h["future_target"] = train_h[target_col].shift(-h)
    train_h["residual"] = train_h["future_target"] - train_h["baseline"]

    feat_cols = select_feature_cols(train_h, target_col)
    # 학습 추가 컬럼 제외
    feat_cols = [c for c in feat_cols if c not in ("baseline", "future_target", "residual")]
    train_clean = train_h.dropna(subset=["baseline", "future_target", "residual"] + feat_cols)
    X = train_clean[feat_cols]
    y_resid = train_clean["residual"]

    common = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  num_leaves=15, random_state=42, verbosity=-1)
    expected = lgb.LGBMRegressor(objective="regression_l1", **common).fit(X, y_resid)
    quantile = lgb.LGBMRegressor(objective="quantile", alpha=PRODUCTION_Q, **common).fit(X, y_resid)
    return {"expected": expected, "quantile": quantile, "feat_cols": feat_cols}


def operational_ensemble_backtest(df, n_thursdays=16, min_train_days=365):
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["dow"] = df["date"].dt.dayofweek
    df_clean = df.dropna().reset_index(drop=True)

    thursdays = df_clean[df_clean["dow"]==3]["date"].tolist()
    thursdays = [t for t in thursdays if (t - df_clean["date"].min()).days >= min_train_days]
    test_ths = thursdays[-n_thursdays:]
    print(f'  test Thursdays: {len(test_ths)}, 기간 {test_ths[0].date()} ~ {test_ths[-1].date()}')

    results = []
    for D in test_ths:
        train = df_clean[df_clean["date"] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df_clean[df_clean["date"] == test_date]
            if len(test_row) == 0:
                continue
            cutoff_row = df_clean[df_clean["date"] == D].copy()
            if len(cutoff_row) == 0:
                continue

            # baseline at cutoff D
            cutoff_row["baseline"] = compute_baseline(df_clean[df_clean["date"] <= D], h, TARGET_COL).iloc[-1]
            baseline_val = cutoff_row["baseline"].iat[0]
            if pd.isna(baseline_val):
                continue

            # Ensemble model
            model = fit_ensemble(train, TARGET_COL, h)
            X = cutoff_row[model["feat_cols"]]
            resid_exp = model["expected"].predict(X)[0]
            resid_q = model["quantile"].predict(X)[0]
            actual = test_row[TARGET_COL].values[0]

            ens_expected   = baseline_val + resid_exp
            ens_production = baseline_val + resid_q

            results.append({
                "cutoff_D": D, "horizon_h": h,
                "test_date": test_date, "test_dow": test_date.dayofweek,
                "actual": actual,
                "baseline":     baseline_val,
                "ens_expected": ens_expected,
                "ens_production": ens_production,
                "resid_exp": resid_exp,
                "resid_q":   resid_q,
            })
    return pd.DataFrame(results)


def main():
    print('=== Ensemble Operational Backtest (baseline + LGBM residual) ===\n')

    cd = build_category_daily(alpha=ALPHA_DEMAND)
    df = build_features(cd, target_col=TARGET_COL)

    print('Running ensemble backtest ...')
    res = operational_ensemble_backtest(df, n_thursdays=16)
    print(f'\nN predictions: {len(res)}')

    # 3 가지 비교: baseline only, ensemble, LGBM-only (이전 결과)
    res["err_baseline"] = (res["actual"] - res["baseline"]).abs()
    res["err_ensemble"] = (res["actual"] - res["ens_expected"]).abs()

    wape_baseline = res["err_baseline"].sum() / res["actual"].sum()
    wape_ensemble = res["err_ensemble"].sum() / res["actual"].sum()

    print('\n=== 전체 비교 (D+4~D+10 합산) ===')
    print(f'  Baseline only (recent_4w):  WAPE {wape_baseline*100:>5.2f}%')
    print(f'  Ensemble (baseline + LGBM): WAPE {wape_ensemble*100:>5.2f}%')
    print(f'  (이전 LGBM-only):           WAPE 19.18%')

    # 매진율 (production < actual)
    under_baseline = (res["baseline"]      < res["actual"]).mean()
    under_ens_prod = (res["ens_production"] < res["actual"]).mean()
    print(f'\n=== 매진 risk (production < actual) ===')
    print(f'  Baseline (point est): {under_baseline*100:.1f}%')
    print(f'  Ensemble production (q=0.90): {under_ens_prod*100:.1f}%')

    # horizon별
    print('\n=== Horizon별 (Ensemble) ===')
    dow_name = {0:'월', 1:'화', 2:'수', 3:'목', 4:'금', 5:'토', 6:'일'}
    print(f'{"horizon":>9s} {"dow":>4s} {"n":>4s} {"WAPE":>7s} {"매진%":>7s} {"avg_pred":>9s} {"avg_actual":>10s}')
    for h in HORIZONS:
        sub = res[res["horizon_h"]==h]
        if len(sub) == 0:
            continue
        wape = sub["err_ensemble"].sum() / sub["actual"].sum()
        under = (sub["ens_production"] < sub["actual"]).mean()
        dn = dow_name[(3 + h) % 7]
        print(f'{f"D+{h}":>9s} {dn:>4s} {len(sub):>4d} {wape*100:>5.2f}% {under*100:>5.1f}% {sub["ens_expected"].mean():>7.1f}  {sub["actual"].mean():>8.1f}')

    # 매진 detail
    print('\n=== Ensemble production (q=0.90) — 토요일 (D+9) detail ===')
    sat = res[res["horizon_h"]==9].copy()
    sat["over_under"] = sat["ens_production"] - sat["actual"]
    print(sat[["cutoff_D","test_date","baseline","ens_expected","ens_production","actual","over_under"]].to_string(index=False))

    res.to_csv('reports/v4_ensemble_operational_backtest.csv', index=False)
    print('\nsaved: reports/v4_ensemble_operational_backtest.csv')

if __name__ == "__main__":
    main()

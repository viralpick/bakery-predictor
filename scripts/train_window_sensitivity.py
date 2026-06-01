"""학습 구간 sensitivity 분석 — sliding window 길이별 4매장 backtest.

window 후보: 3M(90), 6M(180), 1Y(365), 2Y(730), 3Y(1095), 4Y(1460), 5Y(1825) — 5Y는 expanding과 동일

기존: expanding window (모든 과거 데이터)
실험: sliding window (가장 최근 N일만)

가설:
- 짧은 window (3M~1Y) = 최근 트렌드 반영 (2025 brand-wide 폐기율 +5pp 적응)
- 긴 window (3Y~5Y) = 더 많은 데이터 = 안정성 ↑, but 옛 패턴 영향
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from scripts.all4_stores_backtest import (
    build_store_daily, add_features, add_target_date_features,
    compute_baseline, _load_holiday_set, STORE_MAP, TARGET_COL,
    HORIZONS, PRODUCTION_Q, stats,
)


WINDOWS = {
    '3M': 90,
    '6M': 180,
    '1Y': 365,
    '2Y': 730,
    '3Y': 1095,
    '4Y': 1460,
    '5Y (=expand)': None,   # None = expanding (전체 과거)
}


def fit_ensemble_window(train_full, D, h, holiday_dates, window_days):
    """train_window 적용한 model fit."""
    if window_days is not None:
        cutoff = D - pd.Timedelta(days=window_days)
        train = train_full[train_full['date'] > cutoff].copy()
    else:
        train = train_full.copy()
    if len(train) < 60:    # 너무 적으면 skip
        return None

    train_h = train.copy()
    train_h['baseline'] = compute_baseline(train_h, h)
    train_h['future_target'] = train_h[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
    train_h = add_target_date_features(train_h, h, holiday_dates)

    LEAK = ('sold_total','sold_normal','sold_closing','adjusted_demand',
            'baseline','future_target','residual','target_dow')
    feat_cols = [c for c in train_h.columns if c not in ('date', *LEAK)]
    train_clean = train_h.dropna(subset=['baseline','future_target','residual'] + feat_cols)
    if len(train_clean) < 30: return None
    X = train_clean[feat_cols]
    y = train_clean['residual']

    common = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  num_leaves=15, random_state=42, verbosity=-1)
    quantile = lgb.LGBMRegressor(objective='quantile', alpha=PRODUCTION_Q, **common).fit(X, y)
    train_clean = train_clean.copy()
    train_clean['prod_pred'] = train_clean['baseline'] + quantile.predict(X)
    train_clean['shortfall'] = (train_clean['future_target'] - train_clean['prod_pred']).clip(lower=0)
    dow_safety = train_clean.groupby('target_dow')['shortfall'].mean().to_dict()
    return {'quantile': quantile, 'feat_cols': feat_cols, 'dow_safety': dow_safety}


def run_backtest_window(df, holiday_dates, window_days, n_thursdays=16):
    df = df.dropna().reset_index(drop=True)
    df['dow'] = df['date'].dt.dayofweek
    thursdays = df[df['dow']==3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
    for D in test_ths:
        train_full = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            model = fit_ensemble_window(train_full, D, h, holiday_dates, window_days)
            if model is None: continue
            cutoff_row = df[df['date'] == D].copy()
            cutoff_row = add_target_date_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({'D': D, 'h': h, 'baseline': baseline_at_D,
                            'production': production, 'actual': actual})
    return pd.DataFrame(results)


def main():
    print('=== 학습 구간 Sensitivity (sliding window) — 4매장 × 7 windows ===\n')
    holiday_dates = _load_holiday_set()
    all_results = []

    for cd, name in STORE_MAP.items():
        print(f'\n--- {name} ---')
        daily = build_store_daily(cd, exclude_bulk=True)
        df = add_features(daily, holiday_dates)

        for wname, wdays in WINDOWS.items():
            res = run_backtest_window(df, holiday_dates, wdays)
            if len(res) == 0:
                print(f'  {wname}: skip (insufficient train)')
                continue
            r = stats(res)
            all_results.append({'store': name, 'window': wname,
                                'window_days': wdays if wdays else 9999,
                                **r})
            print(f'  {wname:>14s}: WAPE {r["wape"]:>5.2f}%, 매진 {r["under"]:>5.1f}%, '
                  f'폐기 {r["waste_sum"]:>5.0f}, 부족 {r["short_sum"]:>4.0f}, N={r["n"]}')

    # 종합 표
    print(f'\n\n{"="*100}')
    print(f'=== 매장 × 학습구간 WAPE matrix ===')
    print(f'{"="*100}\n')
    df_all = pd.DataFrame(all_results)
    pivot_wape = df_all.pivot(index='store', columns='window', values='wape')
    pivot_under = df_all.pivot(index='store', columns='window', values='under')
    pivot_waste = df_all.pivot(index='store', columns='window', values='waste_sum')
    win_order = list(WINDOWS.keys())
    pivot_wape = pivot_wape[[w for w in win_order if w in pivot_wape.columns]]
    pivot_under = pivot_under[[w for w in win_order if w in pivot_under.columns]]
    pivot_waste = pivot_waste[[w for w in win_order if w in pivot_waste.columns]]

    print('=== WAPE (%) ===')
    print(pivot_wape.round(2).to_string())
    print('\n=== 매진율 (%) ===')
    print(pivot_under.round(1).to_string())
    print('\n=== 폐기합 ===')
    print(pivot_waste.round(0).astype(int).to_string())

    # 매장별 best window
    print(f'\n=== 매장별 Best 학습구간 (WAPE min) ===')
    for store in pivot_wape.index:
        wins = pivot_wape.loc[store].dropna()
        best_w = wins.idxmin()
        print(f'  {store}: {best_w} (WAPE {wins[best_w]:.2f}%) vs 5Y (WAPE {wins["5Y (=expand)"]:.2f}%)')

    pd.DataFrame(all_results).to_csv('reports/train_window_sensitivity.csv', index=False)
    print('\nsaved: reports/train_window_sensitivity.csv')


if __name__ == '__main__':
    main()

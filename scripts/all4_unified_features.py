"""4매장 통합 feature set으로 backtest 재실행.

통합 features (4매장 Permutation Top 5 union + 기본 lag/cyclic) = 15개:
- Lag/Rolling/EWMA: lag1, lag7, lag14, lag28, rmean7, ewma7, ewma28 (7)
- Cyclic D: month_sin, month_cos (2)
- Holiday D: is_holiday (1)
- Target_date: tgt_is_holiday, tgt_dow_sin, tgt_dow_cos, tgt_month_sin, tgt_days_to_chuseok (5)

이전 simple spec (22~30개)에서 dom/dow_sin/dow_cos/is_weekend/대부분 days_to_* 제거.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from bakery.features.category_aggregate import (
    EVENTS, LUNAR_EVENTS, _signed_days_to_event, _days_to_lunar_event,
)
from scripts.all4_stores_backtest import (
    build_store_daily, _load_holiday_set, STORE_MAP, ALPHA, TARGET_COL,
    HORIZONS, PRODUCTION_Q, stats,
)


# 통합 features (15개)
UNIFIED_BASE_FEATURES = [
    'lag1', 'lag7', 'lag14', 'lag28',
    'rmean7', 'ewma7', 'ewma28',
    'month_sin', 'month_cos',
    'is_holiday',
]
UNIFIED_TARGET_FEATURES = [
    'tgt_is_holiday', 'tgt_dow_sin', 'tgt_dow_cos', 'tgt_month_sin',
    'tgt_days_to_chuseok',
]


def add_unified_features(df, holiday_dates):
    d = df.sort_values('date').copy()
    for lag in [1, 7, 14, 28]:
        d[f'lag{lag}'] = d[TARGET_COL].shift(lag)
    d['rmean7'] = d[TARGET_COL].shift(1).rolling(7).mean()
    d['ewma7'] = d[TARGET_COL].shift(1).ewm(halflife=7).mean()
    d['ewma28'] = d[TARGET_COL].shift(1).ewm(halflife=28).mean()
    month = d['date'].dt.month
    d['month_sin'] = np.sin(2 * np.pi * month / 12)
    d['month_cos'] = np.cos(2 * np.pi * month / 12)
    d['is_holiday'] = d['date'].isin(holiday_dates).astype(int)
    return d


def add_unified_target_features(df, h, holiday_dates):
    d = df.copy()
    target_date = d['date'] + pd.Timedelta(days=h)
    target_dow = target_date.dt.dayofweek
    target_month = target_date.dt.month
    d['tgt_is_holiday'] = target_date.isin(holiday_dates).astype(int)
    d['tgt_dow_sin'] = np.sin(2 * np.pi * target_dow / 7)
    d['tgt_dow_cos'] = np.cos(2 * np.pi * target_dow / 7)
    d['tgt_month_sin'] = np.sin(2 * np.pi * target_month / 12)
    tds = pd.Series(target_date.values)
    chuseok_dates = LUNAR_EVENTS['days_to_chuseok']
    d['tgt_days_to_chuseok'] = _days_to_lunar_event(tds, chuseok_dates).astype('int16')
    return d


def compute_baseline(df, h):
    shifts = [7*k - h for k in [1,2,3,4] if 7*k - h > 0]
    return pd.concat([df[TARGET_COL].shift(s) for s in shifts], axis=1).mean(axis=1)


def fit_ensemble(train, h, holiday_dates):
    train_h = train.copy()
    train_h['baseline'] = compute_baseline(train_h, h)
    train_h['future_target'] = train_h[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
    train_h = add_unified_target_features(train_h, h, holiday_dates)

    feat_cols = UNIFIED_BASE_FEATURES + UNIFIED_TARGET_FEATURES
    train_clean = train_h.dropna(subset=['baseline', 'future_target', 'residual'] + feat_cols)
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


def run_backtest(df, holiday_dates, n_thursdays=16):
    df = df.dropna(subset=UNIFIED_BASE_FEATURES + [TARGET_COL]).reset_index(drop=True)
    df['dow'] = df['date'].dt.dayofweek
    thursdays = df[df['dow']==3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
    for D in test_ths:
        train = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            model = fit_ensemble(train, h, holiday_dates)
            cutoff_row = df[df['date'] == D].copy()
            cutoff_row = add_unified_target_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': target_dow,
                'baseline': baseline_at_D, 'production': production, 'actual': actual,
            })
    return pd.DataFrame(results)


def main():
    print(f'=== 4매장 통합 features ({len(UNIFIED_BASE_FEATURES + UNIFIED_TARGET_FEATURES)}개) backtest ===\n')
    print(f'Base features: {UNIFIED_BASE_FEATURES}')
    print(f'Target features: {UNIFIED_TARGET_FEATURES}\n')

    holiday_dates = _load_holiday_set()
    all_results = []

    for cd, name in STORE_MAP.items():
        print(f'\n--- {name} ---')
        daily = build_store_daily(cd, exclude_bulk=True)
        df = add_unified_features(daily, holiday_dates)
        res = run_backtest(df, holiday_dates)
        r = stats(res)
        all_results.append((name, r))
        print(f'  WAPE: {r["wape"]:.2f}%, 매진율: {r["under"]:.1f}%, '
              f'폐기/일: {r["waste_per_day"]:.1f}, 부족/일: {r["short_per_day"]:.1f}')
        print(f'  폐기합: {r["waste_sum"]:.0f}, 부족합: {r["short_sum"]:.0f}')

    # 비교: 이전 simple spec (22~30 features) vs 통합 (15 features)
    PREV_SIMPLE = {
        '광교':       {'wape': 14.33, 'under': 13.9, 'waste_sum': 3715, 'short_sum': 193},
        '광화문':     {'wape': 13.40, 'under': 24.1, 'waste_sum': 3340, 'short_sum': 337},
        '메세나폴리스': {'wape': 13.72, 'under': 8.3, 'waste_sum': 2727, 'short_sum': 122},
        '삼성타운':   {'wape': 21.70, 'under': 13.9, 'waste_sum': 4402, 'short_sum': 306},
    }

    print(f'\n\n{"="*100}')
    print(f'=== 통합 features vs 이전 simple spec 비교 ===')
    print(f'{"="*100}\n')
    print(f'{"매장":>10s} {"이전 WAPE":>10s} {"통합 WAPE":>10s} {"ΔWAPE":>8s} '
          f'{"이전 폐기":>10s} {"통합 폐기":>10s} {"Δ":>7s}')
    for name, r in all_results:
        prev = PREV_SIMPLE[name]
        dw = r['wape'] - prev['wape']
        d_waste = r['waste_sum'] - prev['waste_sum']
        print(f'  {name:>8s} {prev["wape"]:>9.2f}% {r["wape"]:>9.2f}% {dw:>+7.2f}pp '
              f'{prev["waste_sum"]:>9.0f} {r["waste_sum"]:>9.0f} {d_waste:>+7.0f}')


if __name__ == '__main__':
    main()

"""Phase 4-1: v2 시즌 제외 backtest 재실행 (fair 비교용).

v4 backtest와 동일 조건:
- 시즌/프리미엄 16개 품목 제외
- 4 fold × 30일 (마지막 120일을 30일씩 4 fold)
- expanding window
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from bakery.evaluation.backtest import run_backtest, aggregate_by_model
from bakery.evaluation.split import generate_time_splits
from bakery.models.lightgbm_regressor import GlobalLGBM, LGBMParams
from bakery.models.seasonal_naive import SeasonalNaive
from bakery.models.moving_average import MovingAverage
from bakery.analysis.seasonal import filter_seasonal


print('=== Phase 4-1: v2 시즌 제외 backtest ===\n')

# Load + 시즌 제외
daily = pd.read_parquet('data/internal/bonavi_daily.parquet')
daily['item_id'] = daily['item_id'].astype(str)
daily['date'] = pd.to_datetime(daily['date'])
print(f'Raw daily: {len(daily):,} rows, {daily["item_id"].nunique()} items')

daily_f = filter_seasonal(daily)
print(f'After seasonal exclusion: {len(daily_f):,} rows, {daily_f["item_id"].nunique()} items')

# windows: 4 fold × 30일 (v4와 같은 조건)
windows = generate_time_splits(
    daily_f['date'], n_splits=4, val_horizon_days=30, step_days=30, min_train_days=365,
)
print(f'\nWindows ({len(windows)} folds):')
for w in windows:
    print(f'  fold {w.fold_index}: train [{w.train_start.date()}~{w.train_end.date()}] / val [{w.val_start.date()}~{w.val_end.date()}]')

# Forecasters — baselines + v2/v3 (production model)
forecasters = [
    SeasonalNaive(n_weeks=4),
    MovingAverage(window=28),
    GlobalLGBM(feature_set='v0'),
    GlobalLGBM(feature_set='v1'),
    GlobalLGBM(feature_set='v2'),
    GlobalLGBM(feature_set='v3'),
    GlobalLGBM(feature_set='v2', params=LGBMParams(objective='quantile', alpha=0.85)),
    GlobalLGBM(feature_set='v3', params=LGBMParams(objective='quantile', alpha=0.85)),
]
print(f'\nForecasters: {len(forecasters)}')

print('\nRunning backtest ...')
fold_df, pred_df = run_backtest(daily_f, forecasters, windows)

# Save
fold_df.to_csv('reports/v2_seasonal_excluded_folds.csv', index=False)
pred_df.to_csv('reports/v2_seasonal_excluded_predictions.csv', index=False)

# Item-level summary
print('\n=== Item-level WAPE (시즌 제외, fair 비교 v4와 같은 4 fold × 30일) ===')
print(aggregate_by_model(fold_df).round(4).to_string(index=False))

# === Category-aggregate 비교 ===
print('\n=== 카테고리 합 비교 (시즌 제외, bread+pastry+sandwich) ===')
TARGET_CATS = ('bread', 'pastry', 'sandwich')
pred_cat = pred_df[pred_df['category_id'].isin(TARGET_CATS)].copy()
agg_cat = pred_cat.groupby(['model', 'fold', 'date']).agg(
    yhat_sum=('yhat', 'sum'),
    sold_sum=('sold_units', 'sum'),
).reset_index()

print(f'{"model":25s} {"카테고리 WAPE":>15s} {"under_day":>10s} {"mean_yhat":>12s} {"mean_sold":>12s}')
for model, g in agg_cat.groupby('model'):
    wape = np.abs(g['sold_sum'] - g['yhat_sum']).sum() / g['sold_sum'].sum()
    under = (g['yhat_sum'] < g['sold_sum']).mean()
    print(f'{model:25s} {wape*100:>13.2f}%  {under*100:>8.1f}%  {g["yhat_sum"].mean():>10.1f}   {g["sold_sum"].mean():>10.1f}')

# === v4와 직접 비교 ===
v4_folds = pd.read_csv('reports/v4_category_total_folds.csv')
v4_05_09 = v4_folds[v4_folds['alpha']==0.5]
print(f'\n=== v4 reference (α=0.5, q=0.85, 외부 features 없음, 다른 fold) ===')
print(f'  WAPE mean: {v4_05_09["wape"].mean()*100:.2f}%')
print(f'  prod_pct_under: {v4_05_09["prod_pct_under"].mean()*100:.1f}%')

print('\n=== v4 (with external) Step B 결과 — Phase 1 메모리 기록 ===')
print('  α=0.5 q=0.85 + 외부: WAPE 8.50% / under 14.2-20% (보수)')
print('  α=0.5 q=0.90 + 외부: WAPE 8.50% / 18시전 0% / 실제 매진 1.7%')

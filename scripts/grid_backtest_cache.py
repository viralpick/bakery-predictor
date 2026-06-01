"""4매장 × 6 windows × 5 N steps grid backtest predictions cache.

검증 단계 (A: hold-out, B: weight sensitivity, C: bootstrap, E: ensemble)에서
모두 활용 가능. 한 번 만들면 분석은 cache에서.

Cache schema (parquet):
  store, window, N, D (date), h (int), baseline, production, actual

Grid size: 4 stores × 6 windows × 5 N = 120 backtests. 추정 30~60분.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
from pathlib import Path
import pandas as pd

from scripts.all4_stores_backtest import (
    build_store_daily, _load_holiday_set, STORE_MAP,
)
from scripts.auto_feature_selection import add_all_features_for_set, split_features
from scripts.window_rfecv_pipeline import run_backtest_window_features

WINDOWS = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, '4Y': 1460, '5Y': None}
N_STEPS = [4, 8, 14, 20, 33]
OUT = Path('reports/grid_backtest_cache.parquet')


def load_store_ranking(store_name):
    df = pd.read_csv('reports/permutation_importance_all4.csv')
    sub = df[df['store'] == store_name].sort_values('delta_wape_pp', ascending=False)
    return sub['feature'].tolist()


def main():
    print(f'=== Grid Backtest Cache 생성 ===')
    print(f'  Windows: {list(WINDOWS.keys())}')
    print(f'  N steps: {N_STEPS}')
    print(f'  총 backtest: {len(STORE_MAP)} × {len(WINDOWS)} × {len(N_STEPS)} '
          f'= {len(STORE_MAP) * len(WINDOWS) * len(N_STEPS)}')

    holiday_dates = _load_holiday_set()
    all_rows = []
    t0 = time.time()
    backtest_count = 0
    total_backtest = len(STORE_MAP) * len(WINDOWS) * len(N_STEPS)

    for cd, name in STORE_MAP.items():
        print(f'\n--- {name} ---')
        ranking = load_store_ranking(name)
        daily = build_store_daily(cd, exclude_bulk=True)

        for n in N_STEPS:
            features = ranking[:n]
            base, target = split_features(features)
            df = add_all_features_for_set(daily, holiday_dates, base)

            for wname, wdays in WINDOWS.items():
                t_start = time.time()
                res = run_backtest_window_features(df, holiday_dates, wdays, base, target)
                backtest_count += 1
                elapsed = time.time() - t_start
                if len(res) == 0:
                    print(f'  N={n:>2d} {wname:>3s}: skip (insufficient)')
                    continue
                res = res.copy()
                res['store'] = name
                res['window'] = wname
                res['N'] = n
                all_rows.append(res[['store', 'window', 'N', 'D', 'h',
                                       'baseline', 'production', 'actual']])
                total_elapsed = time.time() - t0
                eta = total_elapsed / backtest_count * (total_backtest - backtest_count)
                print(f'  N={n:>2d} {wname:>3s}: {len(res)} preds, '
                      f'{elapsed:.1f}s, ETA {eta/60:.1f}분')

    cache = pd.concat(all_rows, ignore_index=True)
    cache.to_parquet(OUT, index=False)
    print(f'\nsaved: {OUT} ({len(cache):,} rows)')
    print(f'total elapsed: {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()

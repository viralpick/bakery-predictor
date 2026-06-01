"""C. Bootstrap Stability.

16 Thursdays 중 12개 random sampling 50회 (without replacement).
각 sample에서 매장별 best (window, N) 선택. 분포 측정.

광교 (4Y, N=4) 같은 극단 선택이 sample에 robust한가?
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

CACHE = Path('reports/grid_backtest_cache.parquet')
UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}
WASTE_COST_RATIO, SHORT_COST_RATIO = 0.5, 0.6
W_UNDER, W_COST = 0.6, 0.4
STORES = ['광교', '광화문', '메세나폴리스', '삼성타운']
N_BOOTSTRAP = 50
SAMPLE_SIZE = 12
SEED = 42


def compute_stats(pred):
    if len(pred) == 0: return None
    actual = pred['actual'].sum()
    err = (pred['actual'] - pred['production']).abs().sum()
    wape = err / actual * 100 if actual > 0 else np.nan
    under = (pred['production'] < pred['actual']).mean() * 100
    waste = (pred['production'] - pred['actual']).clip(lower=0).sum()
    short = (pred['actual'] - pred['production']).clip(lower=0).sum()
    return {'wape': wape, 'under': under, 'waste_sum': waste, 'short_sum': short}


def cost_calc(stats, store):
    price = UNIT_PRICE[store]
    return stats['waste_sum'] * price * WASTE_COST_RATIO + stats['short_sum'] * price * SHORT_COST_RATIO


def select_best(grid_results):
    unders = np.array([r['under'] for r in grid_results])
    costs = np.array([r['cost'] for r in grid_results])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, r in enumerate(grid_results):
        r['composite'] = W_UNDER * u_norm[i] + W_COST * c_norm[i]
    return min(grid_results, key=lambda r: r['composite'])


def main():
    print(f'=== C. Bootstrap Stability ({N_BOOTSTRAP}회, {SAMPLE_SIZE}/16 Thursdays) ===\n')
    cache = pd.read_parquet(CACHE)
    cache['D'] = pd.to_datetime(cache['D'])
    thursdays = sorted(cache['D'].unique())

    rng = np.random.RandomState(SEED)
    store_picks = {store: [] for store in STORES}

    for b in range(N_BOOTSTRAP):
        sampled_idx = rng.choice(len(thursdays), SAMPLE_SIZE, replace=False)
        sampled = [thursdays[i] for i in sampled_idx]
        sub_cache = cache[cache['D'].isin(sampled)]

        for store in STORES:
            c = sub_cache[sub_cache['store'] == store]
            grid = []
            for (w, n), grp in c.groupby(['window', 'N']):
                st = compute_stats(grp)
                if st is None: continue
                grid.append({'window': w, 'N': int(n), **st,
                              'cost': cost_calc(st, store)})
            best = select_best(grid)
            store_picks[store].append((best['window'], best['N']))

    # 매장별 분포
    print(f'=== 매장별 best (window, N) 분포 ({N_BOOTSTRAP}회 중) ===\n')
    rows = []
    for store in STORES:
        picks = store_picks[store]
        counter = Counter(picks)
        most_common = counter.most_common(5)
        print(f'  {store}:')
        for (w, n), cnt in most_common:
            pct = cnt / N_BOOTSTRAP * 100
            mark = ' ★' if (w, n) == most_common[0][0] else ''
            print(f'    ({w:>3s}, N={n:>2d}): {cnt:>2d}/{N_BOOTSTRAP} ({pct:.0f}%){mark}')
            rows.append({'store': store, 'window': w, 'N': n,
                          'count': cnt, 'pct': pct})
        top1_pct = most_common[0][1] / N_BOOTSTRAP * 100
        n_unique = len(counter)
        stable = 'STABLE' if top1_pct >= 60 else ('UNSTABLE' if top1_pct < 40 else 'MODERATE')
        print(f'    [{stable}] top1 {top1_pct:.0f}% / unique cells {n_unique}\n')

    df = pd.DataFrame(rows)
    df.to_csv('reports/bootstrap_stability.csv', index=False)
    print('saved: reports/bootstrap_stability.csv')


if __name__ == '__main__':
    main()

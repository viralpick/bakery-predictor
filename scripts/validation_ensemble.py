"""E. Ensemble — window × N grid 평균.

Cache의 모든 (window, N) cells의 production qty 평균.
Selection variance를 평균화 → 단일 best 선택의 noise 제거.

3가지 ensemble:
1. Simple (uniform): 모든 30 cells 평균
2. Window-only: window 평균 (N 고정)
3. N-only: N 평균 (window 고정)
4. Weighted (BMA): composite 역수 weight
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path('reports/grid_backtest_cache.parquet')
UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}
WASTE_COST_RATIO, SHORT_COST_RATIO = 0.5, 0.6
STORES = ['광교', '광화문', '메세나폴리스', '삼성타운']


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


def ensemble_simple(cache_store):
    """모든 cells의 production 평균. 매 (D, h)별 평균.
    Returns: DataFrame (D, h, production, actual)."""
    g = cache_store.groupby(['D', 'h']).agg(
        production=('production', 'mean'),
        actual=('actual', 'first'),
    ).reset_index()
    return g


def ensemble_median(cache_store):
    """매 (D, h)별 production median."""
    g = cache_store.groupby(['D', 'h']).agg(
        production=('production', 'median'),
        actual=('actual', 'first'),
    ).reset_index()
    return g


def ensemble_trimmed(cache_store, trim_pct=0.2):
    """매 (D, h)별 trimmed mean (양쪽 trim_pct 제외 후 평균)."""
    def tmean(x):
        n = len(x); k = int(n * trim_pct)
        s = sorted(x); s = s[k:n-k] if n - 2*k > 0 else s
        return np.mean(s)
    rows = []
    for (d, h), grp in cache_store.groupby(['D', 'h']):
        rows.append({'D': d, 'h': h, 'production': tmean(grp['production']),
                      'actual': grp['actual'].iat[0]})
    return pd.DataFrame(rows)


def ensemble_weighted(cache_store, weights_df):
    """weights_df: (window, N) → weight.
    매 (D, h)별 production을 weight로 가중평균."""
    cs = cache_store.merge(weights_df, on=['window', 'N'])
    cs['weighted_prod'] = cs['production'] * cs['weight']
    g = cs.groupby(['D', 'h']).agg(
        sum_w_prod=('weighted_prod', 'sum'),
        sum_w=('weight', 'sum'),
        actual=('actual', 'first'),
    ).reset_index()
    g['production'] = g['sum_w_prod'] / g['sum_w']
    return g[['D', 'h', 'production', 'actual']]


def grid_stats(cache_store, store):
    """매장의 모든 (window, N) cells의 stats."""
    rows = []
    for (w, n), grp in cache_store.groupby(['window', 'N']):
        st = compute_stats(grp)
        if st is None: continue
        rows.append({'window': w, 'N': int(n), **st,
                      'cost': cost_calc(st, store)})
    return rows


def main():
    print('=== E. Ensemble — Selection Variance 평균화 ===\n')
    cache = pd.read_parquet(CACHE)
    cache['D'] = pd.to_datetime(cache['D'])

    all_results = []
    print(f'{"매장":>10s} {"방식":>22s} {"WAPE":>7s} {"매진율":>8s} {"폐기":>7s} {"부족":>6s} {"cost":>13s}')

    for store in STORES:
        c = cache[cache['store'] == store]

        # 1. Simple ensemble (모든 30 cells 평균)
        ens_simple = ensemble_simple(c)
        st = compute_stats(ens_simple)
        cost = cost_calc(st, store)
        print(f'  {store:>8s} {"Simple ensemble (all 30)":>22s} '
              f'{st["wape"]:>6.2f}% {st["under"]:>6.1f}% '
              f'{st["waste_sum"]:>6.0f} {st["short_sum"]:>5.0f} {cost:>12,.0f}')
        all_results.append({'store': store, 'method': 'simple_ensemble', **st, 'cost': cost})

        # 1b. Median ensemble
        ens_med = ensemble_median(c)
        st = compute_stats(ens_med)
        cost = cost_calc(st, store)
        print(f'  {store:>8s} {"Median ensemble":>22s} '
              f'{st["wape"]:>6.2f}% {st["under"]:>6.1f}% '
              f'{st["waste_sum"]:>6.0f} {st["short_sum"]:>5.0f} {cost:>12,.0f}')
        all_results.append({'store': store, 'method': 'median_ensemble', **st, 'cost': cost})

        # 1c. Trimmed mean (20% trim)
        ens_trim = ensemble_trimmed(c, 0.2)
        st = compute_stats(ens_trim)
        cost = cost_calc(st, store)
        print(f'  {store:>8s} {"Trimmed ensemble (20%)":>22s} '
              f'{st["wape"]:>6.2f}% {st["under"]:>6.1f}% '
              f'{st["waste_sum"]:>6.0f} {st["short_sum"]:>5.0f} {cost:>12,.0f}')
        all_results.append({'store': store, 'method': 'trimmed_ensemble', **st, 'cost': cost})

        # 2. Window ensemble (N=14 fix, 6 windows 평균)
        c_n14 = c[c['N'] == 14]
        ens_w = ensemble_simple(c_n14)
        st = compute_stats(ens_w)
        cost = cost_calc(st, store)
        print(f'  {store:>8s} {"Window-only (N=14)":>22s} '
              f'{st["wape"]:>6.2f}% {st["under"]:>6.1f}% '
              f'{st["waste_sum"]:>6.0f} {st["short_sum"]:>5.0f} {cost:>12,.0f}')
        all_results.append({'store': store, 'method': 'window_only_N14', **st, 'cost': cost})

        # 3. N-only ensemble (5Y fix, 5 N 평균)
        c_5y = c[c['window'] == '5Y']
        ens_n = ensemble_simple(c_5y)
        st = compute_stats(ens_n)
        cost = cost_calc(st, store)
        print(f'  {store:>8s} {"N-only (5Y)":>22s} '
              f'{st["wape"]:>6.2f}% {st["under"]:>6.1f}% '
              f'{st["waste_sum"]:>6.0f} {st["short_sum"]:>5.0f} {cost:>12,.0f}')
        all_results.append({'store': store, 'method': 'n_only_5Y', **st, 'cost': cost})

        # 4. Weighted ensemble — composite 역수 weight (BMA-style)
        grid = grid_stats(c, store)
        unders = np.array([r['under'] for r in grid])
        costs = np.array([r['cost'] for r in grid])
        u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
        c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
        composites = 0.6 * u_norm + 0.4 * c_norm
        # softmax 역수 weight (낮은 composite → 높은 weight)
        weights = np.exp(-composites * 5)
        weights = weights / weights.sum()
        weights_df = pd.DataFrame([{'window': r['window'], 'N': r['N'],
                                      'weight': w}
                                     for r, w in zip(grid, weights)])
        ens_wt = ensemble_weighted(c, weights_df)
        st = compute_stats(ens_wt)
        cost = cost_calc(st, store)
        print(f'  {store:>8s} {"Weighted (BMA softmax)":>22s} '
              f'{st["wape"]:>6.2f}% {st["under"]:>6.1f}% '
              f'{st["waste_sum"]:>6.0f} {st["short_sum"]:>5.0f} {cost:>12,.0f}')
        all_results.append({'store': store, 'method': 'bma_softmax', **st, 'cost': cost})

        print()

    # 종합
    print(f'\n=== 4매장 합 cost (108일) ===')
    df = pd.DataFrame(all_results)
    for method in ['simple_ensemble', 'median_ensemble', 'trimmed_ensemble',
                    'window_only_N14', 'n_only_5Y', 'bma_softmax']:
        sub = df[df['method'] == method]
        tot = sub['cost'].sum()
        avg_und = sub['under'].mean()
        print(f'  {method:>22s}: cost {tot:>13,.0f}원 / 평균 매진 {avg_und:.2f}%')

    df.to_csv('reports/ensemble_validation.csv', index=False)
    print('\nsaved: reports/ensemble_validation.csv')


if __name__ == '__main__':
    main()

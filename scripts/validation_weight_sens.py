"""B. Composite Weight Sensitivity.

weight (w_under : w_cost) 변경 → 매장별 (window, N) 선택 robust한지.

5 weight 시나리오: (0.4:0.6) / (0.5:0.5) / (0.6:0.4) / (0.7:0.3) / (0.8:0.2)

전제: scripts/grid_backtest_cache.py 완료.
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
WEIGHTS = [(0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
STORES = ['광교', '광화문', '메세나폴리스', '삼성타운']


def compute_stats(pred):
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


def main():
    print('=== B. Composite Weight Sensitivity ===\n')
    cache = pd.read_parquet(CACHE)

    # 매장별 grid stats (전체 16 Thursdays)
    rows = []
    for store in STORES:
        c = cache[cache['store'] == store]
        for (w, n), grp in c.groupby(['window', 'N']):
            st = compute_stats(grp)
            rows.append({'store': store, 'window': w, 'N': int(n), **st,
                          'cost': cost_calc(st, store)})
    grid_df = pd.DataFrame(rows)

    # weight별 매장 best 선택
    selections = []
    for w_u, w_c in WEIGHTS:
        print(f'--- Weight (under {w_u} : cost {w_c}) ---')
        scenario_total_cost = 0
        scenario_total_under = []
        for store in STORES:
            sub = grid_df[grid_df['store'] == store].copy()
            u_norm = (sub['under'] - sub['under'].min()) / (sub['under'].max() - sub['under'].min() + 1e-9)
            c_norm = (sub['cost'] - sub['cost'].min()) / (sub['cost'].max() - sub['cost'].min() + 1e-9)
            sub['composite'] = w_u * u_norm + w_c * c_norm
            best = sub.loc[sub['composite'].idxmin()]
            selections.append({'w_under': w_u, 'w_cost': w_c, 'store': store,
                                'window': best['window'], 'N': int(best['N']),
                                'wape': best['wape'], 'under': best['under'],
                                'cost': best['cost']})
            scenario_total_cost += best['cost']
            scenario_total_under.append(best['under'])
            print(f'  {store:>10s}: ({best["window"]:>3s}, N={int(best["N"]):>2d}) '
                  f'WAPE {best["wape"]:.2f}% 매진 {best["under"]:.1f}% cost {best["cost"]:,.0f}')
        avg_under = np.mean(scenario_total_under)
        print(f'  4매장 합: cost {scenario_total_cost:,.0f}원 / 평균 매진 {avg_under:.2f}%\n')

    sel_df = pd.DataFrame(selections)

    # 매장별 selection table — weight에 따라 selection 변하는지
    print(f'\n=== 매장별 Selection by Weight ===\n')
    for store in STORES:
        sub = sel_df[sel_df['store'] == store]
        choices = set((r['window'], r['N']) for _, r in sub.iterrows())
        print(f'  {store}: {len(choices)} 가지 선택')
        for _, r in sub.iterrows():
            print(f'    weight ({r["w_under"]}:{r["w_cost"]}) → ({r["window"]}, N={r["N"]}) '
                  f'cost {r["cost"]:,.0f}원')

    # Stability metric
    print(f'\n=== Stability ===\n')
    stability = []
    for store in STORES:
        sub = sel_df[sel_df['store'] == store]
        unique_picks = sub.groupby(['window', 'N']).size()
        max_consistency = unique_picks.max() / len(sub)
        stability.append({'store': store, 'max_consistency': max_consistency,
                           'n_unique': len(unique_picks)})
        print(f'  {store}: weight 5개 중 가장 자주 선택된 비율 {max_consistency:.0%} '
              f'(unique cells: {len(unique_picks)})')

    sel_df.to_csv('reports/weight_sensitivity.csv', index=False)
    print('\nsaved: reports/weight_sensitivity.csv')


if __name__ == '__main__':
    main()

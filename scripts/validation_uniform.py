"""D. Uniform Baseline 비교.

"모든 매장 = (X, Y)" fixed setting → cache에서 stats 추출.
customization (Composite-window + RFECV) vs uniform 차이.

후보:
- (5Y, N=33) — 모든 features + 전체 데이터 (no selection)
- (5Y, N=8)  — 통합 N=8 (기존 baseline)
- (5Y, N=14) — 통합 N=14
- (1Y, N=8)  — 짧은 window + 적은 features
- (2Y, N=14) — middle ground
- 매장별 customized (참조)
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

UNIFORM_CANDIDATES = [
    ('5Y', 33),
    ('5Y', 20),
    ('5Y', 14),
    ('5Y', 8),
    ('2Y', 14),
    ('2Y', 8),
    ('1Y', 14),
    ('1Y', 8),
    ('6M', 8),
]

# Composite-window+RFECV 매장별 customized 선택 (memory에서)
CUSTOMIZED = {
    '광교': ('5Y', 4),         # 4Y → 5Y로 근사 (cache에 4Y 있음, but Memory 표는 4Y → use 4Y가 더 정확)
    '광화문': ('3M', 33),
    '메세나폴리스': ('5Y', 12),
    '삼성타운': ('3M', 8),
}
# 실제 customized 값 (cache에서 가능한 것만):
CUSTOMIZED_CACHE = {
    '광교': ('4Y', 4),         # memory says 4Y, N=4
    '광화문': ('3M', 33),       # 3M, N=33
    '메세나폴리스': ('4Y', 14),  # 4Y, N=12 → cache N=14 사용 (인접)
    '삼성타운': ('3M', 8),       # 3M, N=8
}


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


def main():
    print('=== D. Uniform Baseline vs Customized ===\n')
    cache = pd.read_parquet(CACHE)

    # Uniform candidates
    rows = []
    print(f'{"Setting":>18s} {"4매장 cost":>15s} {"매진 (평균)":>13s} ' +
          ' '.join(f'{s:>10s}' for s in STORES))
    print('-' * 110)
    for win, n in UNIFORM_CANDIDATES:
        per_store_costs = []
        per_store_unders = []
        per_store_wapes = []
        for store in STORES:
            sub = cache[(cache['store'] == store) & (cache['window'] == win) & (cache['N'] == n)]
            if len(sub) == 0:
                per_store_costs.append(None); per_store_unders.append(None); per_store_wapes.append(None)
                continue
            st = compute_stats(sub)
            cost = cost_calc(st, store)
            per_store_costs.append(cost)
            per_store_unders.append(st['under'])
            per_store_wapes.append(st['wape'])
        valid = [c for c in per_store_costs if c is not None]
        valid_und = [u for u in per_store_unders if u is not None]
        total_cost = sum(valid) if valid else 0
        avg_under = np.mean(valid_und) if valid_und else np.nan
        print(f'  ({win:>3s}, N={n:>2d}) {total_cost:>13,.0f} {avg_under:>11.2f}% ' +
              ' '.join(f'{c:>9,.0f}' if c is not None else '       n/a' for c in per_store_costs))
        rows.append({'setting': f'({win},N={n})', 'window': win, 'N': n,
                      'total_cost': total_cost, 'avg_under': avg_under,
                      **{f'cost_{s}': c for s, c in zip(STORES, per_store_costs)},
                      **{f'under_{s}': u for s, u in zip(STORES, per_store_unders)},
                      **{f'wape_{s}': w for s, w in zip(STORES, per_store_wapes)}})

    # Customized (참조)
    print()
    print(f'--- Customized (Composite-window + RFECV from memory) ---')
    cust_costs = []
    cust_unders = []
    for store, (win, n) in CUSTOMIZED_CACHE.items():
        sub = cache[(cache['store'] == store) & (cache['window'] == win) & (cache['N'] == n)]
        if len(sub) == 0:
            print(f'  {store} ({win}, N={n}): cache MISS')
            continue
        st = compute_stats(sub)
        cost = cost_calc(st, store)
        cust_costs.append(cost)
        cust_unders.append(st['under'])
        print(f'  {store} ({win:>3s}, N={n:>2d}): WAPE {st["wape"]:.2f}% '
              f'매진 {st["under"]:.1f}% cost {cost:,.0f}원')
    total_cust = sum(cust_costs)
    avg_cust_und = np.mean(cust_unders)
    print(f'  4매장 합 cost: {total_cust:,.0f}원 / 평균 매진 {avg_cust_und:.2f}%')

    # 종합
    print(f'\n=== 종합 (4매장 합 cost, 108일) ===')
    rows.append({'setting': 'Customized', 'window': 'mixed', 'N': 0,
                  'total_cost': total_cust, 'avg_under': avg_cust_und})
    df = pd.DataFrame(rows).sort_values('total_cost')
    for _, r in df.iterrows():
        mark = ''
        if r['setting'] == 'Customized':
            mark = ' ★ customized'
        print(f'  {r["setting"]:>15s}: cost {r["total_cost"]:>13,.0f}원 / 매진 {r["avg_under"]:.2f}%{mark}')

    df.to_csv('reports/uniform_baseline.csv', index=False)
    print('\nsaved: reports/uniform_baseline.csv')


if __name__ == '__main__':
    main()

"""A. Hold-out Validation — Nested Time-Series CV.

16 Thursdays → 8 (selection) + 8 (hold-out).
Selection set으로 매장별 best (window, N) 선택. Hold-out에서 평가.

핵심 측정:
- Selection composite vs Hold-out composite gap
- Gap이 크면 selection이 noise/overfit
- Hold-out에서 customization (매장별 차등) vs uniform fix 차이

전제: scripts/grid_backtest_cache.py 먼저 실행.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path('reports/grid_backtest_cache.parquet')
UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}
WASTE_COST_RATIO = 0.5
SHORT_COST_RATIO = 0.6
W_UNDER, W_COST = 0.6, 0.4
STORES = ['광교', '광화문', '메세나폴리스', '삼성타운']


def compute_stats(pred):
    if len(pred) == 0:
        return None
    actual = pred['actual'].sum()
    err = (pred['actual'] - pred['production']).abs().sum()
    wape = err / actual * 100 if actual > 0 else np.nan
    under = (pred['production'] < pred['actual']).mean() * 100
    waste = (pred['production'] - pred['actual']).clip(lower=0).sum()
    short = (pred['actual'] - pred['production']).clip(lower=0).sum()
    return {'wape': wape, 'under': under, 'waste_sum': waste,
            'short_sum': short, 'n': len(pred)}


def cost_calc(stats, store):
    price = UNIT_PRICE[store]
    return stats['waste_sum'] * price * WASTE_COST_RATIO + stats['short_sum'] * price * SHORT_COST_RATIO


def select_best(grid_results, store):
    """grid_results: list of {window, N, under, cost}.
    Composite (within store) min cell return."""
    unders = np.array([r['under'] for r in grid_results])
    costs = np.array([r['cost'] for r in grid_results])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, r in enumerate(grid_results):
        r['composite'] = W_UNDER * u_norm[i] + W_COST * c_norm[i]
    best = min(grid_results, key=lambda r: r['composite'])
    return best


def main():
    print('=== A. Hold-out Validation (8 select / 8 hold-out) ===\n')
    cache = pd.read_parquet(CACHE)
    cache['D'] = pd.to_datetime(cache['D'])

    # 16 Thursdays
    thursdays = sorted(cache['D'].unique())
    print(f'  Total Thursdays in cache: {len(thursdays)}')
    print(f'  Selection (앞 8): {[t.strftime("%Y-%m-%d") for t in thursdays[:8]]}')
    print(f'  Hold-out (뒤 8):  {[t.strftime("%Y-%m-%d") for t in thursdays[8:]]}\n')

    sel_thursdays = thursdays[:8]
    ho_thursdays = thursdays[8:]
    sel_cache = cache[cache['D'].isin(sel_thursdays)]
    ho_cache = cache[cache['D'].isin(ho_thursdays)]

    # 매장별 selection + hold-out
    all_rows = []
    for store in STORES:
        print(f'--- {store} ---')
        s_sub = sel_cache[sel_cache['store'] == store]
        h_sub = ho_cache[ho_cache['store'] == store]

        # Grid: window × N
        grid_sel = []
        for (w, n), grp in s_sub.groupby(['window', 'N']):
            st = compute_stats(grp)
            if st is None: continue
            grid_sel.append({'window': w, 'N': int(n),
                              **st, 'cost': cost_calc(st, store)})
        selected = select_best(grid_sel, store)

        # Hold-out stats for selected (window, N)
        ho_pred = h_sub[(h_sub['window'] == selected['window']) &
                         (h_sub['N'] == selected['N'])]
        ho_stats = compute_stats(ho_pred)
        ho_cost = cost_calc(ho_stats, store)

        # 비교: 전체 grid hold-out stats (selected가 hold-out best였는지)
        grid_ho = []
        for (w, n), grp in h_sub.groupby(['window', 'N']):
            st = compute_stats(grp)
            if st is None: continue
            grid_ho.append({'window': w, 'N': int(n),
                             **st, 'cost': cost_calc(st, store)})
        ho_best = select_best(grid_ho, store)

        print(f'  Selection 선택: window={selected["window"]} N={selected["N"]:>2d} '
              f'WAPE {selected["wape"]:.2f}% 매진 {selected["under"]:.1f}% cost {selected["cost"]:,.0f}원')
        print(f'  Hold-out 평가:  window={selected["window"]} N={selected["N"]:>2d} '
              f'WAPE {ho_stats["wape"]:.2f}% 매진 {ho_stats["under"]:.1f}% cost {ho_cost:,.0f}원')
        print(f'  Hold-out best:  window={ho_best["window"]} N={ho_best["N"]:>2d} '
              f'WAPE {ho_best["wape"]:.2f}% 매진 {ho_best["under"]:.1f}% cost {ho_best["cost"]:,.0f}원')
        match = (selected['window'] == ho_best['window']) and (selected['N'] == ho_best['N'])
        print(f'  Selection == Hold-out best? {"YES" if match else "NO"}')
        print()

        all_rows.append({
            'store': store,
            'sel_window': selected['window'], 'sel_N': selected['N'],
            'sel_wape': selected['wape'], 'sel_under': selected['under'], 'sel_cost': selected['cost'],
            'ho_wape': ho_stats['wape'], 'ho_under': ho_stats['under'], 'ho_cost': ho_cost,
            'ho_best_window': ho_best['window'], 'ho_best_N': ho_best['N'],
            'ho_best_wape': ho_best['wape'], 'ho_best_under': ho_best['under'],
            'ho_best_cost': ho_best['cost'],
            'gap_wape': ho_stats['wape'] - selected['wape'],
            'gap_under': ho_stats['under'] - selected['under'],
            'gap_cost': ho_cost - selected['cost'],
            'oracle_gain_cost': ho_cost - ho_best['cost'],
        })

    df = pd.DataFrame(all_rows)

    # 종합
    print(f'\n{"="*100}')
    print(f'=== 종합 — Selection 결과를 Hold-out에서 측정 ===')
    print(f'{"="*100}\n')
    sel_cost = df['sel_cost'].sum()
    ho_cost = df['ho_cost'].sum()
    ho_best_cost = df['ho_best_cost'].sum()
    print(f'  4매장 합 (8 Thursdays):')
    print(f'    Selection cost:  {sel_cost:>13,.0f}원')
    print(f'    Hold-out cost:   {ho_cost:>13,.0f}원  ({(ho_cost-sel_cost)/sel_cost*100:+.1f}%)')
    print(f'    Hold-out oracle: {ho_best_cost:>13,.0f}원  ({(ho_best_cost-ho_cost)/ho_cost*100:+.1f}%)')
    print(f'\n  → Selection→Hold-out 격차: {ho_cost-sel_cost:+,.0f}원 (overfit signal)')
    print(f'  → Oracle gap (post-hoc best): {ho_cost-ho_best_cost:+,.0f}원 (selection이 hold-out best를 못 맞춤)')

    # Selection 일치율 — 매장 몇 개나 hold-out best와 같은 선택?
    matches = sum(1 for r in all_rows
                  if r['sel_window'] == r['ho_best_window'] and r['sel_N'] == r['ho_best_N'])
    print(f'\n  매장별 선택 일치율: {matches}/4 매장 (selection==hold-out best)')

    df.to_csv('reports/holdout_validation.csv', index=False)
    print('\nsaved: reports/holdout_validation.csv')


if __name__ == '__main__':
    main()

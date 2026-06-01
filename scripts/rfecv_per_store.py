"""매장별 독립 RFECV — 각 매장에 최적화된 feature subset 자동 선택.

방식:
1. 매장별 Permutation Importance ranking (사전 측정)
2. 매장별 ranking 하위부터 점차 제거 (RFE)
3. 매장별 N steps × backtest × composite score
4. 매장별 best N + features 자동 결정

매장별 composite (단일 매장 기준):
- normalized within 그 매장의 N steps
- 0.6 × norm_under + 0.4 × norm_cost
- cost = 폐기×단가×0.5 + 부족×단가×0.6 (마진 50%, 매진 ×1.2)
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd

from scripts.all4_stores_backtest import (
    build_store_daily, _load_holiday_set, STORE_MAP, TARGET_COL, stats,
)
from scripts.auto_feature_selection import (
    add_all_features_for_set, run_backtest_custom, split_features,
)

UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}
WASTE_COST_RATIO = 0.5
SHORT_COST_RATIO = 0.6

N_STEPS = [33, 28, 24, 20, 17, 14, 12, 10, 8, 6, 4]


def load_store_ranking(store_name):
    """매장별 Permutation Importance ranking."""
    df = pd.read_csv('reports/permutation_importance_all4.csv')
    sub = df[df['store'] == store_name].sort_values('delta_wape_pp', ascending=False)
    return sub['feature'].tolist(), sub.set_index('feature')['delta_wape_pp'].to_dict()


def store_composite_score(under_rate, waste_sum, short_sum, unit_price):
    """매장 단일 composite cost 계산 (normalize는 candidate set에서)."""
    cost = waste_sum * unit_price * WASTE_COST_RATIO + short_sum * unit_price * SHORT_COST_RATIO
    return under_rate, cost


def run_rfe_for_store(store_cd, store_name, holiday_dates):
    print(f'\n{"="*80}\n=== {store_name} 매장 독립 RFE ===\n{"="*80}')

    ranking, perm_dict = load_store_ranking(store_name)
    print(f'전체 features ({len(ranking)}) — top 10:')
    for f in ranking[:10]:
        print(f'  {f}: {perm_dict[f]:+.3f}pp')

    candidates = []
    daily = build_store_daily(store_cd, exclude_bulk=True)

    for n in N_STEPS:
        features = ranking[:n]
        base, target = split_features(features)
        df = add_all_features_for_set(daily, holiday_dates, base)
        res = run_backtest_custom(df, holiday_dates, base, target)
        r = stats(res)
        under, cost = store_composite_score(
            r['under'], r['waste_sum'], r['short_sum'], UNIT_PRICE[store_name])
        candidates.append({
            'N': n,
            'features': features.copy(),
            'wape': r['wape'],
            'under': under,
            'waste_sum': r['waste_sum'],
            'short_sum': r['short_sum'],
            'cost': cost,
        })
        print(f'  N={n:>3}: WAPE {r["wape"]:>5.2f}%, 매진 {under:>5.1f}%, '
              f'폐기 {r["waste_sum"]:>5.0f}, 부족 {r["short_sum"]:>4.0f}, cost {cost:>10,.0f}원')

    # Normalize within store
    unders = np.array([c['under'] for c in candidates])
    costs = np.array([c['cost'] for c in candidates])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, c in enumerate(candidates):
        c['composite'] = 0.6 * u_norm[i] + 0.4 * c_norm[i]

    best = min(candidates, key=lambda c: c['composite'])
    print(f'\n--- {store_name} Best N={best["N"]} (composite={best["composite"]:.3f}) ---')
    print(f'WAPE {best["wape"]:.2f}%, 매진 {best["under"]:.1f}%, cost {best["cost"]:,.0f}원')
    print(f'features ({len(best["features"])}):')
    for f in best['features']:
        print(f'  - {f} ({perm_dict[f]:+.3f}pp)')

    return store_name, candidates, best


def main():
    print('=== 매장별 독립 RFECV (각 매장 ranking + best N) ===')
    holiday_dates = _load_holiday_set()

    all_results = {}
    for cd, name in STORE_MAP.items():
        _, candidates, best = run_rfe_for_store(cd, name, holiday_dates)
        all_results[name] = {'candidates': candidates, 'best': best}

    # 종합 요약
    print(f'\n\n{"="*110}')
    print(f'=== 매장별 Best N 종합 ===')
    print(f'{"="*110}\n')
    print(f'{"매장":>10s} {"Best N":>7s} {"WAPE":>7s} {"매진율":>8s} {"폐기":>7s} {"부족":>6s} {"cost(원)":>13s}')
    total_cost = 0
    for name, data in all_results.items():
        b = data['best']
        total_cost += b['cost']
        print(f'  {name:>8s} {b["N"]:>6d}  {b["wape"]:>5.2f}% {b["under"]:>6.1f}% '
              f'{b["waste_sum"]:>6.0f} {b["short_sum"]:>5.0f} {b["cost"]:>12,.0f}')
    print(f'\n  4매장 합 cost (108일): {total_cost:,.0f}원')
    print(f'  연 환산 (×3.38): {total_cost * 3.38:,.0f}원/년')
    print(f'  5년 환산 (×16.9): {total_cost * 16.9:,.0f}원/5년')

    # 매장별 feature 교집합/합집합
    print(f'\n=== 매장별 best features 비교 ===')
    sets = {name: set(data['best']['features']) for name, data in all_results.items()}
    union = set.union(*sets.values())
    inter = set.intersection(*sets.values())
    print(f'4매장 합집합: {len(union)}개')
    print(f'4매장 교집합 (모든 매장 공통): {len(inter)}개')
    print(f'  공통 features: {sorted(inter)}')

    # 매장별 feature presence 표
    sorted_union = sorted(union)
    print(f'\n매장별 feature 사용 표:')
    print(f'{"feature":>40s} {"광교":>5s} {"광화문":>6s} {"메세나":>7s} {"삼성타운":>9s}')
    for f in sorted_union:
        row = []
        for name in ['광교', '광화문', '메세나폴리스', '삼성타운']:
            row.append('✓' if f in sets[name] else '-')
        print(f'  {f:>40s} {row[0]:>5s} {row[1]:>6s} {row[2]:>7s} {row[3]:>9s}')

    # CSV
    flat = []
    for name, data in all_results.items():
        for c in data['candidates']:
            flat.append({
                'store': name,
                'N': c['N'],
                'wape': c['wape'],
                'under': c['under'],
                'waste_sum': c['waste_sum'],
                'short_sum': c['short_sum'],
                'cost': c['cost'],
                'composite': c['composite'],
                'is_best': c['N'] == data['best']['N'],
                'features': ','.join(c['features']),
            })
    pd.DataFrame(flat).to_csv('reports/rfecv_per_store.csv', index=False)
    print('\nsaved: reports/rfecv_per_store.csv')


if __name__ == '__main__':
    main()

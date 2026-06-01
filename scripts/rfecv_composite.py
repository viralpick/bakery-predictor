"""RFECV + Composite Score 평가.

방식:
1. Permutation Importance ranking 사용 (사전 측정)
2. 4매장 평균 ranking → 하위 feature 점차 제거 (RFE)
3. 각 N에 대해 4매장 backtest + composite score 측정
4. Composite min N 선택

Composite score 정의 (마진 50% 가정):
- 폐기 cost = qty × 단가 × 0.5 (마진 빼고 재료/인건)
- 매진 cost = qty × 단가 × 0.5 × 1.2 (마진 손실 + 간접 손실 1.2배)
- 즉 effective_cost = waste_sum × 0.5 × 단가 + short_sum × 0.6 × 단가
- composite = 0.6 × normalized_매진율 + 0.4 × normalized_cost
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
MARGIN_RATE = 0.5         # 마진율 50% 가정
WASTE_COST_RATIO = 1 - MARGIN_RATE   # 폐기 cost = 단가 × (1-마진) = 재료비
SHORT_COST_RATIO = MARGIN_RATE * 1.2  # 매진 cost = 단가 × 마진 × 1.2 (간접손실)


def load_permutation_ranking():
    """4매장 Permutation Importance 평균 ranking."""
    df = pd.read_csv('reports/permutation_importance_all4.csv')
    avg = df.groupby('feature')['delta_wape_pp'].mean().sort_values(ascending=False)
    return avg


def composite_score(stats_list, store_names, prices):
    """단가 + 마진 가정 반영한 composite score.

    business_cost = waste × 단가 × (1-margin) + short × 단가 × margin × 1.2
    composite     = 0.6 × norm(매장 평균 매진율) + 0.4 × norm(4매장 cost 합)
    """
    avg_under = np.mean([s['under'] for s in stats_list])
    total_cost = sum(
        (s['waste_sum'] * prices[name] * WASTE_COST_RATIO
         + s['short_sum'] * prices[name] * SHORT_COST_RATIO)
        for s, name in zip(stats_list, store_names)
    )
    return avg_under, total_cost


def main():
    print(f'=== RFECV + Composite Score (마진 50%, 매진 ×1.2, weight 0.6:0.4) ===\n')
    avg_rank = load_permutation_ranking()
    all_features = avg_rank.index.tolist()
    print(f'전체 features ({len(all_features)}) — 4매장 평균 Permutation Importance ranking:')
    print(avg_rank.head(15).to_string())

    holiday_dates = _load_holiday_set()
    # 매 N마다 ranking 상위 N 사용 (하위부터 점차 제거)
    N_STEPS = [33, 28, 24, 20, 17, 14, 12, 10, 8, 6, 4]   # 11 steps

    candidate_results = []
    for n in N_STEPS:
        features = all_features[:n]
        base, target = split_features(features)
        print(f'\n--- N={n} (base {len(base)}, target {len(target)}) ---')

        store_stats = []
        for cd, name in STORE_MAP.items():
            daily = build_store_daily(cd, exclude_bulk=True)
            df = add_all_features_for_set(daily, holiday_dates, base)
            res = run_backtest_custom(df, holiday_dates, base, target)
            r = stats(res)
            store_stats.append(r)
            print(f'  {name}: WAPE {r["wape"]:.2f}%, 매진 {r["under"]:.1f}%, '
                  f'폐기 {r["waste_sum"]:.0f}, 부족 {r["short_sum"]:.0f}')

        avg_und, tot_cost = composite_score(store_stats, list(STORE_MAP.values()), UNIT_PRICE)
        candidate_results.append({
            'N': n,
            'n_features': len(features),
            'features': features.copy(),
            'store_stats': store_stats,
            'avg_under': avg_und,
            'total_cost': tot_cost,
        })

    # Normalize + composite
    unders = np.array([c['avg_under'] for c in candidate_results])
    costs = np.array([c['total_cost'] for c in candidate_results])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, c in enumerate(candidate_results):
        c['composite'] = 0.6 * u_norm[i] + 0.4 * c_norm[i]

    print(f'\n\n{"="*110}')
    print(f'=== RFECV 결과 (Composite Score = 0.6 × norm_매진율 + 0.4 × norm_cost) ===')
    print(f'{"="*110}\n')
    print(f'{"N":>3s} {"평균 매진율":>11s} {"4매장 cost(원)":>16s} {"norm_und":>9s} {"norm_cost":>10s} {"composite":>10s}')
    best_composite = min(c['composite'] for c in candidate_results)
    for c in candidate_results:
        mark = ' ★ best' if c['composite'] == best_composite else ''
        print(f'  {c["N"]:>3d} {c["avg_under"]:>9.2f}% {c["total_cost"]:>15,.0f} '
              f'{(c["avg_under"]-unders.min())/(unders.max()-unders.min()+1e-9):>8.3f} '
              f'{(c["total_cost"]-costs.min())/(costs.max()-costs.min()+1e-9):>9.3f} '
              f'{c["composite"]:>9.3f}{mark}')

    # best N 매장별 결과
    best = min(candidate_results, key=lambda c: c['composite'])
    print(f'\n=== Best N={best["N"]} 매장별 상세 ===')
    print(f'features ({len(best["features"])}):')
    for f in best['features']:
        print(f'  - {f} (avg perm = {avg_rank.get(f, 0):+.3f}pp)')
    print()
    for s, name in zip(best['store_stats'], STORE_MAP.values()):
        unit = UNIT_PRICE[name]
        waste_cost = s['waste_sum'] * unit * WASTE_COST_RATIO
        short_cost = s['short_sum'] * unit * SHORT_COST_RATIO
        total = waste_cost + short_cost
        print(f'  {name:>10s}: WAPE {s["wape"]:>5.2f}%, 매진 {s["under"]:>5.1f}%, '
              f'폐기 {s["waste_sum"]:>5.0f}, 부족 {s["short_sum"]:>4.0f}, '
              f'cost {total:>12,.0f}원 (폐기 {waste_cost:,.0f} + 매진 {short_cost:,.0f})')

    # CSV
    pd.DataFrame([{
        'N': c['N'],
        'avg_under': c['avg_under'],
        'total_cost': c['total_cost'],
        'composite': c['composite'],
        'features': ','.join(c['features']),
    } for c in candidate_results]).to_csv('reports/rfecv_composite.csv', index=False)
    print('\nsaved: reports/rfecv_composite.csv')


if __name__ == '__main__':
    main()

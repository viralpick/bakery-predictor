"""α sensitivity 분석 — 4매장 × 6 α 값 (0.3~0.8) backtest.

각 매장×α 조합:
- target = adjusted_demand = sold_normal + sold_closing × α
- features 동일 (lag/cyclic + holiday + target_date)
- backtest 16 Thursdays × 7 horizons
- 4 metric (WAPE / 매진율 / 폐기합 / 부족합)

best α 결정 기준:
- 비지니스 임팩트 = -폐기 cost - 부족 cost (cost min)
- 또는 4 metric balance
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from scripts.all4_stores_backtest import (
    build_store_daily, add_features, run_v4,
    _load_holiday_set, stats, STORE_MAP,
)
from scripts.all4_stores_backtest import ALPHA, TARGET_COL  # 원본 가져오기 위해
import scripts.all4_stores_backtest as base


def run_for_alpha(store_cd: str, alpha: float, holiday_dates):
    """매장×α 단일 조합 backtest."""
    base.ALPHA = alpha    # module-level override
    daily = build_store_daily(store_cd, exclude_bulk=True)
    df = add_features(daily, holiday_dates)
    res = run_v4(df, holiday_dates)
    return stats(res)


def main():
    print('=== α Sensitivity Analysis (4매장 × α 6값) ===\n')
    holiday_dates = _load_holiday_set()
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []

    for store_cd, store_name in STORE_MAP.items():
        print(f'\n--- {store_name} ---')
        for alpha in alphas:
            r = run_for_alpha(store_cd, alpha, holiday_dates)
            rows.append({
                'store': store_name,
                'alpha': alpha,
                **r,
            })
            print(f'  α={alpha}: WAPE {r["wape"]:.2f}%, 매진율 {r["under"]:.1f}%, '
                  f'폐기합 {r["waste_sum"]:.0f}, 부족합 {r["short_sum"]:.0f}')

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv('reports/alpha_sensitivity.csv', index=False)

    # 매장별 best α (비지니스 임팩트 기준)
    # cost = waste_sum + short_sum (단위 동일, qty 합)
    # 단가 5,000원 가정 동일 → cost min = qty min
    df['total_cost_qty'] = df['waste_sum'] + df['short_sum']

    print(f'\n\n{"="*100}')
    print(f'=== 매장별 α별 결과 + Best α 선택 ===')
    print(f'{"="*100}\n')

    for store_name in ['광교', '광화문', '메세나폴리스', '삼성타운']:
        sub = df[df['store']==store_name].copy()
        print(f'\n[{store_name}]')
        print(f'  {"α":>4s} {"WAPE":>7s} {"매진율":>8s} {"폐기합":>8s} {"부족합":>8s} {"총 cost":>8s}')
        for _, r in sub.iterrows():
            mark = ''
            if r['total_cost_qty'] == sub['total_cost_qty'].min():
                mark = ' ★ best (cost)'
            print(f'  {r["alpha"]:>4.1f} {r["wape"]:>6.2f}% {r["under"]:>6.1f}% '
                  f'{r["waste_sum"]:>7.0f} {r["short_sum"]:>7.0f} {r["total_cost_qty"]:>7.0f}{mark}')
        best_idx = sub['total_cost_qty'].idxmin()
        best_a = sub.loc[best_idx, 'alpha']
        print(f'  → Best α = {best_a} (cost min)')

    # 4 metric 종합 best α (Pareto 분석)
    print(f'\n\n{"="*100}')
    print(f'=== Pareto 분석: 매장별 α 별 4 metric (WAPE↓, 매진율↓, 폐기↓, 부족↓) ===')
    print(f'{"="*100}\n')
    for store_name in ['광교', '광화문', '메세나폴리스', '삼성타운']:
        sub = df[df['store']==store_name].copy().reset_index(drop=True)
        # 각 metric normalize (min=0, max=1)
        for col in ['wape', 'under', 'waste_sum', 'short_sum']:
            min_v, max_v = sub[col].min(), sub[col].max()
            sub[f'{col}_norm'] = (sub[col] - min_v) / (max_v - min_v) if max_v > min_v else 0
        sub['total_score'] = sub[['wape_norm','under_norm','waste_sum_norm','short_sum_norm']].sum(axis=1)
        best_idx = sub['total_score'].idxmin()
        best_a = sub.loc[best_idx, 'alpha']
        print(f'  {store_name}: best α = {best_a} (normalized score min)')


if __name__ == '__main__':
    main()

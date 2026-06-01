"""매장별 Window 선택 기준 = Composite Score (cost + 매진율).

Phase 1: 매장별 7 windows × full features → composite score 측정 → window 선택
Phase 2: 선택된 window에서 RFECV (11 N steps) → best N + features 선택

각 매장 final: (window, N, features) 조합 자동 결정
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
from scripts.auto_feature_selection import add_all_features_for_set, split_features
from scripts.window_rfecv_pipeline import (
    fit_ensemble_window, run_backtest_window_features, store_cost,
    WASTE_COST_RATIO, SHORT_COST_RATIO,
)

UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}
WINDOWS = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, '3Y': 1095, '4Y': 1460, '5Y': None}
N_STEPS = [33, 28, 24, 20, 17, 14, 12, 10, 8, 6, 4]


def load_store_ranking(store_name):
    df = pd.read_csv('reports/permutation_importance_all4.csv')
    sub = df[df['store'] == store_name].sort_values('delta_wape_pp', ascending=False)
    return sub['feature'].tolist()


def pick_best_window_composite(store_cd, store_name, holiday_dates):
    """매장 7 windows × full features → composite min window 선택."""
    print(f'\n{"="*80}\n=== {store_name}: Phase 1 — best window 선택 (composite 기준) ===\n{"="*80}')
    ranking = load_store_ranking(store_name)
    base, target = split_features(ranking)   # 전체 features 사용
    daily = build_store_daily(store_cd, exclude_bulk=True)
    df = add_all_features_for_set(daily, holiday_dates, base)

    results = []
    for wname, wdays in WINDOWS.items():
        res = run_backtest_window_features(df, holiday_dates, wdays, base, target)
        if len(res) == 0: continue
        r = stats(res)
        under, cost = store_cost(r['under'], r['waste_sum'], r['short_sum'], UNIT_PRICE[store_name])
        results.append({'window': wname, 'wdays': wdays, 'wape': r['wape'],
                         'under': under, 'cost': cost,
                         'waste_sum': r['waste_sum'], 'short_sum': r['short_sum']})
        print(f'  {wname:>4s}: WAPE {r["wape"]:>5.2f}%, 매진 {under:>5.1f}%, '
              f'폐기 {r["waste_sum"]:>5.0f}, 부족 {r["short_sum"]:>4.0f}, cost {cost:>10,.0f}원')

    unders = np.array([r['under'] for r in results])
    costs = np.array([r['cost'] for r in results])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, r in enumerate(results):
        r['composite'] = 0.6 * u_norm[i] + 0.4 * c_norm[i]
    print(f'\n  {"window":>6s} {"composite":>10s}')
    for r in results:
        mark = ' ★ best' if r['composite'] == min(rr['composite'] for rr in results) else ''
        print(f'  {r["window"]:>6s} {r["composite"]:>9.3f}{mark}')

    best = min(results, key=lambda r: r['composite'])
    print(f'\n  → Best window: {best["window"]} ({best["wdays"]}일) — composite {best["composite"]:.3f}')
    return best['window'], best['wdays']


def run_rfecv(store_cd, store_name, window_days, win_label, holiday_dates):
    """Phase 2: 선택된 window에서 RFECV."""
    print(f'\n=== {store_name}: Phase 2 — RFECV (window={win_label}) ===')
    ranking = load_store_ranking(store_name)
    daily = build_store_daily(store_cd, exclude_bulk=True)

    candidates = []
    for n in N_STEPS:
        features = ranking[:n]
        base, target = split_features(features)
        df = add_all_features_for_set(daily, holiday_dates, base)
        res = run_backtest_window_features(df, holiday_dates, window_days, base, target)
        if len(res) == 0: continue
        r = stats(res)
        under, cost = store_cost(r['under'], r['waste_sum'], r['short_sum'], UNIT_PRICE[store_name])
        candidates.append({'N': n, 'features': features.copy(), 'wape': r['wape'],
                           'under': under, 'cost': cost,
                           'waste_sum': r['waste_sum'], 'short_sum': r['short_sum']})
        print(f'  N={n:>3}: WAPE {r["wape"]:>5.2f}%, 매진 {under:>5.1f}%, '
              f'폐기 {r["waste_sum"]:>5.0f}, 부족 {r["short_sum"]:>4.0f}, cost {cost:>10,.0f}원')

    unders = np.array([c['under'] for c in candidates])
    costs = np.array([c['cost'] for c in candidates])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, c in enumerate(candidates):
        c['composite'] = 0.6 * u_norm[i] + 0.4 * c_norm[i]

    best = min(candidates, key=lambda c: c['composite'])
    print(f'\n  Best N={best["N"]} (composite={best["composite"]:.3f})')
    return best


def main():
    print('=== 매장별 Window + RFECV (window 선택도 composite 기준) ===')
    holiday_dates = _load_holiday_set()
    all_final = []

    for cd, name in STORE_MAP.items():
        win_label, wdays = pick_best_window_composite(cd, name, holiday_dates)
        best_n = run_rfecv(cd, name, wdays, win_label, holiday_dates)
        all_final.append((name, win_label, best_n))

    print(f'\n\n{"="*110}')
    print(f'=== 최종 매장별 (window, N) 조합 — composite 기반 선택 ===')
    print(f'{"="*110}\n')
    print(f'{"매장":>10s} {"window":>8s} {"N":>4s} {"WAPE":>7s} {"매진율":>8s} {"폐기":>7s} {"부족":>6s} {"cost(원)":>13s}')
    total_cost = 0
    for name, win, b in all_final:
        total_cost += b['cost']
        print(f'  {name:>8s} {win:>7s} {b["N"]:>3d} {b["wape"]:>5.2f}% {b["under"]:>6.1f}% '
              f'{b["waste_sum"]:>6.0f} {b["short_sum"]:>5.0f} {b["cost"]:>12,.0f}')
    print(f'\n  4매장 합 cost (108일): {total_cost:,.0f}원')
    print(f'  연 환산 (×3.38): {total_cost * 3.38:,.0f}원/년')
    print(f'  5년 환산 (×16.9): {total_cost * 16.9:,.0f}원/5년')

    # 비교
    PREV = {
        '5Y RFECV': 39886212,
        'WAPE-window+RFECV': 36332507,
    }
    print(f'\n=== 모델 비교 ===')
    print(f'  통합 N=8 (4매장 평균 ranking): 40,780,528원')
    print(f'  5Y RFECV (매장별): 39,886,212원')
    print(f'  WAPE-window + RFECV (이전): 36,332,507원')
    print(f'  Composite-window + RFECV (현재): {total_cost:,.0f}원')

    pd.DataFrame([{
        'store': name,
        'window': win,
        'N': b['N'],
        'wape': b['wape'],
        'under': b['under'],
        'waste_sum': b['waste_sum'],
        'short_sum': b['short_sum'],
        'cost': b['cost'],
        'composite': b['composite'],
        'features': ','.join(b['features']),
    } for name, win, b in all_final]).to_csv('reports/window_composite_pipeline.csv', index=False)
    print('\nsaved: reports/window_composite_pipeline.csv')


if __name__ == '__main__':
    main()

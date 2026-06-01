"""Phase D: 4매장 각각 (a) 단독 학습 vs (b) multi-store 통합 학습 비교.

광교에선 multi -1.4pp 악화. 데이터 짧은 광화문(3.5년)에서 도움 되는지?
삼성타운/메세나처럼 광교와 다른 패턴 매장에선?
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd

from scripts.multistore_lgbm import (
    build_multistore_daily, add_features, run_backtest,
    HORIZONS, _load_holiday_set,
)


def run_for_store(df_all, target_store):
    """target_store 평가, (a) 단독 vs (b) multi-store 학습 둘 다."""
    holiday = _load_holiday_set()
    # (a) 단독
    df_solo = add_features(df_all[df_all['store'] == target_store].copy(), holiday)
    res_solo = run_backtest(df_solo, target_store=target_store, n_thursdays=16)

    # (b) multi-store
    df_multi = add_features(df_all.copy(), holiday)
    res_multi = run_backtest(df_multi, target_store=target_store, n_thursdays=16)

    def stats(res):
        if len(res) == 0:
            return None, None, None, None, None
        wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
        under = (res['production'] < res['actual']).mean()
        over = (res['production'] - res['actual']).mean()
        return wape, under, over, res['production'].mean(), res['actual'].mean()

    s_solo = stats(res_solo)
    s_multi = stats(res_multi)
    return target_store, s_solo, s_multi, res_solo, res_multi


def main():
    print('=== Phase D: 4매장 단독 vs multi-store 통합 ===\n')
    daily = build_multistore_daily()
    holiday = _load_holiday_set()

    targets = ['광교', '광화문', '메세나폴리스', '삼성타운']
    all_results = []

    for st in targets:
        print(f'\n--- target: {st} ---')
        store_name, s_solo, s_multi, res_solo, res_multi = run_for_store(daily, st)
        all_results.append((store_name, s_solo, s_multi))

    print('\n\n=== 종합 비교 ===')
    print(f'{"매장":>10s} {"":>3s} {"WAPE":>8s} {"매진율":>8s} {"발주over":>9s} {"avg_pred":>9s} {"avg_act":>9s}')
    for st, solo, multi in all_results:
        if solo[0] is not None:
            print(f'  {st:>8s} {"단독":>3s} {solo[0]*100:>6.2f}% {solo[1]*100:>6.1f}% '
                  f'{solo[2]:>+8.1f} {solo[3]:>9.1f} {solo[4]:>9.1f}')
        if multi[0] is not None:
            print(f'  {st:>8s} {"multi":>3s} {multi[0]*100:>6.2f}% {multi[1]*100:>6.1f}% '
                  f'{multi[2]:>+8.1f} {multi[3]:>9.1f} {multi[4]:>9.1f}')
        # delta
        if solo[0] is not None and multi[0] is not None:
            dw = (multi[0] - solo[0]) * 100
            du = (multi[1] - solo[1]) * 100
            sign = '✓' if dw < 0 else '✗'
            print(f'  {st:>8s} {"Δ":>3s} {dw:>+5.2f}pp {du:>+5.1f}pp  ({sign} multi {"better" if dw<0 else "worse"})\n')


if __name__ == '__main__':
    main()

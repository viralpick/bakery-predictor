"""광교 단독 backtest — target = normal_qty (예약 제외 평소 수요).

비교:
- 기존 광교 단독 (qty 단순합 target): WAPE 13.95% (N=108)
- 신규 광교 단독 (normal_qty target = 예약 제외): WAPE ?

기존 multistore_lgbm.py 코드 재사용, target 컬럼만 변경.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd
import numpy as np

from scripts.multistore_lgbm import (
    add_features, run_backtest, _load_holiday_set, HORIZONS,
)

V2 = Path('data/internal/v2')
STORE_MAP = {'1000000047': '광교', '1000000009': '삼성타운',
             '1000000029': '메세나폴리스', '1000000485': '광화문'}


def build_normal_daily() -> pd.DataFrame:
    """4매장 일별 normal_qty (예약 제외)."""
    daily = pd.read_parquet(V2 / 'daily_normal_vs_bulk.parquet')
    daily['date'] = pd.to_datetime(daily['date'])
    # multistore_lgbm.py의 run_backtest는 'qty' column 기대
    daily = daily[['store', 'date', 'normal_qty', 'total_qty', 'bulk_qty']].copy()
    daily['qty'] = daily['normal_qty']
    return daily


def run_for_store(target_store, daily_all, holiday_dates):
    sub = daily_all[daily_all['store'] == target_store].copy()
    print(f'\n--- {target_store} daily: {len(sub)}, avg normal={sub["normal_qty"].mean():.1f}, '
          f'total={sub["total_qty"].mean():.1f}, bulk%={sub["bulk_qty"].sum()/sub["total_qty"].sum()*100:.2f} ---')

    # (a) total qty target
    sub_total = sub.copy()
    sub_total['qty'] = sub_total['total_qty']
    df_t = add_features(sub_total, holiday_dates=holiday_dates)
    res_t = run_backtest(df_t, target_store=target_store, n_thursdays=16)

    # (b) normal qty target
    sub_normal = sub.copy()
    sub_normal['qty'] = sub_normal['normal_qty']
    df_n = add_features(sub_normal, holiday_dates=holiday_dates)
    res_n = run_backtest(df_n, target_store=target_store, n_thursdays=16)

    def stat(r):
        wape = (r['actual'] - r['production']).abs().sum() / r['actual'].sum()
        u = (r['production'] < r['actual']).mean()
        return wape, u, r['actual'].mean(), r['production'].mean()
    return stat(res_t), stat(res_n)


def main():
    print('=== 매장별 backtest — target = total_qty vs normal_qty (예약 제외) ===\n')
    daily_all = build_normal_daily()
    holiday_dates = _load_holiday_set()

    results = []
    for st in ['광교', '광화문', '메세나폴리스', '삼성타운']:
        stat_total, stat_normal = run_for_store(st, daily_all, holiday_dates)
        results.append((st, stat_total, stat_normal))

    print('\n\n=== 종합 비교 ===')
    print(f'{"매장":>10s} {"target":>10s} {"WAPE":>7s} {"매진율":>8s} {"평균실제":>10s} {"평균예측":>10s}')
    for st, (wt, ut, at, pt), (wn, un, an, pn) in results:
        print(f'  {st:>8s} {"total":>9s} {wt*100:>6.2f}% {ut*100:>6.1f}% {at:>9.1f} {pt:>9.1f}')
        print(f'  {st:>8s} {"normal":>9s} {wn*100:>6.2f}% {un*100:>6.1f}% {an:>9.1f} {pn:>9.1f}')
        dw = (wn - wt) * 100
        du = (un - ut) * 100
        sign = '↓ 개선' if dw < 0 else '↑ 악화' if dw > 0 else '='
        print(f'  {st:>8s} {"Δ":>9s} {dw:>+5.2f}pp {du:>+5.1f}pp  ({sign})\n')

    print('\n저장: reports/backtest_normal_target_4stores.txt')


if __name__ == '__main__':
    main()

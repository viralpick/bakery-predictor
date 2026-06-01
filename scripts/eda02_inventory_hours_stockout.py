"""EDA 2: 폐기율 + 마감시간 + 매진 시간 분포.

- 매장별/품목별 평균 폐기율 (폐기량 / 생산량)
- 매장별 dow별 마감시간 분포
- 매장별 매진 시간 cumulative 분포
- α=0.6 가정 검증 위한 sanity: 생산 - 판매 - 폐기 ≈ 0?
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('data/internal/v2')

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}


def parse_hhmm(x):
    """1954 → datetime.time(19,54) 같은 정수형 시각 → 분."""
    x = pd.to_numeric(x, errors='coerce')
    if pd.isna(x): return np.nan
    h = int(x) // 100
    m = int(x) % 100
    return h * 60 + m


def main():
    # ---------- 폐기율 ----------
    print('=== 재고정보: 폐기율 분석 ===')
    inv = pd.read_parquet(OUT / 'inventory.parquet')
    inv['date'] = pd.to_datetime(inv['DT_SALE'].astype(str), errors='coerce')
    inv['cd'] = inv['CD_PARTNER'].astype(str)
    inv['made'] = pd.to_numeric(inv['QT_MADE'], errors='coerce').fillna(0)
    inv['out'] = pd.to_numeric(inv['QT_OUT'], errors='coerce').fillna(0)
    inv['store'] = inv['cd'].map(STORE_MAP)
    inv = inv.dropna(subset=['store'])
    print(f'rows: {len(inv):,}, 매장: {inv["store"].unique()}')

    # 생산량 0 제외 (= 미생산일)
    inv_made = inv[inv['made'] > 0].copy()
    inv_made['waste_rate'] = inv_made['out'] / inv_made['made']

    print('\n=== 매장별 폐기율 (생산>0 row만) ===')
    for st, g in inv_made.groupby('store'):
        avg = g['waste_rate'].mean()
        wt_avg = g['out'].sum() / g['made'].sum()
        print(f'  {st}: 단순평균 {avg*100:.1f}%, 생산가중 {wt_avg*100:.1f}%, '
              f'생산 {g["made"].sum():,.0f} → 폐기 {g["out"].sum():,.0f}')

    # 품목별 폐기율 (광교만, top 10 생산량)
    print('\n=== 광교 품목별 폐기율 top 10 (생산량 기준) ===')
    gw = inv_made[inv_made['store']=='광교'].copy()
    by_item = gw.groupby('CD_ITEM').agg(
        made=('made','sum'), out=('out','sum'),
    )
    by_item['waste_rate'] = by_item['out'] / by_item['made']
    top = by_item.sort_values('made', ascending=False).head(10)
    items = pd.read_parquet(OUT / 'items.parquet')[['CD_ITEM','NM_ITEM']]
    top = top.merge(items, left_index=True, right_on='CD_ITEM')
    print(top[['CD_ITEM','NM_ITEM','made','out','waste_rate']].to_string())

    # ---------- 마감시간 ----------
    print('\n\n=== 영업시간: 마감시간 분포 ===')
    hours = pd.read_parquet(OUT / 'hours.parquet')
    hours['date'] = pd.to_datetime(hours['DT_SALE'].astype(str), errors='coerce')
    hours['cd'] = hours['CD_PARTNER'].astype(str)
    hours['close_min'] = hours['SALE_TIME'].apply(parse_hhmm)
    hours['store'] = hours['cd'].map(STORE_MAP)
    hours = hours.dropna(subset=['store','close_min'])
    hours['dow'] = hours['date'].dt.dayofweek

    print('\n매장별 평균 마감시간 (시:분 환산):')
    for st, g in hours.groupby('store'):
        avg = g['close_min'].mean()
        std = g['close_min'].std()
        print(f'  {st}: 평균 {int(avg)//60:02d}:{int(avg)%60:02d}, '
              f'std {std:.0f}분, n={len(g)}')

    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    print('\n매장별 dow별 평균 마감시간:')
    pivot = hours.groupby(['store','dow'])['close_min'].mean().unstack('dow')
    pivot.columns = [dn[d] for d in pivot.columns]
    print((pivot/60).round(2).to_string(), '\n(소수점 = 시간)')

    # 광교 2025 11-12월 마감시간 변화
    print('\n광교 2025 월별 평일 평균 마감시간:')
    gw_h = hours[(hours['store']=='광교') & (hours['date'] >= '2025-01-01') & (hours['dow'] < 5)]
    for m in range(1, 13):
        sub = gw_h[gw_h['date'].dt.month == m]
        if len(sub) == 0: continue
        avg = sub['close_min'].mean()
        print(f'  {m:>2}월: {int(avg)//60:02d}:{int(avg)%60:02d} (n={len(sub)})')

    # ---------- 품절 시간 ----------
    print('\n\n=== 품절정보: 매진 시간 분포 ===')
    so = pd.read_parquet(OUT / 'stockout.parquet')
    so['date'] = pd.to_datetime(so['DT_SALE'].astype(str), errors='coerce')
    so['cd'] = so['CD_PARTNER'].astype(str)
    so['sold_min'] = so['SOLD_TIME'].apply(parse_hhmm)
    so['store'] = so['cd'].map(STORE_MAP)
    so = so.dropna(subset=['store','sold_min'])

    print('\n매장별 매진 발생 row 수 + 평균 매진시각:')
    for st, g in so.groupby('store'):
        avg = g['sold_min'].mean()
        print(f'  {st}: n={len(g):,}, 평균 매진 {int(avg)//60:02d}:{int(avg)%60:02d}')

    # 시간대별 cumulative 매진 비율 (매장별)
    print('\n매장별 매진 시각 분위수 (10/25/50/75/90 percentile):')
    for st, g in so.groupby('store'):
        qs = g['sold_min'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        fmt = [f'{int(q)//60:02d}:{int(q)%60:02d}' for q in qs]
        print(f'  {st}: 10%={fmt[0]}, 25%={fmt[1]}, 50%={fmt[2]}, 75%={fmt[3]}, 90%={fmt[4]}')


if __name__ == '__main__':
    main()

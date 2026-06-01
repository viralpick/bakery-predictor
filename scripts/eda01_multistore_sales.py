"""EDA 1: 4매장 매출 trend 비교.

- 매장별 일별 매출 (수량/매출액)
- 광교 11-12월 평일 -10.4% drop이 4매장 공통 패턴인지
- 월별 / dow별 매출 분포
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


def main():
    print('=== sales load ===')
    sales = pd.read_parquet(OUT / 'sales.parquet')
    sales['DT_SALE'] = pd.to_datetime(sales['DT_SALE'].astype(str), errors='coerce')
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce')
    sales['AM_PAYMENT'] = pd.to_numeric(sales['AM_PAYMENT'], errors='coerce')
    # 4매장만 + 정상판매만
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']  # 정상판매
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)
    print(f'rows: {len(sales):,}')

    # 매장별 기간 / 일평균 매출
    print('\n=== 매장별 기간 + 일평균 ===')
    daily = sales.groupby(['store', 'DT_SALE']).agg(
        qty=('QT_SALE', 'sum'),
        rev=('AM_PAYMENT', 'sum'),
        receipts=('SLIP_NO', 'nunique'),
    ).reset_index()
    for st, g in daily.groupby('store'):
        print(f'  {st}: {g["DT_SALE"].min().date()} ~ {g["DT_SALE"].max().date()}, '
              f'일평균 qty={g["qty"].mean():.0f}, rev={g["rev"].mean():,.0f}원, '
              f'영수증={g["receipts"].mean():.0f}건')

    # 매장별 월별 매출 (2024 기준 trend 비교)
    daily['ym'] = daily['DT_SALE'].dt.to_period('M')
    daily['dow'] = daily['DT_SALE'].dt.dayofweek
    daily['is_weekend'] = daily['dow'] >= 5

    print('\n=== 매장별 11월/12월 평일 매출 변화 (2025) ===')
    print(f'{"매장":>10s} {"oct":>10s} {"nov":>10s} {"dec":>10s} {"nov_vs_oct":>10s} {"dec_vs_oct":>10s}')
    for st, g in daily.groupby('store'):
        g25 = g[(g['DT_SALE'] >= '2025-09-01') & (g['DT_SALE'] < '2026-01-01') & (~g['is_weekend'])]
        oct_avg = g25[g25['DT_SALE'].dt.month == 10]['qty'].mean()
        nov_avg = g25[g25['DT_SALE'].dt.month == 11]['qty'].mean()
        dec_avg = g25[g25['DT_SALE'].dt.month == 12]['qty'].mean()
        nov_pct = (nov_avg / oct_avg - 1) * 100 if oct_avg else 0
        dec_pct = (dec_avg / oct_avg - 1) * 100 if oct_avg else 0
        print(f'{st:>10s} {oct_avg:>10.0f} {nov_avg:>10.0f} {dec_avg:>10.0f} '
              f'{nov_pct:>+9.1f}% {dec_pct:>+9.1f}%')

    # dow별 매출 (전체 기간)
    print('\n=== 매장별 dow별 일평균 qty ===')
    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    pivot = daily.groupby(['store', 'dow'])['qty'].mean().unstack('dow')
    pivot.columns = [dn[d] for d in pivot.columns]
    print(pivot.round(0).to_string())

    # 연도별 매출 trend (매장별)
    print('\n=== 매장별 연도별 일평균 qty ===')
    daily['year'] = daily['DT_SALE'].dt.year
    yr_pivot = daily.groupby(['store', 'year'])['qty'].mean().unstack('year')
    print(yr_pivot.round(0).to_string())

    # save
    daily.to_parquet(OUT / 'daily_4stores.parquet', index=False)
    print(f'\nsaved: {OUT / "daily_4stores.parquet"}')


if __name__ == '__main__':
    main()

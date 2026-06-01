"""EDA 3: 4매장 카테고리 매출 분포 + 상관 비교.

- 매장별 카테고리 매출 비중
- 매장별 카테고리간 일별 매출 상관 (광교 한묶음 가설 검증)
- bread / pastry 비율
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd
import numpy as np

from bakery.data.bonavi_loader import map_category

OUT = Path('data/internal/v2')

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}

TARGETS = ('bread', 'pastry', 'sandwich', 'sweets', 'cake', 'beverage', 'etc')


def main():
    # 품목 카테고리 매핑
    items = pd.read_parquet(OUT / 'items.parquet')
    items = items[items['FG_ITEM']=='SS'].copy()  # 단품만
    items['category'] = items['NM_ITEM'].apply(map_category)
    print('=== 품목 카테고리 분포 ===')
    print(items['category'].value_counts())

    # sales 로드 + 4매장 + 정상 + 카테고리 join
    print('\n=== sales + category join ===')
    sales = pd.read_parquet(OUT / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']  # 단품만
    sales['DT_SALE'] = pd.to_datetime(sales['DT_SALE'].astype(str), errors='coerce')
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce')
    sales['AM_PAYMENT'] = pd.to_numeric(sales['AM_PAYMENT'], errors='coerce')
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)
    sales = sales.merge(items[['CD_ITEM','category']], on='CD_ITEM', how='left')
    sales['category'] = sales['category'].fillna('etc')
    print(f'rows: {len(sales):,}')

    # 매장별 카테고리 매출 비중
    print('\n=== 매장별 카테고리 매출 비중 (수량 기준) ===')
    pivot_qty = sales.groupby(['store', 'category'])['QT_SALE'].sum().unstack('category').fillna(0)
    pivot_pct = pivot_qty.div(pivot_qty.sum(axis=1), axis=0) * 100
    cats_present = [c for c in TARGETS if c in pivot_pct.columns]
    print(pivot_pct[cats_present].round(1).to_string())

    print('\n=== 매장별 카테고리 매출 비중 (금액 기준) ===')
    pivot_rev = sales.groupby(['store', 'category'])['AM_PAYMENT'].sum().unstack('category').fillna(0)
    pivot_rev_pct = pivot_rev.div(pivot_rev.sum(axis=1), axis=0) * 100
    cats_rev_present = [c for c in TARGETS if c in pivot_rev_pct.columns]
    print(pivot_rev_pct[cats_rev_present].round(1).to_string())

    # 매장별 카테고리간 매출 상관 (일별)
    print('\n=== 매장별 카테고리간 일별 매출 상관 (광교 한묶음 가설) ===')
    main_cats = ['bread', 'pastry', 'sandwich']
    for st, g in sales.groupby('store'):
        daily = g.groupby(['DT_SALE','category'])['QT_SALE'].sum().unstack('category').fillna(0)
        daily = daily[[c for c in main_cats if c in daily.columns]]
        corr = daily.corr()
        print(f'\n[{st}]')
        print(corr.round(3).to_string())

    # bread / pastry 비율 (시간순 변화)
    print('\n=== 매장별 bread / pastry 비율 (연도별) ===')
    sales['year'] = sales['DT_SALE'].dt.year
    bp = sales[sales['category'].isin(['bread','pastry'])].groupby(['store','year','category'])['QT_SALE'].sum().unstack('category').fillna(0)
    bp['bread_pct'] = bp['bread'] / (bp['bread'] + bp['pastry']) * 100
    print(bp.round(1).to_string())


if __name__ == '__main__':
    main()

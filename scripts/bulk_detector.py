"""예약 주문 검출 + 매장별 daily normal/bulk 분리.

정의:
    예약 영수증 = (단일 품목 qty ≥ 5)
                  AND (qty / 그 매장×품목 active day 일평균 ≥ 2.5)
                  AND (active days ≥ 14)

매장×품목 active day = inventory.parquet에서 made>0 OR out>0 인 일자.

출력:
    data/internal/v2/bulk_flagged_sales.parquet  — 영수증 단위 + is_bulk flag
    data/internal/v2/daily_normal_vs_bulk.parquet — 매장×일자 normal_qty/bulk_qty 분리
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd

V2 = Path('data/internal/v2')

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}

K_RATIO = 2.5             # 단일 품목: 평소 일평균의 K배 이상이면 예약
N_FLOOR = 5               # 단일 품목 절대 floor
MIN_ACTIVE_DAYS = 14      # 일평균 신뢰성 위한 최소 active days
DIVERSE_TOTAL_THRESH = 15 # 다양 품목 예약: 영수증 total ≥ N 이면 예약 (max_item < N_FLOOR 조건과 함께)


def main():
    print('[load] inventory + sales + items...')
    inv = pd.read_parquet(V2 / 'inventory.parquet')
    inv['date'] = pd.to_datetime(inv['DT_SALE'].astype(str))
    inv['cd'] = inv['CD_PARTNER'].astype(str)
    inv['item_id'] = inv['CD_ITEM'].astype(str)
    inv['made'] = pd.to_numeric(inv['QT_MADE'], errors='coerce').fillna(0)
    inv['out'] = pd.to_numeric(inv['QT_OUT'], errors='coerce').fillna(0)
    inv = inv[inv['cd'].isin(STORE_MAP.keys())]

    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']
    sales['date'] = pd.to_datetime(sales['DT_SALE'].astype(str))
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['cd'] = sales['CD_PARTNER'].astype(str)
    sales['item_id'] = sales['CD_ITEM'].astype(str)
    sales['store'] = sales['cd'].map(STORE_MAP)
    sales['receipt_id'] = (sales['cd'] + '_' + sales['DT_SALE'].astype(str) + '_'
                          + sales['NO_POS'].astype(str) + '_' + sales['SLIP_NO'].astype(str))

    # ----- 1. 매장×품목 active day + 일평균 ------------------------------
    print('[stats] 매장×품목 active days + 일평균 계산...')
    active = inv[(inv['made'] > 0) | (inv['out'] > 0)]
    active_stats = active.groupby(['cd', 'item_id']).agg(
        active_days=('date', 'nunique'),
    ).reset_index()

    sold_stats = sales.groupby(['cd', 'item_id'])['QT_SALE'].sum().reset_index().rename(columns={'QT_SALE': 'total_sold'})

    item_stats = active_stats.merge(sold_stats, on=['cd', 'item_id'], how='outer')
    item_stats['active_days'] = item_stats['active_days'].fillna(0).astype(int)
    item_stats['total_sold'] = item_stats['total_sold'].fillna(0)
    item_stats['daily_avg'] = item_stats['total_sold'] / item_stats['active_days'].replace(0, pd.NA)
    item_stats['threshold'] = (item_stats['daily_avg'] * K_RATIO).clip(lower=N_FLOOR)
    item_stats['threshold'] = item_stats['threshold'].fillna(N_FLOOR)
    item_stats['reliable'] = item_stats['active_days'] >= MIN_ACTIVE_DAYS
    print(f'  매장×품목 unique: {len(item_stats):,}')
    print(f'  reliable (active≥14): {item_stats["reliable"].sum():,}')

    # ----- 2. 영수증 단위 max(단일 품목 qty) -----
    print('\n[detect] 영수증 단위 예약 분류...')
    receipt_item = sales.groupby(['cd', 'receipt_id', 'item_id', 'date']).agg(
        item_qty=('QT_SALE', 'sum'),
    ).reset_index()

    # join item stats
    receipt_item = receipt_item.merge(item_stats[['cd', 'item_id', 'daily_avg', 'threshold', 'reliable']],
                                       on=['cd', 'item_id'], how='left')
    receipt_item['is_bulk_line'] = (
        (receipt_item['item_qty'] >= N_FLOOR) &
        (receipt_item['item_qty'] >= receipt_item['threshold']) &
        (receipt_item['reliable'].fillna(False))
    )

    # 영수증 단위 단일 품목 bulk
    receipt_single = receipt_item.groupby(['cd', 'receipt_id', 'date']).agg(
        is_bulk_single=('is_bulk_line', 'any'),
        total_qty=('item_qty', 'sum'),
        max_item_qty=('item_qty', 'max'),
    ).reset_index()

    # 다양 품목 bulk: total ≥ 15 (max_item 조건 없음 — 예약+추가구매 섞인 케이스도 포함)
    receipt_single['is_bulk_diverse'] = (
        receipt_single['total_qty'] >= DIVERSE_TOTAL_THRESH
    )
    receipt_single['is_bulk'] = receipt_single['is_bulk_single'] | receipt_single['is_bulk_diverse']

    n_single = receipt_single['is_bulk_single'].sum()
    n_diverse = receipt_single['is_bulk_diverse'].sum()
    n_both = (receipt_single['is_bulk_single'] & receipt_single['is_bulk_diverse']).sum()
    print(f'  단일 품목 예약: {n_single:,} 영수증')
    print(f'  다양 품목 예약: {n_diverse:,} 영수증')
    print(f'  (둘 다 만족: {n_both:,} = 단일에 자동 포함되므로 중복 X)')
    print(f'  총 예약: {receipt_single["is_bulk"].sum():,}')

    receipt_flag = receipt_single[['cd', 'receipt_id', 'date', 'is_bulk',
                                    'is_bulk_single', 'is_bulk_diverse']]

    # ----- 3. 매장×일자 normal/bulk 분리 -----
    print('\n[aggregate] 매장×일자 normal vs bulk...')
    sales_with_flag = sales.merge(receipt_flag[['cd', 'receipt_id', 'is_bulk']],
                                   on=['cd', 'receipt_id'], how='left')
    sales_with_flag['is_bulk'] = sales_with_flag['is_bulk'].fillna(False)

    daily = sales_with_flag.groupby(['store', 'date']).agg(
        total_qty=('QT_SALE', 'sum'),
        normal_qty=('QT_SALE', lambda x: x[~sales_with_flag.loc[x.index, 'is_bulk']].sum()),
        bulk_qty=('QT_SALE', lambda x: x[sales_with_flag.loc[x.index, 'is_bulk']].sum()),
        normal_receipts=('receipt_id', lambda x: x[~sales_with_flag.loc[x.index, 'is_bulk']].nunique()),
        bulk_receipts=('receipt_id', lambda x: x[sales_with_flag.loc[x.index, 'is_bulk']].nunique()),
    ).reset_index()
    daily['bulk_pct'] = daily['bulk_qty'] / daily['total_qty'].replace(0, pd.NA) * 100

    # ----- 4. 매장별 요약 통계 -----
    print('\n=== 매장별 5년 요약 ===')
    print(f'{"매장":>10s} {"총 qty":>10s} {"normal":>10s} {"bulk":>10s} {"bulk%":>8s} '
          f'{"영수증":>9s} {"normal":>9s} {"bulk":>9s}')
    for st in ['광교', '광화문', '메세나폴리스', '삼성타운']:
        sub = daily[daily['store'] == st]
        tot = sub['total_qty'].sum()
        nor = sub['normal_qty'].sum()
        bul = sub['bulk_qty'].sum()
        n_recp = sub['normal_receipts'].sum() + sub['bulk_receipts'].sum()
        nor_recp = sub['normal_receipts'].sum()
        bul_recp = sub['bulk_receipts'].sum()
        print(f'  {st:>8s} {int(tot):>10,} {int(nor):>10,} {int(bul):>10,} {bul/tot*100:>7.2f}% '
              f'{n_recp:>9,} {nor_recp:>9,} {bul_recp:>9,}')

    # ----- 5. 매장별 bulk top 10 일자 -----
    print('\n=== 매장별 bulk qty top 10 일자 ===')
    for st in ['광교', '광화문', '메세나폴리스', '삼성타운']:
        sub = daily[daily['store'] == st].sort_values('bulk_qty', ascending=False).head(10)
        print(f'\n[{st}]')
        for _, r in sub.iterrows():
            dow = '월화수목금토일'[r['date'].dayofweek]
            print(f'  {r["date"].date()} ({dow}) total={int(r["total_qty"]):>4}, '
                  f'bulk={int(r["bulk_qty"]):>3}, normal={int(r["normal_qty"]):>4}, '
                  f'bulk%={r["bulk_pct"]:>5.1f}%')

    # save
    item_stats.to_parquet(V2 / 'item_active_stats.parquet', index=False)
    daily.to_parquet(V2 / 'daily_normal_vs_bulk.parquet', index=False)
    sales_with_flag[['cd', 'store', 'date', 'receipt_id', 'item_id', 'QT_SALE', 'is_bulk']].to_parquet(
        V2 / 'sales_with_bulk_flag.parquet', index=False)
    print(f'\nsaved:')
    print(f'  {V2 / "item_active_stats.parquet"}')
    print(f'  {V2 / "daily_normal_vs_bulk.parquet"}')
    print(f'  {V2 / "sales_with_bulk_flag.parquet"}')


if __name__ == '__main__':
    main()

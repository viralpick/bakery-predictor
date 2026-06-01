"""EDA 4 (Phase C): 폐기 실측 + α 캘리브레이션.

1. 항등식 검증: production = normal_sales + closing_sales + waste + carry_over?
2. 매장별 폐기 손실 실측 (4매장, 5년치)
3. 마감할인 회수율 = closing_sales / (closing_sales + waste)
4. α 추정 단서: 마감판매 시각 vs 매진 시각, 매장별 마감판매 비중
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd
import numpy as np

V2 = Path('data/internal/v2')

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}


def load_closing_codes() -> set:
    """마감 할인코드만 추출 (기존 광교 분석 결과 재사용)."""
    # discount_categorize.py의 분류 결과 재사용 — 광교 30개 할인코드 중 마감 분류된 것
    # 새 discount_codes.parquet 의 비고에서 '마감/폐기/closing' 키워드 매칭
    dc = pd.read_parquet(V2 / 'discount_codes.parquet')
    dc['NM'] = dc['NM_DISC'].astype(str)
    dc['RMK'] = dc['DC_RMK'].astype(str).fillna('')
    # 키워드 기반 추출
    mask = (
        dc['NM'].str.contains('마감|할인빵|클로징|closing', case=False, na=False) |
        dc['RMK'].str.contains('마감|폐기직전|소진', na=False)
    )
    closing_codes = set(dc.loc[mask, 'CD_DISC'].astype(str))
    print(f'마감 할인코드 후보 {len(closing_codes)}개:')
    print(dc.loc[mask, ['CD_DISC','NM_DISC','RT_DISC','DC_RMK']].to_string(max_rows=30))
    return closing_codes


def build_daily_item_4stores(closing_codes: set) -> pd.DataFrame:
    """매장×일×품목 단위로 정상판매 / 마감판매 / 매출 aggregate."""
    print('\n[build] sales aggregate...')
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']
    sales['DT_SALE'] = pd.to_datetime(sales['DT_SALE'].astype(str), errors='coerce')
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['AM_PAYMENT'] = pd.to_numeric(sales['AM_PAYMENT'], errors='coerce').fillna(0)
    sales['AM_DC'] = pd.to_numeric(sales['AM_DC'], errors='coerce').fillna(0)
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['is_closing'] = sales['CD_USERDEF1'].isin(closing_codes)

    g = sales.groupby(['CD_PARTNER', 'DT_SALE', 'CD_ITEM']).agg(
        normal_qty=('QT_SALE', lambda x: x[~sales.loc[x.index, 'is_closing']].sum()),
        closing_qty=('QT_SALE', lambda x: x[sales.loc[x.index, 'is_closing']].sum()),
        normal_rev=('AM_PAYMENT', lambda x: x[~sales.loc[x.index, 'is_closing']].sum()),
        closing_rev=('AM_PAYMENT', lambda x: x[sales.loc[x.index, 'is_closing']].sum()),
        discount_amt=('AM_DC', lambda x: x[sales.loc[x.index, 'is_closing']].sum()),
    ).reset_index()
    g = g.rename(columns={'CD_PARTNER': 'cd', 'DT_SALE': 'date', 'CD_ITEM': 'item_id'})
    return g


def main():
    closing_codes = load_closing_codes()

    sales_daily = build_daily_item_4stores(closing_codes)
    print(f'\nsales aggregate: {len(sales_daily):,} rows')

    # inventory
    print('\n[load] inventory...')
    inv = pd.read_parquet(V2 / 'inventory.parquet')
    inv['date'] = pd.to_datetime(inv['DT_SALE'].astype(str), errors='coerce')
    inv['cd'] = inv['CD_PARTNER'].astype(str)
    inv['item_id'] = inv['CD_ITEM'].astype(str)
    inv['made'] = pd.to_numeric(inv['QT_MADE'], errors='coerce').fillna(0)
    inv['out'] = pd.to_numeric(inv['QT_OUT'], errors='coerce').fillna(0)
    inv = inv[['cd', 'date', 'item_id', 'made', 'out']]

    # join
    df = inv.merge(sales_daily, on=['cd', 'date', 'item_id'], how='left')
    df[['normal_qty','closing_qty','normal_rev','closing_rev','discount_amt']] = \
        df[['normal_qty','closing_qty','normal_rev','closing_rev','discount_amt']].fillna(0)
    df['store'] = df['cd'].map(STORE_MAP)
    df['sold_total'] = df['normal_qty'] + df['closing_qty']
    df['identity_diff'] = df['made'] - df['sold_total'] - df['out']  # 항등식

    # 1. 항등식 검증
    print('\n=== 1. 항등식 검증: 생산 - 판매(정상+마감) - 폐기 = ? ===')
    print('매장별 (made > 0 row만):')
    df_made = df[df['made'] > 0]
    for st, g in df_made.groupby('store'):
        match = (g['identity_diff'].abs() <= 0.5).mean() * 100
        avg_diff = g['identity_diff'].mean()
        sum_made = g['made'].sum()
        sum_sold = g['sold_total'].sum()
        sum_out = g['out'].sum()
        print(f'  {st}: 정확 일치 {match:.1f}%, 평균 diff {avg_diff:+.2f}, '
              f'made={sum_made:,.0f} sold={sum_sold:,.0f} out={sum_out:,.0f}, '
              f'diff_sum={sum_made - sum_sold - sum_out:+,.0f}')

    # 2. 폐기율 + 마감회수율
    print('\n=== 2. 매장별 폐기율 + 마감회수율 ===')
    print(f'{"매장":>10s} {"폐기율":>8s} {"마감비중":>10s} {"마감회수율":>12s} {"광교손실 환산":>15s}')
    for st, g in df_made.groupby('store'):
        waste_rate = g['out'].sum() / g['made'].sum()
        closing_share = g['closing_qty'].sum() / g['sold_total'].sum() if g['sold_total'].sum() > 0 else 0
        recovery = g['closing_qty'].sum() / (g['closing_qty'].sum() + g['out'].sum()) if (g['closing_qty'].sum() + g['out'].sum()) > 0 else 0
        # 폐기 비용 = QT_OUT × 단가 (품목정보에서)
        print(f'  {st:>8s} {waste_rate*100:>6.1f}% {closing_share*100:>8.1f}% {recovery*100:>10.1f}%')

    # 3. 매장별 5년 폐기 손실 (실측) + 마감 손실
    print('\n=== 3. 매장별 5년 폐기 + 마감 손실 (실측) ===')
    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['UM_SO'] = pd.to_numeric(items['UM_SO'], errors='coerce').fillna(4000)
    price_map = items.set_index('item_id')['UM_SO'].to_dict()
    df_made['unit_price'] = df_made['item_id'].map(price_map).fillna(4000)
    df_made['waste_cost'] = df_made['out'] * df_made['unit_price']

    print(f'{"매장":>10s} {"기간(연)":>10s} {"폐기수량":>10s} {"폐기 비용(원)":>15s} {"마감 손실(원)":>15s} {"연환산 폐기":>15s} {"연환산 마감":>15s}')
    for st, g in df_made.groupby('store'):
        years = (g['date'].max() - g['date'].min()).days / 365.25
        waste_qty = g['out'].sum()
        waste_cost = g['waste_cost'].sum()
        closing_loss = g['discount_amt'].sum()
        ann_waste = waste_cost / years
        ann_closing = closing_loss / years
        print(f'  {st:>8s} {years:>8.1f} {waste_qty:>10,.0f} {waste_cost:>13,.0f} {closing_loss:>13,.0f} {ann_waste:>13,.0f} {ann_closing:>13,.0f}')

    # 4. α 추정 단서: 매장별 마감판매 시각 vs 매진 시각
    print('\n=== 4. 매장별 마감판매 비중 + 마감판매 시각 분포 ===')
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['is_closing'] = sales['CD_USERDEF1'].isin(closing_codes)
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)
    # 시각 추출 (SALES_TIME format: YYYYMMDDHHMMSS)
    sales['hhmm'] = sales['SALES_TIME'].astype(str).str.slice(8, 12)
    sales['hour'] = pd.to_numeric(sales['hhmm'].str.slice(0, 2), errors='coerce')

    closing_only = sales[sales['is_closing']].copy()
    print(f'{"매장":>10s} {"마감 거래 수":>12s} {"평균 시각":>10s} {"중앙값":>8s} {"75%":>8s}')
    for st, g in closing_only.groupby('store'):
        avg = g['hour'].mean()
        med = g['hour'].quantile(0.5)
        q75 = g['hour'].quantile(0.75)
        print(f'  {st:>8s} {len(g):>10,} {avg:>8.2f}시 {med:>6.1f}시 {q75:>6.1f}시')

    # 5. 광교 7,846만원 갱신 (실측 vs 기존 추정)
    print('\n=== 5. 광교 마감/폐기 손실 갱신 (실측) ===')
    gw = df_made[df_made['store'] == '광교']
    if len(gw):
        years = (gw['date'].max() - gw['date'].min()).days / 365.25
        closing_loss = gw['discount_amt'].sum()
        waste_cost = gw['waste_cost'].sum()
        print(f'  기간: {years:.1f}년')
        print(f'  기존 추정 (마감 할인 손실, 5년): 7,846만원')
        print(f'  실측 (5년 마감 할인 손실): {closing_loss/10000:,.0f}만원')
        print(f'  실측 (5년 폐기 비용): {waste_cost/10000:,.0f}만원')
        print(f'  총 손실 (마감+폐기): {(closing_loss + waste_cost)/10000:,.0f}만원')
        print(f'  연환산: 마감 {closing_loss/years/10000:,.0f}만원, '
              f'폐기 {waste_cost/years/10000:,.0f}만원, 합 {(closing_loss+waste_cost)/years/10000:,.0f}만원')

    # save
    df_made.to_parquet(V2 / 'waste_alpha_4stores.parquet', index=False)
    print(f'\nsaved: {V2 / "waste_alpha_4stores.parquet"}')


if __name__ == '__main__':
    main()

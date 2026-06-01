"""검증 A: closing 외 28개 할인 코드의 매장×시각 분포 + 비중.

가설: PAYCO/면세/이벤트 등 종일 promo 코드가 정상가 demand에 노이즈?
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
CLOSING_CODES = {'0069', '0077'}


def main():
    # 할인코드 마스터
    dc = pd.read_parquet(V2 / 'discount_codes.parquet')
    dc['CD_DISC'] = dc['CD_DISC'].astype(str)
    dc['NM_DISC'] = dc['NM_DISC'].astype(str)
    dc['RT_DISC'] = pd.to_numeric(dc['RT_DISC'], errors='coerce').fillna(0)

    # sales 4매장
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['AM_PAYMENT'] = pd.to_numeric(sales['AM_PAYMENT'], errors='coerce').fillna(0)
    sales['AM_DC'] = pd.to_numeric(sales['AM_DC'], errors='coerce').fillna(0)
    sales['hhmm'] = sales['SALES_TIME'].astype(str).str.slice(8, 12)
    sales['hour'] = pd.to_numeric(sales['hhmm'].str.slice(0, 2), errors='coerce')

    # 할인 사용된 row만
    has_disc = sales[
        (sales['CD_USERDEF1'].notna()) &
        (sales['CD_USERDEF1'] != 'nan') &
        (sales['CD_USERDEF1'] != '') &
        (sales['AM_DC'] > 0)
    ].copy()
    print(f'할인 사용 rows: {len(has_disc):,} (전체 sales {len(sales):,}의 {len(has_disc)/len(sales)*100:.1f}%)')

    # ----- 1. 매장별 사용 코드 top 10 (closing 외) -----
    print('\n=== 1. 매장별 closing 외 할인 코드 top 10 (사용량 기준) ===')
    other = has_disc[~has_disc['CD_USERDEF1'].isin(CLOSING_CODES)].copy()
    print(f'closing 외 할인 rows: {len(other):,}')

    for st in ['광교','광화문','메세나폴리스','삼성타운']:
        sub = other[other['store']==st]
        if len(sub) == 0: continue
        print(f'\n[{st}]')
        top = sub.groupby('CD_USERDEF1').agg(
            n=('QT_SALE','size'),
            qty=('QT_SALE','sum'),
            dc_amt=('AM_DC','sum'),
            avg_hour=('hour','mean'),
            peak_hour=('hour', lambda x: x.value_counts().index[0] if len(x) else None),
        ).sort_values('n', ascending=False).head(10)
        top = top.merge(dc[['CD_DISC','NM_DISC','RT_DISC']], left_index=True, right_on='CD_DISC', how='left')
        print(top[['CD_DISC','NM_DISC','RT_DISC','n','qty','dc_amt','avg_hour','peak_hour']].to_string(index=False))

    # ----- 2. closing vs other 시각 분포 비교 (광교) -----
    print('\n\n=== 2. 광교 closing vs other 할인 시각 분포 ===')
    gw = has_disc[has_disc['store']=='광교'].copy()
    gw['kind'] = gw['CD_USERDEF1'].isin(CLOSING_CODES).map({True:'closing', False:'other'})
    pivot = gw.groupby(['kind','hour']).size().unstack('hour').fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    print(pivot_pct.round(1).to_string())

    # ----- 3. 광교 other 할인의 시간대별 분포 + 매출 비중 -----
    print('\n=== 3. 광교 other 할인 시간대별 사용 + 정상매출 대비 비중 ===')
    sales_gw = sales[sales['store']=='광교']
    other_gw = other[other['store']=='광교']

    # 시간대별
    def bucket(h):
        if pd.isna(h): return 'unknown'
        if h < 9: return '<9시'
        if h < 14: return '9-14시'
        if h < 17: return '14-17시'
        if h < 19: return '17-19시'
        if h < 21: return '19-21시 (마감)'
        return '21+시'
    buckets = ['<9시','9-14시','14-17시','17-19시','19-21시 (마감)','21+시']
    sales_gw = sales_gw.assign(b=sales_gw['hour'].apply(bucket))
    other_gw = other_gw.assign(b=other_gw['hour'].apply(bucket))

    summary = []
    for b in buckets:
        tot_qty = sales_gw.loc[sales_gw['b']==b, 'QT_SALE'].sum()
        oth_qty = other_gw.loc[other_gw['b']==b, 'QT_SALE'].sum()
        summary.append({
            'bucket': b,
            'total_qty': int(tot_qty),
            'other_disc_qty': int(oth_qty),
            'other_disc_pct': round(oth_qty/tot_qty*100, 2) if tot_qty else 0,
        })
    print(pd.DataFrame(summary).to_string(index=False))

    # ----- 4. 100%/높은 할인 코드 (성격 의심) -----
    print('\n=== 4. 광교 50%+ 할인 코드 (종일 promo 의심) ===')
    high_disc = dc[dc['RT_DISC'] >= 50]
    print(f'50%+ 할인 코드 {len(high_disc)}개:')
    print(high_disc[['CD_DISC','NM_DISC','RT_DISC','DC_RMK']].to_string(max_rows=20))

    print('\n광교에서 50%+ 할인 코드 사용:')
    gw_high = has_disc[(has_disc['store']=='광교') & (has_disc['CD_USERDEF1'].isin(high_disc['CD_DISC'].astype(str)))]
    if len(gw_high):
        agg = gw_high.groupby('CD_USERDEF1').agg(
            n=('QT_SALE','size'),
            qty=('QT_SALE','sum'),
            avg_hour=('hour','mean'),
        )
        agg = agg.merge(dc[['CD_DISC','NM_DISC','RT_DISC']], left_index=True, right_on='CD_DISC')
        print(agg[['CD_DISC','NM_DISC','RT_DISC','n','qty','avg_hour']].sort_values('qty', ascending=False).to_string(index=False))
    else:
        print('  사용 없음')


if __name__ == '__main__':
    main()

"""검증 1: 0077/0069 closing 할인코드의 시각 분포 (매장별).

가설: 0077/0069는 "마감 직전 사용"이라 가정. 평시 사용 시 misclassification.
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
CLOSING_CODES = ['0069', '0077']  # 마감 코드 후보


def main():
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)
    # 시각
    sales['hhmm'] = sales['SALES_TIME'].astype(str).str.slice(8, 12)
    sales['hour'] = pd.to_numeric(sales['hhmm'].str.slice(0, 2), errors='coerce')

    # 1. 매장별 closing 코드 사용량
    print('=== 1. 매장별 0077/0069 사용 횟수 ===')
    for code in CLOSING_CODES:
        sub = sales[sales['CD_USERDEF1'] == code]
        print(f'\n[{code}] 총 {len(sub):,}건')
        print(sub.groupby('store').size())

    # 2. 매장 × 시각 분포 (closing 코드 0077/0069 합산)
    print('\n\n=== 2. 매장별 closing 코드 시각 분포 ===')
    closing = sales[sales['CD_USERDEF1'].isin(CLOSING_CODES)]
    print(f'\n전체 closing rows: {len(closing):,}')
    print(f'\n매장 × 시각 (시) 분포 (%):')
    pivot = closing.groupby(['store','hour']).size().unstack('hour').fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    print(pivot_pct.round(1).to_string())

    # 3. 평시 (9-17시) vs 마감 (18-22시) 비율 매장별
    print('\n=== 3. 매장별 시간대 분포 ===')
    def bucket(h):
        if pd.isna(h): return 'unknown'
        if h < 9: return '<9시'
        if h < 17: return '9-17시 (평시)'
        if h < 19: return '17-19시 (저녁)'
        if h < 21: return '19-21시 (마감)'
        return '21+시 (심야)'
    closing = closing.assign(bucket=closing['hour'].apply(bucket))
    pivot = closing.groupby(['store','bucket']).size().unstack('bucket').fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    order = ['<9시','9-17시 (평시)','17-19시 (저녁)','19-21시 (마감)','21+시 (심야)']
    cols = [c for c in order if c in pivot_pct.columns]
    print(pivot_pct[cols].round(1).to_string())

    # 4. "평시 9-17시 사용" 비율 매장별 (의심스러운 케이스)
    print('\n=== 4. 매장별 평시(9-17시) 사용 비율 (misclassification 의심) ===')
    daytime_pct = (closing['hour'] < 17) & (closing['hour'] >= 9)
    by_store = closing.assign(is_daytime=daytime_pct).groupby('store').agg(
        total=('CD_USERDEF1','size'),
        daytime=('is_daytime','sum'),
    )
    by_store['daytime_pct'] = by_store['daytime'] / by_store['total'] * 100
    print(by_store.round(1).to_string())

    # 5. 광화문 평시 사용 패턴 deep dive (마감시각 19:05라 의심)
    print('\n=== 5. 광화문 closing 코드 평시 (9-17시) 사용 — 일자 패턴 ===')
    gwm_day = closing[(closing['store']=='광화문') & (closing['hour'] >= 9) & (closing['hour'] < 17)].copy()
    if len(gwm_day):
        gwm_day['date'] = pd.to_datetime(gwm_day['DT_SALE'].astype(str))
        gwm_day['ym'] = gwm_day['date'].dt.to_period('M')
        print(f'  rows: {len(gwm_day):,}')
        print(f'  월별 발생 빈도 (top 5):')
        print(gwm_day['ym'].value_counts().head(5))
        print(f'\n  시간 분포:')
        print(gwm_day['hour'].value_counts().sort_index().to_string())

    # 6. 각 매장에서 closing 코드별 시각 detail (0077 vs 0069)
    print('\n\n=== 6. 코드별 시각 통계 (매장 × 코드) ===')
    print(f'{"매장":>10s} {"코드":>5s} {"n":>10s} {"평균":>8s} {"중앙값":>8s} {"75%":>6s} {"평시%":>8s}')
    for st in ['광교','광화문','메세나폴리스','삼성타운']:
        for code in CLOSING_CODES:
            sub = sales[(sales['store']==st) & (sales['CD_USERDEF1']==code)]
            if len(sub) == 0: continue
            mean = sub['hour'].mean()
            med = sub['hour'].median()
            q75 = sub['hour'].quantile(0.75)
            day = ((sub['hour'] >= 9) & (sub['hour'] < 17)).mean() * 100
            print(f'  {st:>8s} {code:>5s} {len(sub):>10,} {mean:>6.2f}시 {med:>6.1f}시 {q75:>4.1f}시 {day:>6.1f}%')


if __name__ == '__main__':
    main()

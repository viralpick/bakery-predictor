"""EDA 5 (C-2): 매장 × 카테고리 × dow 폐기율 패턴.

- 어떤 카테고리가 과잉생산인지
- 어떤 요일이 과잉생산인지
- 폐기율 시계열 trend
- top 폐기 품목 (절대 비용 기준)
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd

from bakery.data.bonavi_loader import map_category

V2 = Path('data/internal/v2')


def main():
    print('=== load ===')
    df = pd.read_parquet(V2 / 'waste_alpha_4stores.parquet')
    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['category'] = items['NM_ITEM'].apply(map_category)
    df = df.merge(items[['item_id','NM_ITEM','category']], on='item_id', how='left')
    df['category'] = df['category'].fillna('etc')
    df['dow'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    print(f'rows: {len(df):,}')

    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}

    # 1. 매장 × 카테고리 폐기율
    print('\n=== 1. 매장 × 카테고리 폐기율 (생산가중) ===')
    g = df.groupby(['store','category']).agg(made=('made','sum'), out=('out','sum'))
    g['waste_rate'] = g['out'] / g['made']
    pivot = g['waste_rate'].unstack('category').fillna(0) * 100
    cats = ['bread','pastry','sandwich','cake','etc']
    cats = [c for c in cats if c in pivot.columns]
    print(pivot[cats].round(1).to_string())

    # 2. 매장 × dow 폐기율
    print('\n=== 2. 매장 × dow 폐기율 ===')
    g = df.groupby(['store','dow']).agg(made=('made','sum'), out=('out','sum'))
    g['waste_rate'] = g['out'] / g['made']
    pivot = g['waste_rate'].unstack('dow').fillna(0) * 100
    pivot.columns = [dn[d] for d in pivot.columns]
    print(pivot.round(1).to_string())

    # 3. 매장 × 카테고리 × dow 폐기율 (광교만 — 상세)
    print('\n=== 3. 광교 카테고리 × dow 폐기율 ===')
    gw = df[df['store']=='광교']
    g = gw.groupby(['category','dow']).agg(made=('made','sum'), out=('out','sum'))
    g['waste_rate'] = g['out'] / g['made']
    pivot = g['waste_rate'].unstack('dow').fillna(0) * 100
    pivot.columns = [dn[d] for d in pivot.columns]
    print(pivot.round(1).to_string())

    # 4. 폐기율 시계열 (연도별, 매장별)
    print('\n=== 4. 매장 × 연도 폐기율 ===')
    g = df.groupby(['store','year']).agg(made=('made','sum'), out=('out','sum'))
    g['waste_rate'] = g['out'] / g['made']
    pivot = g['waste_rate'].unstack('year').fillna(0) * 100
    print(pivot.round(1).to_string())

    # 5. top 폐기 품목 (절대 비용 기준, 매장별 top 5)
    print('\n=== 5. 매장별 top 5 폐기 비용 품목 (5년 누적) ===')
    g = df.groupby(['store','NM_ITEM']).agg(
        out=('out','sum'),
        waste_cost=('waste_cost','sum'),
        made=('made','sum'),
    )
    g['waste_rate'] = g['out'] / g['made']
    for st in ['광교','광화문','메세나폴리스','삼성타운']:
        if st not in g.index.get_level_values('store').unique():
            continue
        top = g.xs(st, level='store').sort_values('waste_cost', ascending=False).head(5)
        print(f'\n[{st}]')
        print(top.round({'waste_rate': 3}).to_string())

    # 6. 매장 × 카테고리 폐기 비용 (절대값)
    print('\n=== 6. 매장 × 카테고리 5년 폐기 비용 (만원) ===')
    g = df.groupby(['store','category'])['waste_cost'].sum().unstack('category').fillna(0) / 10000
    g['합계'] = g.sum(axis=1)
    print(g.round(0).astype(int).to_string())

    # 7. 광교 11-12월 vs 다른 달 폐기율 (D+7 drop 시즌과 일치?)
    print('\n=== 7. 광교 월별 폐기율 + 평일 매출 변화 ===')
    gw['month'] = gw['date'].dt.month
    g = gw[gw['date'] >= '2025-01-01'].groupby('month').agg(
        made=('made','sum'), out=('out','sum'),
        days=('date','nunique'),
    )
    g['waste_rate'] = g['out'] / g['made']
    g['avg_made'] = g['made'] / g['days']
    print(g[['days','made','out','avg_made','waste_rate']].round({'avg_made':0, 'waste_rate':3}).to_string())


if __name__ == '__main__':
    main()

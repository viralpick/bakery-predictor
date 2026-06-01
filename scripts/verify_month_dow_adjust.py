"""검증: 광교 월 × 요일 12×7 매트릭스 — adjust 전후 비교.

raw qty (sold_total) vs adjusted_demand (sold_normal + sold_closing × 0.5)
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd
import numpy as np

V2 = Path('data/internal/v2')
GWANGYO_CD = '1000000047'
CLOSING_CODES = {'0069', '0077'}
ALPHA = 0.5


def main():
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str) == GWANGYO_CD]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']  # 단품
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['date'] = pd.to_datetime(sales['DT_SALE'].astype(str))
    sales['is_closing'] = sales['CD_USERDEF1'].isin(CLOSING_CODES)

    # 일별 sold_total, sold_closing
    daily = sales.groupby('date').agg(
        sold_total=('QT_SALE', 'sum'),
        sold_closing=('QT_SALE', lambda x: x[sales.loc[x.index, 'is_closing']].sum()),
    ).reset_index()
    daily['sold_normal'] = daily['sold_total'] - daily['sold_closing']
    daily['adjusted'] = daily['sold_normal'] + daily['sold_closing'] * ALPHA
    daily['month'] = daily['date'].dt.month
    daily['dow'] = daily['date'].dt.dayofweek
    print(f'일별 row: {len(daily):,}, 기간 {daily["date"].min().date()} ~ {daily["date"].max().date()}')
    print(f'평균 sold_total: {daily["sold_total"].mean():.1f}, '
          f'평균 sold_closing: {daily["sold_closing"].mean():.1f} ({daily["sold_closing"].sum()/daily["sold_total"].sum()*100:.1f}%), '
          f'평균 adjusted: {daily["adjusted"].mean():.1f}')

    # ---------------------------------------------------------------
    # 12 × 7 매트릭스
    # ---------------------------------------------------------------
    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    raw = daily.groupby(['month','dow'])['sold_total'].mean().unstack('dow')
    adj = daily.groupby(['month','dow'])['adjusted'].mean().unstack('dow')
    closing = daily.groupby(['month','dow'])['sold_closing'].mean().unstack('dow')

    raw.columns = [dn[d] for d in raw.columns]
    adj.columns = [dn[d] for d in adj.columns]
    closing.columns = [dn[d] for d in closing.columns]

    print('\n=== 1. raw (sold_total) — 월 × 요일 일평균 ===')
    print(raw.round(0).astype(int).to_string())

    print('\n=== 2. adjusted_demand (adjust 후) — 월 × 요일 일평균 ===')
    print(adj.round(0).astype(int).to_string())

    print('\n=== 3. closing qty (마감 판매량) — 월 × 요일 일평균 ===')
    print(closing.round(1).to_string())

    diff_pct = (raw - adj) / raw * 100
    print('\n=== 4. adjust로 인한 감소율 (%) — 월 × 요일 ===')
    print(diff_pct.round(2).to_string())

    # 통계
    print('\n=== 5. 요약 통계 ===')
    print(f'  raw 평균       : {raw.mean().mean():.1f}')
    print(f'  adjusted 평균  : {adj.mean().mean():.1f}')
    print(f'  diff % 평균    : {diff_pct.mean().mean():.2f}%')
    print(f'  diff % 최대    : {diff_pct.max().max():.2f}% ({diff_pct.stack().idxmax()})')
    print(f'  diff % 최소    : {diff_pct.min().min():.2f}% ({diff_pct.stack().idxmin()})')

    # 요일별 평균
    print('\n=== 6. 요일별 평균 diff % ===')
    by_dow = daily.groupby('dow').agg(
        raw=('sold_total','mean'),
        adj=('adjusted','mean'),
        closing=('sold_closing','mean'),
    )
    by_dow.index = [dn[d] for d in by_dow.index]
    by_dow['diff_pct'] = (by_dow['raw'] - by_dow['adj']) / by_dow['raw'] * 100
    by_dow['closing_share_pct'] = by_dow['closing'] / by_dow['raw'] * 100
    print(by_dow.round(2).to_string())

    # 월별 평균
    print('\n=== 7. 월별 평균 diff % ===')
    by_mon = daily.groupby('month').agg(
        raw=('sold_total','mean'),
        adj=('adjusted','mean'),
        closing=('sold_closing','mean'),
    )
    by_mon['diff_pct'] = (by_mon['raw'] - by_mon['adj']) / by_mon['raw'] * 100
    by_mon['closing_share_pct'] = by_mon['closing'] / by_mon['raw'] * 100
    print(by_mon.round(2).to_string())


if __name__ == '__main__':
    main()

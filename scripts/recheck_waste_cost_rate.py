"""폐기·마감 손실 재계산 (2026-07-14).

배경: 초기(5/26) eda04에서 폐기 비용 = QT_OUT × UM_SO(판매단가), 원가율 미적용.
     → 광교 9,102만원/년은 '판매가 전액' 기준. 현재 표준(business_metrics.py)은
     waste_cost = units × unit_price × cost_rate(0.30, 원가율).
이 스크립트: 동일 실측 폐기수량(QT_OUT)에 원가율을 적용해 재계산 + 항등식 재검증.
마감 손실(AM_DC 실측 할인액)은 원가율 무관(실제 지출된 할인).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda04_waste_alpha_calibration import (  # noqa: E402
    STORE_MAP, V2, load_closing_codes, build_daily_item_4stores,
)

COST_RATES = [1.00, 0.30]  # 1.00 = 옛 naive(판매가 전액), 0.30 = 현재 표준 원가율


def main() -> None:
    closing_codes = load_closing_codes()
    sales_daily = build_daily_item_4stores(closing_codes)

    inv = pd.read_parquet(V2 / 'inventory.parquet')
    inv['date'] = pd.to_datetime(inv['DT_SALE'].astype(str), errors='coerce')
    inv['cd'] = inv['CD_PARTNER'].astype(str)
    inv['item_id'] = inv['CD_ITEM'].astype(str)
    inv['made'] = pd.to_numeric(inv['QT_MADE'], errors='coerce').fillna(0)
    inv['out'] = pd.to_numeric(inv['QT_OUT'], errors='coerce').fillna(0)
    inv = inv[['cd', 'date', 'item_id', 'made', 'out']]

    df = inv.merge(sales_daily, on=['cd', 'date', 'item_id'], how='left')
    cols = ['normal_qty', 'closing_qty', 'normal_rev', 'closing_rev', 'discount_amt']
    df[cols] = df[cols].fillna(0)
    df['store'] = df['cd'].map(STORE_MAP)
    df['sold_total'] = df['normal_qty'] + df['closing_qty']

    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['UM_SO'] = pd.to_numeric(items['UM_SO'], errors='coerce').fillna(4000)
    price_map = items.set_index('item_id')['UM_SO'].to_dict()
    df['unit_price'] = df['item_id'].map(price_map).fillna(4000)

    made = df[df['made'] > 0].copy()

    # 항등식 재검증 (QT_OUT이 실측 폐기인지)
    made['identity_diff'] = made['made'] - made['sold_total'] - made['out']
    print('\n=== 항등식 재검증: made - sold - out ≈ 0 (QT_OUT=실측 폐기 확인) ===')
    for st, g in made.groupby('store'):
        match = (g['identity_diff'].abs() <= 0.5).mean() * 100
        print(f'  {st:>10s}: 정확일치 {match:5.1f}%  평균 diff {g["identity_diff"].mean():+.3f}')

    made['waste_retail'] = made['out'] * made['unit_price']
    print('\n=== 매장별 5년 손실: 폐기(원가율별) + 마감(실측 할인액) ===')
    hdr = f'{"매장":>10s} {"연수":>5s} {"폐기qty":>9s}'
    for r in COST_RATES:
        hdr += f' {"폐기@"+format(r,".0%"):>12s}'
    hdr += f' {"마감손실":>11s}'
    print(hdr)
    rows = []
    for st, g in made.groupby('store'):
        years = (g['date'].max() - g['date'].min()).days / 365.25
        wqty = g['out'].sum()
        retail = g['waste_retail'].sum()
        closing = g['discount_amt'].sum()
        line = f'  {st:>8s} {years:>5.1f} {wqty:>9,.0f}'
        rec = {'store': st, 'years': years, 'waste_qty': wqty,
               'closing_loss': closing}
        for r in COST_RATES:
            wc = retail * r
            line += f' {wc/1e8:>10.2f}억'
            rec[f'waste_cr{int(r*100)}'] = wc
        line += f' {closing/1e4:>9,.0f}만'
        print(line)
        rows.append(rec)

    print('\n=== 연환산 총손실 (폐기 원가 + 마감) — 원가율별 비교 ===')
    print(f'{"매장":>10s}' + ''.join(f' {"@"+format(r,".0%")+" 연":>14s}' for r in COST_RATES))
    for rec in rows:
        y = rec['years']
        line = f'  {rec["store"]:>8s}'
        for r in COST_RATES:
            total_ann = (rec[f'waste_cr{int(r*100)}'] + rec['closing_loss']) / y
            line += f' {total_ann/1e4:>11,.0f}만'
        print(line)

    print('\n※ 옛 9,102만원/년(광교) = @100%(판매가 전액) 열. 현재 표준 원가율=@30% 열.')


if __name__ == '__main__':
    main()

"""매장별 기존 운영(실측) vs 모델 적용 시 추정 비교 — 절감 효과.

전제:
- 기존 운영 = 매장이 실제로 만들고 폐기한 결과 (inventory.QT_OUT)
- 모델 적용 = 우리 모델(v4 + bulk 제외) 발주 권장 시 예상 폐기/부족

비교 조건:
- 카테고리 한정 (bread/pastry/sandwich) — 모델 target과 일치
- backtest 기간 (108일, 2025-09-15 ~ 12-31) 동일 비교
- 5년 환산 (×5×365/108 ≈ ×16.9)
- 매장×품목 실측 단가 사용 (모델 폐기는 카테고리 평균 단가)
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd

from bakery.data.bonavi_loader import map_category

V2 = Path('data/internal/v2')
TARGET_CATEGORIES = ('bread', 'pastry', 'sandwich')
BACKTEST_START = pd.Timestamp('2025-09-15')
BACKTEST_END = pd.Timestamp('2025-12-31')
DAYS_BACKTEST = (BACKTEST_END - BACKTEST_START).days + 1   # 108일
YEARS_FULL = 5
SCALE_TO_5YEAR = YEARS_FULL * 365 / DAYS_BACKTEST            # ≈ 16.9

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}

# 모델 backtest 결과 (α=0.7 통일, 2026-05-29)
MODEL_BACKTEST = {
    '광교':       {'waste_sum': 3715, 'short_sum': 193},
    '광화문':     {'waste_sum': 3340, 'short_sum': 337},
    '메세나폴리스': {'waste_sum': 2727, 'short_sum': 122},
    '삼성타운':   {'waste_sum': 4402, 'short_sum': 306},
}


def main():
    print(f'=== 매장 실측 vs 모델 적용 절감 효과 ===')
    print(f'기간: {BACKTEST_START.date()} ~ {BACKTEST_END.date()} ({DAYS_BACKTEST}일)')
    print(f'5년 환산 비율: ×{SCALE_TO_5YEAR:.2f}\n')

    # 카테고리 한정 매장 실측 데이터 계산
    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['category'] = items['NM_ITEM'].apply(map_category)
    items['UM_SO'] = pd.to_numeric(items['UM_SO'], errors='coerce').fillna(4000)
    items_target = items[items['category'].isin(TARGET_CATEGORIES)]
    price_map = items_target.set_index('item_id')['UM_SO'].to_dict()

    inv = pd.read_parquet(V2 / 'inventory.parquet')
    inv['date'] = pd.to_datetime(inv['DT_SALE'].astype(str))
    inv['cd'] = inv['CD_PARTNER'].astype(str)
    inv['item_id'] = inv['CD_ITEM'].astype(str)
    inv['out'] = pd.to_numeric(inv['QT_OUT'], errors='coerce').fillna(0)
    inv = inv[inv['item_id'].isin(items_target['item_id'])]
    inv['unit_price'] = inv['item_id'].map(price_map).fillna(4000)
    inv['waste_cost'] = inv['out'] * inv['unit_price']

    rows = []
    for cd, name in STORE_MAP.items():
        sub = inv[inv['cd'] == cd]
        # backtest 기간 실측 폐기
        bt = sub[(sub['date'] >= BACKTEST_START) & (sub['date'] <= BACKTEST_END)]
        actual_waste_qty_bt = bt['out'].sum()
        actual_waste_cost_bt = bt['waste_cost'].sum()
        avg_price = (actual_waste_cost_bt / actual_waste_qty_bt) if actual_waste_qty_bt else 5000

        # 모델 backtest 결과
        m = MODEL_BACKTEST[name]
        model_waste_qty_bt = m['waste_sum']
        model_short_qty_bt = m['short_sum']
        model_waste_cost_bt = model_waste_qty_bt * avg_price
        model_short_cost_bt = model_short_qty_bt * avg_price

        # save (backtest 기간)
        save_qty_bt = actual_waste_qty_bt - model_waste_qty_bt
        save_cost_bt = actual_waste_cost_bt - model_waste_cost_bt - model_short_cost_bt  # 폐기 절감 - 매진 cost

        # 5년 환산
        save_5y = save_cost_bt * SCALE_TO_5YEAR
        save_yr = save_5y / YEARS_FULL

        rows.append({
            'store': name,
            'avg_price': avg_price,
            'actual_waste_qty_bt': actual_waste_qty_bt,
            'actual_waste_cost_bt': actual_waste_cost_bt,
            'model_waste_qty_bt': model_waste_qty_bt,
            'model_short_qty_bt': model_short_qty_bt,
            'model_waste_cost_bt': model_waste_cost_bt,
            'model_short_cost_bt': model_short_cost_bt,
            'save_cost_bt': save_cost_bt,
            'save_yr': save_yr,
            'save_5y': save_5y,
        })

    df = pd.DataFrame(rows)

    print(f'\n=== Backtest 기간 ({DAYS_BACKTEST}일, 카테고리: bread/pastry/sandwich) ===\n')
    print(f'{"매장":>10s} {"평균단가":>9s} {"매장 폐기 qty":>13s} {"매장 폐기 비용":>15s} '
          f'{"모델 폐기 qty":>13s} {"모델 부족 qty":>13s}')
    for _, r in df.iterrows():
        print(f'  {r["store"]:>8s} {r["avg_price"]:>7.0f}원 {r["actual_waste_qty_bt"]:>12,.0f} '
              f'{r["actual_waste_cost_bt"]:>13,.0f}원 {r["model_waste_qty_bt"]:>12,.0f} '
              f'{r["model_short_qty_bt"]:>12,.0f}')

    print(f'\n=== 절감 효과 (모델 적용 시) ===\n')
    print(f'{"매장":>10s} {"매장 폐기":>11s} {"모델 폐기":>11s} {"모델 매진(cost)":>15s} '
          f'{"108일 save":>12s} {"연 save":>12s} {"5년 save":>12s}')
    for _, r in df.iterrows():
        print(f'  {r["store"]:>8s} {r["actual_waste_cost_bt"]:>10,.0f}원 '
              f'{r["model_waste_cost_bt"]:>10,.0f}원 {r["model_short_cost_bt"]:>13,.0f}원 '
              f'{r["save_cost_bt"]:>+10,.0f}원 {r["save_yr"]:>+10,.0f}원 {r["save_5y"]:>+10,.0f}원')

    total_save_yr = df['save_yr'].sum()
    total_save_5y = df['save_5y'].sum()
    print(f'\n  {"4매장 합":>8s} {df["actual_waste_cost_bt"].sum():>10,.0f}원 '
          f'{df["model_waste_cost_bt"].sum():>10,.0f}원 {df["model_short_cost_bt"].sum():>13,.0f}원 '
          f'{df["save_cost_bt"].sum():>+10,.0f}원 {total_save_yr:>+10,.0f}원 {total_save_5y:>+10,.0f}원')

    # 매장 실측 5년 폐기 비용 (이전 분석)
    print(f'\n\n=== 매장 5년 실측 폐기 비용 (전 카테고리, 참고) ===')
    print(f'  광교        3.7억 / 광화문 (3.5년) 2.6억 / 메세나 3.3억 / 삼성타운 3.3억 = 합 12.9억')
    print(f'  (위는 매장 운영 절대 손실. 모델 도입 시 위 중 일부 절감 가능)')

    # 카테고리 매장 비중 — 매장 실측 5년 폐기 중 카테고리 한정 비중
    print(f'\n=== 카테고리 한정 매장 5년 폐기 비용 ===')
    inv_all = pd.read_parquet(V2 / 'inventory.parquet')
    inv_all['cd'] = inv_all['CD_PARTNER'].astype(str)
    inv_all['item_id'] = inv_all['CD_ITEM'].astype(str)
    inv_all['out'] = pd.to_numeric(inv_all['QT_OUT'], errors='coerce').fillna(0)
    items_all_price = items.set_index('item_id')['UM_SO'].to_dict()
    inv_all['unit_price'] = inv_all['item_id'].map(items_all_price).fillna(4000)
    inv_all['waste_cost'] = inv_all['out'] * inv_all['unit_price']
    inv_all = inv_all[inv_all['cd'].isin(STORE_MAP.keys())]
    inv_all = inv_all.merge(items[['item_id', 'category']], on='item_id', how='left')
    inv_all_target = inv_all[inv_all['category'].isin(TARGET_CATEGORIES)]
    for cd, name in STORE_MAP.items():
        full_waste_cost = inv_all[inv_all['cd'] == cd]['waste_cost'].sum()
        target_waste_cost = inv_all_target[inv_all_target['cd'] == cd]['waste_cost'].sum()
        target_pct = target_waste_cost / full_waste_cost * 100 if full_waste_cost else 0
        print(f'  {name:>10s}: 전체 {full_waste_cost/10000:>6,.0f}만 / target {target_waste_cost/10000:>6,.0f}만 ({target_pct:.1f}%)')

    df.to_csv('reports/savings_analysis.csv', index=False)
    print('\nsaved: reports/savings_analysis.csv')


if __name__ == '__main__':
    main()

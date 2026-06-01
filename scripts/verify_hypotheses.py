"""4 가설 검증 — modeling_v4 framework 가정.

A. 영수증 패턴 (1-1-a 95% "빵 자체")
B. 카테고리 시간 분포 (1-1-b bread/pastry 경계)
C. 인기품 매진 → 매장 시간당 매출 (2-1-a 손실 0)
D. 매장 매출 trend vs 매진 빈도 (2-1-b 장기 만족도)
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from bakery.analysis.seasonal import filter_seasonal


def header(s):
    print(f'\n{"="*70}\n{s}\n{"="*70}')


# ───────── Load ─────────
header('Loading data')
receipts = pd.read_parquet('data/internal/bonavi_receipts.parquet')
daily = pd.read_parquet('data/internal/bonavi_daily.parquet')
receipts['item_id'] = receipts['item_id'].astype(str)
daily['item_id'] = daily['item_id'].astype(str)

# 시즌 제외
daily_f = filter_seasonal(daily)
receipts_f = filter_seasonal(receipts)
cat_map = daily_f.drop_duplicates('item_id').set_index('item_id')['category_id']
receipts_f['category'] = receipts_f['item_id'].map(cat_map)
receipts_f = receipts_f.dropna(subset=['category'])

print(f'receipts (after seasonal filter): {len(receipts_f):,}')
print(f'unique receipt_ids: {receipts_f["receipt_id"].nunique():,}')

# ════════════════════════════════════════════════════════════════════
# 검증 A: 영수증 패턴 (1-1-a)
# ════════════════════════════════════════════════════════════════════
header('A. 영수증 패턴 — 가정 1-1-a "95% 빵 자체"')

receipt_agg = receipts_f.groupby('receipt_id').agg(
    n_lineitems=('item_id', 'size'),
    n_unique_items=('item_id', 'nunique'),
    n_unique_cats=('category', 'nunique'),
    cats=('category', lambda s: tuple(sorted(set(s)))),
).reset_index()

n_total = len(receipt_agg)
print(f'\n영수증 총: {n_total:,}')

# A-1: 영수증당 unique 품목 수 분포
print(f'\n[A-1] 영수증당 unique 품목 수 분포:')
dist = receipt_agg['n_unique_items'].value_counts().sort_index()
for k, v in dist.head(10).items():
    print(f'  {k}품목  {v:>7,}  ({v/n_total*100:>5.1f}%)')
print(f'  평균: {receipt_agg["n_unique_items"].mean():.2f}')
print(f'  중앙: {receipt_agg["n_unique_items"].median():.0f}')

# A-2: 단품 영수증 비율
single = (receipt_agg['n_unique_items'] == 1).sum()
multi = n_total - single
print(f'\n[A-2] 단품 vs 다품목:')
print(f'  단품 (1품목):    {single:>7,} ({single/n_total*100:.1f}%)')
print(f'  다품목 (2+):     {multi:>7,} ({multi/n_total*100:.1f}%)')

# A-3: 다품목 영수증의 카테고리 다양성
multi_df = receipt_agg[receipt_agg['n_unique_items'] >= 2]
print(f'\n[A-3] 다품목 영수증의 카테고리 혼합:')
cat_div = multi_df['n_unique_cats'].value_counts(normalize=True).sort_index()
for k, p in cat_div.items():
    print(f'  카테고리 {k}개:  {p*100:>5.1f}%')

# A-4: 다품목 영수증의 카테고리 조합 빈도
print(f'\n[A-4] 다품목 영수증의 카테고리 조합 (top 10):')
combo_str = multi_df['cats'].apply(lambda t: ' + '.join(t))
combo_counts = combo_str.value_counts().head(10)
for combo, count in combo_counts.items():
    print(f'  {combo:50s}  {count:>6,}  ({count/len(multi_df)*100:.1f}%)')

# A-5: 같은 카테고리 vs 다른 카테고리 다품목 비율
same_cat = (multi_df['n_unique_cats'] == 1).sum()
diff_cat = len(multi_df) - same_cat
print(f'\n[A-5] 다품목 영수증 — 카테고리 같음 vs 다름:')
print(f'  같은 카테고리만:  {same_cat:>7,} ({same_cat/len(multi_df)*100:.1f}%)')
print(f'  다른 카테고리 섞임: {diff_cat:>7,} ({diff_cat/len(multi_df)*100:.1f}%)')

# 해석 가이드
print(f'\n[A 해석]:')
print(f'  - 단품 비율 = "특정 타겟 손님" 추정. 낮을수록 "빵 자체" 가설 강함.')
print(f'  - 다품목 + 다카테고리 혼합 비율 ↑ = 카테고리 경계 약함 (1-1-b 보강)')

# tuple → str for parquet 호환
receipt_agg['cats_str'] = receipt_agg['cats'].apply(lambda t: ' + '.join(t))
receipt_agg.drop(columns=['cats']).to_parquet('reports/_verify_A_receipt_pattern.parquet')

# ════════════════════════════════════════════════════════════════════
# 검증 B: 카테고리 시간 분포 (1-1-b)
# ════════════════════════════════════════════════════════════════════
header('B. 카테고리 시간 분포 — 가정 1-1-b "bread/pastry 경계 흐릿"')

by_hr = receipts_f.groupby(['hour', 'category']).size().unstack(fill_value=0)
total_by_hr = by_hr.sum(axis=1)
print(f'\n[B-1] 시간대별 카테고리 비율 (% of 그 시간 총 unit):')
ratio = by_hr.div(total_by_hr, axis=0) * 100
print(ratio.round(1).to_string())

# B-2: bread vs pastry 분포 비교 — KS test
print(f'\n[B-2] bread vs pastry 시간 분포 KS test:')
bread_hrs = receipts_f.loc[receipts_f['category']=='bread', 'hour']
pastry_hrs = receipts_f.loc[receipts_f['category']=='pastry', 'hour']
ks_stat, ks_p = stats.ks_2samp(bread_hrs, pastry_hrs)
print(f'  KS statistic: {ks_stat:.4f}')
print(f'  p-value:      {ks_p:.4e}')
print(f'  → p < 0.01 = 분포 다름 (경계 있음), p > 0.05 = 분포 같음 (경계 흐릿)')

# B-3: 카테고리 비율 시간대간 일관성 (8-21시 비교)
print(f'\n[B-3] 카테고리 비율 시간대간 변동성 (8-21시):')
ratio_business = ratio.loc[8:21].copy()
print('  bread CV (변동계수, σ/μ):', f'{ratio_business["bread"].std()/ratio_business["bread"].mean():.3f}')
print('  pastry CV:               ', f'{ratio_business["pastry"].std()/ratio_business["pastry"].mean():.3f}')
print('  (CV 낮으면 시간대 무관 일정 비율 → 카테고리 경계 약함)')

# B-4: 평균 영수증 시각 비교
print(f'\n[B-4] 카테고리별 평균 구매 시각:')
for cat in ['bread', 'pastry', 'sandwich', 'cake']:
    if cat in receipts_f['category'].unique():
        hrs = receipts_f.loc[receipts_f['category']==cat, 'hour']
        print(f'  {cat:10s}: 평균 {hrs.mean():.2f}시  중앙 {hrs.median():.0f}시  ({len(hrs):,} unit)')

ratio.to_csv('reports/_verify_B_category_hour_ratio.csv')

# ════════════════════════════════════════════════════════════════════
# 검증 C: 인기품 매진 시각 → 매장 시간당 매출 (2-1-a)
# ════════════════════════════════════════════════════════════════════
header('C. 인기품 매진 → 매장 unit/hr 변화 — 가정 2-1-a "stockout = 매출 손실 0"')

# top-10 인기품
top_items = daily_f.groupby('item_id')['sold_units'].sum().sort_values(ascending=False).head(10).index.tolist()
print(f'\nTop 10 인기품 (시즌 제외):')
xl = pd.ExcelFile('data/internal/보나비 데이터_20260520.xlsx')
items = pd.read_excel(xl, sheet_name='품목정보')
items['item_id'] = items['품목코드'].astype(str)
name_map = items.set_index('item_id')['POS메뉴명']
for iid in top_items:
    sold = daily_f[daily_f['item_id']==iid]['sold_units'].sum()
    print(f'  {iid}  {name_map.get(iid, "?"):30s}  sold {int(sold):,}')

# 각 인기품에 대해 매진 시각별 매장 시간당 매출 비교
print(f'\n[C-1] 인기품 매진일 vs 비매진일 — 매진 후 매장 unit/hr 비교')

# 매장 매출 시간당
hourly_store = receipts_f.groupby(['date', 'hour']).size().rename('store_unit').reset_index()

results_c = []
for iid in top_items[:5]:  # top 5 only (시간 절약)
    item_daily = daily_f[daily_f['item_id']==iid][['date','is_stockout','stockout_time']].copy()
    item_daily['stockout_hr'] = pd.to_datetime(item_daily['stockout_time']).dt.hour

    # 매진 시각 ≤ 14시인 날 vs 매진 안 한 날 / 매진 ≥ 20시
    early = item_daily[item_daily['is_stockout'] & (item_daily['stockout_hr'] <= 14)]['date'].tolist()
    late_or_none = item_daily[(~item_daily['is_stockout']) | (item_daily['stockout_hr'] >= 20)]['date'].tolist()

    if len(early) < 30 or len(late_or_none) < 30:
        continue

    # 14-17시 매장 unit/hr 비교 (early 매진 영향 받는 시간)
    after_hrs = [14, 15, 16, 17]
    s_early = hourly_store[(hourly_store['date'].isin(early)) & (hourly_store['hour'].isin(after_hrs))]['store_unit']
    s_norm  = hourly_store[(hourly_store['date'].isin(late_or_none)) & (hourly_store['hour'].isin(after_hrs))]['store_unit']
    t_stat, p_val = stats.ttest_ind(s_early, s_norm, equal_var=False)

    results_c.append({
        'item_id': iid,
        'name': name_map.get(iid, '?'),
        'n_early': len(early),
        'n_norm': len(late_or_none),
        'early_mean': s_early.mean(),
        'norm_mean':  s_norm.mean(),
        'diff_pct':   (s_early.mean() - s_norm.mean()) / s_norm.mean() * 100,
        't_stat':     t_stat,
        'p_val':      p_val,
    })

df_c = pd.DataFrame(results_c)
print(df_c.round(2).to_string(index=False))

print(f'\n[C 해석]:')
print(f'  diff_pct < 0 + p < 0.05 = 인기품 매진 시 매장 매출 실제 감소 (가정 2-1-a 약화)')
print(f'  diff_pct ≈ 0 또는 p > 0.05 = 매장 매출 보존 (가정 2-1-a 지지)')

df_c.to_csv('reports/_verify_C_stockout_store_impact.csv', index=False)

# ════════════════════════════════════════════════════════════════════
# 검증 D: 매장 매출 trend vs 매진 빈도 trend (2-1-b)
# ════════════════════════════════════════════════════════════════════
header('D. 매장 매출 trend vs 매진 빈도 trend — 가정 2-1-b 장기 만족도')

# 6개월 윈도우 매장 매출 평균
daily_f['date'] = pd.to_datetime(daily_f['date'])
store_daily = daily_f.groupby('date')['sold_units'].sum().reset_index().rename(columns={'sold_units':'store_total'})

# 6개월 (180일) 롤링 평균
store_daily = store_daily.sort_values('date').reset_index(drop=True)
store_daily['store_total_180d'] = store_daily['store_total'].rolling(180, min_periods=90).mean()

# top-10 매진 빈도 — 일별 매진 품목 수 (조기 매진만; <16시)
def early_stockout_count(row_df):
    so = row_df[row_df['is_stockout']]
    so = so.copy()
    so['so_hr'] = pd.to_datetime(so['stockout_time']).dt.hour
    return (so['so_hr'] <= 16).sum()

so_daily = daily_f.groupby('date').apply(early_stockout_count).rename('early_so_count').reset_index()
merged = store_daily.merge(so_daily, on='date', how='left').fillna(0)
merged['early_so_180d'] = merged['early_so_count'].rolling(180, min_periods=90).mean()

# 두 trend 상관관계
merged_v = merged.dropna(subset=['store_total_180d', 'early_so_180d'])
corr, p_corr = stats.pearsonr(merged_v['store_total_180d'], merged_v['early_so_180d'])
print(f'\n[D-1] 매장 매출 180d 평균 vs 조기 매진 품목수 180d 평균 상관:')
print(f'  Pearson r: {corr:.4f}')
print(f'  p-value:   {p_corr:.4e}')
print(f'  → r < 0 + p < 0.05 = 매진 ↑ ⇒ 매출 ↓ (장기 만족도 가설 지지)')
print(f'  → r ≈ 0 또는 r > 0 = 매진과 매출 trend 무관 (가설 약함)')

# 분기별 평균
merged['quarter'] = merged['date'].dt.to_period('Q')
qa = merged.groupby('quarter').agg(
    store_total_avg=('store_total', 'mean'),
    early_so_count_avg=('early_so_count', 'mean'),
).reset_index()
print(f'\n[D-2] 분기별 매장 매출 vs 조기매진 빈도:')
print(qa.round(2).to_string(index=False))

merged.to_csv('reports/_verify_D_trend.csv', index=False)

print('\n' + '='*70)
print('완료. 결과 파일:')
print('  reports/_verify_A_receipt_pattern.parquet')
print('  reports/_verify_B_category_hour_ratio.csv')
print('  reports/_verify_C_stockout_store_impact.csv')
print('  reports/_verify_D_trend.csv')

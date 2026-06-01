"""Phase F: 새 데이터로 광교 v4 fair backtest.

기존 v4 (5/20 데이터, WAPE 29.28%) vs 새 데이터 + 동일 정의 (α=0.5, filter_seasonal).
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

from bakery.data.bonavi_loader import map_category
from bakery.features.category_aggregate import (
    build_category_daily, build_features,
    EVENTS, LUNAR_EVENTS, _signed_days_to_event, _days_to_lunar_event,
)

V2 = Path('data/internal/v2')

GWANGYO_CD = '1000000047'
TARGET_COL = 'adjusted_demand_unit'
HORIZONS = [4, 5, 6, 7, 8, 9, 10]
PRODUCTION_Q = 0.90
ALPHA = 0.7  # α sensitivity 결과 (2026-05-29) — 4매장 best 평균. 0.5 → 0.7

# 마감 할인코드 (eda04에서 확인된 5개 — 0069/0077/0150/329/320)
CLOSING_CODES = {'0069', '0077', '320'}  # 마감할인만 (음료/쿠차라 제외)

SHORT_TERM_TO_DROP = [
    f'{TARGET_COL}_rmean7',
    f'{TARGET_COL}_rstd7',
    f'{TARGET_COL}_ewma7',
]


def _load_holiday_set():
    cal = pd.read_parquet("data/external/calendar_raw.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return set(cal.loc[cal["is_holiday"] == True, "date"])


def build_new_data_daily(exclude_bulk: bool = False) -> pd.DataFrame:
    """새 데이터 광교 → bonavi_daily 형식.
    exclude_bulk=True 시 예약(bulk) 영수증 line 제외 — normal_qty target 용.
    """
    print(f'[build] sales → gwangyo daily (exclude_bulk={exclude_bulk})...')
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str) == GWANGYO_CD]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']
    sales['date'] = pd.to_datetime(sales['DT_SALE'].astype(str))
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)

    if exclude_bulk:
        flag = pd.read_parquet(V2 / 'sales_with_bulk_flag.parquet')
        flag = flag[flag['cd'] == GWANGYO_CD]
        sales['receipt_id'] = (sales['CD_PARTNER'].astype(str) + '_' + sales['DT_SALE'].astype(str)
                              + '_' + sales['NO_POS'].astype(str) + '_' + sales['SLIP_NO'].astype(str))
        bulk_receipts = set(flag[flag['is_bulk']]['receipt_id'])
        before = len(sales)
        sales = sales[~sales['receipt_id'].isin(bulk_receipts)]
        print(f'  bulk 영수증 line 제외: {before:,} → {len(sales):,}')

    daily = sales.groupby(['date', 'CD_ITEM'])['QT_SALE'].sum().reset_index()
    daily = daily.rename(columns={'CD_ITEM': 'item_id', 'QT_SALE': 'sold_units'})
    daily['item_id'] = daily['item_id'].astype(str)
    daily['store_id'] = 'store_gw01'

    # category mapping
    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['category_id'] = items['NM_ITEM'].apply(map_category)
    cat_map = items.set_index('item_id')['category_id'].to_dict()
    daily['category_id'] = daily['item_id'].map(cat_map).fillna('etc')

    # stockout join (광교)
    print('[build] stockout join...')
    so = pd.read_parquet(V2 / 'stockout.parquet')
    so = so[so['CD_PARTNER'].astype(str) == GWANGYO_CD].copy()
    so['date'] = pd.to_datetime(so['DT_SALE'].astype(str))
    so['item_id'] = so['CD_ITEM'].astype(str)
    so['SOLD_TIME'] = pd.to_numeric(so['SOLD_TIME'], errors='coerce')
    so['hh'] = so['SOLD_TIME'].astype('Int64') // 100
    so['mm'] = so['SOLD_TIME'].astype('Int64') % 100
    so['stockout_time'] = pd.to_datetime(
        so['date'].astype(str) + ' ' + so['hh'].astype(str) + ':' + so['mm'].astype(str),
        errors='coerce',
    )
    so_first = so.sort_values('stockout_time').groupby(['date', 'item_id'])['stockout_time'].first().reset_index()
    daily = daily.merge(so_first, on=['date', 'item_id'], how='left')
    daily['is_stockout'] = daily['stockout_time'].notna()

    print(f'  daily rows: {len(daily):,}, gwangyo unique items: {daily["item_id"].nunique()}')
    return daily


def build_closing_rows() -> pd.DataFrame:
    """새 데이터에서 마감할인 row → date/item_id/qty/closing_revenue."""
    print('[build] closing discount rows...')
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str) == GWANGYO_CD]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales = sales[sales['CD_USERDEF1'].isin(CLOSING_CODES)]
    sales['date'] = pd.to_datetime(sales['DT_SALE'].astype(str))
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['AM_PAYMENT'] = pd.to_numeric(sales['AM_PAYMENT'], errors='coerce').fillna(0)
    sales['AM_DC'] = pd.to_numeric(sales['AM_DC'], errors='coerce').fillna(0)

    out = sales.groupby(['date', 'CD_ITEM']).agg(
        qty=('QT_SALE', 'sum'),
        closing_revenue=('AM_PAYMENT', 'sum'),
        discount_amt=('AM_DC', 'sum'),
    ).reset_index().rename(columns={'CD_ITEM': 'item_id'})
    out['item_id'] = out['item_id'].astype(str)
    print(f'  closing rows: {len(out):,}, 총 마감수량: {out["qty"].sum():,.0f}')
    return out


def add_target_date_features(df: pd.DataFrame, h: int, holiday_dates: set) -> pd.DataFrame:
    d = df.copy()
    target_date = d['date'] + pd.Timedelta(days=h)
    target_dow = target_date.dt.dayofweek
    target_month = target_date.dt.month
    target_dom = target_date.dt.day
    target_dim = target_date.dt.days_in_month
    d['tgt_dow_sin'] = np.sin(2 * np.pi * target_dow / 7)
    d['tgt_dow_cos'] = np.cos(2 * np.pi * target_dow / 7)
    d['tgt_month_sin'] = np.sin(2 * np.pi * target_month / 12)
    d['tgt_month_cos'] = np.cos(2 * np.pi * target_month / 12)
    d['tgt_dom_sin'] = np.sin(2 * np.pi * (target_dom - 1) / target_dim)
    d['tgt_dom_cos'] = np.cos(2 * np.pi * (target_dom - 1) / target_dim)
    d['tgt_is_weekend'] = (target_dow >= 5).astype(int)
    d['tgt_is_holiday'] = target_date.isin(holiday_dates).astype(int)
    next_target = target_date + pd.Timedelta(days=1)
    next_dow = (target_dow + 1) % 7
    is_off_target = (d['tgt_is_holiday'] == 1) | (target_dow >= 5)
    is_off_next = next_target.isin(holiday_dates) | (next_dow >= 5)
    d['tgt_is_before_holiday'] = (~is_off_target & is_off_next).astype(int)

    tds = pd.Series(target_date.values)
    for feat, (m, day) in EVENTS.items():
        new_name = feat.replace('days_to_', 'tgt_days_to_')
        d[new_name] = _signed_days_to_event(tds, m, day).astype('int16')
        d[new_name.replace('tgt_days_to_', 'tgt_is_within7_')] = (np.abs(d[new_name]) <= 7).astype(int)
    for feat, year_dates in LUNAR_EVENTS.items():
        new_name = feat.replace('days_to_', 'tgt_days_to_')
        d[new_name] = _days_to_lunar_event(tds, year_dates).astype('int16')
        d[new_name.replace('tgt_days_to_', 'tgt_is_within7_')] = (np.abs(d[new_name]) <= 7).astype(int)
    return d


def compute_baseline(df, h, target_col=TARGET_COL):
    shifts = [7*k - h for k in [1,2,3,4] if 7*k - h > 0]
    lag_cols = pd.concat([df[target_col].shift(s) for s in shifts], axis=1)
    return lag_cols.mean(axis=1)


def fit_ensemble(train, h, holiday_dates):
    train_h = train.copy()
    train_h['baseline'] = compute_baseline(train_h, h)
    train_h['future_target'] = train_h[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
    train_h = add_target_date_features(train_h, h, holiday_dates)

    LEAK = ('sold_total_unit','sold_total_revenue','sold_normal_unit','sold_normal_revenue',
            'sold_closing','sold_closing_revenue','adjusted_demand_unit','adjusted_demand_revenue',
            'n_stockout_items','n_early_stockout','n_items_active',
            'baseline','future_target','residual','target_dow','dow')
    feat_cols = [c for c in train_h.columns
                 if c not in ('date', TARGET_COL, *LEAK)
                 and c not in SHORT_TERM_TO_DROP]

    train_clean = train_h.dropna(subset=['baseline','future_target','residual'] + feat_cols)
    X = train_clean[feat_cols]
    y = train_clean['residual']

    common = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  num_leaves=15, random_state=42, verbosity=-1)
    expected = lgb.LGBMRegressor(objective='regression_l1', **common).fit(X, y)
    quantile = lgb.LGBMRegressor(objective='quantile', alpha=PRODUCTION_Q, **common).fit(X, y)

    train_clean = train_clean.copy()
    train_clean['prod_pred'] = train_clean['baseline'] + quantile.predict(X)
    train_clean['shortfall'] = (train_clean['future_target'] - train_clean['prod_pred']).clip(lower=0)
    dow_safety = train_clean.groupby('target_dow')['shortfall'].mean().to_dict()

    return {'expected': expected, 'quantile': quantile, 'feat_cols': feat_cols, 'dow_safety': dow_safety}


def run_backtest(df, n_thursdays=16):
    holiday_dates = _load_holiday_set()
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['dow'] = df_clean['date'].dt.dayofweek

    thursdays = df_clean[df_clean['dow'] == 3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df_clean['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]
    print(f'test Thursdays: {len(test_ths)}')

    results = []
    for D in test_ths:
        train = df_clean[df_clean['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df_clean[df_clean['date'] == test_date]
            if len(test_row) == 0:
                continue
            model = fit_ensemble(train, h, holiday_dates)
            cutoff_row = df_clean[df_clean['date'] == D].copy()
            cutoff_row = add_target_date_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df_clean[df_clean['date'] <= D], h).iloc[-1]
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': target_dow,
                'baseline': baseline_at_D, 'production': production, 'actual': actual,
                'n_features': len(model['feat_cols']),
            })
    return pd.DataFrame(results)


def run_one_experiment(exclude_bulk: bool, label: str, csv_path: str):
    print(f'\n{"="*70}\n=== {label} ===\n{"="*70}')
    daily_raw = build_new_data_daily(exclude_bulk=exclude_bulk)
    closing_rows = build_closing_rows()

    cd = build_category_daily(daily_raw=daily_raw, discount_rows=closing_rows, alpha=ALPHA)
    print(f'\nbuild_category_daily 결과: rows={len(cd.df)}, '
          f'avg adjusted_demand={cd.df[TARGET_COL].mean():.1f}')
    df = build_features(cd, target_col=TARGET_COL)
    print(f'features: {df.shape}')

    res = run_backtest(df, n_thursdays=16)
    print(f'N predictions: {len(res)}')

    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under = (res['production'] < res['actual']).mean()
    over = (res['production'] - res['actual']).mean()
    print(f'  WAPE: {wape*100:.2f}%, 매진율: {under*100:.1f}%, 발주 over: {over:+.1f}, '
          f'평균 실제: {res["actual"].mean():.1f}')
    res.to_csv(csv_path, index=False)
    print(f'  saved: {csv_path}')
    return wape, under, over, res['actual'].mean(), len(res)


def main():
    print('=== Phase G: v4 framework sumRn fix + normal_qty target 비교 ===')

    # 실험 A: v4 framework with bulk 포함 (기존 adjusted_demand)
    wape_a, und_a, over_a, act_a, n_a = run_one_experiment(
        exclude_bulk=False,
        label='실험 A: v4 + bulk 포함 (기존 adjusted_demand)',
        csv_path='reports/v4_new_data_bulk_incl.csv')

    # 실험 B: v4 framework with bulk 제외 (normal target)
    wape_b, und_b, over_b, act_b, n_b = run_one_experiment(
        exclude_bulk=True,
        label='실험 B: v4 + bulk 제외 (normal_qty 기반 adjusted_demand)',
        csv_path='reports/v4_new_data_bulk_excl.csv')

    print(f'\n{"="*70}\n=== 최종 비교 (광교, 16 Thursdays × 7 horizons) ===\n{"="*70}')
    print(f'{"Model":>55s} {"WAPE":>7s} {"매진율":>8s} {"발주over":>10s} {"평균실제":>10s} {"N":>5s}')
    print(f'{"v4 기존 (5/20, sumRn 버그)":>55s} {"29.28%":>7s} {"11.4%":>8s} {"+60.0":>10s} {"~223":>10s} {"44":>5s}')
    print(f'{"광교 단독 normal (5/26, lag/cyclic만)":>55s} {"14.15%":>7s} {"12.0%":>8s} {"+34.4":>10s} {"268.6":>10s} {"108":>5s}')
    print(f'{"A: v4 + bulk 포함 (sumRn fix)":>55s} {wape_a*100:>5.2f}% {und_a*100:>6.1f}% {over_a:>+9.1f} {act_a:>9.1f} {n_a:>5d}')
    print(f'{"B: v4 + bulk 제외 (sumRn fix)":>55s} {wape_b*100:>5.2f}% {und_b*100:>6.1f}% {over_b:>+9.1f} {act_b:>9.1f} {n_b:>5d}')
    print(f'\n  Δ (B - A): WAPE {(wape_b-wape_a)*100:+.2f}pp, 매진율 {(und_b-und_a)*100:+.1f}pp\n')
    sys.exit(0)


def _DISABLED_OLD():
    daily_raw = build_new_data_daily()
    closing_rows = build_closing_rows()
    cd = build_category_daily(daily_raw=daily_raw, discount_rows=closing_rows, alpha=ALPHA)
    print(f'\nbuild_category_daily 결과: rows={len(cd.df)}, alpha={cd.alpha}')
    print(f'  컬럼: {list(cd.df.columns)}')
    print(f'  평균 adjusted_demand_unit: {cd.df[TARGET_COL].mean():.1f}')

    print('\n[build features...]')
    df = build_features(cd, target_col=TARGET_COL)
    print(f'features: {df.shape}, cols 일부: {list(df.columns[:20])}')

    res = run_backtest(df, n_thursdays=16)
    print(f'\nN predictions: {len(res)}')

    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under = (res['production'] < res['actual']).mean()
    over = (res['production'] - res['actual']).mean()
    print(f'\n=== 결과 ===')
    print(f'  WAPE        : {wape*100:.2f}%')
    print(f'  매진율      : {under*100:.1f}%')
    print(f'  발주 over   : {over:+.1f}')
    print(f'  평균 발주   : {res["production"].mean():.1f}')
    print(f'  평균 실제   : {res["actual"].mean():.1f}')
    print(f'  n_features  : {res["n_features"].iloc[0]}')

    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    print(f'\n=== Horizon별 ===')
    for h in HORIZONS:
        sub = res[res['h'] == h]
        if len(sub) == 0: continue
        w = (sub['actual'] - sub['production']).abs().sum() / sub['actual'].sum()
        u = (sub['production'] < sub['actual']).mean()
        dow_name = dn[(3+h) % 7]
        print(f'  D+{h} ({dow_name}): n={len(sub):>2}, WAPE {w*100:>5.2f}%, 매진율 {u*100:>5.1f}%, '
              f'pred {sub["production"].mean():>6.1f}, actual {sub["actual"].mean():>6.1f}')

    print(f'\n=== 비교 ===')
    print(f'{"Model (광교, 16 Thursdays × 7 horizons)":>50s} {"WAPE":>7s} {"매진율":>8s} {"발주over":>9s}')
    print(f'{"v4 기존 (5/20 데이터, target_date features)":>50s} {"29.28%":>7s} {"11.4%":>8s} {"+60.0":>9s}')
    print(f'{"광교 단독 (5/26 데이터, qty 단순합, no α)":>50s} {"13.95%":>7s} {"12.0%":>8s} {"+32.1":>9s}')
    print(f'{"v4 fair (5/26 데이터, α=0.5+filter_seasonal)":>50s} {wape*100:>5.2f}% {under*100:>6.1f}% {over:>+7.1f}')

    res.to_csv('reports/v4_new_data_fair.csv', index=False)
    print('\nsaved: reports/v4_new_data_fair.csv')


if __name__ == '__main__':
    main()

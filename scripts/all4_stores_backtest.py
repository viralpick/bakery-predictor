"""4매장 종합 backtest — Naive 단순 / Naive+safety / v4 simple + bulk 제외.

매장별 fair 비교. 4 metric (WAPE / 매진율 / 폐기합 / 부족합) 종합.

Spec:
- 매장별 daily (예약 bulk 제외, closing 분리, 카테고리 bread/pastry/sandwich)
- target = adjusted_demand_unit = sold_normal + sold_closing × 0.5 (α=0.5)
- features: lag/rolling/ewma + cyclic + holiday + 특일 + target_date (외부 weather/comp 제외)
- 16 Thursdays × 7 horizons
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from bakery.data.bonavi_loader import map_category
from bakery.features.category_aggregate import (
    EVENTS, LUNAR_EVENTS, EVENT_CLIP,
    _signed_days_to_event, _days_to_lunar_event,
)

V2 = Path('data/internal/v2')
ALPHA = 0.7         # α sensitivity 분석 결과 (2026-05-29) — 매장별 best 평균. 4매장 0.5 → 0.7 통일
TARGET_COL = 'adjusted_demand'
HORIZONS = [4, 5, 6, 7, 8, 9, 10]
PRODUCTION_Q = 0.90
CLOSING_CODES = {'0069', '0077'}
TARGET_CATEGORIES = ('bread', 'pastry', 'sandwich')

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}


def _load_holiday_set():
    cal = pd.read_parquet("data/external/calendar_raw.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return set(cal.loc[cal["is_holiday"] == True, "date"])


def build_store_daily(store_cd: str, exclude_bulk: bool = True) -> pd.DataFrame:
    """매장별 일별 adjusted_demand 계산.

    1. sales (bulk 제외 옵션) + category 한정 + 정상매출 + 단품
    2. closing 분리 (마감 할인코드 row)
    3. sold_total + sold_closing → adjusted_demand
    """
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str) == store_cd]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']
    sales['date'] = pd.to_datetime(sales['DT_SALE'].astype(str))
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['CD_ITEM'] = sales['CD_ITEM'].astype(str)
    sales['is_closing'] = sales['CD_USERDEF1'].isin(CLOSING_CODES)
    sales['receipt_id'] = (sales['CD_PARTNER'].astype(str) + '_'
                          + sales['DT_SALE'].astype(str) + '_'
                          + sales['NO_POS'].astype(str) + '_'
                          + sales['SLIP_NO'].astype(str))

    # bulk 제외
    if exclude_bulk:
        flag = pd.read_parquet(V2 / 'sales_with_bulk_flag.parquet')
        flag = flag[(flag['cd'] == store_cd) & (flag['is_bulk'])]
        bulk_receipts = set(flag['receipt_id'])
        sales = sales[~sales['receipt_id'].isin(bulk_receipts)]

    # 카테고리 한정
    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['category'] = items['NM_ITEM'].apply(map_category)
    items_target = items[items['category'].isin(TARGET_CATEGORIES)]
    sales = sales.merge(items_target[['item_id']], left_on='CD_ITEM', right_on='item_id', how='inner')

    # 일별 sold_total + sold_closing
    daily = sales.groupby('date').agg(
        sold_total=('QT_SALE', 'sum'),
        sold_closing=('QT_SALE', lambda x: x[sales.loc[x.index, 'is_closing']].sum()),
    ).reset_index()
    daily['sold_normal'] = daily['sold_total'] - daily['sold_closing']
    daily[TARGET_COL] = daily['sold_normal'] + daily['sold_closing'] * ALPHA
    return daily


def add_features(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    d = df.sort_values('date').copy()
    for lag in [1, 7, 14, 28]:
        d[f'lag{lag}'] = d[TARGET_COL].shift(lag)
    for w in [7, 28]:
        d[f'rmean{w}'] = d[TARGET_COL].shift(1).rolling(w).mean()
    for h in [7, 28]:
        d[f'ewma{h}'] = d[TARGET_COL].shift(1).ewm(halflife=h).mean()

    dow = d['date'].dt.dayofweek
    d['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    d['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    month = d['date'].dt.month
    d['month_sin'] = np.sin(2 * np.pi * month / 12)
    d['month_cos'] = np.cos(2 * np.pi * month / 12)

    d['is_holiday'] = d['date'].isin(holiday_dates).astype(int)
    d['is_weekend'] = (dow >= 5).astype(int)
    return d


def add_target_date_features(df: pd.DataFrame, h: int, holiday_dates: set) -> pd.DataFrame:
    d = df.copy()
    target_date = d['date'] + pd.Timedelta(days=h)
    target_dow = target_date.dt.dayofweek
    target_month = target_date.dt.month
    d['tgt_dow_sin'] = np.sin(2 * np.pi * target_dow / 7)
    d['tgt_dow_cos'] = np.cos(2 * np.pi * target_dow / 7)
    d['tgt_month_sin'] = np.sin(2 * np.pi * target_month / 12)
    d['tgt_month_cos'] = np.cos(2 * np.pi * target_month / 12)
    d['tgt_is_weekend'] = (target_dow >= 5).astype(int)
    d['tgt_is_holiday'] = target_date.isin(holiday_dates).astype(int)
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


def compute_baseline(df, h):
    shifts = [7*k - h for k in [1,2,3,4] if 7*k - h > 0]
    lag_cols = pd.concat([df[TARGET_COL].shift(s) for s in shifts], axis=1)
    return lag_cols.mean(axis=1)


def fit_ensemble(train, h, holiday_dates):
    train_h = train.copy()
    train_h['baseline'] = compute_baseline(train_h, h)
    train_h['future_target'] = train_h[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
    train_h = add_target_date_features(train_h, h, holiday_dates)

    LEAK = ('sold_total','sold_normal','sold_closing','adjusted_demand',
            'baseline','future_target','residual','target_dow')
    feat_cols = [c for c in train_h.columns if c not in ('date', *LEAK)]
    train_clean = train_h.dropna(subset=['baseline','future_target','residual'] + feat_cols)
    X = train_clean[feat_cols]
    y = train_clean['residual']

    common = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  num_leaves=15, random_state=42, verbosity=-1)
    quantile = lgb.LGBMRegressor(objective='quantile', alpha=PRODUCTION_Q, **common).fit(X, y)
    train_clean = train_clean.copy()
    train_clean['prod_pred'] = train_clean['baseline'] + quantile.predict(X)
    train_clean['shortfall'] = (train_clean['future_target'] - train_clean['prod_pred']).clip(lower=0)
    dow_safety = train_clean.groupby('target_dow')['shortfall'].mean().to_dict()
    return {'quantile': quantile, 'feat_cols': feat_cols, 'dow_safety': dow_safety}


def get_test_thursdays(df, n=16):
    df = df.dropna().reset_index(drop=True)
    df['dow'] = df['date'].dt.dayofweek
    thursdays = df[df['dow'] == 3]['date'].tolist()
    earliest = df['date'].min()
    thursdays = [t for t in thursdays if (t - earliest).days >= 365]
    return df, thursdays[-n:]


def run_naive_simple(df):
    df, ths = get_test_thursdays(df)
    results = []
    for D in ths:
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            results.append({'date': test_date, 'baseline': baseline_at_D,
                            'production': baseline_at_D, 'actual': test_row[TARGET_COL].iat[0]})
    return pd.DataFrame(results)


def run_naive_safety(df):
    df, ths = get_test_thursdays(df)
    results = []
    for D in ths:
        train = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            train_h = train.copy()
            train_h['baseline'] = compute_baseline(train_h, h)
            train_h['future_target'] = train_h[TARGET_COL].shift(-h)
            train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
            train_h['shortfall'] = (train_h['future_target'] - train_h['baseline']).clip(lower=0)
            train_h = train_h.dropna(subset=['shortfall'])
            target_dow = test_date.dayofweek
            sub_dow = train_h[train_h['target_dow'] == target_dow]
            safety = sub_dow['shortfall'].quantile(PRODUCTION_Q) if len(sub_dow) else 0
            results.append({'date': test_date, 'baseline': baseline_at_D,
                            'production': baseline_at_D + safety, 'actual': test_row[TARGET_COL].iat[0]})
    return pd.DataFrame(results)


def run_v4(df, holiday_dates):
    df, ths = get_test_thursdays(df)
    results = []
    for D in ths:
        train = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            model = fit_ensemble(train, h, holiday_dates)
            cutoff_row = df[df['date'] == D].copy()
            cutoff_row = add_target_date_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            results.append({'date': test_date, 'baseline': baseline_at_D,
                            'production': production, 'actual': test_row[TARGET_COL].iat[0]})
    return pd.DataFrame(results)


def stats(res):
    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under_rate = (res['production'] < res['actual']).mean()
    waste = (res['production'] - res['actual']).clip(lower=0)
    short = (res['actual'] - res['production']).clip(lower=0)
    return {
        'wape': wape * 100,
        'under': under_rate * 100,
        'waste_sum': waste.sum(),
        'short_sum': short.sum(),
        'waste_per_day': waste.mean(),
        'short_per_day': short.mean(),
        'avg_actual': res['actual'].mean(),
        'n': len(res),
    }


def main():
    print('=== 4매장 종합 backtest (Naive vs Naive+safety vs v4+bulk제외) ===\n')
    holiday_dates = _load_holiday_set()

    all_results = []
    for store_cd, store_name in STORE_MAP.items():
        print(f'\n--- {store_name} ---')
        daily = build_store_daily(store_cd, exclude_bulk=True)
        print(f'  daily rows: {len(daily)}, avg adjusted_demand: {daily[TARGET_COL].mean():.1f}')
        df = add_features(daily, holiday_dates)

        r_n = stats(run_naive_simple(df))
        r_s = stats(run_naive_safety(df))
        r_v = stats(run_v4(df, holiday_dates))
        all_results.append((store_name, r_n, r_s, r_v))
        print(f'  Naive 단순: WAPE {r_n["wape"]:.2f}%, 매진율 {r_n["under"]:.1f}%')
        print(f'  Naive+safety: WAPE {r_s["wape"]:.2f}%, 매진율 {r_s["under"]:.1f}%')
        print(f'  v4 (bulk 제외): WAPE {r_v["wape"]:.2f}%, 매진율 {r_v["under"]:.1f}%')

    # 종합 표
    print(f'\n\n{"="*120}')
    print(f'=== 4매장 종합 비교 (bulk 제외, 16 Thursdays × 7 horizons) ===')
    print(f'{"="*120}')
    print(f'{"매장":>10s} {"Model":>20s} {"WAPE":>7s} {"매진율":>8s} {"폐기/일":>9s} {"부족/일":>9s} {"폐기합":>9s} {"부족합":>9s} {"N":>4s}')
    for store_name, r_n, r_s, r_v in all_results:
        for label, r in [('Naive 단순', r_n), ('Naive+safety', r_s), ('v4 (bulk제외)', r_v)]:
            print(f'  {store_name:>8s} {label:>18s} {r["wape"]:>6.2f}% {r["under"]:>6.1f}% '
                  f'{r["waste_per_day"]:>8.1f} {r["short_per_day"]:>8.1f} '
                  f'{r["waste_sum"]:>8.0f} {r["short_sum"]:>8.0f} {r["n"]:>4d}')
        print()

    # v4 vs Naive+safety per store
    print(f'=== v4 vs Naive+safety per store (model 가치) ===')
    print(f'{"매장":>10s} {"Δ WAPE":>9s} {"Δ 매진율":>9s} {"Δ 폐기합":>10s} {"Δ 부족합":>10s}')
    for store_name, _, r_s, r_v in all_results:
        dw = r_v['wape'] - r_s['wape']
        du = r_v['under'] - r_s['under']
        dwa = r_v['waste_sum'] - r_s['waste_sum']
        dsh = r_v['short_sum'] - r_s['short_sum']
        print(f'  {store_name:>8s} {dw:>+8.2f}pp {du:>+7.1f}pp {dwa:>+9.0f} {dsh:>+9.0f}')


if __name__ == '__main__':
    main()

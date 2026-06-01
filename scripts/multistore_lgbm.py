"""Multi-store LGBM Phase B: 4매장 통합 학습 + 광교 holdout backtest.

광교 단독 v4 (WAPE 29.28%) vs 4매장 통합 학습 비교.
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

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}
STORE_IDS = list(STORE_MAP.keys())

TARGET_CATEGORIES = ('bread', 'pastry', 'sandwich')
TARGET_COL = 'qty'  # 매장별 일별 카테고리 합 (수량)
HORIZONS = [4, 5, 6, 7, 8, 9, 10]
PRODUCTION_Q = 0.90


def _load_holiday_set():
    cal = pd.read_parquet("data/external/calendar_raw.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return set(cal.loc[cal["is_holiday"] == True, "date"])


def build_multistore_daily() -> pd.DataFrame:
    """4매장 × category × date daily aggregate (수량 + 매출)."""
    print('[build] sales + items load...')
    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_IDS)]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']
    sales['DT_SALE'] = pd.to_datetime(sales['DT_SALE'].astype(str), errors='coerce')
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce')

    items = pd.read_parquet(V2 / 'items.parquet')
    items = items[items['FG_ITEM'] == 'SS'].copy()
    items['category'] = items['NM_ITEM'].apply(map_category)
    items = items[items['category'].isin(TARGET_CATEGORIES)]

    sales = sales.merge(items[['CD_ITEM', 'category']], on='CD_ITEM', how='inner')
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)

    # category 합 daily (매장별)
    daily = sales.groupby(['store', 'DT_SALE']).agg(qty=('QT_SALE', 'sum')).reset_index()
    daily = daily.rename(columns={'DT_SALE': 'date'})
    print(f'[build] multi-store daily: {len(daily):,} rows')
    return daily


def add_features(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """매장별 group 내에서 lag/rolling features 생성. cyclic + holiday features 매장 무관."""
    d = df.sort_values(['store', 'date']).copy()
    # lag (1, 7, 14, 28)
    for lag in [1, 7, 14, 28]:
        d[f'lag{lag}'] = d.groupby('store')[TARGET_COL].shift(lag)
    # rolling (7, 28) mean
    for w in [7, 28]:
        d[f'rmean{w}'] = d.groupby('store')[TARGET_COL].shift(1).rolling(w).mean()
    # ewma (7, 28)
    for h in [7, 28]:
        d[f'ewma{h}'] = d.groupby('store')[TARGET_COL].transform(
            lambda x: x.shift(1).ewm(halflife=h).mean()
        )

    # cyclic 캘린더 (D 기준, target_date는 fit_horizon에서 추가)
    dow = d['date'].dt.dayofweek
    d['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    d['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    month = d['date'].dt.month
    d['month_sin'] = np.sin(2 * np.pi * month / 12)
    d['month_cos'] = np.cos(2 * np.pi * month / 12)

    # store_id categorical
    d['store_id'] = pd.Categorical(d['store']).codes
    return d


def add_target_date_features(df: pd.DataFrame, h: int, holiday_dates: set) -> pd.DataFrame:
    """target_date (D+h) 기준 cyclic + holiday + 특수일."""
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
    next_target = target_date + pd.Timedelta(days=1)
    next_dow = (target_dow + 1) % 7
    is_off_target = (d['tgt_is_holiday'] == 1) | (target_dow >= 5)
    is_off_next = next_target.isin(holiday_dates) | (next_dow >= 5)
    d['tgt_is_before_holiday'] = (~is_off_target & is_off_next).astype(int)

    target_dates_series = pd.Series(target_date.values)
    for feat, (m, day) in EVENTS.items():
        new_name = feat.replace('days_to_', 'tgt_days_to_')
        d[new_name] = _signed_days_to_event(target_dates_series, m, day).astype('int16')
        d[new_name.replace('tgt_days_to_', 'tgt_is_within7_')] = (np.abs(d[new_name]) <= 7).astype(int)
    for feat, year_dates in LUNAR_EVENTS.items():
        new_name = feat.replace('days_to_', 'tgt_days_to_')
        d[new_name] = _days_to_lunar_event(target_dates_series, year_dates).astype('int16')
        d[new_name.replace('tgt_days_to_', 'tgt_is_within7_')] = (np.abs(d[new_name]) <= 7).astype(int)
    return d


def compute_baseline_per_store(df: pd.DataFrame, h: int) -> pd.Series:
    """store별 4주 동일요일 평균 baseline."""
    shifts = [7 * k - h for k in [1, 2, 3, 4] if 7 * k - h > 0]
    parts = [df.groupby('store')[TARGET_COL].shift(s) for s in shifts]
    return pd.concat(parts, axis=1).mean(axis=1)


def fit_multistore_ensemble(train: pd.DataFrame, h: int, holiday_dates: set):
    """train: 4매장 통합. residual = future - baseline. quantile + L1."""
    train_h = train.copy()
    train_h['baseline'] = compute_baseline_per_store(train_h, h)
    train_h['future_target'] = train_h.groupby('store')[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek

    train_h = add_target_date_features(train_h, h, holiday_dates)

    LEAK = ('baseline', 'future_target', 'residual', 'target_dow',
            'store', 'date', TARGET_COL)
    feat_cols = [c for c in train_h.columns if c not in LEAK]

    train_clean = train_h.dropna(subset=['baseline', 'future_target', 'residual'] + feat_cols)
    X = train_clean[feat_cols]
    y = train_clean['residual']

    common = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  num_leaves=15, random_state=42, verbosity=-1)
    expected = lgb.LGBMRegressor(objective='regression_l1', **common).fit(X, y, categorical_feature=['store_id'])
    quantile = lgb.LGBMRegressor(objective='quantile', alpha=PRODUCTION_Q, **common).fit(X, y, categorical_feature=['store_id'])

    # store × target_dow safety
    train_clean = train_clean.copy()
    train_clean['prod_pred'] = train_clean['baseline'] + quantile.predict(X)
    train_clean['shortfall'] = (train_clean['future_target'] - train_clean['prod_pred']).clip(lower=0)
    dow_safety = train_clean.groupby(['store_id', 'target_dow'])['shortfall'].mean().to_dict()

    return {'expected': expected, 'quantile': quantile, 'feat_cols': feat_cols, 'dow_safety': dow_safety}


def run_backtest(df: pd.DataFrame, target_store: str, n_thursdays: int = 16):
    """target_store에서만 평가, 학습은 4매장 통합."""
    holiday_dates = _load_holiday_set()
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df['dow'] = df['date'].dt.dayofweek
    target_df = df[df['store'] == target_store]

    thursdays = target_df[target_df['dow'] == 3]['date'].sort_values().unique().tolist()
    earliest = df['date'].min()
    thursdays = [t for t in thursdays if (t - earliest).days >= 365]
    test_ths = thursdays[-n_thursdays:]
    print(f'test Thursdays for {target_store}: {len(test_ths)}')

    results = []
    for D in test_ths:
        train = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[(df['store'] == target_store) & (df['date'] == test_date)]
            if len(test_row) == 0:
                continue

            model = fit_multistore_ensemble(train, h, holiday_dates)
            cutoff = df[(df['store'] == target_store) & (df['date'] == D)].copy()
            cutoff = add_target_date_features(cutoff, h, holiday_dates)
            baseline_at_D = compute_baseline_per_store(
                df[(df['store'] == target_store) & (df['date'] <= D)], h
            ).iloc[-1]
            resid_q = model['quantile'].predict(cutoff[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            store_id_code = cutoff['store_id'].iloc[0]
            safety = model['dow_safety'].get((store_id_code, target_dow), 0)
            production = baseline_at_D + resid_q + safety
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': target_dow,
                'baseline': baseline_at_D, 'production': production, 'actual': actual,
                'n_features': len(model['feat_cols']),
            })
    return pd.DataFrame(results)


def main():
    print('=== Phase B: Multi-store LGBM ===\n')

    daily = build_multistore_daily()
    df = add_features(daily, holiday_dates=_load_holiday_set())
    print(f'features prepared: {df.shape}')
    print(f'매장 분포: {df["store"].value_counts().to_dict()}')

    # 광교 holdout backtest
    res = run_backtest(df, target_store='광교', n_thursdays=16)
    print(f'\nN predictions: {len(res)}')

    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under = (res['production'] < res['actual']).mean()
    over = (res['production'] - res['actual']).mean()

    print(f'\n=== 광교 holdout 결과 (multi-store 학습) ===')
    print(f'  WAPE        : {wape * 100:.2f}%')
    print(f'  매진율      : {under * 100:.1f}%')
    print(f'  발주 over   : {over:+.1f}')
    print(f'  평균 발주   : {res["production"].mean():.1f}')
    print(f'  평균 실제   : {res["actual"].mean():.1f}')

    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    print(f'\n=== Horizon별 (광교) ===')
    for h in HORIZONS:
        sub = res[res['h'] == h]
        if len(sub) == 0: continue
        w = (sub['actual'] - sub['production']).abs().sum() / sub['actual'].sum()
        u = (sub['production'] < sub['actual']).mean()
        dow_name = dn[(3 + h) % 7]
        print(f'  D+{h} ({dow_name}): n={len(sub):>2}, WAPE {w*100:>5.2f}%, '
              f'매진율 {u*100:>5.1f}%, pred {sub["production"].mean():>6.1f}, '
              f'actual {sub["actual"].mean():>6.1f}')

    print(f'\n=== 비교 ===')
    print(f'{"Model":>40s} {"WAPE":>7s} {"매진율":>8s} {"발주 over":>10s}')
    print(f'{"v4 광교 단독 (target_date features)":>40s} {"29.28%":>7s} {"11.4%":>8s} {"+60.0":>10s}')
    print(f'{"Multi-store (4매장 통합)":>40s} {wape*100:>5.2f}% {under*100:>6.1f}% {over:>+8.1f}')

    res.to_csv('reports/multistore_v5_gwangyo.csv', index=False)
    print('\nsaved: reports/multistore_v5_gwangyo.csv')


if __name__ == '__main__':
    main()

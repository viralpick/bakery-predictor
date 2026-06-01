"""자동 Feature Selection 파이프라인 + Composite Score 평가.

방식:
1. 각 매장 Permutation Importance ranking (사전 측정 결과 사용)
2. Top N union 후보 (N = 3, 5, 8, 12, 15, 20, 30=all)
3. 각 N에 대해 4매장 backtest
4. Composite score = w_risk × normalized_매진율 + w_impact × normalized_cost
   - cost = (폐기합 + 부족합) × 평균단가 (매장별)
5. composite score min N 선택

전제:
- 매진율 (risk) + 폐기/부족 비용 (impact) 균등 50:50
- 단가 매장별 5,228/4,968/5,072/5,374원 (실측)
- 4매장 합 cost 기준 best 선택
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd

from scripts.all4_unified_features import (
    add_unified_features, add_unified_target_features, fit_ensemble,
    run_backtest, compute_baseline,
    UNIFIED_BASE_FEATURES, UNIFIED_TARGET_FEATURES,
)
from scripts.all4_stores_backtest import (
    build_store_daily, _load_holiday_set, STORE_MAP, TARGET_COL, stats,
)

UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}


def load_permutation_results():
    df = pd.read_csv('reports/permutation_importance_all4.csv')
    return df


def get_top_n_union(perm_df, n: int):
    """4매장 Top N features union (delta_wape_pp 기준 매장별 정렬)."""
    union = set()
    for store in ['광교', '광화문', '메세나폴리스', '삼성타운']:
        sub = perm_df[perm_df['store'] == store].sort_values('delta_wape_pp', ascending=False)
        top_n = sub.head(n)['feature'].tolist()
        union.update(top_n)
    return sorted(union)


def split_features(features):
    """features를 base / target_date 그룹으로 분리."""
    base = [f for f in features if not f.startswith('tgt_')]
    target = [f for f in features if f.startswith('tgt_')]
    return base, target


def add_all_features_for_set(df, holiday_dates, base_features):
    """base_features 만 생성 (다른 features는 NaN 또는 0)."""
    d = df.sort_values('date').copy()
    # 모든 가능한 features 생성 (UNIFIED_BASE_FEATURES + more 확장 필요)
    for lag in [1, 7, 14, 28]:
        d[f'lag{lag}'] = d[TARGET_COL].shift(lag)
    for w in [7, 28]:
        d[f'rmean{w}'] = d[TARGET_COL].shift(1).rolling(w).mean()
        d[f'rstd{w}'] = d[TARGET_COL].shift(1).rolling(w).std()
        d[f'ewma{w}'] = d[TARGET_COL].shift(1).ewm(halflife=w).mean()
    dow = d['date'].dt.dayofweek
    month = d['date'].dt.month
    dom = d['date'].dt.day
    d['dow_sin'] = np.sin(2*np.pi*dow/7); d['dow_cos'] = np.cos(2*np.pi*dow/7)
    d['month_sin'] = np.sin(2*np.pi*month/12); d['month_cos'] = np.cos(2*np.pi*month/12)
    d['dom_sin'] = np.sin(2*np.pi*(dom-1)/d['date'].dt.days_in_month)
    d['dom_cos'] = np.cos(2*np.pi*(dom-1)/d['date'].dt.days_in_month)
    d['dom'] = dom
    d['month'] = month
    d['is_holiday'] = d['date'].isin(holiday_dates).astype(int)
    d['is_weekend'] = (dow >= 5).astype(int)
    return d


def fit_with_custom_features(train, h, holiday_dates, base_feats, target_feats):
    """custom feature set으로 ensemble fit."""
    import lightgbm as lgb
    from scripts.all4_stores_backtest import add_target_date_features, PRODUCTION_Q
    train_h = train.copy()
    train_h['baseline'] = compute_baseline(train_h, h)
    train_h['future_target'] = train_h[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
    train_h = add_target_date_features(train_h, h, holiday_dates)

    feat_cols = [f for f in (base_feats + target_feats) if f in train_h.columns]
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


def run_backtest_custom(df, holiday_dates, base_feats, target_feats, n_thursdays=16):
    from scripts.all4_stores_backtest import add_target_date_features, HORIZONS
    df = df.dropna(subset=[f for f in base_feats if f in df.columns] + [TARGET_COL]).reset_index(drop=True)
    df['dow'] = df['date'].dt.dayofweek
    thursdays = df[df['dow']==3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
    for D in test_ths:
        train = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            model = fit_with_custom_features(train, h, holiday_dates, base_feats, target_feats)
            cutoff_row = df[df['date'] == D].copy()
            cutoff_row = add_target_date_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({'D': D, 'h': h, 'target_dow': target_dow,
                            'baseline': baseline_at_D, 'production': production, 'actual': actual})
    return pd.DataFrame(results)


def composite_score(stats_list, store_names, prices):
    """4매장 합산 composite score.
    score = norm(매장 평균 매진율) + norm(4매장 cost 합).
    각 N 후보에서 비교 위해 합산만. normalize는 후보 set 간에 진행.
    """
    avg_under = np.mean([s['under'] for s in stats_list])
    total_cost = sum((s['waste_sum'] + s['short_sum']) * prices[name]
                     for s, name in zip(stats_list, store_names))
    return avg_under, total_cost


def main():
    print('=== 자동 Feature Selection 파이프라인 ===\n')
    perm_df = load_permutation_results()
    holiday_dates = _load_holiday_set()

    N_CANDIDATES = [3, 5, 8, 12, 15, 20, 30]
    candidate_results = []

    for n in N_CANDIDATES:
        features = get_top_n_union(perm_df, n)
        base, target = split_features(features)
        print(f'\n--- N={n}, union 후보 {len(features)}개 (base {len(base)}, target {len(target)}) ---')
        print(f'  features: {features}')

        store_stats = []
        for cd, name in STORE_MAP.items():
            daily = build_store_daily(cd, exclude_bulk=True)
            df = add_all_features_for_set(daily, holiday_dates, base)
            res = run_backtest_custom(df, holiday_dates, base, target)
            r = stats(res)
            store_stats.append(r)
            print(f'  {name}: WAPE {r["wape"]:.2f}%, 매진 {r["under"]:.1f}%, '
                  f'폐기 {r["waste_sum"]:.0f}, 부족 {r["short_sum"]:.0f}')

        avg_und, tot_cost = composite_score(store_stats,
                                              list(STORE_MAP.values()), UNIT_PRICE)
        candidate_results.append({
            'N': n,
            'n_features': len(features),
            'features': features,
            'store_stats': store_stats,
            'avg_under': avg_und,
            'total_cost': tot_cost,
        })

    # Normalize + composite
    unders = np.array([c['avg_under'] for c in candidate_results])
    costs = np.array([c['total_cost'] for c in candidate_results])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, c in enumerate(candidate_results):
        c['composite'] = u_norm[i] + c_norm[i]   # 50:50

    print(f'\n\n{"="*100}')
    print(f'=== Feature Selection 결과 (Composite Score = norm_매진율 + norm_cost) ===')
    print(f'{"="*100}\n')
    print(f'{"N":>3s} {"feats":>6s} {"평균 매진율":>11s} {"4매장 cost(원)":>16s} {"norm_und":>9s} {"norm_cost":>10s} {"composite":>10s}')
    for c in candidate_results:
        mark = ''
        if c['composite'] == min(cc['composite'] for cc in candidate_results):
            mark = ' ★ best'
        print(f'  {c["N"]:>3d} {c["n_features"]:>5d} {c["avg_under"]:>9.2f}% {c["total_cost"]:>15,.0f} '
              f'{(c["avg_under"]-unders.min())/(unders.max()-unders.min()+1e-9):>8.3f} '
              f'{(c["total_cost"]-costs.min())/(costs.max()-costs.min()+1e-9):>9.3f} '
              f'{c["composite"]:>9.3f}{mark}')

    # best N 매장별 결과
    best = min(candidate_results, key=lambda c: c['composite'])
    print(f'\n=== Best N={best["N"]} 매장별 상세 ===')
    print(f'features ({len(best["features"])}): {best["features"]}\n')
    for s, name in zip(best['store_stats'], STORE_MAP.values()):
        print(f'  {name:>10s}: WAPE {s["wape"]:>5.2f}%, 매진 {s["under"]:>5.1f}%, '
              f'폐기 {s["waste_sum"]:>5.0f}, 부족 {s["short_sum"]:>4.0f}, '
              f'cost {(s["waste_sum"]+s["short_sum"])*UNIT_PRICE[name]:>12,.0f}원')

    # CSV
    pd.DataFrame([{
        'N': c['N'],
        'n_features': c['n_features'],
        'avg_under': c['avg_under'],
        'total_cost': c['total_cost'],
        'composite': c['composite'],
        'features': ','.join(c['features']),
    } for c in candidate_results]).to_csv('reports/auto_feature_selection.csv', index=False)
    print('\nsaved: reports/auto_feature_selection.csv')


if __name__ == '__main__':
    main()

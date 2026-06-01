"""매장별 window + RFECV 통합 pipeline.

각 매장:
1. Best window (학습 구간 sensitivity 결과 사용)
   - 광교: 2Y / 광화문: 4Y / 메세나: 6M / 삼성타운: 1Y
2. 그 window 안에서 RFECV (매장별 Permutation ranking 사용)
3. Composite score (마진 0.5, 매진 ×1.2, weight 0.6:0.4) min N 자동 선택
4. 최종 매장별 (window, N, features) 조합
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from scripts.all4_stores_backtest import (
    build_store_daily, add_features, add_target_date_features,
    compute_baseline, _load_holiday_set, STORE_MAP, TARGET_COL,
    HORIZONS, PRODUCTION_Q, stats,
)
from scripts.auto_feature_selection import add_all_features_for_set, split_features

BEST_WINDOWS = {
    '광교': 730,        # 2Y
    '광화문': 1460,     # 4Y
    '메세나폴리스': 180,  # 6M
    '삼성타운': 365,    # 1Y
}

UNIT_PRICE = {'광교': 5228, '광화문': 4968, '메세나폴리스': 5072, '삼성타운': 5374}
WASTE_COST_RATIO = 0.5
SHORT_COST_RATIO = 0.6
N_STEPS = [33, 28, 24, 20, 17, 14, 12, 10, 8, 6, 4]


def load_store_ranking(store_name):
    df = pd.read_csv('reports/permutation_importance_all4.csv')
    sub = df[df['store'] == store_name].sort_values('delta_wape_pp', ascending=False)
    return sub['feature'].tolist(), sub.set_index('feature')['delta_wape_pp'].to_dict()


def fit_ensemble_window(train_full, D, h, holiday_dates, window_days,
                         base_feats, target_feats):
    """train_window + custom features."""
    if window_days is not None:
        cutoff = D - pd.Timedelta(days=window_days)
        train = train_full[train_full['date'] > cutoff].copy()
    else:
        train = train_full.copy()
    if len(train) < 60: return None

    train_h = train.copy()
    train_h['baseline'] = compute_baseline(train_h, h)
    train_h['future_target'] = train_h[TARGET_COL].shift(-h)
    train_h['residual'] = train_h['future_target'] - train_h['baseline']
    train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
    train_h = add_target_date_features(train_h, h, holiday_dates)

    feat_cols = [f for f in (base_feats + target_feats) if f in train_h.columns]
    train_clean = train_h.dropna(subset=['baseline','future_target','residual'] + feat_cols)
    if len(train_clean) < 30: return None
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


def run_backtest_window_features(df, holiday_dates, window_days, base_feats, target_feats,
                                   n_thursdays=16):
    df = df.dropna(subset=[f for f in base_feats if f in df.columns] + [TARGET_COL]).reset_index(drop=True)
    df['dow'] = df['date'].dt.dayofweek
    thursdays = df[df['dow']==3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
    for D in test_ths:
        train_full = df[df['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df[df['date'] == test_date]
            if len(test_row) == 0: continue
            model = fit_ensemble_window(train_full, D, h, holiday_dates, window_days,
                                          base_feats, target_feats)
            if model is None: continue
            cutoff_row = df[df['date'] == D].copy()
            cutoff_row = add_target_date_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df[df['date'] <= D], h).iloc[-1]
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({'D': D, 'h': h, 'baseline': baseline_at_D,
                            'production': production, 'actual': actual})
    return pd.DataFrame(results)


def store_cost(under, waste_sum, short_sum, unit_price):
    return under, waste_sum * unit_price * WASTE_COST_RATIO + short_sum * unit_price * SHORT_COST_RATIO


def run_pipeline_for_store(store_cd, store_name, holiday_dates):
    window_days = BEST_WINDOWS[store_name]
    win_label = next((k for k,v in [('2Y',730),('4Y',1460),('6M',180),('1Y',365)] if v==window_days), str(window_days))
    print(f'\n{"="*80}')
    print(f'=== {store_name} | Best window = {win_label} ({window_days}일) ===')
    print(f'{"="*80}')

    ranking, perm_dict = load_store_ranking(store_name)
    daily = build_store_daily(store_cd, exclude_bulk=True)

    candidates = []
    for n in N_STEPS:
        features = ranking[:n]
        base, target = split_features(features)
        df = add_all_features_for_set(daily, holiday_dates, base)
        res = run_backtest_window_features(df, holiday_dates, window_days, base, target)
        if len(res) == 0: continue
        r = stats(res)
        under, cost = store_cost(r['under'], r['waste_sum'], r['short_sum'], UNIT_PRICE[store_name])
        candidates.append({'N': n, 'features': features.copy(), 'wape': r['wape'],
                           'under': under, 'waste_sum': r['waste_sum'], 'short_sum': r['short_sum'],
                           'cost': cost})
        print(f'  N={n:>3}: WAPE {r["wape"]:>5.2f}%, 매진 {under:>5.1f}%, '
              f'폐기 {r["waste_sum"]:>5.0f}, 부족 {r["short_sum"]:>4.0f}, cost {cost:>10,.0f}원')

    unders = np.array([c['under'] for c in candidates])
    costs = np.array([c['cost'] for c in candidates])
    u_norm = (unders - unders.min()) / (unders.max() - unders.min() + 1e-9)
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for i, c in enumerate(candidates):
        c['composite'] = 0.6 * u_norm[i] + 0.4 * c_norm[i]

    best = min(candidates, key=lambda c: c['composite'])
    print(f'\n--- Best (window={win_label}, N={best["N"]}, composite={best["composite"]:.3f}) ---')
    print(f'WAPE {best["wape"]:.2f}%, 매진 {best["under"]:.1f}%, '
          f'폐기 {best["waste_sum"]:.0f}, 부족 {best["short_sum"]:.0f}, cost {best["cost"]:,.0f}원')
    return store_name, win_label, best, candidates


def main():
    print('=== 매장별 Window + RFECV 통합 Pipeline ===')
    holiday_dates = _load_holiday_set()
    all_best = []

    for cd, name in STORE_MAP.items():
        _, win_label, best, _ = run_pipeline_for_store(cd, name, holiday_dates)
        all_best.append((name, win_label, best))

    print(f'\n\n{"="*110}')
    print(f'=== 최종 매장별 Window + RFECV 결과 ===')
    print(f'{"="*110}\n')
    print(f'{"매장":>10s} {"window":>8s} {"N":>4s} {"WAPE":>7s} {"매진율":>8s} {"폐기":>7s} {"부족":>6s} {"cost(원)":>13s}')
    total_cost = 0
    for name, win, b in all_best:
        total_cost += b['cost']
        print(f'  {name:>8s} {win:>7s} {b["N"]:>3d} {b["wape"]:>5.2f}% {b["under"]:>6.1f}% '
              f'{b["waste_sum"]:>6.0f} {b["short_sum"]:>5.0f} {b["cost"]:>12,.0f}')
    print(f'\n  4매장 합 cost (108일): {total_cost:,.0f}원')
    print(f'  연 환산 (×3.38): {total_cost * 3.38:,.0f}원/년')
    print(f'  5년 환산 (×16.9): {total_cost * 16.9:,.0f}원/5년')

    # 이전 결과 비교
    PREV_RFECV = {'광교': 10224424, '광화문': 9407018, '메세나폴리스': 7279981, '삼성타운': 12974789}
    print(f'\n=== vs 매장별 RFECV (5Y window) 비교 ===')
    print(f'{"매장":>10s} {"5Y RFECV cost":>15s} {"window+RFECV":>14s} {"Δ":>10s}')
    for name, win, b in all_best:
        prev = PREV_RFECV[name]
        delta = b['cost'] - prev
        print(f'  {name:>8s} {prev:>13,.0f} {b["cost"]:>13,.0f} {delta:>+9,.0f}')
    print(f'\n  4매장 합 5Y RFECV: 39,886,212원')
    print(f'  4매장 합 window+RFECV: {total_cost:,.0f}원')
    print(f'  Δ: {total_cost - 39886212:+,.0f}원 ({(total_cost - 39886212) / 39886212 * 100:+.1f}%)')

    # CSV
    pd.DataFrame([{
        'store': name,
        'window': win,
        'N': b['N'],
        'wape': b['wape'],
        'under': b['under'],
        'waste_sum': b['waste_sum'],
        'short_sum': b['short_sum'],
        'cost': b['cost'],
        'features': ','.join(b['features']),
    } for name, win, b in all_best]).to_csv('reports/window_rfecv_pipeline.csv', index=False)
    print('\nsaved: reports/window_rfecv_pipeline.csv')


if __name__ == '__main__':
    main()

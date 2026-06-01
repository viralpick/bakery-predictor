"""광교 v4 Permutation Importance 재측정 (sumRn fix + α=0.7 후).

이전 측정 sumRn 버그 영향 (N=44 subset). 새 측정 (N=108 full).

방식:
1. backtest 한 번 (baseline WAPE)
2. 각 feature 별:
   - 모든 fold의 test features (cutoff_row) 중 그 column을 shuffle
   - 새 prediction → 새 WAPE
   - delta WAPE = 새 WAPE - baseline WAPE
3. delta WAPE 큰 feature = importance 큼
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd

from bakery.features.category_aggregate import build_category_daily, build_features
from scripts.v4_new_data_backtest import (
    build_new_data_daily, build_closing_rows,
    TARGET_COL, HORIZONS, ALPHA, _load_holiday_set,
    add_target_date_features, compute_baseline, fit_ensemble,
)


N_THURSDAYS = 16
RNG_SEED = 42


def run_backtest_collect(df, n_thursdays=N_THURSDAYS):
    """backtest 진행 + 각 D×h fold의 model, cutoff_features, actual 저장."""
    holiday_dates = _load_holiday_set()
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['dow'] = df_clean['date'].dt.dayofweek
    thursdays = df_clean[df_clean['dow'] == 3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df_clean['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]
    print(f'test Thursdays: {len(test_ths)}')

    folds = []
    for D in test_ths:
        train = df_clean[df_clean['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df_clean[df_clean['date'] == test_date]
            if len(test_row) == 0: continue
            model = fit_ensemble(train, h, holiday_dates)
            cutoff_row = df_clean[df_clean['date'] == D].copy()
            cutoff_row = add_target_date_features(cutoff_row, h, holiday_dates)
            baseline_at_D = compute_baseline(df_clean[df_clean['date'] <= D], h).iloc[-1]
            target_dow = test_date.dayofweek
            safety = model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            folds.append({
                'D': D, 'h': h,
                'model_quantile': model['quantile'],
                'feat_cols': model['feat_cols'],
                'X': cutoff_row[model['feat_cols']].values.flatten(),
                'baseline_at_D': baseline_at_D,
                'safety': safety,
                'actual': actual,
            })
    return folds, test_ths


def compute_wape(productions, actuals):
    return (np.abs(np.array(productions) - np.array(actuals)).sum() /
            np.array(actuals).sum() * 100)


def baseline_wape(folds):
    productions = [(f['baseline_at_D'] + f['model_quantile'].predict(f['X'].reshape(1, -1))[0]
                    + f['safety']) for f in folds]
    actuals = [f['actual'] for f in folds]
    return compute_wape(productions, actuals), productions, actuals


def permute_feature(folds, feature_name, rng):
    """그 feature column shuffle 후 새 WAPE 측정."""
    # 모든 fold의 feat_cols index가 같다고 가정. 단, h마다 다를 수 있음. 안전 처리.
    # feature를 사용하는 fold만 shuffle. 같은 feature index는 다를 수 있어 dict 사용.
    fold_idx = []
    fold_vals = []
    for i, f in enumerate(folds):
        if feature_name in f['feat_cols']:
            idx = f['feat_cols'].index(feature_name)
            fold_idx.append((i, idx))
            fold_vals.append(f['X'][idx])

    if not fold_vals: return None

    shuffled_vals = np.array(fold_vals)
    rng.shuffle(shuffled_vals)

    productions = []
    actuals = []
    for n, f in enumerate(folds):
        X = f['X'].copy()
        if feature_name in f['feat_cols']:
            idx = f['feat_cols'].index(feature_name)
            # shuffled value 찾기
            pos = [k for k, (fi, _) in enumerate(fold_idx) if fi == n][0]
            X[idx] = shuffled_vals[pos]
        pred = f['baseline_at_D'] + f['model_quantile'].predict(X.reshape(1, -1))[0] + f['safety']
        productions.append(pred)
        actuals.append(f['actual'])
    return compute_wape(productions, actuals)


def main():
    print('=== 광교 v4 Permutation Importance 재측정 (sumRn fix + α=0.7) ===\n')

    daily_raw = build_new_data_daily(exclude_bulk=True)
    closing_rows = build_closing_rows()
    cd = build_category_daily(daily_raw=daily_raw, discount_rows=closing_rows, alpha=ALPHA)
    df = build_features(cd, target_col=TARGET_COL)
    print(f'features prepared: {df.shape}')

    folds, test_ths = run_backtest_collect(df)
    print(f'\nfolds: {len(folds)}')

    base_wape, base_prods, actuals = baseline_wape(folds)
    print(f'\nbaseline WAPE: {base_wape:.3f}%')

    # 모든 features 추출 (첫 fold의 feat_cols)
    all_features = list(folds[0]['feat_cols'])
    print(f'\nfeatures 수: {len(all_features)}')

    rng = np.random.default_rng(RNG_SEED)
    importance = []
    print('\n각 feature shuffle 후 WAPE 변화 측정...')
    for fname in all_features:
        new_wape = permute_feature(folds, fname, rng)
        if new_wape is None: continue
        delta = new_wape - base_wape
        importance.append((fname, delta, new_wape))

    importance.sort(key=lambda x: -x[1])

    print(f'\n=== Permutation Importance Ranking (Δ WAPE pp) ===')
    print(f'baseline WAPE = {base_wape:.3f}%')
    print(f'{"Rank":>4s} {"Feature":>40s} {"Δ WAPE":>10s} {"NewWAPE":>10s}')
    for i, (fname, delta, new_wape) in enumerate(importance, 1):
        print(f'  {i:>3d}. {fname:>40s} {delta:>+8.3f}pp {new_wape:>8.3f}%')

    pd.DataFrame(importance, columns=['feature', 'delta_wape_pp', 'new_wape']).to_csv(
        'reports/permutation_importance_v2.csv', index=False)
    print('\nsaved: reports/permutation_importance_v2.csv')


if __name__ == '__main__':
    main()

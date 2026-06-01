"""4매장 Permutation Importance — simple spec (all4_stores_backtest.py 모델 기반).

광교 단독 full spec (외부 features 포함) Permutation Importance는 별도 (v2 결과 참조).
이번에는 4매장 fair 비교용 simple spec (lag/cyclic/holiday/target_date).
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd

from scripts.all4_stores_backtest import (
    build_store_daily, add_features, add_target_date_features,
    compute_baseline, fit_ensemble, get_test_thursdays,
    HORIZONS, TARGET_COL, _load_holiday_set, STORE_MAP,
)


def run_backtest_collect(df, holiday_dates, n_thursdays=16):
    df, ths = get_test_thursdays(df, n=n_thursdays)
    folds = []
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
            safety = model['dow_safety'].get(test_date.dayofweek, 0)
            folds.append({
                'D': D, 'h': h,
                'model_quantile': model['quantile'],
                'feat_cols': model['feat_cols'],
                'X': cutoff_row[model['feat_cols']].values.flatten(),
                'baseline_at_D': baseline_at_D,
                'safety': safety,
                'actual': test_row[TARGET_COL].iat[0],
            })
    return folds


def compute_wape(productions, actuals):
    return (np.abs(np.array(productions) - np.array(actuals)).sum() /
            np.array(actuals).sum() * 100)


def baseline_wape(folds):
    productions = [(f['baseline_at_D'] + f['model_quantile'].predict(f['X'].reshape(1, -1))[0]
                    + f['safety']) for f in folds]
    actuals = [f['actual'] for f in folds]
    return compute_wape(productions, actuals)


def permute_feature(folds, feature_name, rng):
    """그 feature column shuffle 후 새 WAPE 측정."""
    fold_idx_pairs = [(i, f['feat_cols'].index(feature_name))
                       for i, f in enumerate(folds) if feature_name in f['feat_cols']]
    if not fold_idx_pairs: return None
    vals = np.array([folds[i]['X'][idx] for i, idx in fold_idx_pairs])
    shuffled = vals.copy()
    rng.shuffle(shuffled)

    productions = []
    actuals = []
    pos = 0
    for n, f in enumerate(folds):
        X = f['X'].copy()
        if feature_name in f['feat_cols']:
            idx = f['feat_cols'].index(feature_name)
            X[idx] = shuffled[pos]
            pos += 1
        pred = f['baseline_at_D'] + f['model_quantile'].predict(X.reshape(1, -1))[0] + f['safety']
        productions.append(pred)
        actuals.append(f['actual'])
    return compute_wape(productions, actuals)


def main():
    print('=== 4매장 Permutation Importance (simple spec, α=0.7, bulk 제외) ===\n')
    holiday_dates = _load_holiday_set()
    rng = np.random.default_rng(42)

    all_results = {}
    for store_cd, store_name in STORE_MAP.items():
        print(f'\n{"="*70}\n=== {store_name} ===\n{"="*70}')
        daily = build_store_daily(store_cd, exclude_bulk=True)
        df = add_features(daily, holiday_dates)
        folds = run_backtest_collect(df, holiday_dates)
        print(f'folds: {len(folds)}')

        base_wape = baseline_wape(folds)
        print(f'baseline WAPE: {base_wape:.3f}%')
        features = list(folds[0]['feat_cols'])
        print(f'features: {len(features)}')

        results = []
        for fname in features:
            new_wape = permute_feature(folds, fname, rng)
            if new_wape is None: continue
            results.append((fname, new_wape - base_wape, new_wape))
        results.sort(key=lambda x: -x[1])

        all_results[store_name] = {
            'baseline_wape': base_wape,
            'results': results,
        }

        print(f'\nTOP 10:')
        for i, (fname, delta, new_w) in enumerate(results[:10], 1):
            print(f'  {i:>2}. {fname:>40s}  Δ {delta:>+7.3f}pp  → {new_w:.3f}%')
        print(f'\nBOTTOM 5 (noise/negative):')
        for fname, delta, new_w in results[-5:]:
            print(f'      {fname:>40s}  Δ {delta:>+7.3f}pp  → {new_w:.3f}%')

    # 통합 ranking 표
    print(f'\n\n{"="*100}')
    print(f'=== 4매장 TOP 5 비교 (Δ WAPE pp) ===')
    print(f'{"="*100}\n')
    print(f'{"Rank":>4s} {"광교":>30s} {"광화문":>30s} {"메세나":>30s} {"삼성타운":>30s}')
    top5 = {st: r['results'][:5] for st, r in all_results.items()}
    for i in range(5):
        row = []
        for st in ['광교', '광화문', '메세나폴리스', '삼성타운']:
            if i < len(top5.get(st, [])):
                fname, delta, _ = top5[st][i]
                row.append(f'{fname[:22]:>22s} +{delta:.2f}pp')
            else:
                row.append('')
        print(f'  {i+1:>3d}. {row[0]:>30s} {row[1]:>30s} {row[2]:>30s} {row[3]:>30s}')

    # CSV 저장
    flat = []
    for store, data in all_results.items():
        for fname, delta, new_w in data['results']:
            flat.append({
                'store': store,
                'feature': fname,
                'delta_wape_pp': delta,
                'new_wape': new_w,
                'baseline_wape': data['baseline_wape'],
            })
    pd.DataFrame(flat).to_csv('reports/permutation_importance_all4.csv', index=False)
    print('\nsaved: reports/permutation_importance_all4.csv')


if __name__ == '__main__':
    main()

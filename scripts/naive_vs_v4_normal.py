"""광교 종합 비교 — Naive vs v4 vs v4+bulk제외 (sumRn fix 후, fair).

조건 통일:
- 5/26 데이터, 광교 단독
- 카테고리 한정 (bread/pastry/sandwich), filter_seasonal 적용
- target: adjusted_demand_unit (α=0.5)
- 16 Thursdays × 7 horizons (N=108)
- sumRn fix 적용
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from bakery.features.category_aggregate import (
    build_category_daily, build_features,
    EVENTS, LUNAR_EVENTS, _signed_days_to_event, _days_to_lunar_event,
)
from scripts.v4_new_data_backtest import (
    build_new_data_daily, build_closing_rows,
    TARGET_COL, HORIZONS, PRODUCTION_Q, ALPHA, SHORT_TERM_TO_DROP,
    _load_holiday_set, add_target_date_features, compute_baseline,
    fit_ensemble,
)


def run_naive(df, n_thursdays=16):
    """Naive baseline: production = baseline + dow별 평균 shortfall (residual 학습 X)."""
    holiday_dates = _load_holiday_set()
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['dow'] = df_clean['date'].dt.dayofweek

    thursdays = df_clean[df_clean['dow'] == 3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df_clean['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
    for D in test_ths:
        train = df_clean[df_clean['date'] <= D].copy()
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df_clean[df_clean['date'] == test_date]
            if len(test_row) == 0: continue

            # baseline = 4주 동일요일 평균 (h shift 고려)
            baseline_at_D = compute_baseline(df_clean[df_clean['date'] <= D], h).iloc[-1]

            # Naive dow safety: train에서 그 dow의 actual 실제 분위수 (production_q)
            train_h = train.copy()
            train_h['baseline'] = compute_baseline(train_h, h)
            train_h['future_target'] = train_h[TARGET_COL].shift(-h)
            train_h['target_dow'] = (train_h['date'] + pd.Timedelta(days=h)).dt.dayofweek
            train_h['shortfall'] = (train_h['future_target'] - train_h['baseline']).clip(lower=0)
            train_h = train_h.dropna(subset=['shortfall'])

            target_dow = test_date.dayofweek
            sub_dow = train_h[train_h['target_dow'] == target_dow]
            # production_q quantile shortfall (안전마진)
            safety = sub_dow['shortfall'].quantile(PRODUCTION_Q) if len(sub_dow) else 0

            production = baseline_at_D + safety
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': target_dow,
                'baseline': baseline_at_D, 'production': production, 'actual': actual,
            })
    return pd.DataFrame(results)


def run_naive_simple(df, n_thursdays=16):
    """Naive 단순: production = baseline (safety margin X)."""
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['dow'] = df_clean['date'].dt.dayofweek
    thursdays = df_clean[df_clean['dow'] == 3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df_clean['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
    for D in test_ths:
        for h in HORIZONS:
            test_date = D + pd.Timedelta(days=h)
            test_row = df_clean[df_clean['date'] == test_date]
            if len(test_row) == 0: continue
            baseline_at_D = compute_baseline(df_clean[df_clean['date'] <= D], h).iloc[-1]
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': test_date.dayofweek,
                'baseline': baseline_at_D, 'production': baseline_at_D, 'actual': actual,
            })
    return pd.DataFrame(results)


def run_v4_backtest(df, n_thursdays=16):
    """v4 framework: ensemble (baseline + LGBM residual + dow_safety)."""
    holiday_dates = _load_holiday_set()
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['dow'] = df_clean['date'].dt.dayofweek
    thursdays = df_clean[df_clean['dow'] == 3]['date'].tolist()
    thursdays = [t for t in thursdays if (t - df_clean['date'].min()).days >= 365]
    test_ths = thursdays[-n_thursdays:]

    results = []
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
            resid_q = model['quantile'].predict(cutoff_row[model['feat_cols']])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model['dow_safety'].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': target_dow,
                'baseline': baseline_at_D, 'production': production, 'actual': actual,
            })
    return pd.DataFrame(results)


def stats(res, label):
    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under_rate = (res['production'] < res['actual']).mean()
    waste_per_day = (res['production'] - res['actual']).clip(lower=0)   # 발주 과잉 = 폐기
    short_per_day = (res['actual'] - res['production']).clip(lower=0)   # 발주 부족 = 매진
    return {
        'label': label,
        'wape': wape * 100,
        'under_rate': under_rate * 100,
        'waste_total': waste_per_day.sum(),
        'short_total': short_per_day.sum(),
        'waste_per_day': waste_per_day.mean(),
        'short_per_day': short_per_day.mean(),
        'avg_actual': res['actual'].mean(),
        'avg_pred': res['production'].mean(),
        'n': len(res),
    }


def main():
    print('=== 광교 종합 비교 — Naive vs v4 vs v4+bulk 제외 ===\n')

    closing_rows = build_closing_rows()
    results_summary = []

    for exclude_bulk, suffix in [(False, '(bulk 포함)'), (True, '(bulk 제외)')]:
        print(f'\n--- 데이터: exclude_bulk={exclude_bulk} ---')
        daily_raw = build_new_data_daily(exclude_bulk=exclude_bulk)
        cd = build_category_daily(daily_raw=daily_raw, discount_rows=closing_rows, alpha=ALPHA)
        df = build_features(cd, target_col=TARGET_COL)
        print(f'features: {df.shape}, avg target {cd.df[TARGET_COL].mean():.1f}')

        # Naive 단순 (no safety)
        if not exclude_bulk:  # bulk 포함만 측정 (Naive simple은 baseline만 측정)
            res_n_simple = run_naive_simple(df)
            results_summary.append(stats(res_n_simple, f'Naive 단순 (baseline만) {suffix}'))

            # Naive + dow safety
            res_n_safety = run_naive(df)
            results_summary.append(stats(res_n_safety, f'Naive + dow safety (q={PRODUCTION_Q}) {suffix}'))

        # v4 framework
        res_v4 = run_v4_backtest(df)
        results_summary.append(stats(res_v4, f'v4 framework {suffix}'))

    # 출력
    print(f'\n{"="*110}')
    print(f'=== 광교 (16 Thursdays × 7 horizons, sumRn fix 적용, N=108) — 4 metric 종합 ===')
    print(f'{"="*110}')
    print(f'{"Model":>45s} {"WAPE":>7s} {"매진율":>8s} {"폐기/일":>9s} {"부족/일":>9s} {"폐기합":>8s} {"부족합":>8s}')
    for r in results_summary:
        print(f'{r["label"]:>45s} {r["wape"]:>6.2f}% {r["under_rate"]:>6.1f}% '
              f'{r["waste_per_day"]:>8.1f} {r["short_per_day"]:>8.1f} '
              f'{r["waste_total"]:>7.0f} {r["short_total"]:>7.0f}')

    print(f'\n  *폐기 = production - actual (발주 과잉으로 남는 빵, 일별 평균/누적)')
    print(f'  *부족 = actual - production (발주 부족으로 매진된 수량, 일별 평균/누적)')
    print(f'  *WAPE 분자 = 폐기합 + 부족합 (둘 다 줄이는 게 진짜 모델 개선)')

    # 핵심 비교 line
    print(f'\n=== 핵심 효과 ===')
    bench = {r['label']: r for r in results_summary}
    pairs = [
        ('Naive 단순 (baseline만) (bulk 포함)', 'v4 framework (bulk 포함)', 'Naive 단순 → v4'),
        ('Naive + dow safety (q=0.9) (bulk 포함)', 'v4 framework (bulk 포함)', 'Naive+safety → v4'),
        ('v4 framework (bulk 포함)', 'v4 framework (bulk 제외)', 'v4 bulk 포함 → 제외'),
    ]
    for k1, k2, label in pairs:
        if k1 in bench and k2 in bench:
            a, b = bench[k1], bench[k2]
            print(f'\n  [{label}]')
            print(f'    WAPE     : {a["wape"]:>5.2f}% → {b["wape"]:>5.2f}% ({b["wape"]-a["wape"]:+.2f}pp)')
            print(f'    매진율   : {a["under_rate"]:>5.1f}% → {b["under_rate"]:>5.1f}% ({b["under_rate"]-a["under_rate"]:+.1f}pp)')
            print(f'    폐기합   : {a["waste_total"]:>5.0f} → {b["waste_total"]:>5.0f} ({b["waste_total"]-a["waste_total"]:+.0f})')
            print(f'    부족합   : {a["short_total"]:>5.0f} → {b["short_total"]:>5.0f} ({b["short_total"]-a["short_total"]:+.0f})')


if __name__ == '__main__':
    main()

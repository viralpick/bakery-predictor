"""1+2+3 적용 backtest:
1. Target date 기준 캘린더 (cyclic + holiday + before_holiday) 추가
2. Target date 기준 특수일 (xmas/valentine/white_day/children_day/chuseok/seollal)
3. 단기 features 제거 (rmean7, rstd7, ewma7)

운영 시 dow=목요일 고정 문제 해결 — target_date 기준 features로 진짜 dow 효과 캡처.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from bakery.features.category_aggregate import (
    build_category_daily, build_features, EVENTS, LUNAR_EVENTS, EVENT_CLIP,
    _signed_days_to_event, _days_to_lunar_event,
)


TARGET_COL = "adjusted_demand_unit"
HORIZONS = [4,5,6,7,8,9,10]
PRODUCTION_Q = 0.90

# 제거할 단기 features (rmean7, rstd7, ewma7)
SHORT_TERM_TO_DROP = [
    f"{TARGET_COL}_rmean7",
    f"{TARGET_COL}_rstd7",
    f"{TARGET_COL}_ewma7",
]


def _load_holiday_set():
    cal = pd.read_parquet("data/external/calendar_raw.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return set(cal.loc[cal["is_holiday"] == True, "date"])


def add_target_date_features(df: pd.DataFrame, h: int, holiday_dates: set, target_col: str = TARGET_COL) -> pd.DataFrame:
    """target_date (D+h) 기준 features.

    1. 캘린더 cyclic (dow/month/dom)
    2. holiday + before_holiday (좁은 정의)
    3. 특수일 days_to_* (xmas/valentine/white_day/children_day + chuseok/seollal)
    """
    d = df.copy()
    target_date = d["date"] + pd.Timedelta(days=h)
    target_dow = target_date.dt.dayofweek
    target_month = target_date.dt.month
    target_dom = target_date.dt.day
    target_dim = target_date.dt.days_in_month

    # 1. Cyclic 캘린더
    d["tgt_dow_sin"]   = np.sin(2 * np.pi * target_dow / 7)
    d["tgt_dow_cos"]   = np.cos(2 * np.pi * target_dow / 7)
    d["tgt_month_sin"] = np.sin(2 * np.pi * target_month / 12)
    d["tgt_month_cos"] = np.cos(2 * np.pi * target_month / 12)
    d["tgt_dom_sin"]   = np.sin(2 * np.pi * (target_dom - 1) / target_dim)
    d["tgt_dom_cos"]   = np.cos(2 * np.pi * (target_dom - 1) / target_dim)
    d["tgt_is_weekend"] = (target_dow >= 5).astype(int)

    # 2. Holiday + before_holiday (좁은 정의)
    d["tgt_is_holiday"]   = target_date.isin(holiday_dates).astype(int)
    next_target = target_date + pd.Timedelta(days=1)
    next_dow = (target_dow + 1) % 7
    is_off_target = d["tgt_is_holiday"] == 1
    is_off_target |= (target_dow >= 5)
    is_off_next = next_target.isin(holiday_dates) | (next_dow >= 5)
    d["tgt_is_before_holiday"] = (~is_off_target & is_off_next).astype(int)

    # 3. 특수일 (target_date 기준)
    target_dates_series = pd.Series(target_date.values)
    for feat, (m, day) in EVENTS.items():
        new_name = feat.replace("days_to_", "tgt_days_to_")
        d[new_name] = _signed_days_to_event(target_dates_series, m, day).astype("int16")
        d[new_name.replace("tgt_days_to_", "tgt_is_within7_")] = (np.abs(d[new_name]) <= 7).astype(int)
    for feat, year_dates in LUNAR_EVENTS.items():
        new_name = feat.replace("days_to_", "tgt_days_to_")
        d[new_name] = _days_to_lunar_event(target_dates_series, year_dates).astype("int16")
        d[new_name.replace("tgt_days_to_", "tgt_is_within7_")] = (np.abs(d[new_name]) <= 7).astype(int)

    return d


def compute_baseline(df, h, target_col=TARGET_COL):
    shifts = [7*k - h for k in [1,2,3,4] if 7*k - h > 0]
    lag_cols = pd.concat([df[target_col].shift(s) for s in shifts], axis=1)
    return lag_cols.mean(axis=1)


def fit_ensemble(train: pd.DataFrame, h: int, holiday_dates: set, target_col: str = TARGET_COL):
    """Train Ensemble with target_date features."""
    train_h = train.copy()
    train_h["baseline"] = compute_baseline(train_h, h, target_col)
    train_h["future_target"] = train_h[target_col].shift(-h)
    train_h["residual"] = train_h["future_target"] - train_h["baseline"]
    train_h["target_dow"] = (train_h["date"] + pd.Timedelta(days=h)).dt.dayofweek

    # target_date features 추가
    train_h = add_target_date_features(train_h, h, holiday_dates, target_col)

    # feature columns 선택
    LEAK = ("sold_total_unit","sold_total_revenue","sold_normal_unit","sold_normal_revenue",
            "sold_closing","sold_closing_revenue","adjusted_demand_unit","adjusted_demand_revenue",
            "n_stockout_items","n_early_stockout","n_items_active",
            "baseline","future_target","residual","target_dow","dow")
    feat_cols = [c for c in train_h.columns
                 if c not in ("date", target_col, *LEAK)
                 and c not in SHORT_TERM_TO_DROP]   # 단기 features 제거

    train_clean = train_h.dropna(subset=["baseline","future_target","residual"] + feat_cols)
    X = train_clean[feat_cols]
    y = train_clean["residual"]

    common = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
                  num_leaves=15, random_state=42, verbosity=-1)
    expected = lgb.LGBMRegressor(objective="regression_l1", **common).fit(X, y)
    quantile = lgb.LGBMRegressor(objective="quantile", alpha=PRODUCTION_Q, **common).fit(X, y)

    # dow safety
    train_clean = train_clean.copy()
    train_clean["prod_pred"] = train_clean["baseline"] + quantile.predict(X)
    train_clean["shortfall"] = (train_clean["future_target"] - train_clean["prod_pred"]).clip(lower=0)
    dow_safety = train_clean.groupby("target_dow")["shortfall"].mean().to_dict()

    return {"expected": expected, "quantile": quantile, "feat_cols": feat_cols, "dow_safety": dow_safety}


def run_backtest(df, n_thursdays=16):
    holiday_dates = _load_holiday_set()
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['dow'] = df_clean['date'].dt.dayofweek

    thursdays = df_clean[df_clean['dow']==3]['date'].tolist()
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
            resid_q = model["quantile"].predict(cutoff_row[model["feat_cols"]])[0]
            target_dow = test_date.dayofweek
            production = baseline_at_D + resid_q + model["dow_safety"].get(target_dow, 0)
            actual = test_row[TARGET_COL].iat[0]
            results.append({
                'D': D, 'h': h, 'target_dow': target_dow,
                'baseline': baseline_at_D, 'production': production, 'actual': actual,
                'n_features': len(model["feat_cols"]),
            })
    return pd.DataFrame(results)


def main():
    print('=== 1+2+3 적용 — target_date features + 단기 features 제거 ===\n')

    cd = build_category_daily(alpha=0.6)
    df = build_features(cd, target_col=TARGET_COL)

    res = run_backtest(df, n_thursdays=16)
    n_features = res['n_features'].iloc[0]
    print(f'\nN predictions: {len(res)}')
    print(f'사용 features 수: {n_features}\n')

    # 전체 결과
    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under = (res['production'] < res['actual']).mean()
    over = (res['production'] - res['actual']).mean()
    print(f'=== 전체 결과 ===')
    print(f'  WAPE        : {wape*100:.2f}%')
    print(f'  매진율      : {under*100:.1f}%')
    print(f'  발주 over   : {over:+.1f}')
    print(f'  평균 발주   : {res["production"].mean():.1f}')
    print(f'  평균 실제   : {res["actual"].mean():.1f}')

    # Horizon별
    print(f'\n=== Horizon별 ===')
    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    for h in HORIZONS:
        sub = res[res['h']==h]
        if len(sub) == 0: continue
        w = (sub['actual'] - sub['production']).abs().sum() / sub['actual'].sum()
        u = (sub['production'] < sub['actual']).mean()
        dow_name = dn[(3+h)%7]
        print(f'  D+{h} ({dow_name}): n={len(sub):>2}, WAPE {w*100:>5.2f}%, 매진율 {u*100:>5.1f}%, '
              f'pred {sub["production"].mean():>6.1f}, actual {sub["actual"].mean():>6.1f}')

    # 비교 (이전 모델 vs 새 모델)
    print(f'\n=== 비교: 이전 (full 45 features) vs 새 (target_date + 단기 제거) ===')
    print(f'{"Model":>40s} {"n_feat":>7s} {"WAPE":>7s} {"매진율":>8s} {"발주 over":>10s}')
    print(f'{"이전 full (D dow 기반)":>40s} {"45":>7s} {"29.44%":>7s} {"11.4%":>8s} {"+63.3":>10s}')
    print(f'{"이전 minimal (lag+dow+특수일)":>40s} {"22":>7s} {"30.31%":>7s} {"13.6%":>8s} {"+65.4":>10s}')
    print(f'{"새 (target_date + 단기 제거)":>40s} {n_features:>7} {wape*100:>5.2f}% {under*100:>6.1f}% {over:>+8.1f}')

    res.to_csv('reports/v4_target_date_features.csv', index=False)
    print('\nsaved: reports/v4_target_date_features.csv')

if __name__ == "__main__":
    main()

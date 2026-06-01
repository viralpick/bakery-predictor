"""검증: multistore_lgbm.py와 동일 데이터/feature, 광교 단독만 학습.

multi-store의 WAPE 15.35%가:
(a) store_id 통합 학습의 효과인지, 또는
(b) 단순히 데이터 정의 차이 (closing 포함 + filter_seasonal 미적용)
구별하기 위해 광교 단독 동일 조건으로 재실험.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd

from scripts.multistore_lgbm import (
    build_multistore_daily, add_features, run_backtest,
    HORIZONS, _load_holiday_set,
)


def main():
    print('=== 검증: 광교 단독 (multi-store와 동일 데이터/feature) ===\n')
    daily = build_multistore_daily()
    # 광교만
    daily = daily[daily['store'] == '광교'].copy()
    df = add_features(daily, holiday_dates=_load_holiday_set())
    print(f'features prepared: {df.shape}, 매장: {df["store"].unique()}')

    res = run_backtest(df, target_store='광교', n_thursdays=16)
    print(f'\nN predictions: {len(res)}')

    wape = (res['actual'] - res['production']).abs().sum() / res['actual'].sum()
    under = (res['production'] < res['actual']).mean()
    over = (res['production'] - res['actual']).mean()

    print(f'\n=== 광교 단독 (동일 데이터/feature, store_id만 빠짐) ===')
    print(f'  WAPE        : {wape * 100:.2f}%')
    print(f'  매진율      : {under * 100:.1f}%')
    print(f'  발주 over   : {over:+.1f}')
    print(f'  평균 발주   : {res["production"].mean():.1f}')
    print(f'  평균 실제   : {res["actual"].mean():.1f}')

    dn = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
    print(f'\n=== Horizon별 (광교 단독) ===')
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
    print(f'{"Model":>50s} {"WAPE":>7s} {"매진율":>8s}')
    print(f'{"v4 광교 단독 (구 데이터, adjusted_demand)":>50s} {"29.28%":>7s} {"11.4%":>8s}')
    print(f'{"광교 단독 (새 데이터, qty 단순합)":>50s} {wape*100:>5.2f}% {under*100:>6.1f}%')
    print(f'{"Multi-store (새 데이터, qty 단순합, store_id 추가)":>50s} {"15.35%":>7s} {"11.1%":>8s}')

    res.to_csv('reports/multistore_v5_gwangyo_solo.csv', index=False)
    print('\nsaved: reports/multistore_v5_gwangyo_solo.csv')


if __name__ == '__main__':
    main()

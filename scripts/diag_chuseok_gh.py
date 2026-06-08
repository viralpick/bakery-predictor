"""[DEPRECATED 2026-06-08] v5 conformal 구간(범위)예측 진단 — 폐기. 점추정+품절/매진 위험수치로 전환 (docs/poc_scope_v6.md §8 D1). 데이터 검증 단계 산출물로만 보존.

광화문 과거 추석 주 실제 수요 — 연도별 비교.

days_to_chuseok ∈ [-3,+3] 구간의 adjusted_demand(타깃)를 연도별로 출력.
q0.9가 추석에 안전마진을 키우는 근거(과거 추석 분산)를 데이터로 확인.
"""
from __future__ import annotations

import pandas as pd

from interval_backtest_4stores import build_store_features
from store_daily import STORE_MAP

pd.set_option("display.width", 200, "display.max_columns", 30)

STORE_CD = "1000000485"  # 광화문
NAME, STORE_ID = STORE_MAP[STORE_CD]
TARGET = "adjusted_demand_unit"


def main():
    feats = build_store_features(STORE_CD, STORE_ID)
    feats["date"] = pd.to_datetime(feats["date"])
    feats["year"] = feats["date"].dt.year
    print(f"[{NAME}] 데이터 범위: {feats['date'].min().date()} ~ {feats['date'].max().date()} "
          f"({feats['date'].nunique()}일)")

    near = feats[feats["days_to_chuseok"].abs() <= 3].copy()
    cols = ["date", "year", "dow", "days_to_chuseok", "is_public_holiday", TARGET]
    near = near[cols].sort_values("date")
    near["date"] = near["date"].dt.strftime("%Y-%m-%d")
    print("\n=== 추석 ±3일 (연도별) ===")
    print(near.round(1).to_string(index=False))

    print("\n=== 연도별 추석주 요약 (±3일) ===")
    g = near.groupby("year")[TARGET].agg(["count", "mean", "min", "max"]).round(1)
    # 평상시(추석 영향권 밖, |days_to_chuseok|>14) 평균과 비교
    normal = feats[feats["days_to_chuseok"].abs() > 14]
    g["평상시평균"] = normal.groupby(normal["date"].dt.year)[TARGET].mean().round(1)
    g["추석주/평상시"] = (g["mean"] / g["평상시평균"]).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()

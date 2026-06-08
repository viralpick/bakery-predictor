"""[DEPRECATED 2026-06-08] v5 conformal 구간(범위)예측 진단 — 폐기. 점추정+품절/매진 위험수치로 전환 (docs/poc_scope_v6.md §8 D1). 데이터 검증 단계 산출물로만 보존.

광화문 9말~10초 급락 구간 진단 — q0.90 vs q0.50 anchor 비교 + 공휴일 feature.

목적: 비교 그래프에서 광화문 10/1~10/7 판매 급락을 q0.9는 못 따라가고 q0.5는
따라가는 것처럼 보이는 이유를 실제 예측값으로 검증. (추석 2025-10-06 군집 의심)
"""
from __future__ import annotations

import pandas as pd

from interval_backtest import run
from interval_backtest_4stores import build_store_features
from store_daily import STORE_MAP

pd.set_option("display.width", 200, "display.max_columns", 30)

STORE_CD = "1000000485"  # 광화문
NAME, STORE_ID = STORE_MAP[STORE_CD]
LO, HI = "2025-09-25", "2025-10-12"


def sym95(feats, q):
    preds = run(feats, n_folds=4, min_train_days=365, calibration_days=210,
                horizon_days=30, production_q=q)
    sub = preds[(preds["variant"] == "symmetric") & (preds["coverage_level"] == 0.95)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    return sub.sort_values("date")


def main():
    feats = build_store_features(STORE_CD, STORE_ID)
    cal_cols = [c for c in feats.columns
                if c in ("date", "is_public_holiday", "is_before_holiday",
                         "is_weekend", "days_to_chuseok")]
    cal = feats[cal_cols].copy()
    cal["date"] = pd.to_datetime(cal["date"])

    a90 = sym95(feats, 0.90)[["date", "dow", "actual", "anchor", "lower", "upper"]]
    a50 = sym95(feats, 0.50)[["date", "anchor", "lower", "upper"]]
    m = a90.merge(a50, on="date", suffixes=("_q90", "_q50")).merge(cal, on="date", how="left")
    win = m[(m["date"] >= LO) & (m["date"] <= HI)].copy()
    win["date"] = win["date"].dt.strftime("%m-%d")
    cols = ["date", "dow", "is_public_holiday", "days_to_chuseok",
            "actual", "anchor_q90", "anchor_q50", "lower_q90", "upper_q90",
            "lower_q50", "upper_q50"]
    cols = [c for c in cols if c in win.columns]
    print(f"\n===== [{NAME}] {LO}~{HI} =====")
    print(win[cols].round(1).to_string(index=False))

    # 형태 비교: anchor_q90 - anchor_q50 가 구간 내내 ~일정한 offset인지
    win["offset"] = win["anchor_q90"] - win["anchor_q50"]
    print(f"\nanchor offset(q90-q50): mean={win['offset'].mean():.1f} "
          f"std={win['offset'].std():.1f} min={win['offset'].min():.1f} max={win['offset'].max():.1f}")
    corr = win[["anchor_q90", "anchor_q50"]].corr().iloc[0, 1]
    print(f"anchor_q90 vs anchor_q50 상관(형태 일치도): {corr:.3f}")


if __name__ == "__main__":
    main()

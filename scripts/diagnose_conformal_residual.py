"""#2 conformal 잔차 진단 (diagnosis-only).

문제: one-sided scale-정규화 split-conformal 발주(PR#30, item 경로)의 실현 초과율이
nominal(1−s)보다 균일하게 +0.02~0.07 높다(s=0.74에서 +0.039). exchangeability 하에서는
coverage가 맞아야 하는데 왜 under-cover 하는가?

이 스크립트는 헤드라인 config(s=0.74, n_folds=8, val_weeks=8, cal_fold_frac=0.5, α=0.5,
광교)를 재현해 test item-day의 초과 indicator (adjusted_demand > our_order)를 세 축으로
분해한다:
  A. test fold(시간 위치 = cal↔test gap) — 시간 드리프트 후보
  B. volume tier(item pre-cutoff 평균 = scale의 3분위) — volume 이질성 후보
  C. 명절/주말/평일 — 특수일 후보(이미 base 피처라 드롭 후보)

+ 드리프트 결정타: cal에서 구한 Q_s vs 각 test fold의 empirical 필요-Q_s 비교.
test Q_s가 체계적으로 크면(과거 cal이 최근 마진을 과소평가) 드리프트가 주범.

진단만 한다 — 코드/모델 변경 없음. 결과로 #2 종료 여부(PoC 충분) 판단.

실행: PYTHONPATH=scripts uv run python scripts/diagnose_conformal_residual.py
산출: reports/conformal_residual_diagnosis.csv (+ stdout 표)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.cli import _load_dataset, _enrich_if_needed, _median_base_fold_predictions
from bakery.features.category_aggregate import build_item_adjusted_demand
from bakery.features.scale import compute_item_scale

SERVICE_LEVEL = 0.74
N_FOLDS = 8
VAL_WEEKS = 8
CAL_FOLD_FRAC = 0.5
ALPHA = 0.5
NOMINAL = 1.0 - SERVICE_LEVEL  # 목표 초과율 0.26
CALENDAR_PATH = "data/external/calendar_raw.parquet"


def _holiday_dates() -> set:
    if not Path(CALENDAR_PATH).exists():
        return set()
    cal = pd.read_parquet(CALENDAR_PATH)
    cal["date"] = pd.to_datetime(cal["date"])
    return set(cal.loc[cal["is_holiday"] == True, "date"])  # noqa: E712


def _day_type(dates: pd.Series, holidays: set) -> pd.Series:
    """공휴일 > 주말 > 평일 우선순위로 라벨."""
    d = pd.to_datetime(dates)
    is_holiday = d.isin(holidays)
    is_weekend = d.dt.dayofweek >= 5
    return pd.Series(
        np.where(is_holiday, "holiday", np.where(is_weekend, "weekend", "weekday")),
        index=dates.index,
    )


def _exceed_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """group별 초과율 + nominal 대비 잔차 + n."""
    g = df.groupby(group_col)
    out = pd.DataFrame({
        "n": g.size(),
        "exceed_rate": g["exceed"].mean(),
    })
    out["residual"] = out["exceed_rate"] - NOMINAL
    return out.reset_index()


def main() -> None:
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    daily = build_item_adjusted_demand(daily, alpha=ALPHA)
    pred_df, windows = _median_base_fold_predictions(daily, val_weeks=VAL_WEEKS, n_folds=N_FOLDS)
    first_val_start = min(w.val_start for w in windows)
    scale = compute_item_scale(daily, before_date=first_val_start, y_col="adjusted_demand")

    # --- conformal 로직 재현 (Q_s 노출을 위해 인라인; _apply_conformal_to_folds와 동일) ---
    fold_order = pred_df.groupby("fold")["date"].min().sort_values().index.tolist()
    n_cal = max(1, int(len(fold_order) * CAL_FOLD_FRAC))
    cal_folds, test_folds = set(fold_order[:n_cal]), set(fold_order[n_cal:])
    fold_min_date = pred_df.groupby("fold")["date"].min()

    def scale_of(items: pd.Series) -> np.ndarray:
        return items.astype(str).map(scale).fillna(1.0).to_numpy()

    cal = pred_df[pred_df["fold"].isin(cal_folds)].copy()
    test = pred_df[pred_df["fold"].isin(test_folds)].copy()
    cal_scale = scale_of(cal["item_id"])
    cal_scores = (cal["adjusted_demand"].to_numpy() - cal["yhat"].to_numpy()) / cal_scale
    q_s = float(np.quantile(cal_scores[~np.isnan(cal_scores)], SERVICE_LEVEL, method="higher"))

    test_scale = scale_of(test["item_id"])
    test["our_order"] = np.clip(test["yhat"].to_numpy() + q_s * test_scale, 0.0, None)
    test["score"] = (test["adjusted_demand"].to_numpy() - test["yhat"].to_numpy()) / test_scale
    test["exceed"] = test["adjusted_demand"] > test["our_order"]

    # 메타데이터
    holidays = _holiday_dates()
    test["day_type"] = _day_type(test["date"], holidays)
    item_scale_series = test["item_id"].astype(str).map(scale).fillna(1.0)
    test["volume_tier"] = pd.qcut(
        item_scale_series, 3, labels=["low", "mid", "high"], duplicates="drop"
    )

    overall = float(test["exceed"].mean())
    print("=" * 70)
    print(f"헤드라인 재현: s={SERVICE_LEVEL} n_folds={N_FOLDS} val_weeks={VAL_WEEKS} "
          f"cal_fold_frac={CAL_FOLD_FRAC} α={ALPHA}")
    print(f"cal folds={sorted(cal_folds)} test folds={sorted(test_folds)}")
    print(f"cal Q_s(s={SERVICE_LEVEL}, method=higher) = {q_s:.4f}")
    print(f"test item-days = {len(test):,} | cal item-days = {len(cal):,}")
    print(f"[전체] 실현 초과율 = {overall:.4f}  (nominal {NOMINAL:.2f}, 잔차 {overall-NOMINAL:+.4f})")

    # --- A. 시간(test fold) ---
    print("\n[A] test fold별 (시간 위치 = 드리프트 축)")
    a = _exceed_summary(test, "fold")
    a["min_date"] = a["fold"].map(fold_min_date).dt.date
    # 각 test fold의 empirical 필요 Q_s (드리프트 결정타)
    a["need_q_s"] = a["fold"].map(
        lambda f: float(np.quantile(
            test.loc[test["fold"] == f, "score"].dropna(), SERVICE_LEVEL, method="higher"
        ))
    )
    a["need_minus_cal"] = a["need_q_s"] - q_s
    a = a.sort_values("min_date")
    print(a.to_string(index=False))

    # --- B. volume tier ---
    print("\n[B] volume tier별 (item pre-cutoff 평균 3분위)")
    b = _exceed_summary(test, "volume_tier")
    # tier별 필요 Q_s
    b["need_q_s"] = b["volume_tier"].map(
        lambda t: float(np.quantile(
            test.loc[test["volume_tier"] == t, "score"].dropna(), SERVICE_LEVEL, method="higher"
        ))
    )
    print(b.to_string(index=False))

    # --- C. 명절/주말/평일 ---
    print("\n[C] day_type별 (공휴일>주말>평일)")
    c = _exceed_summary(test, "day_type")
    print(c.to_string(index=False))

    # --- 드리프트 요약: cal vs test 전체 score 분포 ---
    print("\n[드리프트 요약] normalized score 분포 (cal은 과거, test는 최근)")
    cal_s = cal_scores[~np.isnan(cal_scores)]
    test_s = test["score"].dropna().to_numpy()
    for name, s in [("cal", cal_s), ("test", test_s)]:
        print(f"  {name}: mean={s.mean():+.3f} median={np.median(s):+.3f} "
              f"q74={np.quantile(s, 0.74, method='higher'):+.3f} "
              f"q90={np.quantile(s, 0.90):+.3f}")

    out_path = Path("reports/conformal_residual_diagnosis.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test[["item_id", "date", "fold", "adjusted_demand", "yhat", "our_order",
          "score", "exceed", "day_type", "volume_tier"]].to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

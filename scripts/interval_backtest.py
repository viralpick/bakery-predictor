"""광교 conformal 구간예측 A/B: 대칭 vs 비대칭 (Coverage@80/@95).

q0.90 production을 중심 앵커로 고정한 채, calibration fold에서 요일별 잔차로
구간 마진을 보정한다. 대칭(center±δ)과 비대칭(adaptive q_lo 하한 + center 상한)을
동일 fold·동일 calibration으로 비교하고, 명목 coverage 대비 실측·매진율 수렴을 본다.

산출물:
  reports/interval_predictions.csv  — test 행별 (date, dow, actual, anchor, lower, upper, ...)
  reports/interval_coverage.csv     — variant×coverage 집계 (coverage/width/매진율)
  reports/interval_coverage_by_dow.csv — variant×coverage×요일 coverage

데이터 빌드는 v4_new_data_backtest의 광교 canonical 경로를 재사용한다
(exclude_bulk, 마감할인코드, α=0.7).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bakery.evaluation.metrics import coverage, coverage_by_group, interval_width
from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.models.category_total import (
    expanding_calibration_folds,
    fit_category_total,
)
from bakery.models.conformal_interval import ConformalInterval
from v4_new_data_backtest import (
    ALPHA,
    PRODUCTION_Q,
    TARGET_COL,
    build_closing_rows,
    build_new_data_daily,
)

# coverage 수준 → conformal 보정 전 초기 하한 quantile (설계 §4.2)
COVERAGE_Q_LO = {0.80: 0.10, 0.95: 0.05}
VARIANTS = ("symmetric", "asymmetric")
REPORTS = Path("reports")


def build_gwangyo_features(exclude_bulk: bool = True) -> pd.DataFrame:
    daily_raw = build_new_data_daily(exclude_bulk=exclude_bulk)
    closing_rows = build_closing_rows()
    cd = build_category_daily(daily_raw=daily_raw, discount_rows=closing_rows, alpha=ALPHA)
    return build_features(cd, target_col=TARGET_COL)


def fit_fold(train: pd.DataFrame) -> dict[float, object]:
    """coverage 수준별 모델 (center/expected 공유, q_lo만 다름)."""
    return {
        cov: fit_category_total(
            train,
            target_col=TARGET_COL,
            alpha_demand=ALPHA,
            production_q=PRODUCTION_Q,
            q_lo=q_lo,
        )
        for cov, q_lo in COVERAGE_Q_LO.items()
    }


def _calibrate_and_predict(variant, cov, model, cal, test):
    cal_center = model.predict_production(cal)
    test_center = model.predict_production(test)
    cal_dow = cal["date"].dt.dayofweek.to_numpy()
    test_dow = test["date"].dt.dayofweek.to_numpy()
    cal_actual = cal[TARGET_COL].to_numpy()
    ci = ConformalInterval(mode=variant, coverage=cov)
    if variant == "symmetric":
        ci.calibrate(actual=cal_actual, center_pred=cal_center, dow=cal_dow)
        lower, upper = ci.predict_interval(center_pred=test_center, dow=test_dow)
    else:
        cal_lo = model.predict_production_lo(cal)
        test_lo = model.predict_production_lo(test)
        ci.calibrate(actual=cal_actual, center_pred=cal_center, dow=cal_dow, lo_pred=cal_lo)
        lower, upper = ci.predict_interval(
            center_pred=test_center, dow=test_dow, lo_pred=test_lo
        )
    return test_center, lower, upper


def eval_fold(fold, models) -> list[dict]:
    rows: list[dict] = []
    for cov, model in models.items():
        for variant in VARIANTS:
            center, lower, upper = _calibrate_and_predict(
                variant, cov, model, fold.calibration, fold.test
            )
            test = fold.test
            for i in range(len(test)):
                rows.append(
                    {
                        "fold": fold.fold,
                        "variant": variant,
                        "coverage_level": cov,
                        "date": test["date"].iloc[i],
                        "dow": int(test["date"].iloc[i].dayofweek),
                        "actual": float(test[TARGET_COL].iloc[i]),
                        "anchor": float(center[i]),
                        "lower": float(lower[i]),
                        "upper": float(upper[i]),
                    }
                )
    return rows


def run(df: pd.DataFrame, *, n_folds, min_train_days, calibration_days, horizon_days) -> pd.DataFrame:
    folds = expanding_calibration_folds(
        df,
        target_col=TARGET_COL,
        n_folds=n_folds,
        min_train_days=min_train_days,
        calibration_days=calibration_days,
        horizon_days=horizon_days,
    )
    records: list[dict] = []
    for f in folds:
        print(
            f"[fold {f.fold}] train={f.train['date'].nunique()}d "
            f"cal={f.calibration['date'].nunique()}d test={f.test['date'].nunique()}d "
            f"(test {f.test['date'].min().date()}→{f.test['date'].max().date()})"
        )
        models = fit_fold(f.train)
        records.extend(eval_fold(f, models))
    return pd.DataFrame(records)


def summarize(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _agg(g: pd.DataFrame) -> pd.Series:
        a, lo, up, anchor = g["actual"], g["lower"], g["upper"], g["anchor"]
        return pd.Series(
            {
                "n": len(g),
                "coverage": coverage(a.to_numpy(), lo.to_numpy(), up.to_numpy()),
                "mean_width": interval_width(lo.to_numpy(), up.to_numpy()),
                "stockout_upper": float((a > up).mean()),  # 상한 초과 (구간 miss)
                "stockout_anchor": float((a > anchor).mean()),  # q0.90 앵커 초과 proxy
            }
        )

    report = (
        preds.groupby(["coverage_level", "variant"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    dow_rows = []
    for (cov, variant), g in preds.groupby(["coverage_level", "variant"], observed=True):
        by_dow = coverage_by_group(
            g["actual"].to_numpy(), g["lower"].to_numpy(), g["upper"].to_numpy(), g["dow"].to_numpy()
        )
        for d, c in sorted(by_dow.items()):
            dow_rows.append(
                {"coverage_level": cov, "variant": variant, "dow": int(d), "coverage": c}
            )
    return report, pd.DataFrame(dow_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="광교 conformal 구간예측 대칭/비대칭 A/B")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--min-train-days", type=int, default=365)
    ap.add_argument("--calibration-days", type=int, default=210)  # 요일당 30+ → Mondrian 활성
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--include-bulk", action="store_true", help="bulk 품목 포함 (기본 제외)")
    args = ap.parse_args()

    df = build_gwangyo_features(exclude_bulk=not args.include_bulk)
    print(f"[build] features: {df.shape}, dates {df['date'].nunique()}")

    preds = run(
        df,
        n_folds=args.n_folds,
        min_train_days=args.min_train_days,
        calibration_days=args.calibration_days,
        horizon_days=args.horizon_days,
    )
    report, dow_report = summarize(preds)

    REPORTS.mkdir(exist_ok=True)
    preds.to_csv(REPORTS / "interval_predictions.csv", index=False)
    report.to_csv(REPORTS / "interval_coverage.csv", index=False)
    dow_report.to_csv(REPORTS / "interval_coverage_by_dow.csv", index=False)

    print("\n=== Coverage A/B (대칭 vs 비대칭) ===")
    print(report.to_string(index=False))
    print("\nsaved: reports/interval_predictions.csv, interval_coverage.csv, interval_coverage_by_dow.csv")


if __name__ == "__main__":
    main()

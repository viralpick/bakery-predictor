"""4매장 xmas event-window 재측정: base(LightGBM) vs EventLevelPrior 블렌드 검증.

leave-past-out(expanding-past) 방식으로 각 매장의 xmas 연도마다:
  1. train = feat[(date < xmas) & (date >= xmas - WINDOW_DAYS)] 로
     fit_category_total 학습 -> base 예측 (`predict_expected` / `predict_production`).
  2. hist = feat[feat.date < xmas] (=leakage-safe, 해당 xmas 이전 전체 과거)로
     bakery.models.event_prior.EventLevelPrior 를 fit -> 과거 xmas 실측 레벨.
  3. `EventLevelPrior.blend([xmas], base_exp, base_prod)` 로 블렌드.

인라인으로 블렌드 산식을 재구현하지 않고, Task 1~4에서 구현·배선된 실제
`EventLevelPrior.fit`/`.blend` 를 그대로 호출한다 (scripts/store_predictive_power.py
의 배선 패턴과 동일).

실행: `PYTHONPATH=scripts uv run python scripts/verify_event_prior.py`

기대 방향 (2026-07-12, 브랜치 feat/bulk-filter-alpha-margin-policy 기준, 정확한 숫자는
train window/데이터 갱신에 따라 소폭 달라질 수 있음):
  광교      base WAPE ~0.23 -> blend ~0.09
  메세나    base WAPE ~0.17 -> blend ~0.11
  광화문    base WAPE ~0.15 -> blend ~0.06
  삼성타운  base 과대(over) bias -> blend 완화
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.models.category_total import fit_category_total
from bakery.models.event_prior import EventLevelPrior
from store_daily import build_store_closing_rows, build_store_daily

TARGET = "adjusted_demand_unit"
ALPHA = 0.8
PRODUCTION_Q = 0.85
WINDOW_DAYS = 730
MIN_TRAIN_ROWS = 60
XMAS_MONTH_DAY = (12, 25)

STORES = [
    ("1000000047", "store_gw01", "광교"),
    ("1000000009", "store_ss01", "삼성타운"),
    ("1000000029", "store_mp01", "메세나폴리스"),
    ("1000000485", "store_gh01", "광화문"),
]


def build_feat(cd_code: str, store_id: str) -> pd.DataFrame:
    daily = build_store_daily(cd_code, store_id, exclude_bulk=True)
    closing = build_store_closing_rows(cd_code)
    cd = build_category_daily(daily_raw=daily, discount_rows=closing, alpha=ALPHA)
    feat = build_features(cd, target_col=TARGET)
    feat["date"] = pd.to_datetime(feat["date"])
    return feat.dropna().reset_index(drop=True)


def xmas_dates(feat: pd.DataFrame) -> list[pd.Timestamp]:
    years = sorted(feat["date"].dt.year.unique())
    month, day = XMAS_MONTH_DAY
    return [pd.Timestamp(year, month, day) for year in years]


def evaluate_one_xmas(feat: pd.DataFrame, xmas: pd.Timestamp) -> dict | None:
    """단일 xmas 연도에 대한 leave-past-out base vs blend 예측을 계산한다."""
    test_df = feat[feat["date"] == xmas]
    if test_df.empty:
        return None
    window = pd.Timedelta(days=WINDOW_DAYS)
    train_df = feat[(feat["date"] < xmas) & (feat["date"] >= xmas - window)]
    if len(train_df) < MIN_TRAIN_ROWS:
        return None

    model = fit_category_total(
        train_df, target_col=TARGET, alpha_demand=ALPHA, production_q=PRODUCTION_Q,
    )
    base_exp = model.predict_expected(test_df)
    base_prod = model.predict_production(test_df)

    hist = feat[feat["date"] < xmas]
    prior = EventLevelPrior().fit(hist, target_col=TARGET)
    blend_exp, _blend_prod = prior.blend(test_df["date"].values, base_exp, base_prod)

    return dict(
        date=xmas.date(),
        actual=float(test_df[TARGET].iloc[0]),
        base=float(base_exp[0]),
        blend=float(blend_exp[0]),
    )


def evaluate_store(feat: pd.DataFrame) -> dict | None:
    """매장의 모든 xmas 연도를 순회하며 base/blend WAPE·bias를 누적한다."""
    rows = [r for xmas in xmas_dates(feat) if (r := evaluate_one_xmas(feat, xmas)) is not None]
    if not rows:
        return None

    actual_total = sum(abs(r["actual"]) for r in rows)
    if actual_total == 0:
        return None

    base_abs_err = sum(abs(r["base"] - r["actual"]) for r in rows)
    blend_abs_err = sum(abs(r["blend"] - r["actual"]) for r in rows)
    base_bias = sum(r["base"] - r["actual"] for r in rows)
    blend_bias = sum(r["blend"] - r["actual"] for r in rows)

    return dict(
        rows=rows,
        base_wape=base_abs_err / actual_total,
        blend_wape=blend_abs_err / actual_total,
        base_bias=base_bias / actual_total,
        blend_bias=blend_bias / actual_total,
    )


def print_store_result(label: str, result: dict | None) -> None:
    print(f"\n=== {label} ===")
    if result is None:
        print("  (xmas 평가 가능한 fold 없음)")
        return

    print(
        f"  xmas: base  WAPE={result['base_wape']:.3f} bias={result['base_bias']:+.3f}"
        f"   ->   blend WAPE={result['blend_wape']:.3f} bias={result['blend_bias']:+.3f}"
    )
    for r in result["rows"]:
        pct_base = (r["base"] - r["actual"]) / r["actual"] * 100
        pct_blend = (r["blend"] - r["actual"]) / r["actual"] * 100
        print(
            f"      {r['date']}: actual={r['actual']:6.0f}"
            f"  base={r['base']:6.0f}({pct_base:+5.0f}%)"
            f"  blend={r['blend']:6.0f}({pct_blend:+5.0f}%)"
        )


def main() -> None:
    print("=== xmas event-window 검증: base(LightGBM) vs EventLevelPrior 블렌드 ===")
    print(f"(leave-past-out, train window={WINDOW_DAYS}d, alpha={ALPHA}, K={EventLevelPrior().k})")
    for cd_code, store_id, label in STORES:
        feat = build_feat(cd_code, store_id)
        result = evaluate_store(feat)
        print_store_result(label, result)


if __name__ == "__main__":
    main()

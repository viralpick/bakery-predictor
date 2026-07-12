"""4매장 event-window(xmas + 등록 음력) 재측정: base(LightGBM) vs EventLevelPrior 블렌드 검증.

`store_predictive_power.STORE_EVENT_PRIORS` 에 매장별로 등록된 이벤트 셋(xmas +
opt-in 음력 이벤트)을 그대로 읽어, 등록된 각 이벤트에 대해 leave-past-out(expanding-past)
방식으로 base-vs-blend WAPE 를 검증한다:

  1. train = feat[(date < D) & (date >= D - WINDOW_DAYS)] 로 fit_category_total 학습
     -> base 예측 (`predict_expected` / `predict_production`).
  2. hist = feat[feat.date < D] (=leakage-safe, D 이전 전체 과거)로
     `EventLevelPrior(events=cfg["events"], lunar_events=cfg["lunar_events"])` 를 fit
     -> 과거 동일 이벤트 실측 레벨.
  3. `EventLevelPrior.blend([D], base_exp, base_prod)` 로 블렌드.

인라인으로 블렌드 산식을 재구현하지 않고, 실제 `EventLevelPrior.fit`/`.blend` 를
그대로 호출한다 (scripts/store_predictive_power.py 의 windowed_backtest 배선 패턴과 동일).

실행: `PYTHONPATH=scripts uv run python scripts/verify_event_prior.py`

기대 방향 (2026-07-13, 브랜치 feat/event-prior-lunar-register 기준, 정확한 숫자는
train window/데이터 갱신에 따라 소폭 달라질 수 있음):
  광교      xmas base ~0.10        -> blend 개선 / 추석 base ~0.214 -> blend ~0.145
  삼성타운  xmas base ~0.42        -> blend 개선 (lunar 미등록)
  메세나    xmas base ~0.10        -> blend 개선 / 설   base ~0.179 -> blend ~0.101
  광화문    xmas base ~0.085       -> blend 개선 (lunar 미등록)
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
from store_predictive_power import STORE_EVENT_PRIORS

TARGET = "adjusted_demand_unit"
ALPHA = 0.8
PRODUCTION_Q = 0.85
WINDOW_DAYS = 730
MIN_TRAIN_ROWS = 60

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


def event_days_by_name(
    feat: pd.DataFrame, prior: EventLevelPrior,
) -> dict[str, list[pd.Timestamp]]:
    """prior 에 등록된 이벤트에 해당하는 feat 내 날짜를 이벤트명별로 그룹."""
    groups: dict[str, list[pd.Timestamp]] = {}
    for d in sorted(feat["date"].unique()):
        d = pd.Timestamp(d)
        name = prior._event_name_for(d)
        if name is not None:
            groups.setdefault(name, []).append(d)
    return groups


def evaluate_one_day(feat: pd.DataFrame, day: pd.Timestamp, prior_cfg: dict) -> dict | None:
    """단일 이벤트일에 대한 leave-past-out base vs blend 예측을 계산한다."""
    test_df = feat[feat["date"] == day]
    if test_df.empty:
        return None
    window = pd.Timedelta(days=WINDOW_DAYS)
    train_df = feat[(feat["date"] < day) & (feat["date"] >= day - window)]
    if len(train_df) < MIN_TRAIN_ROWS:
        return None

    model = fit_category_total(
        train_df, target_col=TARGET, alpha_demand=ALPHA, production_q=PRODUCTION_Q,
    )
    base_exp = model.predict_expected(test_df)
    base_prod = model.predict_production(test_df)

    hist = feat[feat["date"] < day]
    prior = EventLevelPrior(**prior_cfg).fit(hist, target_col=TARGET)
    blend_exp, _blend_prod = prior.blend(test_df["date"].values, base_exp, base_prod)

    return dict(
        date=day.date(),
        actual=float(test_df[TARGET].iloc[0]),
        base=float(base_exp[0]),
        blend=float(blend_exp[0]),
    )


def evaluate_event(
    feat: pd.DataFrame, days: list[pd.Timestamp], prior_cfg: dict,
) -> dict | None:
    """단일 이벤트의 모든 발생연도를 순회하며 base/blend WAPE·bias를 누적한다."""
    rows = [r for day in days if (r := evaluate_one_day(feat, day, prior_cfg)) is not None]
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


def print_event_result(label: str, event_name: str, result: dict | None) -> None:
    print(f"\n=== {label} / {event_name} ===")
    if result is None:
        print("  (평가 가능한 fold 없음)")
        return

    print(
        f"  base  WAPE={result['base_wape']:.3f} bias={result['base_bias']:+.3f}"
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
    print("=== event-window 검증 (STORE_EVENT_PRIORS 등록 이벤트): "
          "base(LightGBM) vs EventLevelPrior 블렌드 ===")
    print(f"(leave-past-out, train window={WINDOW_DAYS}d, alpha={ALPHA})")

    concerns: list[tuple[str, str, float, float]] = []
    for cd_code, store_id, label in STORES:
        cfg = STORE_EVENT_PRIORS[label]
        prior_cfg = dict(events=cfg["events"], lunar_events=cfg["lunar_events"])
        prior_template = EventLevelPrior(**prior_cfg)
        feat = build_feat(cd_code, store_id)
        groups = event_days_by_name(feat, prior_template)
        for event_name, days in groups.items():
            result = evaluate_event(feat, days, prior_cfg)
            print_event_result(label, event_name, result)
            if result is not None and result["blend_wape"] > result["base_wape"]:
                concerns.append((label, event_name, result["base_wape"], result["blend_wape"]))

    if concerns:
        print("\n!!! CONCERNS: blend WAPE > base WAPE for registered event(s):")
        for label, event_name, base_wape, blend_wape in concerns:
            print(f"    {label}/{event_name}: base={base_wape:.3f} blend={blend_wape:.3f}")
    else:
        print("\n모든 등록 이벤트 순개선 확인 (blend WAPE <= base WAPE).")


if __name__ == "__main__":
    main()

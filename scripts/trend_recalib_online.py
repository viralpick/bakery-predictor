"""추세 recalibration 레버(c) — 온라인 레벨(잔차) 보정.

730d 창은 그대로 두고, OOS 예측을 시간순으로 훑으며 **직전 M주의 실현 편향**으로
현재 주 expected 레벨을 곱셈 보정한다. 이전 fold들의 actual은 모두 test_start 이전
= leakage-safe. LGBM 트리 밖에서 곱하므로 외삽 한계(레버 b가 죽은 이유)를 우회한다.

보정 계수 b_t = Σ_{직전 M fold} actual / Σ_{직전 M fold} expected  (clip [lo,hi]).
- 하락기: 모델 과대예측(exp>act) → b<1 → expected를 끌어내림.
- 안정기: bias≈0 → b≈1 → 무해.

baseline(730d, 무보정) vs 보정(M 스윕)을 연도별 WPE·WAPE로 비교.

실행: PYTHONPATH=scripts uv run --with matplotlib python scripts/trend_recalib_online.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

import numpy as np
import pandas as pd

from store_predictive_power import (
    build_store_data,
    windowed_backtest,
    STORE_EVENT_PRIORS,
    DEFAULT_WINDOW_DAYS,
    TARGET,
)
from trend_recalib_diagnose import year_metrics, N_FOLDS, OOS_YEARS

CD_CODE, STORE_ID, LABEL = "1000000047", "store_gw01", "광교"
RECENT_M_WEEKS = (4, 8, 12)  # 편향 추정에 쓸 직전 fold(주) 수
CLIP = (0.80, 1.20)  # 과보정 방지
OUT_CSV = Path("reports/trend_recalib_online_by_year.csv")
OUT_MONTHLY = Path("reports/trend_recalib_monthly_wpe.csv")
# baseline 730d preds 캐시 (160-fold 재적합=유일한 비싼 단계 → 저장 후 월별/폐기 분석 무료)
BASE_PREDS_PARQUET = Path("reports/trend_recalib_base_preds.parquet")


def waste_stockout(preds: pd.DataFrame, label: str) -> dict:
    """발주(production, q0.85) 관점 폐기·매진 (헌장 1차 KPI=폐기율).

    online 보정은 expected뿐 아니라 production에도 같은 계수 b_t를 곱해 적용
    (레벨 보정이므로 분위수 발주도 같이 내려가야 정합). 컬럼 'b' 있으면 사용.
    """
    p = preds.copy()
    prod = p["production"].to_numpy()
    if "b" in p.columns:  # online 변형: production도 동일 계수로 보정
        prod = prod * p["b"].to_numpy()
    act = p["actual"].to_numpy()
    denom = max(act.sum(), 1.0)
    surplus = np.clip(prod - act, 0, None)
    short = np.clip(act - prod, 0, None)
    return dict(
        variant=label,
        waste_units=float(surplus.sum()),
        waste_rate=float(surplus.sum() / denom),
        stockout_rate=float((prod < act).mean()),
        shortfall_units=float(short.sum()),
    )


def monthly_wpe_spread(preds: pd.DataFrame, label: str) -> pd.DataFrame:
    """월별 WPE + spread 지표. 연간 aggregate가 상쇄로 숨기는 시기집중 편향을 드러낸다.

    spread = mean|monthly WPE| · std(monthly WPE). 연평균이 작아도 spread가 크면
    편향이 특정 월에 몰려있다는 뜻 → 온라인 보정이 실제로 할 일이 있다.
    """
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["ym"] = p["date"].dt.to_period("M")
    rows = []
    for ym, g in p.groupby("ym"):
        act, exp = g["actual"].to_numpy(), g["expected"].to_numpy()
        rows.append(dict(ym=str(ym), wpe=float((exp - act).sum() / max(np.abs(act).sum(), 1.0))))
    m = pd.DataFrame(rows)
    m["variant"] = label
    return m


def online_bias_correct(preds: pd.DataFrame, m_weeks: int) -> pd.DataFrame:
    """fold(주) 단위 시간순 곱셈 보정. 직전 m_weeks fold의 Σactual/Σexpected.

    fold는 windowed_backtest에서 1주 블록. leakage-safe: fold t는 t 이전 fold만 참조.
    """
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    # fold 대표일(시작일)로 시간순 정렬
    fold_order = (
        p.groupby("fold")["date"].min().sort_values().reset_index()
    )
    ordered_folds = fold_order["fold"].tolist()

    agg = p.groupby("fold").agg(
        act_sum=("actual", "sum"), exp_sum=("expected", "sum")
    )
    corrected = {}
    for idx, fold in enumerate(ordered_folds):
        prior = ordered_folds[max(0, idx - m_weeks):idx]
        if not prior:
            corrected[fold] = 1.0
            continue
        exp_sum = agg.loc[prior, "exp_sum"].sum()
        act_sum = agg.loc[prior, "act_sum"].sum()
        b = act_sum / exp_sum if exp_sum > 0 else 1.0
        corrected[fold] = float(np.clip(b, *CLIP))

    p["b"] = p["fold"].map(corrected)
    p["expected"] = p["expected"] * p["b"]
    return p


def run() -> None:
    print(f"[{LABEL}] build_store_data ...")
    sd = build_store_data(CD_CODE, STORE_ID, LABEL)
    cfg = STORE_EVENT_PRIORS[LABEL]

    if BASE_PREDS_PARQUET.exists():
        print(f"[{LABEL}] baseline preds 캐시 로드 {BASE_PREDS_PARQUET} (백테스트 skip) ...")
        base_preds = pd.read_parquet(BASE_PREDS_PARQUET)
    else:
        print(f"[{LABEL}] windowed_backtest 730d n_folds={N_FOLDS} (baseline) ...")
        res = windowed_backtest(
            sd.feat,
            window_days=DEFAULT_WINDOW_DAYS,
            n_folds=N_FOLDS,
            target_col=TARGET,
            events=cfg.get("events"),
            lunar_events=cfg.get("lunar_events"),
        )
        base_preds = res.predictions
        BASE_PREDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        base_preds.to_parquet(BASE_PREDS_PARQUET, index=False)

    variants = {"baseline(730d,무보정)": base_preds}
    for m in RECENT_M_WEEKS:
        variants[f"online_M{m}"] = online_bias_correct(base_preds, m)

    rows = []
    for name, preds in variants.items():
        ym = year_metrics(preds)
        ym["variant"] = name
        rows.append(ym)
    result = pd.concat(rows, ignore_index=True)
    result = result[result["year"].isin(OOS_YEARS)].copy()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    _print_matrix(result, "WPE (편향, +=과대예측)", "wpe")
    _print_matrix(result, "WAPE (정확도)", "wape")
    print(f"\n저장: {OUT_CSV}")

    # 월별 WPE spread — 연간 aggregate가 상쇄로 숨기는 시기집중 편향 판정 (advisor)
    monthly = pd.concat(
        [monthly_wpe_spread(base_preds, "baseline"),
         monthly_wpe_spread(variants["online_M8"], "online_M8")],
        ignore_index=True,
    )
    monthly.to_csv(OUT_MONTHLY, index=False)
    print("\n=== 월별 WPE spread (baseline vs online_M8) ===")
    for label, g in monthly.groupby("variant"):
        w = g["wpe"].to_numpy() * 100
        print(f"  {label:12s}: mean|WPE|={np.abs(w).mean():.2f}%  std={w.std():.2f}%  "
              f"min={w.min():+.2f}%  max={w.max():+.2f}%  worst|·|={np.abs(w).max():.2f}%")
    print(f"저장: {OUT_MONTHLY}")

    # 폐기·매진 (헌장 1차 KPI) — 채택 기준은 WPE가 아니라 폐기 (advisor)
    ws = pd.DataFrame([waste_stockout(base_preds, "baseline")]
                      + [waste_stockout(variants[f"online_M{m}"], f"online_M{m}")
                         for m in RECENT_M_WEEKS])
    print("\n=== 폐기·매진 (production=q0.85, 3년 OOS) ===")
    print(ws.assign(
        waste_rate=lambda d: (d["waste_rate"] * 100).round(2),
        stockout_rate=lambda d: (d["stockout_rate"] * 100).round(2),
    ).to_string(index=False))

    # iso-waste 비교 (advisor 최종 판정): baseline production을 상수배해 online_M8과
    # 같은 폐기율로 맞춘 뒤 매진율 비교. 시간가변 b_t가 단순 발주 하향보다 나은가?
    _iso_waste_compare(base_preds, variants["online_M8"])


def _iso_waste_compare(base_preds: pd.DataFrame, online_preds: pd.DataFrame) -> None:
    target_waste = waste_stockout(online_preds, "online_M8")["waste_rate"]
    act = base_preds["actual"].to_numpy()
    prod = base_preds["production"].to_numpy()
    denom = max(act.sum(), 1.0)

    def waste_at(c: float) -> float:
        return float(np.clip(c * prod - act, 0, None).sum() / denom)

    lo, hi = 0.5, 1.0  # 상수배 c (폐기율 단조증가)
    for _ in range(40):
        mid = (lo + hi) / 2
        if waste_at(mid) < target_waste:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    scaled = c * prod
    so_rate = float((scaled < act).mean())
    online_so = waste_stockout(online_preds, "online_M8")["stockout_rate"]
    print(f"\n=== iso-waste 비교 (폐기율 {target_waste*100:.2f}%로 통일) ===")
    print(f"  baseline×{c:.4f} (상수 발주하향): 매진율 {so_rate*100:.2f}%")
    print(f"  online_M8 (시간가변 보정)     : 매진율 {online_so*100:.2f}%")
    verdict = ("시간가변 보정 우위(파레토)" if so_rate - online_so > 0.005
               else "상수하향 우위(트랙1 무가치)" if online_so - so_rate > 0.005
               else "무차별(트랙1 순가치≈0)")
    print(f"  → {verdict}")


def _print_matrix(df: pd.DataFrame, title: str, col: str) -> None:
    pivot = df.pivot(index="variant", columns="year", values=col) * 100
    print(f"\n=== {title} (%) — 행=variant, 열=연도 ===")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    run()

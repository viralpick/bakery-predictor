"""트랙3 진단 — 주말·여름 계절 과소예측이 fixed 모델에도 남아있는가.

배경: 이상치 분석(렌즈③④)이 "발주부족 8일 전부 급등·주말/여름 집중, 2024 여름 매진 집중"을
신호로 냈으나, 그 preds는 **broken-feature 구모델**(add_holiday_features가 calendar_raw 읽어
2021-23 공휴일 blind) 산출물. 트랙1이 target 재정의로 편향을 흡수해 drop된 것과 같은 구조 위험.

→ 이 스크립트는 **fixed 모델 fresh 백테스트**에서 계절 결손을 2레이어로 재확인한다:
  - 예측층: WPE by dow/month/season (expected 편향, 음수=과소예측)
  - 발주층: shortfall rate = P(actual>production) by dow/month (q0.85 버퍼가 뚫린 전체 매진)
  - 체계적 계절 under-level vs 산발 spike 분리 (spike는 마진으로 못 막고 폐기만 늚)
  - 메커니즘: feature importance + residual-by-dow (트리가 dow×month를 실제 학습했나)

규율: dow×month=84셀·셀당 ~13관측 → 셀 줍기 금지. 사전등록 3대비만 검정(주말/여름월/주말×여름),
      각 대비는 day-level 부트스트랩 CI로 판정(CI가 0 배제해야 real).

WPE 부호: (expected−actual)/Σ|actual|. 음수 = 과소예측(under). 발주부족과 같은 방향.

실행: PYTHONPATH=scripts uv run python scripts/track3_seasonal_diagnose.py
      (fresh preds 캐시 없으면 백테스트 실행, 있으면 재사용. --refresh 로 강제 재생성)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from store_predictive_power import (
    build_store_data, windowed_backtest, STORE_EVENT_PRIORS,
    DEFAULT_WINDOW_DAYS, HORIZON, TARGET,
)
from bakery.models.category_total import fit_category_total, select_feature_cols
from bakery.models.event_prior import EventLevelPrior

CD_CODE, STORE_ID, LABEL = "1000000047", "store_gw01", "광교"
YEARS = 3
N_FOLDS = 160                       # ≈3.07년 OOS (아래서 3년 컷) — anomaly 스크립트와 동일 span
SUMMER_MONTHS = (6, 7, 8)           # 사전등록: 여름
WEEKEND_DOW = (5, 6)               # 토·일
SPIKE_Z = 3.0                       # robust z 이상 = 산발 spike (체계적 성분과 분리)
N_BOOT = 2000
CACHE = Path("reports/track3_fresh_preds.parquet")
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]
SEED = 42


def _fresh_preds() -> pd.DataFrame:
    """fixed feature 파이프라인으로 3년 OOS preds 생성 (date/actual/expected/production)."""
    sd = build_store_data(CD_CODE, STORE_ID, LABEL)
    cfg = STORE_EVENT_PRIORS.get(LABEL, {})
    res = windowed_backtest(
        sd.feat, window_days=DEFAULT_WINDOW_DAYS, n_folds=N_FOLDS,
        horizon_days=HORIZON, events=cfg.get("events"), lunar_events=cfg.get("lunar_events"),
    )
    p = res.predictions.copy()
    p["date"] = pd.to_datetime(p["date"])
    cutoff = p["date"].max() - pd.DateOffset(years=YEARS)
    p = p[p["date"] > cutoff].sort_values("date").reset_index(drop=True)
    return p


def _load_or_build(refresh: bool) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        print(f"[캐시 재사용] {CACHE}")
        return pd.read_parquet(CACHE)
    print("[fresh 백테스트] fixed feature로 3년 OOS 재생성 (수 분 소요) ...")
    p = _fresh_preds()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(CACHE, index=False)
    print(f"[캐시 저장] {CACHE}  ({len(p)}일)")
    return p


def _enrich(p: pd.DataFrame) -> pd.DataFrame:
    d = p.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["is_weekend"] = d["dow"].isin(WEEKEND_DOW)
    d["is_summer"] = d["month"].isin(SUMMER_MONTHS)
    d["resid"] = d["actual"] - d["expected"]                 # 음수 = expected 과대(actual 낮음)
    d["shortfall"] = (d["actual"] > d["production"]).astype(int)  # 발주 뚫린 매진
    med = np.median(d["resid"])
    mad = np.median(np.abs(d["resid"] - med)) or d["resid"].std()
    d["resid_z"] = 0.6745 * (d["resid"] - med) / mad if mad else 0.0
    return d


def _wpe(sub: pd.DataFrame) -> float:
    """WPE = Σ(expected−actual)/Σ|actual|. 음수 = 과소예측(under)."""
    denom = sub["actual"].abs().sum()
    return float((sub["expected"] - sub["actual"]).sum() / denom * 100) if denom else 0.0


def _boot_wpe_contrast(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator) -> tuple:
    """WPE(a) − WPE(b) 의 day-level 부트스트랩 95% CI. a=관심세그먼트(주말/여름)."""
    diffs = np.empty(N_BOOT)
    a_idx, b_idx = a.index.to_numpy(), b.index.to_numpy()
    for i in range(N_BOOT):
        ra = a.loc[rng.choice(a_idx, len(a_idx), replace=True)]
        rb = b.loc[rng.choice(b_idx, len(b_idx), replace=True)]
        diffs[i] = _wpe(ra) - _wpe(rb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return _wpe(a) - _wpe(b), float(lo), float(hi)


def _print_by_group(d: pd.DataFrame, col: str, labels: list[str] | None) -> None:
    rows = []
    for key, sub in d.groupby(col):
        name = labels[int(key)] if labels else str(key)
        rows.append(dict(
            그룹=name, n=len(sub), WPE_pct=round(_wpe(sub), 2),
            shortfall_pct=round(sub["shortfall"].mean() * 100, 1),
        ))
    print(pd.DataFrame(rows).to_string(index=False))


def _contrast(d: pd.DataFrame, mask_col: str, name: str, rng: np.random.Generator) -> None:
    seg = d[d[mask_col]]
    rest = d[~d[mask_col]]
    diff, lo, hi = _boot_wpe_contrast(seg, rest, rng)
    verdict = "★ CI가 0 배제 (real)" if (lo > 0 or hi < 0) else "0 포함 (noise)"
    print(f"\n[{name}] WPE {name}={_wpe(seg):.2f}% vs 나머지={_wpe(rest):.2f}%  "
          f"diff={diff:+.2f}pp  95%CI=[{lo:+.2f}, {hi:+.2f}]  → {verdict}")
    print(f"   shortfall: {name}={seg['shortfall'].mean()*100:.1f}% vs 나머지={rest['shortfall'].mean()*100:.1f}%")


def _spike_split(d: pd.DataFrame, mask_col: str, name: str) -> None:
    """세그먼트 under-bias 중 산발 spike(|z|≥3) 제거 후 체계적 성분이 남는가."""
    seg = d[d[mask_col]]
    wpe_all = _wpe(seg)
    seg_no_spike = seg[seg["resid_z"].abs() < SPIKE_Z]
    wpe_calm = _wpe(seg_no_spike)
    n_spike = len(seg) - len(seg_no_spike)
    print(f"   {name} spike분리: 전체 WPE={wpe_all:.2f}% → spike({n_spike}일) 제거 후={wpe_calm:.2f}%  "
          f"(체계적 성분 = spike 제거 후 값)")


def _mechanism(rng_seed: int) -> None:
    """마지막 fold의 fitted expected 모델 feature importance — dow/month가 실제 쓰였나."""
    sd = build_store_data(CD_CODE, STORE_ID, LABEL)
    df = sd.feat.sort_values("date").dropna().reset_index(drop=True)
    train = df[df["date"] >= df["date"].max() - pd.Timedelta(days=DEFAULT_WINDOW_DAYS)]
    model = fit_category_total(train, target_col=TARGET, alpha_demand=0.8, production_q=0.85)
    cols = model.feature_cols
    imp = pd.Series(model.expected.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n=== 메커니즘: expected 모델 feature importance (마지막 fold train) ===")
    print(imp.head(15).to_string())
    cal_cols = [c for c in cols if c in ("dow", "month", "dom", "is_weekend",
                                          "dow_sin", "dow_cos", "month_sin", "month_cos")]
    print(f"\n달력 feature importance 합={imp[cal_cols].sum():.0f} / 전체={imp.sum():.0f} "
          f"({imp[cal_cols].sum()/imp.sum()*100:.1f}%)")


def main() -> None:
    refresh = "--refresh" in sys.argv
    p = _load_or_build(refresh)
    d = _enrich(p)
    rng = np.random.default_rng(SEED)

    span = f"{d['date'].min().date()} ~ {d['date'].max().date()}"
    print(f"\n[광교 3년 OOS fixed 모델] {len(d)}일 ({span})  전체 WPE={_wpe(d):.2f}%  "
          f"전체 shortfall={d['shortfall'].mean()*100:.1f}%")

    print("\n=== 예측층/발주층 by 요일 ===")
    _print_by_group(d, "dow", DOW_KR)
    print("\n=== 예측층/발주층 by 월 ===")
    _print_by_group(d, "month", None)

    print("\n=== 사전등록 3대비 (부트스트랩 95%CI, 셀 줍기 금지) ===")
    _contrast(d, "is_weekend", "주말", rng)
    _spike_split(d, "is_weekend", "주말")
    _contrast(d, "is_summer", "여름(6-8월)", rng)
    _spike_split(d, "is_summer", "여름(6-8월)")
    d["is_weekend_summer"] = d["is_weekend"] & d["is_summer"]
    _contrast(d, "is_weekend_summer", "주말×여름", rng)
    _spike_split(d, "is_weekend_summer", "주말×여름")

    _mechanism(SEED)

    print("\n※ WPE 음수=과소예측(발주부족 방향). shortfall=q0.85 버퍼 뚫린 전체매진 비율.")
    print("※ 게이트1 판정: 사전등록 대비 중 CI가 0을 배제하며 spike 제거 후에도 체계적 under가 남는가.")


if __name__ == "__main__":
    main()

"""실제 레버 검증 — 평일(월·수) center 과대예측 트림이 전역 균일 하향을 iso-waste에서 이기는가.

진단(캐시 track3_fresh_preds): expected가 월 −2.84%·수 −3.58%(rel mean) 3년 일관 과대예측
(원인=rolling/ewma가 dow-blind, 주말 +30% 고수요를 평일로 흘림). distributional/σ(x) branch는
dow별 잔차 std 평평(8.5~10.9%)으로 사망 → 남은 레버 = center 편향 제거(KPI waste 1차).

★게이트(advisor + 트랙1 교훈): post-hoc dow 보정이 "전역 균일 하향"을 iso-waste에서 이겨야
  fix 투자 정당. 트랙1 online 시간보정이 상수하향에 진 전례 → dow 곱셈도 검증 필수.

공정한 A/B (둘 다 발주=expected×mult, 캐시 재사용, 모델 재학습 없음):
  - GLOBAL : order = exp × (1+g),                g로 waste 타겟 (전역 균일)
  - DOW    : order = exp × (1+b+e·1[월∪수]),      e=월·수 트림(고정 grid), b로 waste 타겟
    (월·수 과대분을 트림하고 b로 전역 보충 → 같은 waste, center 편향만 제거된 구조)

판정: 동일 waste에서 DOW 매진(sf_freq/mag)이 GLOBAL보다 낮고(gap<0) 주 블록 부트스트랩 CI가
  0을 배제하면 DOW 우위 = center 보정 가치 있음 → 모델fix 투자. CI 0포함/양수면 drop.
안정 요일만: 월·수만 트림(목·금·토·일은 연도별 부호 뒤집혀 노이즈, 트랙2 err25 규율).

실행: PYTHONPATH=scripts uv run python scripts/weekday_bias_isowaste.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("reports/track3_fresh_preds.parquet")
W_TARGETS = (0.06, 0.08, 0.10)       # waste 타겟 (base expected waste=4.8%)
E_TRIMS = (0.02, 0.03, 0.04)         # 월·수 트림 grid (관측 bias ~3%)
N_BOOT = 2000
SEED = 42


def _load() -> pd.DataFrame:
    d = pd.read_parquet(CACHE)
    d["date"] = pd.to_datetime(d["date"])
    d["is_monwed"] = d["date"].dt.dayofweek.isin((0, 2))   # 월=0, 수=2
    d["week"] = d["date"].dt.isocalendar().week.astype(int) + d["date"].dt.year * 100
    return d


def _order(exp: np.ndarray, base: float, trim: float, monwed: np.ndarray) -> np.ndarray:
    """order = exp × (1 + base − trim·1[월·수]). trim>0 = 월·수 발주 삭감."""
    return exp * (1.0 + base - trim * monwed)


def _waste(order: np.ndarray, actual: np.ndarray) -> float:
    denom = actual.sum()
    return float(np.maximum(order - actual, 0).sum() / denom) if denom else 0.0


def _sf_freq(order: np.ndarray, actual: np.ndarray) -> float:
    return float((actual > order).mean())


def _sf_mag(order: np.ndarray, actual: np.ndarray) -> float:
    denom = actual.sum()
    return float(np.maximum(actual - order, 0).sum() / denom) if denom else 0.0


def _bisect(waste_of, w_target: float) -> float:
    lo, hi = -0.5, 6.0
    if waste_of(hi) < w_target:
        return hi
    for _ in range(45):
        mid = (lo + hi) / 2
        if waste_of(mid) < w_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _gap(d: pd.DataFrame, w_target: float, trim: float) -> tuple[float, float]:
    """iso-waste에서 DOW−GLOBAL 매진(빈도·크기). 음수=DOW 우위(매진 적음)."""
    exp, actual, mw = d["expected"].to_numpy(), d["actual"].to_numpy(), d["is_monwed"].to_numpy()
    g = _bisect(lambda b: _waste(_order(exp, b, 0.0, mw), actual), w_target)      # 전역 균일
    b = _bisect(lambda bb: _waste(_order(exp, bb, trim, mw), actual), w_target)   # 월·수 트림+보충
    og, od = _order(exp, g, 0.0, mw), _order(exp, b, trim, mw)
    return _sf_freq(od, actual) - _sf_freq(og, actual), _sf_mag(od, actual) - _sf_mag(og, actual)


def _boot_ci(d: pd.DataFrame, w_target: float, trim: float, rng: np.random.Generator) -> dict:
    weeks = d["week"].unique()
    groups = {w: d[d["week"] == w] for w in weeks}
    fg, mg = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(weeks, len(weeks), replace=True)
        rs = pd.concat([groups[w] for w in pick], ignore_index=True)
        fg[i], mg[i] = _gap(rs, w_target, trim)
    return dict(freq=np.percentile(fg, [2.5, 50, 97.5]), mag=np.percentile(mg, [2.5, 50, 97.5]))


def main() -> None:
    d = _load()
    rng = np.random.default_rng(SEED)
    exp, actual = d["expected"].to_numpy(), d["actual"].to_numpy()
    print(f"[광교 3년 OOS] {len(d)}일 · 월·수 {d['is_monwed'].mean()*100:.1f}%  "
          f"base(expected) waste={_waste(exp, actual)*100:.1f}%")
    print("판정: 동일 waste에서 DOW(월·수 트림)−GLOBAL 매진 gap; 음수+CI0배제=DOW 우위=center보정 가치")

    for w in W_TARGETS:
        print(f"\n=== waste 타겟 {w*100:.0f}% ===")
        for trim in E_TRIMS:
            gf, gm = _gap(d, w, trim)
            ci = _boot_ci(d, w, trim, rng)
            f_lo, _, f_hi = ci["freq"]
            v = "★DOW 우위" if f_hi < 0 else ("GLOBAL 우위" if f_lo > 0 else "0포함(무차)")
            print(f"  트림 {trim*100:.0f}%: 매진빈도 gap={gf*100:+.2f}pp CI=[{f_lo*100:+.2f},{f_hi*100:+.2f}] {v}"
                  f"  |  매진크기 gap={gm*100:+.2f}pp")

    print("\n※ gap 음수=DOW(월·수 트림)가 전역 균일보다 매진 적음(우위). CI 0포함/양수면 center 보정 무가치→drop.")


if __name__ == "__main__":
    main()

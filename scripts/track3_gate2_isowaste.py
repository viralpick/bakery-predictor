"""트랙3 게이트2 — 계절(주말·여름) 마진이 전역 마진을 iso-waste에서 이기는가.

게이트1 결과: expected는 주말·여름에 무편향(계절 과소예측 premise 소멸). 생존 신호 =
발주층 매진빈도가 주말·여름에 초과. 계절 마진이 이 신호를 정당하게 겨냥하는지 판정.
(주: 세그먼트 잔차분산은 평평 — 초과는 이분산이 아니라 평일 과대예측의 뒷면. docs/track3_seasonal_result.md)

★게이트(advisor + project_margin_buffer_optimization): 계절 마진이 "전역 마진 한 단계 상향"보다
  **동일 폐기율(iso-waste)에서 매진을 더 줄이고, OOS 부트스트랩 CI가 0을 배제**해야 채택.
  못 이기면 drop(정률 마진은 OOS 과적합 — CI 0 포함이 기본 예상).

★공정한 A/B (증분 비교): 공통 base 마진 b(전역, 폐기율 w_base 달성)에서 출발해, 추가 폐기를
  전역으로 쓸지 주말·여름에만 쓸지 비교. "타깃일에만 buffer, 비타깃일 0"은 strawman(비타깃일이
  median에 방치돼 매진 폭발) → 반드시 공통 base 위에서 증분만 비교.
  - GLOBAL : order = expected × (1+g),           g로 w_target 달성 (전 요일 균일 상향)
  - SEASONAL: order = expected × (1+b+e·1[타깃]), 공통 base b + 타깃일 추가 e로 w_target 달성

지표(measurement charter): waste% = Σmax(order−actual,0)/Σactual (폐기, 1차 KPI),
  stockout_freq = P(actual>order) (전체매진 빈도), stockout_mag = Σmax(actual−order,0)/Σactual.
  actual = adjusted_demand (매진일 실수요 하한 → 매진은 하한, 폐기는 상한).

실행: PYTHONPATH=scripts uv run python scripts/track3_gate2_isowaste.py
      (track3_fresh_preds.parquet 필요 — track3_seasonal_diagnose.py가 생성)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("reports/track3_fresh_preds.parquet")
SUMMER_MONTHS = (6, 7, 8)
WEEKEND_DOW = (5, 6)
W_BASE = 0.06                        # 공통 base 폐기율 (여기서부터 증분을 전역 vs 계절 비교)
WASTE_TARGETS = (0.08, 0.10)         # 증분 목표 폐기율 (> W_BASE)
N_BOOT = 2000
SEED = 42


def _load() -> pd.DataFrame:
    d = pd.read_parquet(CACHE)
    d["date"] = pd.to_datetime(d["date"])
    dt = d["date"].dt
    d["is_target"] = dt.dayofweek.isin(WEEKEND_DOW) | dt.month.isin(SUMMER_MONTHS)
    d["week"] = dt.isocalendar().week.astype(int) + dt.year * 100  # 블록 부트스트랩 키
    return d


def _order(exp: np.ndarray, base: float, extra: float, target: np.ndarray) -> np.ndarray:
    """order = exp × (1 + base + extra·1[타깃]). extra=0이면 전역 균일 마진."""
    return exp * (1.0 + base + extra * target)


def _waste(order: np.ndarray, actual: np.ndarray) -> float:
    denom = actual.sum()
    return float(np.maximum(order - actual, 0).sum() / denom) if denom else 0.0


def _stockout_freq(order: np.ndarray, actual: np.ndarray) -> float:
    return float((actual > order).mean())


def _stockout_mag(order: np.ndarray, actual: np.ndarray) -> float:
    denom = actual.sum()
    return float(np.maximum(actual - order, 0).sum() / denom) if denom else 0.0


def _bisect(waste_of, waste_target: float) -> float:
    """waste_of(x)=waste_target 되는 x (waste는 x에 단조증가 → bisection). 도달 불가면 상한."""
    lo, hi = 0.0, 6.0
    if waste_of(hi) < waste_target:
        return hi
    for _ in range(45):
        mid = (lo + hi) / 2
        if waste_of(mid) < waste_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _global_frontier(d: pd.DataFrame, margins: np.ndarray) -> pd.DataFrame:
    exp, actual, target = d["expected"].to_numpy(), d["actual"].to_numpy(), d["is_target"].to_numpy()
    rows = []
    for m in margins:
        order = _order(exp, m, 0.0, target)
        rows.append(dict(margin=m, waste=_waste(order, actual),
                         sf_freq=_stockout_freq(order, actual), sf_mag=_stockout_mag(order, actual)))
    return pd.DataFrame(rows)


def _isowaste_gap(d: pd.DataFrame, w_base: float, w_target: float) -> tuple[float, float]:
    """공통 base(w_base 전역) 위에서 증분을 전역 vs 계절로 쓸 때 SEASONAL−GLOBAL 매진(빈도·크기).
    음수 = seasonal이 매진 더 낮음(우위)."""
    exp, actual, target = d["expected"].to_numpy(), d["actual"].to_numpy(), d["is_target"].to_numpy()
    base = _bisect(lambda b: _waste(_order(exp, b, 0.0, target), actual), w_base)   # 공통 base
    g = _bisect(lambda m: _waste(_order(exp, m, 0.0, target), actual), w_target)     # 전역 상향
    e = _bisect(lambda x: _waste(_order(exp, base, x, target), actual), w_target)    # base+타깃 추가
    order_g = _order(exp, g, 0.0, target)
    order_s = _order(exp, base, e, target)
    freq_gap = _stockout_freq(order_s, actual) - _stockout_freq(order_g, actual)
    mag_gap = _stockout_mag(order_s, actual) - _stockout_mag(order_g, actual)
    return freq_gap, mag_gap


def _boot_ci(d: pd.DataFrame, w_base: float, w_target: float, rng: np.random.Generator) -> dict:
    weeks = d["week"].unique()
    groups = {w: d[d["week"] == w] for w in weeks}
    freq_gaps, mag_gaps = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(weeks, len(weeks), replace=True)
        rs = pd.concat([groups[w] for w in pick], ignore_index=True)
        freq_gaps[i], mag_gaps[i] = _isowaste_gap(rs, w_base, w_target)
    return dict(
        freq=np.percentile(freq_gaps, [2.5, 50, 97.5]),
        mag=np.percentile(mag_gaps, [2.5, 50, 97.5]),
    )


def main() -> None:
    d = _load()
    rng = np.random.default_rng(SEED)
    print(f"[광교 3년 OOS] {len(d)}일 · 타깃(주말∪여름) {d['is_target'].mean()*100:.1f}%")

    # 참고: base(margin=0=expected 그대로)와 q0.85 production의 폐기/매진
    exp, actual, prod = d["expected"].to_numpy(), d["actual"].to_numpy(), d["production"].to_numpy()
    print(f"  base(expected)  : waste={_waste(exp, actual)*100:.1f}%  "
          f"sf_freq={_stockout_freq(exp, actual)*100:.1f}%  sf_mag={_stockout_mag(exp, actual)*100:.1f}%")
    print(f"  q0.85 production: waste={_waste(prod, actual)*100:.1f}%  "
          f"sf_freq={_stockout_freq(prod, actual)*100:.1f}%  sf_mag={_stockout_mag(prod, actual)*100:.1f}%")

    fg = _global_frontier(d, np.linspace(0, 0.6, 61))
    print(f"\n=== iso-waste 판정: 공통 base 폐기율 {W_BASE*100:.0f}%에서 증분을 전역 vs 주말·여름에 ===")
    print("(SEASONAL−GLOBAL 매진; 음수=seasonal 우위. CI 0배제해야 채택 근거) ")
    for w in WASTE_TARGETS:
        gap_freq, gap_mag = _isowaste_gap(d, W_BASE, w)
        ci = _boot_ci(d, W_BASE, w, rng)
        f_lo, _, f_hi = ci["freq"]
        m_lo, _, m_hi = ci["mag"]
        v_freq = "★seasonal 우위" if f_hi < 0 else ("★global 우위" if f_lo > 0 else "0포함(무차)")
        v_mag = "★seasonal 우위" if m_hi < 0 else ("★global 우위" if m_lo > 0 else "0포함(무차)")
        print(f"\n폐기율 {W_BASE*100:.0f}%→{w*100:.0f}%:")
        print(f"  매진빈도 gap = {gap_freq*100:+.2f}pp  95%CI=[{f_lo*100:+.2f}, {f_hi*100:+.2f}]  → {v_freq}")
        print(f"  매진크기 gap = {gap_mag*100:+.2f}pp  95%CI=[{m_lo*100:+.2f}, {m_hi*100:+.2f}]  → {v_mag}")

    fg.to_csv("reports/track3_frontier_global.csv", index=False)
    print("\nwrote reports/track3_frontier_global.csv")
    print("※ gap 음수 = 계절 마진이 전역보다 매진 낮음(우위). 양수+CI0배제 = 전역이 오히려 우위(계절=열위).")
    print("※ 어느 쪽도 CI 0포함이면 무차 → 과적합 위험 큰 계절 파라미터 추가 정당성 없음 → drop.")


if __name__ == "__main__":
    main()

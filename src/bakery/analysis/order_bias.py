"""발주 편향 진단 — iso-waste에서 요일 트림이 전역 균일 하향을 이기는가.

출처: scripts/weekday_bias_isowaste.py(2026-07-18). 모델을 재학습하지 않고 이미
계산된 OOS 예측(expected/actual)만 재사용한다 — 발주 정책 A/B 껍질이다.

공정 비교 설계: 두 정책 모두 발주 = expected×배수이고, 같은 waste 수준에 도달하도록
base를 이분탐색으로 맞춘 뒤 매진(빈도/크기)만 비교한다.
  GLOBAL : order = expected × (1 + base)
  DOW    : order = expected × (1 + base − trim·1[대상요일])
gap = DOW − GLOBAL. 음수 = DOW 우위(같은 폐기에서 매진이 적다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W_TARGETS: tuple[float, ...] = (0.06, 0.08, 0.10)   # waste 타겟
E_TRIMS: tuple[float, ...] = (0.02, 0.03, 0.04)     # 대상요일 트림 grid
TARGET_DOWS: tuple[int, ...] = (0, 2)               # 월=0, 수=2 (부호 안정 요일만)
N_BOOT = 2000
SEED = 42
_BISECT_ITERATIONS = 45
_BISECT_LOW, _BISECT_HIGH = -0.5, 6.0
_CI_PERCENTILES = (2.5, 50, 97.5)
WINNER_DOW = "DOW 우위"
WINNER_GLOBAL = "GLOBAL 우위"
WINNER_TIE = "0포함(무차)"


def waste_rate_of(order: np.ndarray, actual: np.ndarray) -> float:
    """Σmax(order−actual,0) / Σactual."""
    denom = actual.sum()
    return float(np.maximum(order - actual, 0).sum() / denom) if denom else 0.0


def soldout_freq(order: np.ndarray, actual: np.ndarray) -> float:
    """발주 부족일 비율."""
    return float((actual > order).mean())


def soldout_mag(order: np.ndarray, actual: np.ndarray) -> float:
    """Σmax(actual−order,0) / Σactual."""
    denom = actual.sum()
    return float(np.maximum(actual - order, 0).sum() / denom) if denom else 0.0


def dow_trimmed_order(expected: np.ndarray, base: float, trim: float,
                      is_target_dow: np.ndarray) -> np.ndarray:
    """order = expected × (1 + base − trim·1[대상요일]). trim>0 = 대상요일 삭감."""
    return expected * (1.0 + base - trim * is_target_dow)


def solve_base_for_waste(expected: np.ndarray, actual: np.ndarray,
                         is_target_dow: np.ndarray, *, trim: float,
                         w_target: float) -> float:
    """주어진 trim에서 waste가 w_target이 되는 base를 이분탐색으로 찾는다."""
    def _waste_at(base: float) -> float:
        return waste_rate_of(dow_trimmed_order(expected, base, trim, is_target_dow), actual)

    low, high = _BISECT_LOW, _BISECT_HIGH
    if _waste_at(high) < w_target:
        return high
    for _ in range(_BISECT_ITERATIONS):
        mid = (low + high) / 2
        if _waste_at(mid) < w_target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _arrays(preds: pd.DataFrame, target_dows: tuple[int, ...]):
    expected = preds["expected"].to_numpy()
    actual = preds["actual"].to_numpy()
    is_target = pd.to_datetime(preds["date"]).dt.dayofweek.isin(target_dows).to_numpy()
    return expected, actual, is_target


def isowaste_dow_gap(preds: pd.DataFrame, *, w_target: float, trim: float,
                     target_dows: tuple[int, ...] = TARGET_DOWS) -> tuple[float, float]:
    """iso-waste에서 (DOW − GLOBAL) 매진 빈도·크기 gap. 음수=DOW 우위."""
    expected, actual, is_target = _arrays(preds, target_dows)
    base_global = solve_base_for_waste(expected, actual, is_target, trim=0.0, w_target=w_target)
    base_dow = solve_base_for_waste(expected, actual, is_target, trim=trim, w_target=w_target)
    order_global = dow_trimmed_order(expected, base_global, 0.0, is_target)
    order_dow = dow_trimmed_order(expected, base_dow, trim, is_target)
    return (soldout_freq(order_dow, actual) - soldout_freq(order_global, actual),
            soldout_mag(order_dow, actual) - soldout_mag(order_global, actual))


def bootstrap_gap_ci(preds: pd.DataFrame, *, w_target: float, trim: float,
                     n_boot: int = N_BOOT, seed: int = SEED,
                     target_dows: tuple[int, ...] = TARGET_DOWS) -> dict[str, np.ndarray]:
    """주(week) 블록 부트스트랩 — 요일 구조를 깨지 않으려 주 단위로 리샘플한다."""
    frame = preds.copy()
    dates = pd.to_datetime(frame["date"])
    frame["week"] = dates.dt.isocalendar().week.astype(int) + dates.dt.year * 100
    weeks = frame["week"].unique()
    groups = {week: frame[frame["week"] == week] for week in weeks}
    rng = np.random.default_rng(seed)
    freq_gaps, mag_gaps = np.empty(n_boot), np.empty(n_boot)
    for index in range(n_boot):
        picked = rng.choice(weeks, len(weeks), replace=True)
        resampled = pd.concat([groups[w] for w in picked], ignore_index=True)
        freq_gaps[index], mag_gaps[index] = isowaste_dow_gap(
            resampled, w_target=w_target, trim=trim, target_dows=target_dows)
    return {"freq": np.percentile(freq_gaps, _CI_PERCENTILES),
            "mag": np.percentile(mag_gaps, _CI_PERCENTILES)}


def _winner(ci_low: float, ci_high: float) -> str:
    if ci_high < 0:
        return WINNER_DOW
    if ci_low > 0:
        return WINNER_GLOBAL
    return WINNER_TIE


def isowaste_grid(preds: pd.DataFrame, *, w_targets: tuple[float, ...] = W_TARGETS,
                  trims: tuple[float, ...] = E_TRIMS, n_boot: int = N_BOOT,
                  seed: int = SEED) -> pd.DataFrame:
    """(waste 타겟 × 트림) 격자에서 gap + 부트스트랩 CI + 승자 판정.

    의도된 common-random-numbers: 매 칸마다 같은 `seed`로 bootstrap_gap_ci를 호출하므로
    9칸이 전부 동일한 주(week) 리샘플 draw를 쓴다(출처 scripts/weekday_bias_isowaste.py는
    rng 하나를 9칸에 공유 — 동작이 다르지만 점추정엔 영향 없다). 칸간 CI가 상관되므로
    "칸 하나가 CI 0을 배제"를 여러 칸에 걸쳐 독립 사건처럼 다중비교하면 안 된다.
    """
    rows = []
    for w_target in w_targets:
        for trim in trims:
            gap_freq, gap_mag = isowaste_dow_gap(preds, w_target=w_target, trim=trim)
            ci = bootstrap_gap_ci(preds, w_target=w_target, trim=trim,
                                  n_boot=n_boot, seed=seed)
            low, median, high = ci["freq"]
            rows.append({"w_target": w_target, "trim": trim,
                         "gap_freq": gap_freq, "gap_mag": gap_mag,
                         "freq_ci_low": low, "freq_median": median, "freq_ci_high": high,
                         "winner": _winner(low, high)})
    return pd.DataFrame(rows)

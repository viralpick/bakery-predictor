"""예측 편향 축 진단 프리미티브 — 이미 계산된 OOS preds만 소비한다.

출처: scripts/track3_seasonal_diagnose.py, scripts/track4_weather_diagnose.py.
모델을 실행하지 않는다 — (date, actual, expected, production) 프레임이 전부다.

WPE 부호 규약: (Σexpected − Σactual)/Σ|actual| × 100. 음수 = 과소예측(발주부족 방향).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUMMER_MONTHS: tuple[int, ...] = (6, 7, 8, 9)
WINTER_MONTHS: tuple[int, ...] = (12, 1, 2)
WEEKEND_DOW: tuple[int, ...] = (5, 6)
EXTREME_THRESHOLDS: dict[str, float] = {
    "heatwave_max_ta": 33.0,
    "coldwave_min_ta": -10.0,
    "heavy_rain_mm": 30.0,
}
N_BOOT = 2000
SEED = 42
_CI_PERCENTILES = (2.5, 97.5)


def wpe_percent(preds: pd.DataFrame) -> float:
    denom = preds["actual"].abs().sum()
    if denom == 0:
        return 0.0
    return float((preds["expected"] - preds["actual"]).sum() / denom * 100)


def stockout_rate_percent(preds: pd.DataFrame) -> float:
    """버퍼발주(production)가 뚫린 전체매진 비율 %."""
    if len(preds) == 0:
        return 0.0
    return float((preds["actual"] > preds["production"]).mean() * 100)


def bias_by_axis(preds: pd.DataFrame, axis_column: str) -> pd.DataFrame:
    """축(요일/월/계절/세그먼트)별 WPE + 매진률."""
    rows = []
    for value, group in preds.groupby(axis_column, observed=True):
        rows.append({axis_column: value, "n": int(len(group)),
                     "wpe": wpe_percent(group),
                     "stockout_rate": stockout_rate_percent(group)})
    return pd.DataFrame(rows)


def segment_contrast(preds: pd.DataFrame, mask: pd.Series, *, n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    """세그먼트 vs 여집합 WPE 차이 + day-level 부트스트랩 95% CI."""
    segment, rest = preds[mask], preds[~mask]
    diff = wpe_percent(segment) - wpe_percent(rest)
    rng = np.random.default_rng(seed)
    seg_index, rest_index = segment.index.to_numpy(), rest.index.to_numpy()
    diffs = np.empty(n_boot)
    for index in range(n_boot):
        resampled_seg = segment.loc[rng.choice(seg_index, len(seg_index), replace=True)]
        resampled_rest = rest.loc[rng.choice(rest_index, len(rest_index), replace=True)]
        diffs[index] = wpe_percent(resampled_seg) - wpe_percent(resampled_rest)
    return {"wpe_diff": diff, "ci": np.percentile(diffs, _CI_PERCENTILES),
            "n_segment": int(len(segment)), "n_rest": int(len(rest))}


def is_signal(contrast: dict) -> bool:
    """CI가 0을 배제하면 신호, 포함하면 noise."""
    low, high = contrast["ci"]
    return bool(low > 0 or high < 0)

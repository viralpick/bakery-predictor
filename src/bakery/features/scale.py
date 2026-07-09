"""Leakage-safe per-item 규모 — conformal 마진 정규화용.

scale_i = cutoff 이전 이력의 item별 target 평균(floor 적용). 예측 시점 이후
데이터를 안 보므로 conformal calibration의 normalized score에 안전하게 쓴다.
"""
from __future__ import annotations

import pandas as pd


def compute_item_scale(
    daily: pd.DataFrame,
    before_date,
    y_col: str = "adjusted_demand",
    floor: float = 1.0,
) -> dict[str, float]:
    """item별 scale=max(floor, cutoff 이전 평균). 평균이 NaN(예: 전부 NaN 값)이면
    floor로 대체 — Python max()는 NaN 비교에서 항상 첫 인자를 반환해 NaN이 새어나가는
    문제를 막는다. pre-cutoff 행이 전혀 없는 item은 groupby 결과에서 아예 빠지며
    dict에도 포함되지 않는다 — 호출자는 누락 키를 처리해야 한다
    (downstream `_apply_conformal_to_folds`는 누락 → floor로 fillna 처리)."""
    before = pd.to_datetime(before_date)
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    hist = d[d["date"] < before]
    means = hist.groupby(hist["item_id"].astype(str))[y_col].mean()
    return {
        item: (floor if not (float(m) == float(m)) else max(float(m), floor))
        for item, m in means.items()
    }

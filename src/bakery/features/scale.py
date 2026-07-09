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
    before = pd.to_datetime(before_date)
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    hist = d[d["date"] < before]
    means = hist.groupby(hist["item_id"].astype(str))[y_col].mean()
    return {item: max(float(m), floor) for item, m in means.items()}

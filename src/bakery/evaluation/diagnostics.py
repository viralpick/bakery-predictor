"""복원(potential_demand) 품질 진단.

Decoupling Score: 복원한 수요가 품절률과 여전히 상관되는지 재는 지표.
FreshRetailNet(2505.16319)의 ρ_DS. 미복원이면 '품절→낮은 수요'라는 검열
편향이 강한 음의 상관으로 남고, 잘 복원하면 상관이 사라진다(≈0). 강한
음수 = 복원 부족(=발주 과소 위험). 품목 레벨은 흡수·검열 이중편향으로
식별 불가이므로 **카테고리 레벨 셀에만** 적용한다.
"""
from __future__ import annotations

import numpy as np


def decoupling_score(
    demand: np.ndarray,
    stockout_rate: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """가중 Pearson 상관(품절률, 복원수요). 분산이 0이면 0을 반환."""
    d = np.asarray(demand, dtype=float)
    s = np.asarray(stockout_rate, dtype=float)
    if d.shape != s.shape:
        raise ValueError(f"demand{d.shape} vs stockout_rate{s.shape} shape mismatch")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != d.shape:
            raise ValueError(f"weights{w.shape} vs demand{d.shape} shape mismatch")
    else:
        w = np.ones_like(d)
    wsum = w.sum()
    if wsum <= 0:
        return float("nan")
    dm = d - np.average(d, weights=w)
    sm = s - np.average(s, weights=w)
    cov = np.sum(w * dm * sm)
    var_d = np.sum(w * dm * dm)
    var_s = np.sum(w * sm * sm)
    if var_d <= 0 or var_s <= 0:
        return 0.0
    return float(cov / np.sqrt(var_d * var_s))

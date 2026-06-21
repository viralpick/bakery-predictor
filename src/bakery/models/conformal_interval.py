"""Split conformal prediction intervals with Mondrian (day-of-week) grouping.

⚠️ DEPRECATED (v6, 2026-06): v6 산출물이 *구간(interval) 예측 → 점추정 + 품절/매진
위험 수치*로 전환되면서 이 모듈은 더 이상 v6 경로에 쓰이지 않는다. 위험 수치는
`bakery.decision`(MC 껍질)이 산출한다. 데이터 검증 단계(v5) 산출물로만 보존하며,
코드 삭제는 참조 정리 후 별도 작업으로 진행한다. 배경:
docs/poc_scope_v6.md §2.1, docs/kinetic_layer_fit_analysis.md §3.

The q0.90 production point stays the order anchor (center); this layer wraps it
with calibrated residual margins so measured coverage converges to the nominal
level. Distribution-free: uses empirical residual quantiles, no normality
assumption (bakery demand is right-skewed). Two modes:

- symmetric:  margin δ = Q_dow(|actual − center|); interval = center ± δ.
- asymmetric: lower anchored on an adaptive q_lo model, upper on the q0.90
              center, with independent per-side residual margins.

Calibration residuals must come from a TIME-ORDERED calibration split that sits
strictly between train and test (CLAUDE.md absolute rules #1 and #3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_VALID_MODES = ("symmetric", "asymmetric")


@dataclass
class ConformalInterval:
    mode: str
    coverage: float
    min_group_n: int = 30
    margins_by_dow: dict = field(default_factory=dict)
    pooled_margin: tuple[float, float] = (0.0, 0.0)

    def calibrate(
        self,
        *,
        actual: np.ndarray,
        center_pred: np.ndarray,
        dow: np.ndarray,
        lo_pred: np.ndarray | None = None,
    ) -> "ConformalInterval":
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        actual = np.asarray(actual, dtype=float)
        center = np.asarray(center_pred, dtype=float)
        dow_arr = np.asarray(dow)

        if self.mode == "symmetric":
            score_lo = np.abs(actual - center)
            score_hi = score_lo
        else:  # asymmetric
            if lo_pred is None:
                raise ValueError("asymmetric mode requires lo_pred")
            lo_base = np.asarray(lo_pred, dtype=float)
            score_hi = actual - center  # upper residual r_hi
            score_lo = lo_base - actual  # lower residual r_lo

        self.pooled_margin = (self._q(score_lo), self._q(score_hi))
        margins: dict = {}
        for d in np.unique(dow_arr):
            mask = dow_arr == d
            if int(mask.sum()) < self.min_group_n:
                margins[d] = self.pooled_margin
            else:
                margins[d] = (self._q(score_lo[mask]), self._q(score_hi[mask]))
        self.margins_by_dow = margins
        return self

    def predict_interval(
        self,
        *,
        center_pred: np.ndarray,
        dow: np.ndarray,
        lo_pred: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(center_pred, dtype=float)
        dow_arr = np.asarray(dow)
        m_lo = np.empty(center.shape)
        m_hi = np.empty(center.shape)
        for i, d in enumerate(dow_arr):
            ml, mh = self.margins_by_dow.get(d, self.pooled_margin)
            m_lo[i] = ml
            m_hi[i] = mh

        if self.mode == "symmetric":
            return center - m_lo, center + m_hi
        if lo_pred is None:
            raise ValueError("asymmetric mode requires lo_pred")
        lo_base = np.asarray(lo_pred, dtype=float)
        return lo_base - m_lo, center + m_hi

    def _q(self, scores: np.ndarray) -> float:
        """Finite-sample-corrected empirical quantile at the coverage level."""
        n = len(scores)
        if n == 0:
            return float("nan")
        level = min(1.0, self.coverage * (1 + 1.0 / n))
        return float(np.quantile(scores, level))

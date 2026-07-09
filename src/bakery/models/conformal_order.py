"""One-sided, scale-normalized split-conformal 발주 보정.

base median 예측 위에 잔차 마진을 씌워 목표 서비스레벨 s의 coverage
(P(demand > order) ≈ 1−s)를 달성한다. 잔차는 item scale로 정규화해 pooled로
분위를 구하므로 희소 품목도 강건. path-agnostic — 배열만 받는다.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SERVICE_LEVEL = 0.74  # cost-optimal Cu/(Cu+Co) = 0.85/(0.85+0.30)


class ConformalOrderCalibrator:
    q_s: float

    def fit(self, scores: np.ndarray, service_level: float) -> "ConformalOrderCalibrator":
        """normalized conformity score E=(y−ŷ)/scale 의 s-분위를 저장.

        method="higher"로 유한표본서 약간 보수적(coverage 하한 보호).
        """
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]
        if scores.size == 0:
            raise ValueError("scores is empty after NaN drop")
        self.q_s = float(np.quantile(scores, service_level, method="higher"))
        return self

    def apply(self, base_pred: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """order = clip(base + q_s × scale, 0, None)."""
        base_pred = np.asarray(base_pred, dtype=float)
        scales = np.asarray(scales, dtype=float)
        return np.clip(base_pred + self.q_s * scales, 0.0, None)

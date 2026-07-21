"""category-total 수요의 분포회귀 발주 모델 (NGBoost LogNormal, μ·σ 동시추정).

PoC(docs/distributional_boosting_poc_result.md)서 채택 방향 확정:
LightGBM 독립분위수의 spread 병리(저수요일 과대마진)를 공유-μ 결합(LogNormal)으로 해소.
발주 = 적합 분포의 분위수(기본 q0.85). CategoryTotalModel(LightGBM)과 무손상 공존하며
drop-in alias(predict_expected/predict_production)로 동일 계약을 만족한다.

event_prior 블렌드·conformal 보정은 이 클래스에 넣지 않는다 — 소비자가 씌우는 post-model 레이어.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import LogNormal

from bakery.models.category_total import select_feature_cols


def fit_distributional_total(
    train: pd.DataFrame,
    target_col: str = "adjusted_demand_unit",
    n_estimators: int = 500,
    learning_rate: float = 0.02,
    random_state: int = 42,
) -> "DistributionalTotalModel":
    """NGBoost(Dist=LogNormal)로 μ(x)·σ(x)를 동시 추정.

    LogNormal은 양수 전용 → train target에 y≤0 있으면 ValueError.
    휴무/0 수요일 처리는 호출자 책임(category-total은 구조적 양수).
    Feature는 numeric만 지원(NGBoost는 범주형/object 미허용; LightGBM과 달리).
    """
    feat_cols = select_feature_cols(train, target_col)
    y = train[target_col].to_numpy()
    n_bad = int((~(y > 0)).sum())
    if n_bad:
        raise ValueError(
            f"LogNormal requires positive target; found {n_bad} non-positive/NaN rows in '{target_col}'"
        )
    model = NGBRegressor(
        Dist=LogNormal,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=False,
    ).fit(train[feat_cols].to_numpy(), y)
    return DistributionalTotalModel(model=model, feature_cols=feat_cols, target_col=target_col)


@dataclass
class DistributionalTotalModel:
    model: NGBRegressor
    feature_cols: list[str]
    target_col: str

    def _pred_dist(self, df: pd.DataFrame):
        return self.model.pred_dist(df[self.feature_cols].to_numpy())

    def predict_dist(self, df: pd.DataFrame):
        """적합 LogNormal의 scipy frozen 분포(배열). 임의 분위수·전체 분포용."""
        return self._pred_dist(df).dist

    def predict_quantile(self, df: pd.DataFrame, q: float) -> np.ndarray:
        return np.ravel(self._pred_dist(df).dist.ppf(q))

    def predict_median(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_quantile(df, 0.5)

    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """σ(x) log-space (LogNormal shape). 진단·coupling용."""
        return np.ravel(self._pred_dist(df).params["s"])

    # --- CategoryTotalModel drop-in 호환 alias ---
    # predict_expected = median(점추정), LogNormal 통계적 기댓값(mean)이 아님 —
    # 기존 CategoryTotalModel(L1=median) 계약과 일치시켜 무손상 swap.
    def predict_expected(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_median(df)

    def predict_production(self, df: pd.DataFrame, production_q: float = 0.85) -> np.ndarray:
        """production_q는 call마다 지정되는 인자(기본값 0.85). CategoryTotalModel과 달리 fit 시 고정되지 않음."""
        return self.predict_quantile(df, production_q)

"""Forecaster 어댑터 — fit/predict_production 규약 차이를 격리(windowed_backtest 순수 유지).

category_total(LightGBM)과 distributional_total(NGBoost)은 동일 타깃(빵 총량)을 예측하되
fit 시그니처·production_q 전달 시점·결정성 처리가 다르다. 어댑터가 이를 흡수해 균일한
FittedForecaster 계약(predict_expected/predict_production)을 제공한다.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from bakery.models.category_total import (
    EXPECTED_OBJECTIVE_L1,
    EXPECTED_OBJECTIVE_L2,
    fit_category_total,
)


@runtime_checkable
class FittedForecaster(Protocol):
    def predict_expected(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_production(self, df: pd.DataFrame) -> np.ndarray: ...


class Forecaster(Protocol):
    name: str
    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster: ...


class CategoryTotalForecaster:
    """헤드라인 forecaster. expected 모델 목적함수는 기본 L1(조건부 중앙값)."""

    name = "category_total"
    expected_objective = EXPECTED_OBJECTIVE_L1

    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster:
        # 반환 CategoryTotalModel이 이미 predict_expected/predict_production(q fit-고정) 계약 만족.
        return fit_category_total(
            train, target_col=target_col, alpha_demand=alpha, production_q=production_q,
            expected_objective=self.expected_objective,
        )


class CategoryTotalL2Forecaster(CategoryTotalForecaster):
    """expected 모델만 L2(조건부 평균)로 바꾼 변이 — 목적함수 축 A/B용.

    생산량(quantile) 모델은 동일하다. 나머지 하이퍼파라미터도 동일하므로 두 forecaster를
    한 실험에 태우면 **같은 fold·같은 데이터**에서 목적함수 효과만 분리된다.
    """

    name = "category_total_l2"
    expected_objective = EXPECTED_OBJECTIVE_L2


class _ProdQBound:
    """distributional 모델을 production_q로 바인딩해 균일 계약(predict_production(df)) 제공."""

    def __init__(self, model, production_q: float):
        self._model = model
        self._production_q = production_q

    def predict_expected(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_expected(df)

    def predict_production(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_production(df, production_q=self._production_q)


class DistributionalTotalForecaster:
    name = "distributional_total"

    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster:
        # ngboost는 무거워 lazy import(category 전용 실행 시 회피). alpha는 미사용(균일 인터페이스).
        from bakery.models.distributional_total import fit_distributional_total

        # hermetic seed: NGBoost 0.5.11은 전역 numpy RNG 사용 → random_state 인자만으론 비결정적.
        # save/restore로 전역 스트림 누수 없이 결정성 확보. 시드 42(fit_distributional_total 기본값과 일치).
        state = np.random.get_state()
        np.random.seed(42)
        try:
            model = fit_distributional_total(train, target_col=target_col)
        finally:
            np.random.set_state(state)
        return _ProdQBound(model, production_q)

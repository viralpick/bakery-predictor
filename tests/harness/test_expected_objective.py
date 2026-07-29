"""expected 모델 목적함수 축(L1 vs L2) — 기본값 불변 + 변이 격리 회귀.

이 축을 노출한 이유: 평가 지표(WAPE/WPE)는 **합계 기준**이라 평균성 대상을 가리키는데
L1은 조건부 중앙값을 맞춘다. 그 불일치가 요일별 편향·예측 압축의 후보 원인으로 실측됐다.
테스트는 (a) 헤드라인 기본값이 안 바뀌었는지 (b) 변이가 expected만 건드리는지를 지킨다.
"""
import numpy as np
import pandas as pd
import pytest

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.forecasters import (
    CategoryTotalForecaster,
    CategoryTotalL2Forecaster,
)
from bakery.harness.registry import build_forecaster, is_runnable, kind_of
from bakery.models.category_total import (
    EXPECTED_OBJECTIVE_L1,
    EXPECTED_OBJECTIVE_L2,
    fit_category_total,
)

TARGET = "adjusted_demand_unit"


@pytest.fixture(scope="module")
def train():
    feat = build_features(build_category_daily(alpha=0.8), target_col=TARGET)
    return feat.dropna().reset_index(drop=True).tail(400)


def test_default_objective_is_l1():
    """★헤드라인 불변: 기본 목적함수는 L1이다(엔진 동등성 게이트의 전제)."""
    assert EXPECTED_OBJECTIVE_L1 == "regression_l1"
    assert CategoryTotalForecaster.expected_objective == EXPECTED_OBJECTIVE_L1


def test_fit_default_matches_explicit_l1(train):
    """인자 없이 fit == expected_objective=L1 명시 (기본값 경로 불변)."""
    implicit = fit_category_total(train, target_col=TARGET, production_q=0.85)
    explicit = fit_category_total(train, target_col=TARGET, production_q=0.85,
                                  expected_objective=EXPECTED_OBJECTIVE_L1)
    assert np.allclose(
        implicit.predict_expected(train), explicit.predict_expected(train), rtol=1e-12,
    )


def test_l2_objective_changes_expected_predictions(train):
    """L2는 다른 예측을 낸다 — 인자가 실제로 배선됐는지 확인(무동작 방지)."""
    l1 = fit_category_total(train, target_col=TARGET, production_q=0.85,
                            expected_objective=EXPECTED_OBJECTIVE_L1)
    l2 = fit_category_total(train, target_col=TARGET, production_q=0.85,
                            expected_objective=EXPECTED_OBJECTIVE_L2)
    assert not np.allclose(l1.predict_expected(train), l2.predict_expected(train), rtol=1e-6)


def test_production_model_unaffected_by_objective(train):
    """★생산량(quantile) 모델은 목적함수 축과 무관하다 — 발주 정책이 섞이면 비교가 오염된다."""
    l1 = fit_category_total(train, target_col=TARGET, production_q=0.85,
                            expected_objective=EXPECTED_OBJECTIVE_L1)
    l2 = fit_category_total(train, target_col=TARGET, production_q=0.85,
                            expected_objective=EXPECTED_OBJECTIVE_L2)
    assert np.allclose(
        l1.predict_production(train), l2.predict_production(train), rtol=1e-12,
    )


def test_l2_forecaster_registered_and_runnable():
    assert is_runnable("category_total_l2")
    assert kind_of("category_total_l2") == kind_of("category_total")
    assert isinstance(build_forecaster("category_total_l2"), CategoryTotalL2Forecaster)


def test_l2_forecaster_shares_all_other_hyperparameters(train):
    """변이는 목적함수만 다르다 — forecaster 경로로도 production이 동일해야 한다."""
    fitted_l1 = CategoryTotalForecaster().fit(
        train, target_col=TARGET, alpha=0.8, production_q=0.85,
    )
    fitted_l2 = CategoryTotalL2Forecaster().fit(
        train, target_col=TARGET, alpha=0.8, production_q=0.85,
    )
    assert np.allclose(
        fitted_l1.predict_production(train), fitted_l2.predict_production(train), rtol=1e-12,
    )
    assert not np.allclose(
        fitted_l1.predict_expected(train), fitted_l2.predict_expected(train), rtol=1e-6,
    )


def test_l2_reduces_prediction_compression(train):
    """L2는 조건부 평균을 맞추므로 예측 분산이 L1보다 크다(압축 완화 방향).

    in-sample 방향 확인 — OOS 크기는 실험(docs/expected_objective_result.md)이 잰다.
    """
    l1 = fit_category_total(train, target_col=TARGET, production_q=0.85,
                            expected_objective=EXPECTED_OBJECTIVE_L1)
    l2 = fit_category_total(train, target_col=TARGET, production_q=0.85,
                            expected_objective=EXPECTED_OBJECTIVE_L2)
    actual_std = float(pd.Series(train[TARGET]).std())
    ratio_l1 = actual_std / float(pd.Series(l1.predict_expected(train)).std())
    ratio_l2 = actual_std / float(pd.Series(l2.predict_expected(train)).std())
    assert ratio_l2 < ratio_l1

import pandas as pd
import pytest

from bakery.analysis.lab.handlers.substitution import (
    DID_BETA_THRESHOLD,
    LAMBDA_INDEPENDENCE_THRESHOLD,
    hour_profiles_from_receipts,
    substitution_verdict,
)


def _receipts():
    rows = []
    for hour, count in ((11, 3), (15, 2), (19, 5)):
        for index in range(count):
            rows.append({"receipt_id": f"r{hour}_{index}", "date": pd.Timestamp("2025-01-01"),
                         "item_id": "b1", "hour": hour, "minute": 0, "qty": 1,
                         "timestamp": pd.Timestamp(f"2025-01-01 {hour}:00"), "is_bulk": False})
    return pd.DataFrame(rows)


def _daily():
    return pd.DataFrame([{"store_id": "store_gw01", "item_id": "b1", "category_id": "bread",
                          "date": pd.Timestamp("2025-01-01"), "sold_units": 10,
                          "is_stockout": False, "stockout_time": pd.NaT}])


def test_thresholds():
    assert LAMBDA_INDEPENDENCE_THRESHOLD == 0.95
    assert DID_BETA_THRESHOLD == 0.02


def test_hour_profiles_are_length_24_and_normalized():
    profiles = hour_profiles_from_receipts(_receipts(), _daily())
    assert list(profiles) == ["store_gw01"]
    profile = profiles["store_gw01"]
    assert profile.shape == (24,)
    assert profile.sum() == pytest.approx(1.0)
    # 11시 3/10, 15시 2/10, 19시 5/10
    assert profile[11] == pytest.approx(0.3)
    assert profile[15] == pytest.approx(0.2)
    assert profile[19] == pytest.approx(0.5)
    assert profile[0] == 0.0


def test_verdict_weak_substitution_when_lambda_near_one_and_did_zero():
    verdict = substitution_verdict(rd_mean_outflow=0.03, did_mean_beta=0.001,
                                   nested_lambda_min=0.99)
    assert verdict == (
        "기각(대체 약함) — nested λ_min 0.990(≈1=독립), DiD 평균 β 0.0010, "
        "RD 평균 유출 0.030. 카테고리는 한 묶음 수요로 취급 가능")


def test_verdict_reports_substitution_when_did_material():
    verdict = substitution_verdict(rd_mean_outflow=0.15, did_mean_beta=0.05,
                                   nested_lambda_min=0.60)
    assert verdict == (
        "지지(대체 있음) — nested λ_min 0.600, DiD 평균 β 0.0500(임계 0.02 초과), "
        "RD 평균 유출 0.150")


def test_verdict_inconclusive_when_signals_conflict():
    verdict = substitution_verdict(rd_mean_outflow=0.15, did_mean_beta=0.001,
                                   nested_lambda_min=0.60)
    assert verdict == (
        "불확실 — nested λ_min 0.600은 대체를 시사하나 DiD 평균 β 0.0010은 0에 가깝다 "
        "(RD 평균 유출 0.150)")


@pytest.mark.slow
def test_handler_produces_four_estimator_tables():
    from bakery.analysis.lab.handlers.substitution import substitution
    from bakery.analysis.lab.inputs import AnalysisInputs
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = substitution(inputs)
    assert [label for label, _ in result.tables] == [
        "rd_coefficients", "did_coefficients", "mnl", "nested_lambda", "sensitivity"]

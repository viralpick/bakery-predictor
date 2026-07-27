import pandas as pd
import pytest

from bakery.analysis.lab.handlers.calendar_bias import holiday_premium, premium_verdict
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _dow_class(weekday_median, weekend_median):
    return pd.DataFrame([
        {"dow_class": "평일", "n": 71, "median_lift": weekday_median, "q25": 1.10, "q75": 1.38},
        {"dow_class": "주말", "n": 22, "median_lift": weekend_median, "q25": 0.78, "q75": 1.00},
    ])


def test_verdict_supports_weekday_premium_without_weekend_premium():
    assert premium_verdict(_dow_class(1.25, 0.89)) == (
        "지지 — 평일 공휴일 프리미엄 +25.0%, 주말 공휴일은 −11.0%(프리미엄 없음)")


def test_verdict_reports_weekend_premium_when_present():
    assert premium_verdict(_dow_class(1.25, 1.12)) == (
        "부분 지지 — 평일 +25.0%, 주말도 +12.0%(주말 프리미엄 존재)")


def test_verdict_rejects_when_no_weekday_premium():
    assert premium_verdict(_dow_class(1.01, 0.99)) == (
        "기각 — 평일 공휴일 프리미엄 +1.0%로 미미(임계 5%)")


@pytest.mark.slow
def test_handler_uses_fresh_category_daily_not_frozen_csv():
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = holiday_premium(inputs)
    assert result.kind == KIND_HYPOTHESIS
    from bakery.analysis.lab.registry import HYPOTHESES, load_handlers
    load_handlers()
    assert HYPOTHESES["holiday_premium"].needs_single_store is True
    assert [label for label, _ in result.tables] == [
        "dow_class", "event_ranking", "streak_buckets", "by_holiday"]
    # vintage 분리를 은폐하지 않는다
    assert any("vintage" in note for note in result.notes)

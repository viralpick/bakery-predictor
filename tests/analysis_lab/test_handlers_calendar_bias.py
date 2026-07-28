import pandas as pd
import pytest

from bakery.analysis.lab.handlers.calendar_bias import (
    _NOTE_MONTH_DOW_SOURCE,
    _NOTE_NOT_A_VERDICT,
    holiday_premium,
    month_dow_adjust,
    month_dow_verdict,
    premium_verdict,
)
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


def test_month_dow_verdict_names_worst_cell():
    table = pd.DataFrame([
        {"month": 1, "dow": 0, "raw_mean": 100.0, "adjusted_mean": 92.0,
         "closing_mean": 10.0, "delta": -8.0, "delta_pct": -8.0},
        {"month": 7, "dow": 5, "raw_mean": 200.0, "adjusted_mean": 196.0,
         "closing_mean": 5.0, "delta": -4.0, "delta_pct": -2.0},
    ])
    assert month_dow_verdict(table) == (
        "조정 효과 최대 칸: 1월 월요일 -8.0%, 칸간 편차 6.0%p")


def _category_daily_two_months():
    """월×요일 칸이 1개로 퇴화하지 않도록 2개월치 관측을 담은 stub."""
    dates = list(pd.date_range("2025-01-06", periods=7, freq="D")) + \
        [pd.Timestamp("2025-02-03")]
    return pd.DataFrame({
        "date": dates,
        "sold_total_unit": [100, 110, 120, 130, 140, 200, 190, 300],
        "sold_closing": [10, 10, 10, 10, 10, 20, 20, 30],
        "adjusted_demand_unit": [92, 102, 112, 122, 132, 184, 174, 276],
    })


def test_month_dow_adjust_handler_structure_and_gate(stub_inputs):
    inputs = stub_inputs(category_daily=_category_daily_two_months())
    result = month_dow_adjust(inputs)
    assert result.kind == KIND_HYPOTHESIS
    assert [label for label, _ in result.tables] == [
        "effect", "raw_matrix", "adjusted_matrix"]
    # 칸이 8개(1월 7요일 + 2월 월요일)라 idxmin/spread가 퇴화(1칸)로 무증상 통과하지 않는다
    effect_table = result.tables[0][1]
    assert len(effect_table) == 8
    assert result.verdict == (
        "조정 효과 최대 칸: 1월 일요일 -8.4%, 칸간 편차 2.7%p")
    from bakery.analysis.lab.registry import HYPOTHESES, load_handlers
    load_handlers()
    assert HYPOTHESES["month_dow_adjust"].needs_single_store is True
    # 출처 α=0.5 직독과의 비등가 고지 + "규모 보고일 뿐 판정 아님" 고지가 누락되면
    # 안 된다(정확 비교, 존재만 확인 X)
    assert result.notes == [_NOTE_MONTH_DOW_SOURCE, _NOTE_NOT_A_VERDICT]

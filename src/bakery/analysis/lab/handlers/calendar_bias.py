"""가설 — 캘린더 축 편향(공휴일 프리미엄 / 월×요일 조정).

모델 예측을 참조하지 않는다(입력 데이터 + 캘린더만). 시리즈는 현 vintage
`build_category_daily(alpha)`에서 만들며, 출처 스크립트가 읽던 동결 CSV와는 다르다.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.holiday_premium import SERIES_VALUE_COLUMN, decompose_holiday_premium
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult

PREMIUM_THRESHOLD = 0.05      # 평일 프리미엄 판정 임계(5%)
_NOTE_VINTAGE = ("시리즈는 현 vintage build_category_daily(alpha)에서 생성 — "
                 "출처 스크립트가 읽던 reports/raw_adjusted_series.csv(2026-07-16 동결)와 "
                 "최대 28단위 차이가 있다. 동결 입력 대조는 tests/test_holiday_premium.py 참조.")
_NOTE_CALENDAR = ("캘린더는 build_calendar_daily 사용 — calendar_raw parquet 직독은 "
                  "2021-23 공휴일이 누락돼 프리미엄을 −18.5% 과소평가한다.")


def _format_signed_pct(value: float) -> str:
    """부호 있는 퍼센트 문자열. 음수는 유니코드 마이너스(U+2212)를 쓴다(ASCII '-' 아님)."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.1f}%"


def premium_verdict(dow_class: pd.DataFrame) -> str:
    """평일/주말 공휴일 median lift로 판정."""
    indexed = dow_class.set_index("dow_class")
    weekday = float(indexed.loc["평일", "median_lift"])
    weekend = float(indexed.loc["주말", "median_lift"])
    weekday_pct = _format_signed_pct((weekday - 1.0) * 100)
    weekend_pct = _format_signed_pct((weekend - 1.0) * 100)
    if weekday - 1.0 < PREMIUM_THRESHOLD:
        return f"기각 — 평일 공휴일 프리미엄 {weekday_pct}로 미미(임계 5%)"
    if weekend > 1.0:
        return (f"부분 지지 — 평일 {weekday_pct}, "
                f"주말도 {weekend_pct}(주말 프리미엄 존재)")
    return (f"지지 — 평일 공휴일 프리미엄 {weekday_pct}, "
            f"주말 공휴일은 {weekend_pct}(프리미엄 없음)")


def _series_from_category_daily(category_daily: pd.DataFrame) -> pd.DataFrame:
    return (category_daily.groupby("date", as_index=False)[SERIES_VALUE_COLUMN].sum()
            .sort_values("date").reset_index(drop=True))


def _ranking_fig(ranking: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=ranking["base_name"], y=ranking["median_lift"]))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="평일 공휴일 median lift 랭킹 (이벤트 고유성)",
                      xaxis_title="공휴일", yaxis_title="lift (평상 동일요일 대비)")
    return fig


def _dow_class_fig(dow_class: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=dow_class["dow_class"], y=dow_class["median_lift"],
        error_y=dict(type="data", symmetric=False,
                     array=dow_class["q75"] - dow_class["median_lift"],
                     arrayminus=dow_class["median_lift"] - dow_class["q25"]),
    ))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="평일 vs 주말 공휴일 프리미엄 (median, IQR)",
                      xaxis_title="구분", yaxis_title="lift")
    return fig


@register_hypothesis("holiday_premium", "공휴일 프리미엄 분해 (요일·연휴·대체 축)",
                     needs_single_store=True)
def holiday_premium(inputs: AnalysisInputs) -> AnalysisResult:
    series = _series_from_category_daily(inputs.category_daily)
    tables = decompose_holiday_premium(series, inputs.calendar,
                                       **inputs.params_for("holiday_premium"))
    return AnalysisResult(
        name="holiday_premium", kind=KIND_HYPOTHESIS,
        title="공휴일 프리미엄 분해 (요일·연휴·대체 축)",
        tables=[("dow_class", tables["dow_class"]),
                ("event_ranking", tables["event_ranking"]),
                ("streak_buckets", tables["streak_buckets"]),
                ("by_holiday", tables["by_holiday"])],
        figures=[_dow_class_fig(tables["dow_class"]), _ranking_fig(tables["event_ranking"])],
        verdict=premium_verdict(tables["dow_class"]),
        notes=[_NOTE_VINTAGE, _NOTE_CALENDAR],
    )

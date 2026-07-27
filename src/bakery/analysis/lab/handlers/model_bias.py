"""가설 — 모델 예측 편향 진단(preds artifact 의존).

★경계: 이 레이어는 모델을 실행하지 않는다. harness-run이 남긴 predictions.csv를
읽기만 하며, spec.predictions가 없으면 runner가 preds_required로 스킵한다.

수치 게이트 주의: 출처 스크립트들은 비-canonical 엔진(store_predictive_power)의
캐시 preds를 썼다. canonical harness preds(category_total + event_prior)로 돌리면
수치가 다르므로 방향/판정만 비교 가능하다 — 동결 입력 대조는 tests/test_order_bias.py.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.order_bias import (
    TARGET_DOWS,
    WINNER_DOW,
    WINNER_GLOBAL,
    isowaste_grid,
    waste_rate_of,
)

_NOTE_ENGINE = ("preds는 harness-run(canonical category_total+event_prior) 산출물 — "
                "출처 스크립트가 쓴 비-canonical 엔진 캐시와 수치가 다르다. "
                "판정/방향만 비교하라.")
_NOTE_NO_REFIT = "모델 재학습 없음 — 이미 계산된 expected/actual에 발주 정책 껍질만 씌운 A/B다."


def weekday_bias_verdict(grid: pd.DataFrame) -> str:
    """격자에서 DOW 트림이 CI 0 배제로 이긴 칸이 하나라도 있으면 지지."""
    n_dow = int((grid["winner"] == WINNER_DOW).sum())
    n_global = int((grid["winner"] == WINNER_GLOBAL).sum())
    if n_dow > 0:
        return (f"지지 — {len(grid)}칸 중 {n_dow}칸에서 DOW 트림 우위(CI 0 배제). "
                "center 보정 가치 있음")
    if n_global > 0:
        return f"기각 — {n_global}칸에서 GLOBAL(전역 균일) 우위, DOW 우위 0칸"
    return "기각 — 전 격자에서 CI가 0을 포함. center 보정은 전역 균일 하향을 못 이김"


def _dow_bias_table(preds: pd.DataFrame) -> pd.DataFrame:
    """요일별 상대편향 — 진단 근거(음수=과대예측)."""
    frame = preds.copy()
    frame["dow"] = pd.to_datetime(frame["date"]).dt.dayofweek
    frame["rel_error"] = (frame["actual"] - frame["expected"]) / frame["actual"]
    return (frame.groupby("dow")
            .agg(n=("rel_error", "size"), rel_mean=("rel_error", "mean"),
                 rel_median=("rel_error", "median"))
            .reset_index())


def _gap_fig(grid: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for w_target, group in grid.groupby("w_target"):
        fig.add_trace(go.Scatter(
            x=group["trim"], y=group["gap_freq"], mode="lines+markers",
            name=f"waste {w_target:.0%}",
            error_y=dict(type="data", symmetric=False,
                         array=group["freq_ci_high"] - group["gap_freq"],
                         arrayminus=group["gap_freq"] - group["freq_ci_low"]),
        ))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="iso-waste 매진빈도 gap (DOW − GLOBAL), 음수=DOW 우위",
                      xaxis_title="대상요일 트림", yaxis_title="gap (빈도 차)")
    return fig


def _dow_bias_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=table["dow"], y=table["rel_mean"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="요일별 상대오차 평균 (음수=과대예측)",
                      xaxis_title="요일(월=0)", yaxis_title="(actual−expected)/actual")
    return fig


@register_hypothesis("weekday_bias", "평일(월·수) 과대예측 트림의 iso-waste 가치",
                     needs_predictions=True)
def weekday_bias(inputs: AnalysisInputs) -> AnalysisResult:
    preds = inputs.predictions
    grid = isowaste_grid(preds, **inputs.params_for("weekday_bias"))
    base_waste = waste_rate_of(preds["expected"].to_numpy(), preds["actual"].to_numpy())
    return AnalysisResult(
        name="weekday_bias", kind=KIND_HYPOTHESIS,
        title="평일(월·수) 과대예측 트림의 iso-waste 가치",
        tables=[("isowaste_grid", grid), ("dow_bias", _dow_bias_table(preds))],
        figures=[_gap_fig(grid), _dow_bias_fig(_dow_bias_table(preds))],
        verdict=weekday_bias_verdict(grid),
        notes=[_NOTE_ENGINE, _NOTE_NO_REFIT,
               f"base(expected) waste={base_waste:.3f}, 대상요일={TARGET_DOWS}(월·수)"],
    )

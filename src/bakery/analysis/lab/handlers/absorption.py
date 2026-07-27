"""가설 — 카테고리 총량 수요이전 흡수(W0 게이트).

품목 조기품절이 같은 카테고리 총 sold를 떨어뜨리는가(β<0=walk-away) 아니면
카테고리 안에서 흡수되는가(β≈0). 흡수면 v4 Stage1(카테고리 합)→Stage2(비율 배분)
설계가 정당하다. 계산은 전부 `bakery.analysis.demand_absorption` 프리미티브 호출.
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.demand_absorption import (
    GATE_CATEGORIES,
    AbsorptionResult,
    placebo_absorption,
    run_absorption,
)
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult

ARM_REAL = "real"
ARM_PLACEBO = "placebo"
PLACEBO_ONLY_PARAMS = frozenset({"horizon_days"})   # placebo_absorption 전용 kwarg
_NOTE_CENSORING = ("품절일 판매량은 censored — β는 흡수의 하한 추정. "
                   "처치변수는 마감시각 기준 품절강도(시간)이다.")
_NOTE_PLACEBO = ("placebo(미래 d+7 품절강도) β가 real β와 비슷하면 그 β는 인과가 아니라 "
                 "confound다 — 두 arm을 함께 읽어야 한다.")


def _split_params(params: dict) -> tuple[dict, dict]:
    """공유 params를 수신 함수별로 분리한다 — run_absorption엔 horizon_days가 없어
    그대로 전개하면 TypeError가 난다(placebo 전용 튜너블)."""
    real = {k: v for k, v in params.items() if k not in PLACEBO_ONLY_PARAMS}
    return real, params


def results_to_frame(results: list[AbsorptionResult], *, arm: str) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(r) for r in results])
    frame["arm"] = arm
    return frame


def _gate_results(results: list[AbsorptionResult]) -> list[AbsorptionResult]:
    return [r for r in results if r.category_id in GATE_CATEGORIES]


def absorption_verdict(results: list[AbsorptionResult]) -> str:
    """게이트 카테고리(bread/pastry) 기준 판정. walk-away 1건이라도 있으면 기각."""
    gate = _gate_results(results)
    if not gate:
        return "불확실 — 게이트 카테고리(bread/pastry) 결과 없음"
    walkaways = [(r.store_id, r.category_id) for r in gate if r.verdict == "walkaway"]
    n_absorb = sum(r.verdict == "absorb" for r in gate)
    if walkaways:
        return (f"기각 — walk-away 발견: {walkaways} "
                f"(게이트 {len(gate)}건 중 absorb {n_absorb}건)")
    if n_absorb == len(gate):
        return f"지지 — 게이트 카테고리 {len(gate)}건 전부 absorb, walk-away 0건"
    n_inconclusive = len(gate) - n_absorb
    return (f"불확실 — 게이트 {len(gate)}건 중 absorb {n_absorb}건 / "
            f"inconclusive {n_inconclusive}건, walk-away 0건")


def _gate_summary(results: list[AbsorptionResult]) -> pd.DataFrame:
    gate = _gate_results(results)
    return pd.DataFrame([{
        "n_gate": len(gate),
        "n_absorb": sum(r.verdict == "absorb" for r in gate),
        "n_walkaway": sum(r.verdict == "walkaway" for r in gate),
        "n_inconclusive": sum(r.verdict == "inconclusive" for r in gate),
    }])


def _beta_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for arm, group in table.groupby("arm", observed=True):
        label = group["store_id"] + "/" + group["category_id"]
        fig.add_trace(go.Bar(
            x=label, y=group["beta"], name=str(arm),
            error_y=dict(type="data", symmetric=False,
                         array=group["ci_high"] - group["beta"],
                         arrayminus=group["beta"] - group["ci_low"]),
        ))
    fig.update_layout(title="흡수 계수 β (90% CI) — real vs placebo",
                      xaxis_title="매장/카테고리", yaxis_title="β (품절 1시간당 총량 변화)",
                      barmode="group")
    return fig


def _delta_fig(table: pd.DataFrame) -> go.Figure:
    real = table[table["arm"] == ARM_REAL]
    label = real["store_id"] + "/" + real["category_id"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=label, y=real["delta"], name="δ(등가 경계)"))
    fig.add_trace(go.Scatter(x=label, y=real["beta"].abs(), mode="markers", name="|β|"))
    fig.update_layout(title="TOST 등가 경계 δ 대비 |β|", xaxis_title="매장/카테고리",
                      yaxis_title="단위/품절시간")
    return fig


@register_hypothesis("demand_absorption", "카테고리 총량 수요이전 흡수 (W0 게이트)")
def demand_absorption(inputs: AnalysisInputs) -> AnalysisResult:
    daily = inputs.daily
    real_params, placebo_params = _split_params(inputs.params_for("demand_absorption"))
    real = run_absorption(daily, **real_params)
    placebo = placebo_absorption(daily, **placebo_params)
    table = pd.concat([results_to_frame(real, arm=ARM_REAL),
                       results_to_frame(placebo, arm=ARM_PLACEBO)], ignore_index=True)
    return AnalysisResult(
        name="demand_absorption", kind=KIND_HYPOTHESIS,
        title="카테고리 총량 수요이전 흡수 (W0 게이트)",
        tables=[("results", table), ("gate_summary", _gate_summary(real))],
        figures=[_beta_fig(table), _delta_fig(table)],
        verdict=absorption_verdict(real),
        notes=[_NOTE_CENSORING, _NOTE_PLACEBO],
    )

"""가설 — 품목 매진 시 수요가 다른 품목으로 대체되는가(4추정기).

RD(회귀 불연속) / DiD / MNL / Nested logit을 모두 돌려 결론을 교차 확인한다.
계산은 `bakery.analysis.{substitution, substitution_did, mnl_substitution, nested_logit}` 호출.
출처 스크립트: substitution_4stores.

과거 결론(광교): MNL/Nested λ≈0.99, DiD β≈0 → 개별 substitution 효과 약함,
카테고리는 한 묶음 수요. 이 핸들러는 그 결론을 현 vintage에서 재확인하는 수단이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.mnl_substitution import fit_mnl_per_category
from bakery.analysis.nested_logit import fit_nested_logit
from bakery.analysis.substitution import compute_substitution_matrix, sensitivity_summary
from bakery.analysis.substitution_did import compute_did_substitution

LAMBDA_INDEPENDENCE_THRESHOLD = 0.95   # λ≈1 = nest 내 독립 = 대체 약함
DID_BETA_THRESHOLD = 0.02              # DiD 평균 β 임계
HOURS_IN_DAY = 24
_NOTE_RECEIPTS = ("영수증은 canonical bonavi_receipts(bulk 제외 빌드) — "
                  "is_bulk 컬럼은 진단용이며 재필터하지 않는다.")
_NOTE_COST = "4추정기 전부 실행하므로 실행 시간이 길다(수 분). 필요 없으면 off로 둔다."


def hour_profiles_from_receipts(receipts: pd.DataFrame,
                                daily: pd.DataFrame) -> dict[str, np.ndarray]:
    """매장별 시간대 판매 분포(길이 24, 합=1). DiD가 매진 노출 시간을 배분하는 데 쓴다."""
    stores = daily["store_id"].unique()
    if "store_id" in receipts.columns:
        grouped = {store: group for store, group in receipts.groupby("store_id",
                                                                    observed=True)}
    else:
        grouped = {store: receipts for store in stores}      # 단매장 receipts
    profiles: dict[str, np.ndarray] = {}
    for store in stores:
        group = grouped.get(store)
        counts = np.zeros(HOURS_IN_DAY, dtype=float)
        if group is None or len(group) == 0:
            profiles[store] = counts
            continue
        by_hour = group.groupby("hour")["qty"].sum()
        for hour, qty in by_hour.items():
            if 0 <= int(hour) < HOURS_IN_DAY:
                counts[int(hour)] = float(qty)
        total = counts.sum()
        profiles[store] = counts / total if total else counts
    return profiles


def substitution_verdict(*, rd_mean_outflow: float, did_mean_beta: float,
                         nested_lambda_min: float) -> str:
    """세 추정기 신호를 합쳐 판정. λ≈1 + DiD β≈0 = 대체 약함(카테고리 한 묶음)."""
    is_lambda_independent = nested_lambda_min >= LAMBDA_INDEPENDENCE_THRESHOLD
    is_did_material = abs(did_mean_beta) > DID_BETA_THRESHOLD
    if is_lambda_independent and not is_did_material:
        return (f"기각(대체 약함) — nested λ_min {nested_lambda_min:.3f}(≈1=독립), "
                f"DiD 평균 β {did_mean_beta:.4f}, RD 평균 유출 {rd_mean_outflow:.3f}. "
                "카테고리는 한 묶음 수요로 취급 가능")
    if is_did_material:
        return (f"지지(대체 있음) — nested λ_min {nested_lambda_min:.3f}, "
                f"DiD 평균 β {did_mean_beta:.4f}(임계 0.02 초과), "
                f"RD 평균 유출 {rd_mean_outflow:.3f}")
    return (f"불확실 — nested λ_min {nested_lambda_min:.3f}은 대체를 시사하나 "
            f"DiD 평균 β {did_mean_beta:.4f}은 0에 가깝다 "
            f"(RD 평균 유출 {rd_mean_outflow:.3f})")


def _outflow_fig(rd_outflow: pd.Series, did_outflow: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rd_outflow.index.astype(str), y=rd_outflow.values, name="RD"))
    fig.add_trace(go.Bar(x=did_outflow.index.astype(str), y=did_outflow.values, name="DiD"))
    fig.update_layout(title="품목별 유출 비율(대체 강도)", barmode="group",
                      xaxis_title="품목", yaxis_title="Σ 대체율")
    return fig


def _lambda_fig(lambdas: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=lambdas["nest"].astype(str), y=lambdas["lambda"]))
    fig.add_hline(y=1.0, line_dash="dash")
    fig.update_layout(title="Nested logit λ (1=nest 내 독립)",
                      xaxis_title="nest", yaxis_title="λ")
    return fig


@register_hypothesis("substitution", "품목 매진 시 수요 대체(RD/DiD/MNL/Nested)")
def substitution(inputs: AnalysisInputs) -> AnalysisResult:
    daily, receipts = inputs.daily, inputs.receipts
    profiles = hour_profiles_from_receipts(receipts, daily)
    rd = compute_substitution_matrix(daily, receipts, hour_profiles=profiles)
    did = compute_did_substitution(daily, receipts, profiles)
    mnl = fit_mnl_per_category(receipts, daily)
    nested = fit_nested_logit(receipts, daily)
    lambdas = pd.Series(nested.lambdas).rename("lambda").rename_axis("nest").reset_index()
    verdict = substitution_verdict(
        rd_mean_outflow=float(rd.outflow_ratio.mean()),
        did_mean_beta=float(did.coefficients["beta_did"].mean()),
        nested_lambda_min=float(lambdas["lambda"].min()),
    )
    return AnalysisResult(
        name="substitution", kind=KIND_HYPOTHESIS,
        title="품목 매진 시 수요 대체(RD/DiD/MNL/Nested)",
        tables=[("rd_coefficients", rd.coefficients),
                ("did_coefficients", did.coefficients),
                ("mnl", mnl.substitution),
                ("nested_lambda", lambdas),
                ("sensitivity", sensitivity_summary(rd.outflow_ratio))],
        figures=[_outflow_fig(rd.outflow_ratio, did.outflow_ratio), _lambda_fig(lambdas)],
        verdict=verdict, notes=[_NOTE_RECEIPTS, _NOTE_COST],
    )

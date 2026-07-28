"""가설 — 품목 매진 시 수요가 다른 품목으로 대체되는가(4추정기).

RD(회귀 불연속) / DiD / MNL / Nested logit을 모두 돌려 결론을 교차 확인한다.
계산은 `bakery.analysis.{substitution, substitution_did, mnl_substitution, nested_logit}` 호출.
출처 스크립트: substitution_4stores.

과거 결론(광교): MNL/Nested λ≈0.99, DiD β≈0 → 개별 substitution 효과 약함,
카테고리는 한 묶음 수요. 이 핸들러는 그 결론을 현 vintage에서 재확인하는 수단이다.

λ_min 판정은 게이트 카테고리(bread/pastry, `demand_absorption.GATE_CATEGORIES`)에서만
구한다 — 비타깃 카테고리(sweets 등) 하나가 λ_min을 끌어내려 헤드라인 판정을 뒤집는 것을
막기 위함이다. 게이트 밖 카테고리는 verdict 문구에 별도로 열거한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.demand_absorption import GATE_CATEGORIES
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
_MISSING_VERDICT = "판정 불가 — 추정기 산출에 결측(NaN)이 있어 신호를 해석할 수 없다"
_NOTE_RECEIPTS = ("영수증은 canonical bonavi_receipts(bulk 제외 빌드) — "
                  "is_bulk 컬럼은 진단용이며 재필터하지 않는다.")
_NOTE_COST = "4추정기 전부 실행하므로 실행 시간이 길다(수 분). 필요 없으면 off로 둔다."
_NOTE_GATE = (f"nested_lambda 표는 전체 카테고리를 담지만, 판정은 게이트"
             f"({list(GATE_CATEGORIES)})만 사용한다 — 게이트 밖은 verdict에 별도 열거된다.")


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


def _gate_lambda_min(lambdas: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """게이트 카테고리(bread/pastry)의 λ 최소값 + 게이트 밖 카테고리 표.

    λ_min을 전체 nest에 걸어 구하면 비타깃 카테고리 하나(예: sweets)가 헤드라인 판정을
    뒤집는다. 프로젝트가 실제로 예측하는 집합에서 판정하고, 밖은 따로 보고한다.
    """
    is_gate = lambdas["nest"].astype(str).isin(GATE_CATEGORIES)
    gate, non_gate = lambdas[is_gate], lambdas[~is_gate]
    return float(gate["lambda"].min()) if len(gate) else float("nan"), non_gate


def _non_gate_text(non_gate: pd.DataFrame) -> str:
    """게이트 밖 카테고리 λ를 사람이 읽는 문구로. 빈 프레임이면 '없음'."""
    if len(non_gate) == 0:
        return "없음"
    parts = [f"{row['nest']} λ={row['lambda']:.3f}" for _, row in non_gate.iterrows()]
    return ", ".join(parts)


def _missing_signal_names(rd_mean_outflow: float, did_mean_beta: float,
                          gate_lambda_min: float) -> list[str]:
    """NaN인 신호 이름을 모은다 — mean()/min()의 skipna 기본값이 결측을 조용히
    지우고 well-formed 문구를 내는 것을 막는다(fails-loud)."""
    signals = {"rd_mean_outflow": rd_mean_outflow, "did_mean_beta": did_mean_beta,
              "gate_lambda_min": gate_lambda_min}
    return [name for name, value in signals.items() if pd.isna(value)]


def _did_mean_beta(did_coefficients: pd.DataFrame) -> float:
    """DiD 평균 β. 빈 프레임(추정 실패로 컬럼조차 없는 경우) 시 KeyError 대신 NaN."""
    if len(did_coefficients) == 0 or "beta_did" not in did_coefficients.columns:
        return float("nan")
    return float(did_coefficients["beta_did"].mean())


def _estimator_health(lambdas: pd.DataFrame, did_coefficients: pd.DataFrame) -> pd.DataFrame:
    """NaN 결측 카운트 표 — 부분 실패가 verdict 문구 뒤에 숨지 않게 드러낸다."""
    did_total = len(did_coefficients)
    has_beta_column = did_total and "beta_did" in did_coefficients.columns
    did_nan = int(did_coefficients["beta_did"].isna().sum()) if has_beta_column else 0
    return pd.DataFrame([
        {"estimator": "nested_lambda", "n_total": len(lambdas),
         "n_nan": int(lambdas["lambda"].isna().sum())},
        {"estimator": "did_beta", "n_total": did_total, "n_nan": did_nan},
    ])


def substitution_verdict(*, rd_mean_outflow: float, did_mean_beta: float,
                         gate_lambda_min: float, non_gate: pd.DataFrame) -> str:
    """세 추정기 신호를 합쳐 판정. λ≈1 + DiD β≈0 = 대체 약함(카테고리 한 묶음).

    λ_min은 게이트 카테고리(bread/pastry)에서만 구한다 — 비타깃 카테고리 하나가
    전체 판정을 뒤집는 것을 막는다. 게이트 밖 카테고리는 접미사로 별도 보고한다.
    """
    missing = _missing_signal_names(rd_mean_outflow, did_mean_beta, gate_lambda_min)
    if missing:
        return f"{_MISSING_VERDICT} (결측 신호: {', '.join(missing)})"
    suffix = f" | 게이트={list(GATE_CATEGORIES)} 기준, 게이트 밖: {_non_gate_text(non_gate)}"
    is_lambda_independent = gate_lambda_min >= LAMBDA_INDEPENDENCE_THRESHOLD
    is_did_material = abs(did_mean_beta) > DID_BETA_THRESHOLD
    if is_lambda_independent and not is_did_material:
        return (f"기각(대체 약함) — nested λ_min {gate_lambda_min:.3f}(≈1=독립), "
                f"DiD 평균 β {did_mean_beta:.4f}, RD 평균 유출 {rd_mean_outflow:.3f}. "
                "카테고리는 한 묶음 수요로 취급 가능" + suffix)
    if is_did_material:
        return (f"지지(대체 있음) — nested λ_min {gate_lambda_min:.3f}, "
                f"DiD 평균 β {did_mean_beta:.4f}(임계 0.02 초과), "
                f"RD 평균 유출 {rd_mean_outflow:.3f}" + suffix)
    return (f"불확실 — nested λ_min {gate_lambda_min:.3f}은 대체를 시사하나 "
            f"DiD 평균 β {did_mean_beta:.4f}은 0에 가깝다 "
            f"(RD 평균 유출 {rd_mean_outflow:.3f})" + suffix)


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
    fig.update_layout(title="Nested logit λ (1=nest 내 독립, 게이트 밖 카테고리 포함)",
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
    gate_lambda_min, non_gate = _gate_lambda_min(lambdas)
    verdict = substitution_verdict(
        rd_mean_outflow=float(rd.outflow_ratio.mean()),
        did_mean_beta=_did_mean_beta(did.coefficients),
        gate_lambda_min=gate_lambda_min, non_gate=non_gate,
    )
    return AnalysisResult(
        name="substitution", kind=KIND_HYPOTHESIS,
        title="품목 매진 시 수요 대체(RD/DiD/MNL/Nested)",
        tables=[("rd_coefficients", rd.coefficients),
                ("did_coefficients", did.coefficients),
                ("mnl", mnl.substitution),
                ("nested_lambda", lambdas),
                ("sensitivity", sensitivity_summary(rd.outflow_ratio)),
                ("estimator_health", _estimator_health(lambdas, did.coefficients))],
        figures=[_outflow_fig(rd.outflow_ratio, did.outflow_ratio), _lambda_fig(lambdas)],
        verdict=verdict, notes=[_NOTE_RECEIPTS, _NOTE_COST, _NOTE_GATE],
    )

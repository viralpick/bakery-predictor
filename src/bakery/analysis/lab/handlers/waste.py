"""입력 데이터 분석 — 폐기율 / 항등식 / 과잉생산 분해.

소스는 `waste_alpha_4stores`(생산 made, 폐기 out, 정상/마감 수량, 단가, 폐기비용).
레거시 eda02/eda04/eda05는 `data/internal/v2/inventory.parquet`를 직독했다 —
수치 등가가 아니며 게이트는 구조 불변식(비율 범위, 항등식 잔차)이다.

폐기비용 주의: `waste_cost`는 판매가 기준이다. 원가율(≈0.3)을 곱하지 않은 값이므로
사업 임팩트로 인용할 때 반드시 원가율을 적용해야 한다(과대계상 방지).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_data
from bakery.analysis.lab.result import KIND_DATA, AnalysisResult

MIN_PRODUCTION_FOR_ITEM_RATE = 30      # 생산 누적 30 미만은 비율이 불안정 → 제외
_IDENTITY_TOLERANCE = 1e-9
_NOTE_COST_BASIS = ("waste_cost는 판매가 기준 — 사업 임팩트 인용 시 원가율(≈0.3)을 곱해야 한다.")
_NOTE_CARRY_IN = ("폐기율은 **순합(net)** 기준이다. waste_qty 음수 행(전일 재고 이월로 판매가 "
                  "당일 생산을 초과, 실측 2.89%)을 clip하지 않으므로 gross 폐기보다 낮다 — "
                  "clip하면 폐기율이 부풀려진다(광교 0.12532→0.12933).")
_NOTE_IDENTITY_BASELINE = ("항등식 잔차 0 비율 실측(2026-07-28): 전체 91.83% / out<0 88.80% / "
                           "out≥0 91.92%. 8%대 잔차는 정상 기대값이며 실패 신호가 아니다 — "
                           "재계산 공식은 저장 identity_diff와 100% 일치함이 확인됐다.")
_NOTE_LEGACY = ("레거시 eda02/04/05(inventory.parquet 직독)와 수치 등가가 아니다 — "
                "canonical waste_alpha_4stores 기준 재표현이다.")


def _rate(waste_qty: pd.Series, production_qty: pd.Series) -> pd.Series:
    return (waste_qty / production_qty).where(production_qty > 0, 0.0)


def waste_rate_by_store(waste: pd.DataFrame) -> pd.DataFrame:
    """매장별 폐기율(순합 기준). carry-in 음수 행수/합계를 함께 실어 은폐하지 않는다."""
    frame = waste.copy()
    frame["is_carry_in"] = frame["waste_qty"] < 0
    grouped = (frame.groupby(["cd", "store"], observed=True)
               .agg(production_qty=("production_qty", "sum"),
                    waste_qty=("waste_qty", "sum"), waste_cost=("waste_cost", "sum"),
                    n_carry_in=("is_carry_in", "sum"))
               .reset_index())
    carry_in_sum = (frame[frame["is_carry_in"]].groupby(["cd", "store"], observed=True)
                    ["waste_qty"].sum().rename("carry_in_units"))
    grouped = grouped.merge(carry_in_sum, on=["cd", "store"], how="left")
    grouped["carry_in_units"] = grouped["carry_in_units"].fillna(0.0)
    grouped["waste_rate"] = _rate(grouped["waste_qty"], grouped["production_qty"])
    return grouped


def waste_rate_by_item(waste: pd.DataFrame, item_to_category: pd.Series, *,
                       min_production: int = MIN_PRODUCTION_FOR_ITEM_RATE) -> pd.DataFrame:
    grouped = (waste.groupby(["cd", "item_id"], observed=True)
               .agg(production_qty=("production_qty", "sum"),
                    waste_qty=("waste_qty", "sum"), waste_cost=("waste_cost", "sum"))
               .reset_index())
    grouped = grouped[grouped["production_qty"] >= min_production].copy()
    grouped["category_id"] = grouped["item_id"].map(item_to_category)
    grouped["waste_rate"] = _rate(grouped["waste_qty"], grouped["production_qty"])
    return grouped.sort_values("waste_rate", ascending=False).reset_index(drop=True)


def identity_residual(waste: pd.DataFrame) -> pd.DataFrame:
    """production − (normal + closing) − waste 잔차 검증(재계산해서 대조)."""
    frame = waste.copy()
    frame["recomputed_diff"] = (frame["production_qty"]
                                - (frame["normal_qty"] + frame["closing_qty"])
                                - frame["waste_qty"])
    rows = []
    for (cd, store), group in frame.groupby(["cd", "store"], observed=True):
        diffs = group["recomputed_diff"].abs()
        n_nonzero = int((diffs > _IDENTITY_TOLERANCE).sum())
        rows.append({"cd": cd, "store": store, "n_rows": int(len(group)),
                     "n_nonzero": n_nonzero, "max_abs_diff": float(diffs.max()),
                     "mean_abs_diff": float(diffs.mean()),
                     "zero_frac": 1.0 - n_nonzero / len(group)})
    return pd.DataFrame(rows)


def overproduction_by_category(waste: pd.DataFrame,
                               item_to_category: pd.Series) -> pd.DataFrame:
    frame = waste.copy()
    frame["category_id"] = frame["item_id"].map(item_to_category)
    grouped = (frame.groupby(["cd", "category_id"], observed=True)
               .agg(production_qty=("production_qty", "sum"),
                    waste_qty=("waste_qty", "sum"), waste_cost=("waste_cost", "sum"))
               .reset_index())
    grouped["waste_rate"] = _rate(grouped["waste_qty"], grouped["production_qty"])
    total_cost = grouped.groupby("cd")["waste_cost"].transform("sum")
    grouped["cost_share"] = (grouped["waste_cost"] / total_cost).where(total_cost > 0, 0.0)
    return grouped.sort_values(["cd", "waste_cost"], ascending=[True, False]) \
                  .reset_index(drop=True)


def _bar_fig(frame: pd.DataFrame, x: str, y: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=frame[x].astype(str), y=frame[y]))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y_title)
    return fig


@register_data("waste_rate", "매장별/품목별 폐기율")
def waste_rate(inputs: AnalysisInputs) -> AnalysisResult:
    by_store = waste_rate_by_store(inputs.waste)
    by_item = waste_rate_by_item(inputs.waste, inputs.item_to_category)
    return AnalysisResult(
        name="waste_rate", kind=KIND_DATA, title="매장별/품목별 폐기율",
        tables=[("by_store", by_store), ("by_item", by_item)],
        figures=[_bar_fig(by_store, "store", "waste_rate", "매장별 폐기율", "폐기율"),
                 _bar_fig(by_item.head(20), "item_id", "waste_rate",
                          "품목별 폐기율 상위 20", "폐기율")],
        notes=[_NOTE_COST_BASIS, _NOTE_CARRY_IN, _NOTE_LEGACY],
    )


@register_data("waste_alpha_identity", "생산 = 정상+마감+폐기 항등식 잔차")
def waste_alpha_identity(inputs: AnalysisInputs) -> AnalysisResult:
    residual = identity_residual(inputs.waste)
    return AnalysisResult(
        name="waste_alpha_identity", kind=KIND_DATA,
        title="생산 = 정상+마감+폐기 항등식 잔차",
        tables=[("residual", residual)],
        figures=[_bar_fig(residual, "store", "zero_frac",
                          "매장별 항등식 성립 비율", "잔차 0 비율")],
        notes=[_NOTE_IDENTITY_BASELINE, _NOTE_LEGACY],
    )


@register_data("overproduction_breakdown", "과잉생산 카테고리 분해")
def overproduction_breakdown(inputs: AnalysisInputs) -> AnalysisResult:
    by_category = overproduction_by_category(inputs.waste, inputs.item_to_category)
    return AnalysisResult(
        name="overproduction_breakdown", kind=KIND_DATA, title="과잉생산 카테고리 분해",
        tables=[("by_category", by_category)],
        figures=[_bar_fig(by_category, "category_id", "cost_share",
                          "카테고리별 폐기비용 비중", "비중"),
                 _bar_fig(by_category, "category_id", "waste_rate",
                          "카테고리별 폐기율", "폐기율")],
        notes=[_NOTE_COST_BASIS, _NOTE_CARRY_IN, _NOTE_LEGACY],
    )

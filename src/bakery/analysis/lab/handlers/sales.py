"""입력 데이터 분석 — 매출 분포 / 카테고리 비중.

레거시 eda01/eda03은 `data/internal/v2/` 원본 시트를 다른 필터(FG_ITEM=='SS',
beverage/etc 포함)로 읽었다. 여기서는 canonical daily(bulk 제외·5카테고리) 위에서
재표현하므로 옛 스크립트와 수치 등가가 아니다 — 게이트는 구조 불변식(비중 합=1.0)이다.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_data
from bakery.analysis.lab.result import KIND_DATA, AnalysisResult

MONTH_STD_DDOF = 0        # 관측 월 전체가 모집단 — 표본 보정 없이 재현 가능한 값


def median_unit_price(waste: pd.DataFrame) -> pd.Series:
    """item_id → 단가 중앙값. 폐기 실측 테이블이 유일한 매장 단가 소스."""
    return waste.groupby("item_id")["unit_price"].median().astype(float)


def _with_revenue(daily: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    out = daily.copy()
    out["unit_price"] = out["item_id"].map(prices)
    out["revenue"] = out["sold_units"] * out["unit_price"].fillna(0.0)
    return out


def category_share(daily: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """매장×카테고리 수량/매출 비중."""
    priced = _with_revenue(daily, prices)
    grouped = (priced.groupby(["store_id", "category_id"], observed=True)
               .agg(sold_units=("sold_units", "sum"), revenue=("revenue", "sum"))
               .reset_index())
    totals = grouped.groupby("store_id")[["sold_units", "revenue"]].transform("sum")
    grouped["share"] = grouped["sold_units"] / totals["sold_units"]
    grouped["revenue_share"] = grouped["revenue"] / totals["revenue"]
    return grouped.sort_values(["store_id", "share"], ascending=[True, False]) \
                  .reset_index(drop=True)


def monthly_share_stability(daily: pd.DataFrame) -> pd.DataFrame:
    """월별 카테고리 비중의 산포 — 믹스가 안정적이면 카테고리 합 예측이 정당하다."""
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = (frame.groupby(["store_id", "month", "category_id"], observed=True)
               ["sold_units"].sum().reset_index())
    month_total = monthly.groupby(["store_id", "month"])["sold_units"].transform("sum")
    monthly["share"] = monthly["sold_units"] / month_total
    return (monthly.groupby(["store_id", "category_id"], observed=True)["share"]
            .agg(n_months="count",
                 share_std=lambda s: s.std(ddof=MONTH_STD_DDOF),
                 share_min="min", share_max="max")
            .reset_index())


def _share_fig(share: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for category, group in share.groupby("category_id", observed=True):
        fig.add_trace(go.Bar(x=group["store_id"], y=group["share"], name=str(category)))
    fig.update_layout(title="매장별 카테고리 수량 비중", barmode="stack",
                      xaxis_title="매장", yaxis_title="비중")
    return fig


def _stability_fig(daily: pd.DataFrame) -> go.Figure:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = (frame.groupby(["store_id", "month", "category_id"], observed=True)
               ["sold_units"].sum().reset_index())
    total = monthly.groupby(["store_id", "month"])["sold_units"].transform("sum")
    monthly["share"] = monthly["sold_units"] / total
    fig = go.Figure()
    for (store, category), group in monthly.groupby(["store_id", "category_id"],
                                                    observed=True):
        fig.add_trace(go.Scatter(x=group["month"], y=group["share"], mode="lines+markers",
                                 name=f"{store}/{category}"))
    fig.update_layout(title="월별 카테고리 비중 안정성", xaxis_title="월", yaxis_title="비중")
    return fig


def _coverage_notes(daily: pd.DataFrame, prices: pd.Series) -> list[str]:
    mapped = daily["item_id"].isin(prices.index)
    total = float(daily["sold_units"].sum())
    if total == 0.0:
        return []
    coverage = float(daily.loc[mapped, "sold_units"].sum()) / total
    if coverage >= 1.0:
        return []
    return [f"단가 매핑 커버리지 {coverage:.3f} — 미매핑 품목의 revenue는 0으로 계산됨"]


@register_data("category_mix", "카테고리 매출 비중 + 월별 안정성")
def category_mix(inputs: AnalysisInputs) -> AnalysisResult:
    daily = inputs.daily
    prices = median_unit_price(inputs.waste)
    share = category_share(daily, prices)
    stability = monthly_share_stability(daily)
    return AnalysisResult(
        name="category_mix", kind=KIND_DATA, title="카테고리 매출 비중 + 월별 안정성",
        tables=[("share", share), ("monthly_stability", stability)],
        figures=[_share_fig(share), _stability_fig(daily)],
        notes=_coverage_notes(daily, prices),
    )

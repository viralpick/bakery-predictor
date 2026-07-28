"""가설 — 마감할인 실수요 α / 마감 외 할인 분포 / 할인 레짐 전환.

계산은 `bakery.analysis.{discount, closing_demand, discount_regime}` 프리미티브 호출.
출처 스크립트: verify_closing_codes / verify_other_discounts (+ discount_regime는 신규 노출).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.closing_demand import run_closing_demand
from bakery.analysis.discount import (
    DiscountSales,
    closing_by_category_hour,
    discount_summary,
    label_summary,
)
from bakery.analysis.discount_regime import run_discount_regime
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult

CLOSING_CATEGORY_DEFAULT = "bread"
CLOSING_LABEL = "closing"
_NOTE_ALPHA_STRUCTURAL = ("광교 저녁 상시할인 때문에 A1의 floor가 검증되지 않아 제외되고 "
                          "A2가 degenerate다. 증거는 높은 α 방향이며 헌장 기본값은 0.8이다.")
_NOTE_WASTE_SOURCE = "폐기는 waste_alpha_4stores(생산−판매 실측) 기준이다."
_NOTE_REGIME_POOLING = ("multistore spec에서 build_regime_panel은 store 컬럼(cd)이 없어 "
                        "4매장을 store 항 없이 한 회귀로 pool한다 — β는 매장별 효과가 아니라 "
                        "pooled 평균이다. discount_rows 교차 join이 없어 needs_single_store "
                        "게이트는 걸지 않는다(closing_discount/other_discounts/"
                        "popularity_stockout보다 약한 문제).")


def closing_waste_frame(waste: pd.DataFrame) -> pd.DataFrame:
    """run_closing_demand의 waste 인자 계약 = (item_id, date, waste_qty)."""
    return waste[["item_id", "date", "waste_qty"]].reset_index(drop=True)


def regime_rows_from_waste(waste: pd.DataFrame) -> pd.DataFrame:
    """run_discount_regime의 rows 계약 = item-day 패널(date/item_id/normal_qty/closing_qty/made).

    `AnalysisInputs.waste`는 같은 waste_alpha_4stores를 `made→production_qty`로 리네임해
    돌려준다(폐기 핸들러 관용). discount_regime.build_regime_panel은 원래 이름 `made`만
    읽으므로(out은 본문에서 미사용) 여기서 되돌린다 — CLI `cmd_regime_alpha`가 같은 원본
    parquet을 리네임 없이 읽는 것과 동일 계약(2026-07-28 확인, src/bakery/cli.py:1846).
    """
    return waste.rename(columns={"production_qty": "made"})


def discount_hour_table(ds: DiscountSales, item_to_category: pd.Series) -> pd.DataFrame:
    """카테고리×시각 마감할인 수량. 반환 컬럼(실측) = category_id, hour, qty, loss_won, rows."""
    return closing_by_category_hour(ds, item_to_category)


def _estimator_text(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "없음"


def alpha_verdict(alpha) -> str:
    """AlphaEstimate(alpha_low/alpha_high/a1/a2/a3_slope/note) → 판정 문구."""
    return (f"구간 추정 α ∈ [{alpha.alpha_low:.3f}, {alpha.alpha_high:.3f}] "
            f"(A1 {_estimator_text(alpha.a1)} / A2 {_estimator_text(alpha.a2)} / "
            f"A3 {_estimator_text(alpha.a3_slope)}) — {alpha.note}")


def alpha_estimates_table(report: dict) -> pd.DataFrame:
    """추정기별 원시 산출 — SurplusResult에는 alpha가 없어 slope를 싣는다."""
    kink, depth, surplus = report["kink"], report["depth"], report["surplus"]
    return pd.DataFrame([
        {"estimator": "A1 kink", "alpha": kink.alpha, "statistic": kink.base,
         "n": kink.n_days, "note": kink.note},
        {"estimator": "A2 depth", "alpha": depth.alpha, "statistic": depth.slope,
         "n": depth.n, "note": depth.note},
        {"estimator": "A3 surplus", "alpha": report["alpha"].a3_slope,
         "statistic": surplus.slope, "n": surplus.n, "note": surplus.note},
    ])


def regime_verdict(report: dict) -> str:
    """run_discount_regime 반환 dict → 판정. p_value가 없어 CI로 읽는다."""
    share = report["closing_share"]
    # CI95: discount_regime.Z_95(1.9599…) 기준 — demand_absorption의 CI90(norm.ppf(0.95)
    # two-sided)과 신뢰수준이 다르다. 프리미티브가 만드는 구간을 그대로 라벨링한다.
    return (f"레짐 전환 {report['verdict']} — closing_share β={share.beta:.4f} "
            f"CI95[{share.ci_low:.4f},{share.ci_high:.4f}], "
            f"placebo {len(report['placebo'])}건, n={report['n']} "
            f"(cut={pd.Timestamp(report['cut_date']).date()})")


def _hour_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for category, group in table.groupby("category_id", observed=True):
        fig.add_trace(go.Bar(x=group["hour"], y=group["qty"], name=str(category)))
    fig.update_layout(title="시각별 할인 판매 수량", barmode="stack",
                      xaxis_title="시(hour)", yaxis_title="수량")
    return fig


def _panel_fig(panel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=panel["date"], y=panel["closing_qty"],
                             mode="lines", name="마감 수량"))
    fig.add_trace(go.Scatter(x=panel["date"], y=panel["waste_qty"],
                             mode="lines", name="폐기 수량"))
    fig.update_layout(title="마감 vs 폐기 추이(잉여의 두 배출구)",
                      xaxis_title="날짜", yaxis_title="수량")
    return fig


@register_hypothesis("closing_discount", "마감할인 실수요 비율 α 추정",
                     needs_single_store=True)
def closing_discount(inputs: AnalysisInputs) -> AnalysisResult:
    """params: `category`(기본 bread)만 읽는다 — 그 외 키는 무시된다."""
    ds = DiscountSales(rows=inputs.discount_rows)
    params = inputs.params_for("closing_discount")
    category = params.get("category", CLOSING_CATEGORY_DEFAULT)
    report = run_closing_demand(inputs.discount_rows, closing_waste_frame(inputs.waste),
                               inputs.item_to_category, category=category)
    estimates = alpha_estimates_table(report)
    by_hour = discount_hour_table(ds, inputs.item_to_category)
    return AnalysisResult(
        name="closing_discount", kind=KIND_HYPOTHESIS, title="마감할인 실수요 비율 α 추정",
        tables=[("estimates", estimates), ("panel", report["panel"]), ("by_hour", by_hour)],
        figures=[_panel_fig(report["panel"]), _hour_fig(by_hour)],
        verdict=alpha_verdict(report["alpha"]),
        notes=[_NOTE_ALPHA_STRUCTURAL, _NOTE_WASTE_SOURCE, f"카테고리={category}"],
    )


# 빈 결과에서도 테이블 스키마를 유지한다 — 라벨과 컬럼이 어긋나면 CSV 소비자가 깨진다.
# ★컬럼은 2026-07-28 실측(discount_summary/label_summary 실제 반환)과 정확히 일치시킨 것이다.
_EMPTY_BY_HOUR = pd.DataFrame({"discount_code": pd.Series(dtype="object"),
                               "hour": pd.Series(dtype="int64"),
                               "qty": pd.Series(dtype="float64")})
_EMPTY_BY_CODE = pd.DataFrame({"discount_code": pd.Series(dtype="object"),
                               "label": pd.Series(dtype="object"),
                               "rows": pd.Series(dtype="int64"),
                               "qty_total": pd.Series(dtype="float64"),
                               "amt_total": pd.Series(dtype="float64"),
                               "avg_amt": pd.Series(dtype="float64"),
                               "peak_hour": pd.Series(dtype="float64"),
                               "share_at_pm8": pd.Series(dtype="float64")})
_EMPTY_BY_LABEL = pd.DataFrame({"label": pd.Series(dtype="object"),
                                "rows": pd.Series(dtype="int64"),
                                "qty_total": pd.Series(dtype="float64"),
                                "amt_total": pd.Series(dtype="float64"),
                                "share_at_pm8": pd.Series(dtype="float64")})
# placebo가 0건이면 list comprehension이 컬럼 없는 0x0 프레임을 만든다 — 스키마를 고정한다.
_EMPTY_PLACEBO = pd.DataFrame({"cut_date": pd.Series(dtype="object"),
                               "beta": pd.Series(dtype="float64"),
                               "ci_low": pd.Series(dtype="float64"),
                               "ci_high": pd.Series(dtype="float64")})


def _discount_code_hour_fig(by_hour: pd.DataFrame) -> go.Figure:
    """할인코드별 시각 분포. 빈 프레임이면 빈 축만 그린다(is_empty 은폐 방지는 verdict가 담당)."""
    fig = go.Figure()
    for code, group in by_hour.groupby("discount_code", observed=True):
        fig.add_trace(go.Bar(x=group["hour"], y=group["qty"], name=str(code)))
    fig.update_layout(title="마감 외 할인코드 시각별 수량", barmode="stack",
                      xaxis_title="시(hour)", yaxis_title="수량")
    return fig


@register_hypothesis("other_discounts", "마감 외 할인코드 시각 분포",
                     needs_single_store=True)
def other_discounts(inputs: AnalysisInputs) -> AnalysisResult:
    rows = inputs.discount_rows
    others = rows[(rows["label"] != CLOSING_LABEL) & (rows["discount_amt"] > 0)]
    ds_others = DiscountSales(rows=others)
    is_empty = len(others) == 0
    by_hour = (_EMPTY_BY_HOUR.copy() if is_empty else
               others.groupby(["discount_code", "hour"], observed=True)["qty"]
               .sum().reset_index())
    verdict = ("마감 외 할인 0건 — 이 매장은 마감할인이 전부다" if is_empty
               else f"마감 외 할인 {len(others):,}건 / 코드 "
                    f"{others['discount_code'].nunique()}종 — 시각 분포로 성격 판별")
    return AnalysisResult(
        name="other_discounts", kind=KIND_HYPOTHESIS, title="마감 외 할인코드 시각 분포",
        tables=[("by_code", _EMPTY_BY_CODE.copy() if is_empty else discount_summary(ds_others)),
                ("by_label", _EMPTY_BY_LABEL.copy() if is_empty else label_summary(ds_others)),
                ("by_hour", by_hour)],
        figures=[_discount_code_hour_fig(by_hour)],
        verdict=verdict,
    )


def _regime_summary_table(report: dict) -> pd.DataFrame:
    rows = [{"outcome": name, "beta": result.beta, "se": result.se,
             "ci_low": result.ci_low, "ci_high": result.ci_high, "n": result.n,
             "ill_posed": result.ill_posed}
            for name, result in (("closing_share", report["closing_share"]),
                                 ("closing_intensity", report["closing_intensity"]))]
    return pd.DataFrame(rows)


def _regime_placebo_table(report: dict) -> pd.DataFrame:
    if not report["placebo"]:
        return _EMPTY_PLACEBO.copy()
    return pd.DataFrame([{"cut_date": r.cut_date, "beta": r.beta,
                          "ci_low": r.ci_low, "ci_high": r.ci_high}
                         for r in report["placebo"]])


def _regime_fig(report: dict) -> go.Figure:
    betas = [abs(report["closing_share"].beta)] + [abs(r.beta) for r in report["placebo"]]
    labels = ["real β"] + [f"placebo {pd.Timestamp(r.cut_date).date()}"
                           for r in report["placebo"]]
    fig = go.Figure(go.Bar(x=labels, y=betas))
    fig.update_layout(title="레짐 전환 |β| — real vs placebo cut", yaxis_title="|β|")
    return fig


@register_hypothesis("discount_regime", "할인 레짐 전환(마감 비중 구조변화)")
def discount_regime(inputs: AnalysisInputs) -> AnalysisResult:
    """params: `category`(기본 bread) + `cut_date`/`placebo_cut_dates`를 그대로 전달한다."""
    params = inputs.params_for("discount_regime")
    category = params.get("category", CLOSING_CATEGORY_DEFAULT)
    rows = regime_rows_from_waste(inputs.waste)
    report = run_discount_regime(rows, inputs.item_to_category, category,
                                 **{k: v for k, v in params.items() if k != "category"})
    return AnalysisResult(
        name="discount_regime", kind=KIND_HYPOTHESIS,
        title="할인 레짐 전환(마감 비중 구조변화)",
        tables=[("summary", _regime_summary_table(report)),
                ("placebo", _regime_placebo_table(report))],
        figures=[_regime_fig(report)],
        verdict=regime_verdict(report),
        notes=[f"카테고리={category}",
               "placebo cut이 real과 비슷한 β를 내면 그 전환은 구조가 아니라 추세다.",
               _NOTE_REGIME_POOLING],
    )

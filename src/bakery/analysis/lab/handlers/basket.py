"""가설 — modeling_v4 framework의 4가정(카테고리 합 → 품목 비율 3-stage 전제).

출처 스크립트: verify_hypotheses.py, docs/modeling_v4.md.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.seasonal import filter_seasonal

ASSUMPTION_THRESHOLDS: dict[str, float] = {
    "1-1-b": 0.7,     # cv(카테고리 총량) / 평균 cv(품목) — 낮을수록 총량이 안정
    "2-1-a": 0.05,    # 월별 품목 비율 std 중앙값
    "2-1-b": 0.10,    # 신제품 도입 시 기존 비율 변화 중앙값
    "basket": 0.30,   # 다중 카테고리 바스켓 비율(> 임계여야 통과)
}
_LOWER_IS_BETTER = ("1-1-b", "2-1-a", "2-1-b")
_NOTE_SEASONAL = ("시즌 제외 품목은 filter_seasonal로 제거(광교 기준) — 계절 특수품이 "
                  "비율 안정성을 왜곡하지 않게 한다.")
_NOTE_THRESHOLDS = ("임계값은 modeling_v4 설계 문서 기준의 실무 기준선이며 통계적 "
                    "검정이 아니다.")
_NOTE_BASKET_SCOPE = (
    "basket 가정은 영수증당 카테고리 다양성(_multi_category_basket_share)으로 측정한다. "
    "analysis/basket_composition.py의 basket_composition_summary는 마감할인 바스켓 분류라 "
    "label/paid 컬럼을 요구하고(bonavi_receipts에 없음) 의미도 다르므로 여기서 쓰지 않는다 "
    "— 다른 범위의 지표를 4가정 리포트에 섞지 않기 위한 결정(2026-07-28)."
)


def _cv(series: pd.Series) -> float:
    """변동계수(std/mean). mean==0이면 0.0 — 호출부가 n_observations로 표본 규모를 드러낸다."""
    mean = float(series.mean())
    return float(series.std()) / mean if mean else 0.0


def _total_vs_item_cv(daily: pd.DataFrame) -> tuple[float, int]:
    """카테고리 총량 cv 평균 / 품목 cv 평균(작을수록 총량이 예측 쉽다).

    n_observations=분모(품목 cv 평균)를 구성하는 품목 수 — 소표본이면 평균이 불안정하다.
    """
    category_daily = daily.groupby(["category_id", "date"])["sold_units"].sum()
    category_cv_mean = float(category_daily.groupby("category_id").apply(_cv).mean())
    item_daily = daily.groupby(["item_id", "date"])["sold_units"].sum()
    item_cv = item_daily.groupby("item_id").apply(_cv)
    item_cv_mean = float(item_cv.mean())
    ratio = category_cv_mean / item_cv_mean if item_cv_mean else 0.0
    return float(ratio), int(len(item_cv))


def _monthly_proportion(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = (frame.groupby(["category_id", "month", "item_id"], observed=True)
               ["sold_units"].sum().reset_index())
    total = monthly.groupby(["category_id", "month"])["sold_units"].transform("sum")
    monthly["proportion"] = (monthly["sold_units"] / total).where(total > 0, 0.0)
    return monthly


def _proportion_stability(monthly: pd.DataFrame) -> tuple[float, int]:
    """카테고리×품목별 월간 비율 std의 중앙값.

    n_observations=유효 std 개수(월 관측이 1개뿐인 품목은 std가 NaN이라 제외).
    """
    stds = monthly.groupby(["category_id", "item_id"], observed=True)["proportion"].std()
    stds = stds.dropna()
    value = float(stds.median()) if len(stds) else 0.0
    return value, int(len(stds))


def _shared_change(previous: pd.Series | None, snapshot: pd.Series,
                   new_items: set[str], seen: set[str]) -> float | None:
    """직전월 대비 공유(비신규) 품목 비율 변화의 중앙 절대값. 해당 없으면 None."""
    if previous is None or not new_items or not seen:
        return None
    shared = previous.index.intersection(snapshot.index).difference(new_items)
    if not len(shared):
        return None
    return float((snapshot[shared] - previous[shared]).abs().median())


def _category_new_item_changes(group: pd.DataFrame) -> list[float]:
    months = sorted(group["month"].unique())
    seen: set[str] = set()
    previous: pd.Series | None = None
    changes: list[float] = []
    for month in months:
        snapshot = group[group["month"] == month].set_index("item_id")["proportion"]
        new_items = set(snapshot.index) - seen
        change = _shared_change(previous, snapshot, new_items, seen)
        if change is not None:
            changes.append(change)
        seen |= set(snapshot.index)
        previous = snapshot
    return changes


def _new_item_disruption(monthly: pd.DataFrame) -> tuple[float, int]:
    """신제품 첫 등장 월에 기존 품목 비율이 얼마나 흔들리는지(중앙 절대변화).

    n_observations=changes 리스트 길이 — 0이면 신제품 등장 사례가 없었다는 뜻이라
    0.0(자동 통과처럼 보임)이 실제로는 '미검증'임을 verdict가 드러내야 한다.
    """
    changes: list[float] = []
    for (_category,), group in monthly.groupby(["category_id"], observed=True):
        changes.extend(_category_new_item_changes(group))
    value = float(pd.Series(changes).median()) if changes else 0.0
    return value, len(changes)


def _multi_category_basket_share(receipts: pd.DataFrame,
                                 item_to_category: pd.Series) -> tuple[float, int]:
    """영수증(receipt_id)당 서로 다른 카테고리 수 > 1인 비율. n_observations=영수증 수."""
    frame = receipts.copy()
    frame["category_id"] = frame["item_id"].map(item_to_category)
    per_receipt = frame.groupby("receipt_id")["category_id"].nunique()
    share = float((per_receipt > 1).mean()) if len(per_receipt) else 0.0
    return share, int(len(per_receipt))


def _assumption_row(key: str, statistic: str, value: float, n_observations: int) -> dict:
    threshold = ASSUMPTION_THRESHOLDS[key]
    is_supported = value < threshold if key in _LOWER_IS_BETTER else value > threshold
    return {"assumption": key, "statistic": statistic, "value": value,
            "threshold": threshold, "is_supported": bool(is_supported),
            "n_observations": n_observations}


def assumption_table(daily: pd.DataFrame, receipts: pd.DataFrame,
                     item_to_category: pd.Series) -> pd.DataFrame:
    filtered = filter_seasonal(daily)
    monthly = _monthly_proportion(filtered)
    cv_ratio, cv_n = _total_vs_item_cv(filtered)
    stability, stability_n = _proportion_stability(monthly)
    disruption, disruption_n = _new_item_disruption(monthly)
    basket_share, basket_n = _multi_category_basket_share(receipts, item_to_category)
    rows = [
        _assumption_row("1-1-b", "cv(카테고리 총량)/평균 cv(품목)", cv_ratio, cv_n),
        _assumption_row("2-1-a", "월별 품목 비율 std 중앙값", stability, stability_n),
        _assumption_row("2-1-b", "신제품 도입 시 기존 비율 변화 중앙값",
                       disruption, disruption_n),
        _assumption_row("basket", "영수증당 다중 카테고리 비율", basket_share, basket_n),
    ]
    return pd.DataFrame(rows)


def _zero_observation_supported(table: pd.DataFrame) -> list[str]:
    """관측 0건인데 통과로 뜬 가정 이름 — 0/0류 무증상 통과를 verdict에 노출한다."""
    hollow = table[table["is_supported"] & (table["n_observations"] == 0)]
    return hollow["assumption"].tolist()


def v4_verdict(table: pd.DataFrame) -> str:
    failed = table[~table["is_supported"]]["assumption"].tolist()
    n_passed = len(table) - len(failed)
    if not failed:
        verdict = "지지 — 4가정 전부 통과(v4 카테고리 합 → 품목 비율 설계 정당)"
    elif n_passed == 0:
        verdict = "기각 — 4가정 전부 미통과"
    else:
        verdict = f"부분 지지 — 4가정 중 {n_passed}건 통과, 미통과: {failed}"
    hollow = _zero_observation_supported(table)
    if hollow:
        verdict += f" (관측 0건: {hollow} — 통과가 아니라 미검증)"
    return verdict


def _assumption_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["assumption"], y=table["value"], name="관측값"))
    fig.add_trace(go.Scatter(x=table["assumption"], y=table["threshold"],
                             mode="markers", name="임계"))
    fig.update_layout(title="v4 4가정 통계량 vs 임계", xaxis_title="가정", yaxis_title="값")
    return fig


@register_hypothesis("modeling_v4_assumptions", "modeling_v4 framework 4가정")
def modeling_v4_assumptions(inputs: AnalysisInputs) -> AnalysisResult:
    table = assumption_table(inputs.daily, inputs.receipts, inputs.item_to_category)
    return AnalysisResult(
        name="modeling_v4_assumptions", kind=KIND_HYPOTHESIS,
        title="modeling_v4 framework 4가정",
        tables=[("assumptions", table)],
        figures=[_assumption_fig(table)],
        verdict=v4_verdict(table),
        notes=[_NOTE_SEASONAL, _NOTE_THRESHOLDS, _NOTE_BASKET_SCOPE],
    )

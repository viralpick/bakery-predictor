"""Pre-registered grounding question set + deterministic gold generator.

Questions are fixed in code (fairness §8: no cherry-picking). Gold answers are
produced by calling the OntologyFunction directly — never hand-labeled — so the
eval is reproducible and the grounded arm's job is to reach the same number
through tool calls, while the rag-only arm must guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .. import functions as fn
from ...data.loader import DailyDataset


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    grader_type: str            # numeric | ranking | decomposition
    source_fn: str
    fn_kwargs: dict = field(default_factory=dict)
    tolerance: float = 0.05     # relative, numeric only


def _ctx(dataset: DailyDataset):
    """Resolve a stable (store, period, top_item) from the dataset."""
    daily = dataset.daily
    store = daily["store_id"].iloc[0]
    dd = pd.to_datetime(daily.loc[daily["store_id"] == store, "date"])
    period = (str(dd.min().date()), str(dd.max().date()))
    sub = daily[daily["store_id"] == store]
    top_item = sub.groupby("item_id", observed=True)["sold_units"].sum().idxmax()
    return store, period, top_item


QUESTIONS: list[Question] = [
    Question("q_rank_top3", "광교에서 매진 위험이 가장 높은 상위 3개 품목은?",
             "ranking", "rank_stockout_risk", {"k": 3}),
    Question("q_rank_top5", "매진 위험 상위 5개 품목은?", "ranking", "rank_stockout_risk", {"k": 5}),
    Question("q_waste", "이 기간 이 매장의 폐기(capacity-sold) 수량 합계는?",
             "numeric", "waste_cost", {}),
    Question("q_diff_weekend", "주말일 때와 아닐 때 일 판매량 평균 차이는?",
             "numeric", "demand_diff_by_condition", {"condition_col": "is_weekend", "frame": "calendar"}),
    Question("q_diff_rain", "비 올 때와 안 올 때 일 판매량 평균 차이는?",
             "numeric", "demand_diff_by_condition", {"condition_col": "is_rain", "frame": "weather"}),
    Question("q_order_top", "상위 품목의 권장 발주량은?", "decomposition", "explain_order", {}),
    Question("q_whatif_up", "수요 30, 발주 30에서 발주를 10 늘리면 기대비용은?",
             "numeric", "what_if", {"demand_point": 30.0, "base_order": 30.0, "delta_order": 10.0}),
    Question("q_whatif_down", "수요 30, 발주 40에서 발주를 -10 줄이면 기대비용은?",
             "numeric", "what_if", {"demand_point": 30.0, "base_order": 40.0, "delta_order": -10.0}),
    Question("q_rank_top1", "매진 위험이 가장 높은 1개 품목은?", "ranking", "rank_stockout_risk", {"k": 1}),
    Question("q_diff_offday", "휴무일 여부에 따른 일 판매량 평균 차이는?",
             "numeric", "demand_diff_by_condition", {"condition_col": "is_off_day", "frame": "calendar"}),
]


def build_gold(question: Question, dataset: DailyDataset) -> dict:
    store, period, top_item = _ctx(dataset)
    k = question.fn_kwargs
    if question.source_fn == "rank_stockout_risk":
        ranked = fn.rank_stockout_risk(dataset.daily, store, period, k["k"])
        return {"top_items": list(ranked["item_id"])}
    if question.source_fn == "waste_cost":
        return {"answer_value": float(fn.waste_cost(dataset.daily, store, period)["waste_cost"])}
    if question.source_fn == "demand_diff_by_condition":
        frame = dataset.calendar if k["frame"] == "calendar" else dataset.weather
        out = fn.demand_diff_by_condition(dataset.daily, frame, store, k["condition_col"])
        value = float(out["diff"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"answer_value": value}
    if question.source_fn == "explain_order":
        lin = fn.explain_order(dataset.daily, store, top_item, period)
        value = float(lin["contribution"].sum())
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"order_qty": value}
    if question.source_fn == "what_if":
        r = fn.what_if(k["demand_point"], k["base_order"], k["delta_order"])
        return {"answer_value": float(r.new_expected_cost)}
    raise KeyError(question.source_fn)

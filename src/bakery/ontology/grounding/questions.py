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

from .. import explain
from .. import functions as fn
from ...data.loader import DailyDataset
from ...forecast.forward import forecast_forward
from .constants import CALENDAR, DECOMPOSITION, NUMERIC, RANKING, WEATHER


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    grader_type: str            # numeric | ranking | decomposition
    source_fn: str
    fn_kwargs: dict = field(default_factory=dict)
    tolerance: float = 0.05     # relative, numeric only


def _ctx(dataset: DailyDataset):
    """Resolve a stable (store, period) from the dataset."""
    daily = dataset.daily
    store = sorted(dataset.daily["store_id"].unique())[0]
    dd = pd.to_datetime(daily.loc[daily["store_id"] == store, "date"])
    period = (str(dd.min().date()), str(dd.max().date()))
    return store, period


def _forward_ctx(dataset: DailyDataset, horizon_days: int = 7):
    """explain 질문용 forward 컨텍스트: (store, 마지막 관측일 다음날).

    _ctx(historical)와 달리 forward 대상. forecast_forward가 결정론이라 gold 재현 가능.
    """
    store = sorted(dataset.daily["store_id"].unique())[0]
    dd = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    first_future = (dd.max() + pd.Timedelta(days=1)).date()
    return store, str(first_future)


def _forward_top_item(dataset: DailyDataset, store: str, date: str) -> str:
    """forward our_order 최대 품목(결정론 선택).

    다중 store daily면 store로 필터 후 forecast_forward — explain._forward_at_date와
    동일 규칙(단일-store 가정). 필터 없이 넘기면 여러 매장 판매가 합산돼 store별
    비중과 다른 품목이 뽑혀 explain_item_order(store=store 필터)와 어긋난다.
    """
    daily = dataset.daily
    if "store_id" in daily.columns and daily["store_id"].nunique() > 1:
        daily = daily[daily["store_id"] == store].copy()
    iq = forecast_forward(store, daily=daily, horizon_days=7,
                          use_forecast=False).item_quantities
    iq_d = iq[iq["date"].astype(str).str.startswith(date)].copy()
    iq_d["item_id"] = iq_d["item_id"].astype(str)
    return str(iq_d.sort_values(["our_order", "item_id"], ascending=[False, True]).iloc[0]["item_id"])


QUESTIONS: list[Question] = [
    Question("q_rank_top3", "이 매장에서 매진 위험이 가장 높은 상위 3개 품목은?",
             RANKING, "rank_stockout_risk", {"k": 3}),
    Question("q_rank_top5", "매진 위험 상위 5개 품목은?", RANKING, "rank_stockout_risk", {"k": 5}),
    Question("q_waste", "이 기간 이 매장의 폐기(capacity-sold) 수량 합계는?",
             NUMERIC, "waste_cost", {}),
    Question("q_diff_weekend", "주말일 때 평균에서 평일일 때 평균을 뺀 일 판매량 차이는? (주말이 더 높으면 양수)",
             NUMERIC, "demand_diff_by_condition", {"condition_col": "is_weekend", "frame": CALENDAR}),
    Question("q_diff_rain", "비 올 때 평균에서 비 안 올 때 평균을 뺀 일 판매량 차이는? (비 올 때가 더 높으면 양수)",
             NUMERIC, "demand_diff_by_condition", {"condition_col": "is_rain", "frame": WEATHER}),
    Question("q_order_top",
             "관측 매진 위험(일평균 손실 영업시간 기준)이 가장 높은 품목은 무엇이고, "
             "그 품목의 권장 발주량은?",
             DECOMPOSITION, "explain_order", {}),
    Question("q_rank_earliness",
             "일평균 손실 영업시간 기준 매진 위험 상위 3개 품목은?",
             RANKING, "rank_stockout_earliness", {"k": 3}),
    Question("q_whatif_up", "수요 30, 발주 30에서 발주를 10 늘리면 기대비용은?",
             NUMERIC, "what_if", {"demand_point": 30.0, "base_order": 30.0, "delta_order": 10.0}),
    Question("q_whatif_down", "수요 30, 발주 40에서 발주를 -10 줄이면 기대비용은?",
             NUMERIC, "what_if", {"demand_point": 30.0, "base_order": 40.0, "delta_order": -10.0}),
    Question("q_rank_top1", "매진 위험이 가장 높은 1개 품목은?", RANKING, "rank_stockout_risk", {"k": 1}),
    Question("q_diff_offday", "휴무일일 때 평균에서 비휴무일일 때 평균을 뺀 일 판매량 차이는? (휴무일이 더 높으면 양수)",
             NUMERIC, "demand_diff_by_condition", {"condition_col": "is_off_day", "frame": CALENDAR}),
    Question("q_explain_total",
             "다음주 이 매장의 빵 카테고리 생산총량은? base 예측·특수일 보정·분위수 버퍼로 분해하면?",
             NUMERIC, "explain_category_total", {}),
    Question("q_explain_item",
             "다음주 이 매장에서 가장 많이 생산하는 품목의 생산량은? 카테고리 총량과 품목 비중으로 분해하면?",
             DECOMPOSITION, "explain_item_order", {}),
]


def resolve_eval_context(dataset: DailyDataset) -> tuple[str, tuple[str, str]]:
    """The (store_id, period) the eval targets — same basis as build_gold's gold."""
    return _ctx(dataset)


def build_gold(question: Question, dataset: DailyDataset) -> dict:
    store, period = _ctx(dataset)
    k = question.fn_kwargs
    if question.source_fn == "rank_stockout_risk":
        ranked = fn.rank_stockout_risk(dataset.daily, store, period, k["k"])
        return {"top_items": list(ranked["item_id"])}
    if question.source_fn == "waste_cost":
        value = float(fn.waste_cost(dataset.daily, store, period)["waste_cost"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"answer_value": value}
    if question.source_fn == "demand_diff_by_condition":
        frame_map = {CALENDAR: dataset.calendar, WEATHER: dataset.weather}
        try:
            frame = frame_map[k["frame"]]
        except KeyError as exc:
            raise KeyError(f"unknown frame for {question.id}: {k['frame']}") from exc
        out = fn.demand_diff_by_condition(dataset.daily, frame, store, k["condition_col"])
        value = float(out["diff"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"answer_value": value}
    if question.source_fn == "rank_stockout_earliness":
        ranked = fn.rank_stockout_earliness(dataset.daily, store, period, k=k["k"])
        return {"top_items": list(ranked["item_id"])}
    if question.source_fn == "explain_order":
        top_risk = fn.rank_stockout_earliness(dataset.daily, store, period, k=1)
        risk_item = str(top_risk["item_id"].iloc[0])
        lineage = fn.explain_order(dataset.daily, store, risk_item, period)
        value = float(lineage["contribution"].sum())
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"item_id": risk_item, "order_qty": value}
    if question.source_fn == "what_if":
        r = fn.what_if(k["demand_point"], k["base_order"], k["delta_order"])
        value = float(r.new_expected_cost)
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"answer_value": value}
    if question.source_fn == "explain_category_total":
        store, date = _forward_ctx(dataset)
        rows = explain.explain_category_total(store, daily=dataset.daily, date=date, use_forecast=False)
        value = float(rows.set_index("step")["value"]["prior_prod"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"answer_value": value}
    if question.source_fn == "explain_item_order":
        store, date = _forward_ctx(dataset)
        item = _forward_top_item(dataset, store, date)
        rows = explain.explain_item_order(store, item, daily=dataset.daily, date=date, use_forecast=False)
        value = float(rows.set_index("step")["value"]["final"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"item_id": item, "order_qty": value}
    raise KeyError(question.source_fn)

"""OntologyFunction layer — parameterized, reusable operations over the ontology.

Mirrors AOS OntologyFunction: a stable API the agent calls instead of writing
ad-hoc queries. Every function here is a thin wrapper that REUSES the v6 decision
layer (commit 8d13157) — no new modeling logic. The numbers come from the
deterministic engine; the agent's job is only to call + interpret (docs §2,
"수치 = 엔진, 해석 = LLM").

Demand point estimate: **forward forecast for forward periods, historical
proxy fallback for historical periods** (5a 의도적 변경, commit 이후). When
`period` starts after the store's last observed date, `rank_stockout_risk`/
`explain_order` call `_forward_demand_points` → `forecast_forward` (Task
1-4의 forward 2층 예측), which is what lets the architect ask "왜 K개
생산?" against a real forecast instead of a backward-looking mean. When
`period` is historical (e.g. grounding/eval-gold's full observed range),
they fall back to the original `_item_demand_points` column-mean path via
`_resolve_demand_proxy` — `adjusted_demand` when present (real data), else
`potential_demand` (synthetic). Labeled, not hidden (fairness contract §7).

All functions are *read-only* over the ontology (AOS rule). Writeback lives in
writeback.py (S4) behind a human-approval gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ..decision import (
    PolicyParams,
    RiskParams,
    apply_policy,
    build_recommendation,
    simulate_item_risk,
)
from . import scenario
from .writeback import WritebackStore

DEMAND_PROXY_COL = "potential_demand"  # fallback when adjusted_demand isn't attached
CONDITION_ON, CONDITION_OFF = 1, 0   # 0/1 flag values for demand_diff_by_condition
DEFAULT_CLOSE_HOUR = 22  # 라벨된 가정: bonavi loader 하드코딩·synthetic store_A와 일치 (spec D3)


def _resolve_demand_proxy(daily: pd.DataFrame) -> str:
    """수요 점추정 컬럼 결정: adjusted_demand 있으면 그것(real), 없으면 potential_demand.

    real 데이터의 potential_demand는 stockout_time 로더 버그로 오염 →
    grounding/run.py가 real일 때 adjusted_demand를 부착하면 자동 채택된다.
    """
    return "adjusted_demand" if "adjusted_demand" in daily.columns else DEMAND_PROXY_COL


def _period_slice(daily: pd.DataFrame, store_id: str, start: str, end: str) -> pd.DataFrame:
    """Rows for one store within [start, end] inclusive. Guard clause on emptiness."""
    dates = pd.to_datetime(daily["date"])
    mask = (daily["store_id"] == store_id) & (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sliced = daily.loc[mask]
    if sliced.empty:
        raise ValueError(f"no rows for store={store_id} in [{start}, {end}]")
    return sliced


def _item_demand_points(period: pd.DataFrame, demand_col: str) -> pd.DataFrame:
    """Per-item demand point estimate = mean of the (proxy) demand col over the period."""
    grouped = period.groupby("item_id", observed=True)[demand_col].mean()
    return grouped.reset_index(name="demand_point")


def _forward_demand_points(
    daily: pd.DataFrame, store_id: str, period: tuple[str, str], *, horizon_days: int = 7,
) -> pd.DataFrame:
    """forward 예측 demand_point (과거 평균 _item_demand_points 대체).

    의도적 변경(5a): 온톨로지 수요 = adjusted_demand 컬럼 평균 → forecast_forward.
    period는 다가오는 horizon 내 대상으로 슬라이스. daily 주입·use_forecast=False로 결정론.
    """
    from ..forecast.forward import forecast_forward
    ff = forecast_forward(store_id, daily=daily, horizon_days=horizon_days,
                          use_forecast=False).item_quantities
    dates = pd.to_datetime(ff["date"])
    mask = (dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1]))
    sliced = ff.loc[mask]
    if sliced.empty:
        raise ValueError(f"forward 예측에 period {period} 대상 없음 (horizon 밖)")
    return (sliced.groupby("item_id", observed=True)["demand_point"].mean()
            .reset_index(name="demand_point"))


def _is_forward_period(daily: pd.DataFrame, store_id: str, period: tuple[str, str]) -> bool:
    """period 시작일이 store의 마지막 관측일 이후면 forward(미래) 대상으로 판정.

    forward면 _forward_demand_points(forecast_forward)로, historical이면 기존
    _item_demand_points(컬럼 평균)로 분기 — 그라운딩 eval-gold(historical
    min~max period) 등 기존 소비처와의 호환을 위한 폴백 경로(모듈 docstring 참조).
    """
    dates = pd.to_datetime(daily.loc[daily["store_id"] == store_id, "date"])
    last_observed = dates.max()
    return pd.notna(last_observed) and pd.Timestamp(period[0]) > last_observed


def _resolve_demand_points(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    *,
    demand_col: str | None = None,
) -> pd.DataFrame:
    """forward period면 forecast_forward 예측, 아니면 컬럼평균(historical fallback).

    ⚠️ demand_col은 historical 경로에만 적용된다(forward 경로는 forecast_forward가
    산출). ⚠️ period가 관측경계를 걸치면(시작<관측 last<끝) historical로 분류돼
    과거 부분만 요약된다.
    """
    if _is_forward_period(daily, store_id, period):
        return _forward_demand_points(daily, store_id, period)
    demand_col = demand_col or _resolve_demand_proxy(daily)
    return _item_demand_points(_period_slice(daily, store_id, *period), demand_col)


def rank_stockout_risk(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    k: int = 5,
    *,
    demand_col: str | None = None,
    policy: PolicyParams = PolicyParams(),
    risk: RiskParams = RiskParams(),
) -> pd.DataFrame:
    """Top-k items by P(stockout) for a store over a period (uses risk.py MC).

    demand_point: period가 forward(마지막 관측일 이후)면 forecast_forward 기반
    _forward_demand_points(5a 의도적 변경), historical이면 기존 컬럼-평균
    _item_demand_points로 폴백(그라운딩 eval-gold 등 과거 period 호환).
    """
    items = _resolve_demand_points(daily, store_id, period, demand_col=demand_col)
    rec = build_recommendation(items, policy=policy, risk=risk)
    ranked = rec.table.sort_values("p_stockout", ascending=False).head(k)
    return ranked.reset_index(drop=True)


def rank_stockout_earliness(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    k: int = 5,
    *,
    close_hour: int = DEFAULT_CLOSE_HOUR,
) -> pd.DataFrame:
    """Top-k items by observed stockout earliness: avg selling-hours lost per day.

    Score = mean over ALL the item's days of max(close_hour − stockout
    time-of-day, 0); days without a stockout contribute 0 — earliness and
    frequency in one number. Historical observation (stockout_time), NOT a
    forecast; complements the MC-based rank_stockout_risk.
    """
    sliced = _period_slice(daily, store_id, *period)
    stockout_at = pd.to_datetime(sliced["stockout_time"])
    hour_of_day = stockout_at.dt.hour + stockout_at.dt.minute / 60.0
    lost = (close_hour - hour_of_day).clip(lower=0.0).fillna(0.0)
    per_item = (
        sliced.assign(lost_hours=lost)
        .groupby("item_id", observed=True)
        .agg(lost_hours_total=("lost_hours", "sum"),
             stockout_days=("stockout_time", "count"),
             days=("lost_hours", "size"))
        .reset_index()
    )
    per_item["lost_hours_per_day"] = per_item["lost_hours_total"] / per_item["days"]
    if float(per_item["lost_hours_per_day"].max()) == 0.0:
        raise ValueError(f"no stockouts observed for store={store_id} in {period}")
    ranked = per_item.sort_values(["lost_hours_per_day", "item_id"],
                                  ascending=[False, True]).head(k)
    return ranked[["item_id", "lost_hours_per_day", "stockout_days", "days"]].reset_index(drop=True)


def explain_order(
    daily: pd.DataFrame,
    store_id: str,
    item_id: str,
    period: tuple[str, str],
    *,
    demand_col: str | None = None,
    policy: PolicyParams = PolicyParams(),
) -> pd.DataFrame:
    """Decision lineage for one item's order: base → safety → floor → rounding.

    demand_point: rank_stockout_risk와 동일한 forward/historical 분기(모듈
    docstring·rank_stockout_risk 참조).
    """
    items = _resolve_demand_points(daily, store_id, period, demand_col=demand_col)
    match = items.loc[items["item_id"] == item_id, "demand_point"]
    if match.empty:
        raise ValueError(f"item {item_id} not sold at {store_id} in {period}")
    _, lineage = apply_policy(item_id, float(match.iloc[0]), policy)
    return pd.DataFrame(lineage.to_records(), columns=["item_id", "step", "contribution", "detail"])


@dataclass(frozen=True)
class WhatIfResult:
    """Downstream lever: how risk/cost moves when the order qty is changed."""

    demand_point: float
    base_order: float
    new_order: float
    base_p_stockout: float
    new_p_stockout: float
    base_expected_cost: float
    new_expected_cost: float


def what_if(
    demand_point: float,
    base_order: float,
    delta_order: float,
    *,
    risk: RiskParams = RiskParams(),
) -> WhatIfResult:
    """Downstream what-if: re-score risk/cost for order = base_order + delta_order."""
    new_order = base_order + delta_order
    before = simulate_item_risk(demand_point, base_order, risk)
    after = simulate_item_risk(demand_point, new_order, risk)
    return WhatIfResult(
        demand_point=demand_point, base_order=base_order, new_order=new_order,
        base_p_stockout=before.p_stockout, new_p_stockout=after.p_stockout,
        base_expected_cost=before.expected_cost, new_expected_cost=after.expected_cost,
    )


def waste_cost(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    *,
    unit_cost: float = 1.0,
) -> dict:
    """Aggregate leftover (capacity − sold) cost for a store/period.

    Proxy waste = max(capacity − sold_units, 0); excludes stockout days where
    leftover is structurally zero. Normalized cost unless unit_cost is real KRW.

    Deliberately a *simplified* form of analysis/waste.py CapacityMinusSoldEstimator:
    it omits closing-discount quantities (that frame isn't in the dataset bundle)
    and adds the stockout-day=0 rule. Swap in the full estimator once discount
    data is wired into the ontology's backing frames.
    """
    period_df = _period_slice(daily, store_id, *period)
    leftover = (period_df["capacity"] - period_df["sold_units"]).clip(lower=0)
    # stockout days have structurally zero leftover (item ran out → nothing wasted)
    leftover = leftover.where(~period_df["is_stockout"], 0)
    units = float(leftover.sum())
    return {"store_id": store_id, "leftover_units": units, "waste_cost": units * unit_cost}


def demand_diff_by_condition(
    daily: pd.DataFrame,
    join_frame: pd.DataFrame,
    store_id: str,
    condition_col: str,
) -> dict:
    """Mean daily units when condition_col is on vs off (e.g. is_weekend, is_rain).

    Traverses DailySales → CalendarEvent/Weather via the link's join keys. The
    condition_col must be a 0/1 flag column on join_frame (calendar or weather).
    """
    if condition_col not in join_frame.columns:
        raise ValueError(f"condition_col {condition_col!r} not in join_frame")
    join_keys = ["store_id", "date"] if "store_id" in join_frame.columns else ["date"]
    cols = [*join_keys, condition_col]
    merged = daily.merge(join_frame[cols], on=join_keys, how="inner")
    store_rows = merged.loc[merged["store_id"] == store_id]
    if store_rows.empty:
        raise ValueError(f"no joined rows for store={store_id}")
    daily_units = store_rows.groupby(["date", condition_col], observed=True)["sold_units"].sum().reset_index()
    means = daily_units.groupby(condition_col, observed=True)["sold_units"].mean()
    on, off = float(means.get(CONDITION_ON, float("nan"))), float(means.get(CONDITION_OFF, float("nan")))
    return {"condition": condition_col, "mean_on": on, "mean_off": off, "diff": on - off}


@dataclass(frozen=True)
class OntologyFunctionSpec:
    """Agent-facing metadata for one function (name, params, return, impl)."""

    name: str
    description: str
    params: tuple[str, ...]
    returns: str
    impl: Callable
    side: str = "read"          # "read" | "write" — write는 LLM 도구 surface 제외


# The stable API surface the grounded agent enumerates and calls (S3 consumes this).
FUNCTION_REGISTRY: dict[str, OntologyFunctionSpec] = {
    "rank_stockout_risk": OntologyFunctionSpec(
        "rank_stockout_risk", "Top-k items by stockout probability for a store/period.",
        ("store_id", "period", "k"), "table[item_id, p_stockout, order_qty, ...]", rank_stockout_risk),
    "rank_stockout_earliness": OntologyFunctionSpec(
        "rank_stockout_earliness",
        "Top-k items by observed stockout earliness (avg selling-hours lost per day).",
        ("store_id", "period", "k"),
        "table[item_id, lost_hours_per_day, stockout_days, days]", rank_stockout_earliness),
    "explain_order": OntologyFunctionSpec(
        "explain_order", "Decision lineage breaking down one item's recommended order.",
        ("store_id", "item_id", "period"), "table[step, contribution, detail]", explain_order),
    "what_if": OntologyFunctionSpec(
        "what_if", "Downstream lever: risk/cost delta when an order qty changes.",
        ("demand_point", "base_order", "delta_order"), "WhatIfResult", what_if),
    "waste_cost": OntologyFunctionSpec(
        "waste_cost", "Aggregate leftover (capacity−sold) cost for a store/period.",
        ("store_id", "period"), "{leftover_units, waste_cost}", waste_cost),
    "demand_diff_by_condition": OntologyFunctionSpec(
        "demand_diff_by_condition", "Mean daily sales when a condition is on vs off.",
        ("store_id", "condition_col"), "{mean_on, mean_off, diff}", demand_diff_by_condition),
    "propose_order": OntologyFunctionSpec(
        "propose_order", "Write a PENDING order recommendation (human-approval-gated).",
        ("store_id", "item_id", "date", "proposed_qty"), "OrderRecord",
        WritebackStore.propose_order, side="write"),
    "commit_order": OntologyFunctionSpec(
        "commit_order", "Commit a PENDING order (approve, optionally correcting qty).",
        ("record_id", "approver", "approved_qty"), "OrderRecord",
        WritebackStore.approve, side="write"),
    "what_if_driver": OntologyFunctionSpec(
        "what_if_driver",
        "Upstream lever: perturb weather/calendar driver(s), re-forecast demand, propagate to stockout risk/cost.",
        ("store_id", "item_id", "period", "driver_overrides", "base_order"),
        "WhatIfDriverResult", scenario.what_if_driver, side="read"),
}

"""LLM-facing tool definitions for the 5 OntologyFunctions, plus the dispatch
that binds the dataset (which the LLM never sees) and calls the real function.

The LLM is shown only business arguments (store_id, period, item_id, ...);
dispatch() injects the daily/weather/calendar frames and serializes the result
to a JSON string for the tool-result turn.
"""

from __future__ import annotations

import json

from ...data.loader import DailyDataset
from .. import functions as fn
from .llm import ToolCall, ToolResult, ToolSpec

_PERIOD = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
           "description": "[start_date, end_date] as YYYY-MM-DD"}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec("rank_stockout_risk", "Top-k items by stockout probability for a store over a period.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "period": _PERIOD, "k": {"type": "integer"}},
              "required": ["store_id", "period", "k"], "additionalProperties": False}),
    ToolSpec("explain_order", "Decision lineage breaking down one item's recommended order.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "item_id": {"type": "string"}, "period": _PERIOD},
              "required": ["store_id", "item_id", "period"], "additionalProperties": False}),
    ToolSpec("what_if", "Downstream lever: risk/cost delta when an order qty changes.",
             {"type": "object", "properties": {
                 "demand_point": {"type": "number"}, "base_order": {"type": "number"},
                 "delta_order": {"type": "number"}},
              "required": ["demand_point", "base_order", "delta_order"], "additionalProperties": False}),
    ToolSpec("waste_cost", "Aggregate leftover (capacity-sold) cost for a store/period.",
             {"type": "object", "properties": {"store_id": {"type": "string"}, "period": _PERIOD},
              "required": ["store_id", "period"], "additionalProperties": False}),
    ToolSpec("demand_diff_by_condition", "Mean daily sales when a 0/1 condition is on vs off.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "condition_col": {"type": "string"},
                 "frame": {"type": "string", "enum": ["calendar", "weather"]}},
              "required": ["store_id", "condition_col", "frame"], "additionalProperties": False}),
]


def _to_jsonable(obj):
    """DataFrame → records, dataclass-ish → dict, else as-is."""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def dispatch(call: ToolCall, dataset: DailyDataset) -> ToolResult:
    """Run one tool call against the real OntologyFunction, return JSON result."""
    a = call.arguments
    try:
        result = _call(call.name, a, dataset)
        content = json.dumps(_to_jsonable(result), default=str)
    except Exception as exc:                       # surfaced to the model as a tool error
        content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return ToolResult(call_id=call.id, content=content)


def _call(name: str, a: dict, dataset: DailyDataset):
    if name == "rank_stockout_risk":
        return fn.rank_stockout_risk(dataset.daily, a["store_id"], tuple(a["period"]), a["k"])
    if name == "explain_order":
        return fn.explain_order(dataset.daily, a["store_id"], a["item_id"], tuple(a["period"]))
    if name == "what_if":
        return fn.what_if(a["demand_point"], a["base_order"], a["delta_order"])
    if name == "waste_cost":
        return fn.waste_cost(dataset.daily, a["store_id"], tuple(a["period"]))
    if name == "demand_diff_by_condition":
        frame = dataset.calendar if a["frame"] == "calendar" else dataset.weather
        return fn.demand_diff_by_condition(dataset.daily, frame, a["store_id"], a["condition_col"])
    raise KeyError(f"unknown tool: {name}")

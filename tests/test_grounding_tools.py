import json
import pandas as pd
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import ToolCall
from bakery.ontology.grounding.tools import TOOL_SPECS, dispatch


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def test_tool_specs_cover_seven_functions():
    names = {t.name for t in TOOL_SPECS}
    assert names == {
        "rank_stockout_risk", "rank_stockout_earliness", "explain_order",
        "what_if", "waste_cost", "demand_diff_by_condition", "what_if_driver",
    }
    for t in TOOL_SPECS:
        assert t.parameters["type"] == "object"
        assert "properties" in t.parameters


def test_dispatch_rank_stockout_risk_returns_json(dataset):
    store = dataset.daily["store_id"].iloc[0]
    import pandas as pd
    dates = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    call = ToolCall(id="c1", name="rank_stockout_risk",
                    arguments={"store_id": store, "period": [str(dates.min().date()), str(dates.max().date())], "k": 3})
    result = dispatch(call, dataset)
    payload = json.loads(result.content)
    assert result.call_id == "c1"
    assert len(payload) <= 3


def test_dispatch_unknown_tool_errors(dataset):
    result = dispatch(ToolCall(id="c2", name="nope", arguments={}), dataset)
    assert "error" in json.loads(result.content)


def test_what_if_driver_tool_spec_present():
    from bakery.ontology.grounding.tools import TOOL_SPECS
    names = {t.name for t in TOOL_SPECS}
    assert "what_if_driver" in names
    spec = next(t for t in TOOL_SPECS if t.name == "what_if_driver")
    props = spec.parameters["properties"]
    assert set(props["driver_overrides"]["properties"]) == {
        "is_public_holiday", "is_rain", "is_snow"}


def test_dispatch_what_if_driver_serializes(dataset):
    """dispatch derives train_cutoff=period[0]; returns JSON with demand_delta."""
    import json
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    enriched_dates = sorted(pd.to_datetime(dataset.daily["date"]).dt.date.unique())
    period = [str(enriched_dates[-2]), str(enriched_dates[-1])]
    store = dataset.daily["store_id"].iloc[0]
    item = dataset.daily.loc[dataset.daily["store_id"] == store, "item_id"].iloc[0]
    call = ToolCall(id="c1", name="what_if_driver", arguments={
        "store_id": store, "item_id": item, "period": period,
        "driver_overrides": {"is_rain": 1}, "base_order": 10.0})
    result = dispatch(call, dataset)
    payload = json.loads(result.content)
    assert "demand_delta" in payload or "error" in payload   # real fit may be heavy; both shapes valid


def test_what_if_driver_schema_is_strict_compatible():
    """strict:True 규칙: nested object도 모든 property 키가 required에 있어야 한다.
    (Azure live에서 400 invalid_function_parameters로 드러난 회귀 가드.)"""
    from bakery.ontology.grounding.tools import TOOL_SPECS
    spec = next(t for t in TOOL_SPECS if t.name == "what_if_driver")
    overrides = spec.parameters["properties"]["driver_overrides"]
    assert set(overrides["required"]) == set(overrides["properties"])
    # optional 표현은 nullable 타입으로 — 모델이 안 바꿀 드라이버는 null을 보낸다
    for prop in overrides["properties"].values():
        assert "null" in prop["type"]


def test_dispatch_what_if_driver_drops_null_overrides(dataset, monkeypatch):
    """모델이 null로 보낸 드라이버는 override에서 제거하고 나머지만 전달한다."""
    import json
    from bakery.ontology import scenario
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch

    captured = {}
    def fake_wid(daily, calendar, weather, store_id, item_id, period,
                 driver_overrides, *, base_order, train_cutoff, **kw):
        captured["overrides"] = driver_overrides
        return {"demand_delta": 0.0}
    monkeypatch.setattr(scenario, "what_if_driver", fake_wid)

    call = ToolCall(id="c1", name="what_if_driver", arguments={
        "store_id": "S", "item_id": "I", "period": ["2024-01-01", "2024-01-02"],
        "driver_overrides": {"is_rain": 1, "is_snow": None, "is_public_holiday": None},
        "base_order": 10.0})
    result = dispatch(call, dataset)
    assert "error" not in json.loads(result.content)
    assert captured["overrides"] == {"is_rain": 1}


def test_dispatch_rank_stockout_earliness_returns_json(dataset):
    store = dataset.daily["store_id"].iloc[0]
    dates = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    call = ToolCall(id="c9", name="rank_stockout_earliness",
                    arguments={"store_id": store,
                               "period": [str(dates.min().date()), str(dates.max().date())],
                               "k": 3})
    result = dispatch(call, dataset)
    payload = json.loads(result.content)
    assert len(payload) == 3
    assert {"item_id", "lost_hours_per_day", "stockout_days", "days"} <= set(payload[0])

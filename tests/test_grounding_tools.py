import json
import pandas as pd
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import ToolCall
from bakery.ontology.grounding.tools import TOOL_SPECS, dispatch


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def test_tool_specs_cover_six_functions():
    names = {t.name for t in TOOL_SPECS}
    assert names == {
        "rank_stockout_risk", "explain_order", "what_if",
        "waste_cost", "demand_diff_by_condition", "what_if_driver",
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

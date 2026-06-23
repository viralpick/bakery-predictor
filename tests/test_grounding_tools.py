import json
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import ToolCall
from bakery.ontology.grounding.tools import TOOL_SPECS, dispatch


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def test_tool_specs_cover_five_functions():
    names = {t.name for t in TOOL_SPECS}
    assert names == {
        "rank_stockout_risk", "explain_order", "what_if",
        "waste_cost", "demand_diff_by_condition",
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

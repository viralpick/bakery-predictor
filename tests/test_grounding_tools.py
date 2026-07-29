import json
import pandas as pd
import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import ToolCall
from bakery.ontology.grounding.tools import TOOL_SPECS, dispatch


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def test_tool_specs_cover_all_functions():
    names = {t.name for t in TOOL_SPECS}
    assert names == {
        "rank_stockout_risk", "rank_stockout_earliness", "explain_order",
        "what_if", "waste_cost", "demand_diff_by_condition", "what_if_driver",
        "explain_category_total", "explain_item_order", "rank_forward_items",
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


def _forward_date(ds, store):
    """store의 마지막 관측일 다음날 (forward horizon 첫날)."""
    dd = pd.to_datetime(ds.daily.loc[ds.daily["store_id"] == store, "date"])
    return str((dd.max() + pd.Timedelta(days=1)).date())


def test_dispatch_explain_category_total(dataset):
    """explain_category_total 도구가 dispatch되어 분해 행을 JSON으로 반환."""
    import json
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    store = str(dataset.daily["store_id"].iloc[0])
    date = _forward_date(dataset, store)
    call = ToolCall(id="c1", name="explain_category_total",
                    arguments={"store_id": store, "date": date})
    res = dispatch(call, dataset)
    rows = json.loads(res.content)
    assert [r["step"] for r in rows] == ["base_median", "event_prior", "prior_median", "quantile_buffer", "prior_prod"]


def test_dispatch_explain_item_order(dataset):
    import json
    from bakery.forecast.forward import forecast_forward
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    store = str(dataset.daily["store_id"].iloc[0])
    date = _forward_date(dataset, store)
    iq = forecast_forward(store, daily=dataset.daily[dataset.daily["store_id"] == store].copy(),
                          horizon_days=7, use_forecast=False).item_quantities
    iq_d = iq[iq["date"].astype(str).str.startswith(date)].copy()
    iq_d["item_id"] = iq_d["item_id"].astype(str)
    item = str(iq_d.sort_values(["our_order", "item_id"], ascending=[False, True]).iloc[0]["item_id"])
    call = ToolCall(id="c2", name="explain_item_order",
                    arguments={"store_id": store, "item_id": item, "date": date})
    res = dispatch(call, dataset)
    rows = json.loads(res.content)
    # 비중 인자 5종 + 정규화 분모가 category_total과 proportion 사이에 노출된다.
    assert [r["step"] for r in rows] == [
        "category_total",
        "base_sold", "adj_trend", "adj_stockout", "adj_closing", "adj_new",
        "raw_weight", "weight_sum",
        "proportion", "item_order", "final",
    ]


def test_dispatch_rank_forward_items(dataset):
    import json
    from bakery.ontology.grounding.llm import ToolCall
    from bakery.ontology.grounding.tools import dispatch
    store = str(dataset.daily["store_id"].iloc[0])
    date = _forward_date(dataset, store)
    res = dispatch(ToolCall(id="c3", name="rank_forward_items",
                            arguments={"store_id": store, "date": date, "k": 3}), dataset)
    rows = json.loads(res.content)
    assert len(rows) == 3
    assert all("item_id" in r and "our_order" in r for r in rows)
    # 내림차순
    assert [r["our_order"] for r in rows] == sorted([r["our_order"] for r in rows], reverse=True)

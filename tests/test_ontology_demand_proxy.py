import pandas as pd

from bakery.ontology.functions import _resolve_demand_proxy


def _frame(cols):
    base = {"store_id": ["S"], "item_id": ["A"],
            "date": pd.to_datetime(["2026-01-01"]), "sold_units": [10]}
    base.update({c: [10.0] for c in cols})
    return pd.DataFrame(base)


def test_proxy_prefers_adjusted_when_present():
    df = _frame(["potential_demand", "adjusted_demand"])
    assert _resolve_demand_proxy(df) == "adjusted_demand"


def test_proxy_falls_back_to_potential():
    df = _frame(["potential_demand"])
    assert _resolve_demand_proxy(df) == "potential_demand"


from bakery.data.loader import DailyDataset


def test_run_eval_enriches_real(monkeypatch):
    import bakery.ontology.grounding.run as run_mod

    captured = {}

    def fake_load(source):
        daily = pd.DataFrame({
            "store_id": ["S"], "item_id": ["A"],
            "date": pd.to_datetime(["2026-01-01"]), "sold_units": [10],
            "potential_demand": [10.0],
        })
        empty = pd.DataFrame()
        return DailyDataset(daily=daily, weather=empty, calendar=empty, competitor=empty,
                            living_population=empty, population=empty, consumption=empty)

    def fake_client(provider, model):
        return object()

    def fake_eval_with_client(client, dataset):
        captured["cols"] = list(dataset.daily.columns)
        from bakery.ontology.grounding.scorer import EvalReport
        return EvalReport(results=[], grounded_accuracy=0.0, rag_accuracy=0.0, delta=0.0)

    monkeypatch.setattr(run_mod, "load_dataset", fake_load)
    monkeypatch.setattr(run_mod, "make_llm_client", fake_client)
    monkeypatch.setattr(run_mod, "run_eval_with_client", fake_eval_with_client)
    monkeypatch.setattr(
        "bakery.ontology.grounding.run.build_item_adjusted_demand",
        lambda daily, alpha=0.5: daily.assign(adjusted_demand=daily["sold_units"].astype(float)),
    )

    run_mod.run_eval(source="real")
    assert "adjusted_demand" in captured["cols"]

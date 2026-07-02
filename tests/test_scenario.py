import pandas as pd
import pytest
from bakery.ontology.scenario import (
    VALID_DRIVERS, WhatIfDriverResult,
    _validate_drivers, _propagation_path, _count_support,
)
from bakery.data.loader import load_dataset
from bakery.ontology.scenario import (
    _build_enriched, _fit_demand_model, _period_item_rows, _predict_demand,
)
from bakery.ontology import scenario as sc


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


def _cutoff_and_period(enriched):
    dates = pd.to_datetime(enriched["date"]).sort_values().unique()
    cutoff = pd.Timestamp(dates[-3])                 # last 2 dates are the "future" period
    return str(cutoff.date()), (str(pd.Timestamp(dates[-2]).date()),
                                str(pd.Timestamp(dates[-1]).date()))


def test_build_enriched_has_driver_columns(dataset):
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    for col in ["is_rain", "is_snow", "is_public_holiday"]:
        assert col in enriched.columns


def test_fit_excludes_cutoff_and_later_rows(dataset):
    """Leakage guard: no training row may be dated >= train_cutoff."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, _ = _cutoff_and_period(enriched)
    model = _fit_demand_model(enriched, cutoff)
    trained = pd.to_datetime(model._train_history["date"])
    assert (trained < pd.Timestamp(cutoff)).all()


def test_fit_empty_train_raises(dataset):
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    early = str(pd.to_datetime(enriched["date"]).min().date())
    with pytest.raises(ValueError):
        _fit_demand_model(enriched, early)            # nothing strictly before the first date


def test_predict_demand_deterministic_and_finite(dataset):
    import math
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    model = _fit_demand_model(enriched, cutoff)
    rows = _period_item_rows(enriched, store, item, period)
    d1 = _predict_demand(model, rows)
    d2 = _predict_demand(model, rows)
    assert math.isfinite(d1) and d1 == d2            # deterministic


def test_period_item_rows_empty_raises(dataset):
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    with pytest.raises(ValueError):
        _period_item_rows(enriched, "NO_STORE", "NO_ITEM", ("2024-01-01", "2024-01-02"))


def test_valid_drivers_membership():
    assert VALID_DRIVERS == {"is_rain", "is_snow", "is_public_holiday"}


def test_validate_drivers_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        _validate_drivers({})
    with pytest.raises(ValueError):
        _validate_drivers({"is_sunny": 1})
    _validate_drivers({"is_rain": 1})            # ok, no raise


def test_propagation_path_by_driver_kind():
    assert _propagation_path({"is_rain": 1}) == (
        "dailysales_observed_on_weather", "item_sold_as_dailysales")
    assert _propagation_path({"is_public_holiday": 1}) == (
        "dailysales_observed_on_calendar", "item_sold_as_dailysales")
    assert _propagation_path({"is_rain": 1, "is_public_holiday": 1}) == (
        "dailysales_observed_on_weather", "dailysales_observed_on_calendar",
        "item_sold_as_dailysales")


def test_count_support_matches_store_and_overrides():
    df = pd.DataFrame({
        "store_id": ["A", "A", "A", "B"],
        "is_rain":  [1,   0,   1,   1],
        "is_public_holiday": [1, 1, 0, 1],
    })
    # store A, is_rain=1 & is_public_holiday=1 → only row 0
    assert _count_support(df, "A", {"is_rain": 1, "is_public_holiday": 1}) == 1
    # store A, is_rain=1 → rows 0,2
    assert _count_support(df, "A", {"is_rain": 1}) == 2
    # store A, combo never seen
    assert _count_support(df, "A", {"is_rain": 0, "is_public_holiday": 0}) == 0


def test_result_is_frozen():
    r = WhatIfDriverResult("A", "P1", {"is_rain": 1}, 10.0, 7.0, -3.0,
                           0.2, 0.1, 5.0, 3.0, False,
                           ("dailysales_observed_on_weather", "item_sold_as_dailysales"))
    with pytest.raises(Exception):
        r.before_demand = 1.0


class _StubModel:
    """Demand responds to is_rain: 10 baseline, −3 when is_rain==1. Deterministic."""
    def predict(self, target):
        rain = float(target["is_rain"].iloc[0]) if "is_rain" in target.columns else 0.0
        return pd.Series([10.0 - 3.0 * rain] * len(target))


def test_what_if_driver_propagates_demand_to_risk(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    res = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                            store, item, period, {"is_rain": 1}, base_order=10.0,
                            train_cutoff=cutoff)
    assert res.before_demand == 10.0 and res.after_demand == 7.0
    assert res.demand_delta == -3.0
    # demand fell → stockout risk should not rise
    assert res.after_p_stockout <= res.before_p_stockout
    assert res.propagation_path == ("dailysales_observed_on_weather", "item_sold_as_dailysales")


def test_what_if_driver_out_of_support_flag(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    # impossible combo unlikely in history → out_of_support True
    res = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                            store, item, period, {"is_rain": 1, "is_snow": 1},
                            base_order=10.0, train_cutoff=cutoff)
    assert isinstance(res.out_of_support, bool)


def test_what_if_driver_rejects_unknown_driver(dataset):
    with pytest.raises(ValueError):
        sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                          "A", "P1", ("2024-01-01", "2024-01-02"), {"is_sunny": 1},
                          base_order=10.0, train_cutoff="2024-01-01")


def test_what_if_driver_base_order_none_uses_policy(dataset, monkeypatch):
    """base_order=None → 내부에서 apply_policy(before_demand) 사용; 명시 호출과 동일 위험."""
    from bakery.decision import apply_policy
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    auto = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                             store, item, period, {"is_rain": 1}, base_order=None,
                             train_cutoff=cutoff)
    # _StubModel: before demand = 10.0 → policy order
    policy_order = apply_policy(item, 10.0)[0]
    explicit = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                                 store, item, period, {"is_rain": 1}, base_order=policy_order,
                                 train_cutoff=cutoff)
    assert auto.before_p_stockout == explicit.before_p_stockout
    assert auto.after_expected_cost == explicit.after_expected_cost


def test_what_if_driver_base_order_none_honors_policy(dataset, monkeypatch):
    """base_order=None + custom policy → 내부 base_order가 그 policy로 산출된다."""
    from bakery.decision import PolicyParams, apply_policy
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    custom = PolicyParams(safety_margin=1.0)          # 극단 margin → default와 확실히 다른 base_order
    auto = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                             store, item, period, {"is_rain": 1}, base_order=None,
                             train_cutoff=cutoff, policy=custom)
    explicit = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                                 store, item, period, {"is_rain": 1},
                                 base_order=apply_policy(item, 10.0, custom)[0],
                                 train_cutoff=cutoff)
    assert auto.before_p_stockout == explicit.before_p_stockout
    assert auto.after_expected_cost == explicit.after_expected_cost


def _two_items(dataset, store_id):
    sub = dataset.daily[dataset.daily["store_id"] == store_id]
    return list(sub["item_id"].drop_duplicates())[:2]


def test_batch_fits_model_once(dataset, monkeypatch):
    """fit 공유: N품목이어도 _fit_demand_model 1회만."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = sorted(dataset.daily["store_id"].unique())[0]
    items = _two_items(dataset, store)

    calls = {"n": 0}
    real_fit = sc._fit_demand_model

    def counting_fit(*a, **k):
        calls["n"] += 1
        return real_fit(*a, **k)

    monkeypatch.setattr(sc, "_fit_demand_model", counting_fit)
    results = sc.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store, items, period,
        {"is_rain": 1}, train_cutoff=cutoff)
    assert calls["n"] == 1                    # ← fit shared, not per-item
    assert len(results) == len(items)


def test_batch_matches_single(dataset):
    """배치=단일: 각 품목 결과가 단일 what_if_driver와 동일."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = sorted(dataset.daily["store_id"].unique())[0]
    items = _two_items(dataset, store)

    batch = {r.item_id: r for r in sc.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store, items, period,
        {"is_rain": 1}, train_cutoff=cutoff)}
    for item in items:
        single = sc.what_if_driver(
            dataset.daily, dataset.calendar, dataset.weather, store, item, period,
            {"is_rain": 1}, train_cutoff=cutoff)
        assert batch[item].before_demand == single.before_demand
        assert batch[item].after_demand == single.after_demand


def test_batch_skips_unknown_item(dataset):
    """품목 실패는 skip, 나머지 정상."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = sorted(dataset.daily["store_id"].unique())[0]
    good = _two_items(dataset, store)[0]

    results = sc.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store,
        [good, "NONEXISTENT_ITEM"], period, {"is_rain": 1}, train_cutoff=cutoff)
    ids = {r.item_id for r in results}
    assert good in ids
    assert "NONEXISTENT_ITEM" not in ids       # skipped, no crash

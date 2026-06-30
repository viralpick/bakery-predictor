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

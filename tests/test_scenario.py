import pandas as pd
import pytest
from bakery.ontology.scenario import (
    VALID_DRIVERS, WhatIfDriverResult,
    _validate_drivers, _propagation_path, _count_support,
)


def test_valid_drivers_membership():
    assert VALID_DRIVERS == {"is_rain", "is_snow", "is_weekend", "is_off_day", "is_public_holiday"}


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
    assert _propagation_path({"is_rain": 1, "is_off_day": 1}) == (
        "dailysales_observed_on_weather", "dailysales_observed_on_calendar",
        "item_sold_as_dailysales")


def test_count_support_matches_store_and_overrides():
    df = pd.DataFrame({
        "store_id": ["A", "A", "A", "B"],
        "is_rain":  [1,   0,   1,   1],
        "is_off_day": [1, 1,   0,   1],
    })
    # store A, is_rain=1 & is_off_day=1 → only row 0
    assert _count_support(df, "A", {"is_rain": 1, "is_off_day": 1}) == 1
    # store A, is_rain=1 → rows 0,2
    assert _count_support(df, "A", {"is_rain": 1}) == 2
    # store A, combo never seen
    assert _count_support(df, "A", {"is_rain": 0, "is_off_day": 0}) == 0


def test_result_is_frozen():
    r = WhatIfDriverResult("A", "P1", {"is_rain": 1}, 10.0, 7.0, -3.0,
                           0.2, 0.1, 5.0, 3.0, False,
                           ("dailysales_observed_on_weather", "item_sold_as_dailysales"))
    with pytest.raises(Exception):
        r.before_demand = 1.0

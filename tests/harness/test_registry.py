import pytest
from bakery.harness.registry import (
    ForecasterKind, kind_of, LAYER_NAMES, is_runnable, build_forecaster,
)


def test_kind_taxonomy():
    assert kind_of("category_total") == ForecasterKind.CATEGORY_TOTAL
    assert kind_of("distributional_total") == ForecasterKind.DISTRIBUTIONAL
    assert kind_of("lightgbm_v2") == ForecasterKind.POINT
    assert kind_of("category_v4") == ForecasterKind.COMPOSITE


def test_unknown_raises():
    with pytest.raises(KeyError):
        kind_of("bogus")


def test_layers_registered():
    assert "event_prior" in LAYER_NAMES
    assert "decision" in LAYER_NAMES


def test_is_runnable():
    assert is_runnable("category_total") is True
    assert is_runnable("distributional_total") is True
    assert is_runnable("lightgbm_v2") is False
    assert is_runnable("bogus") is False


def test_build_forecaster():
    from bakery.harness.forecasters import CategoryTotalForecaster, DistributionalTotalForecaster
    assert isinstance(build_forecaster("category_total"), CategoryTotalForecaster)
    assert isinstance(build_forecaster("distributional_total"), DistributionalTotalForecaster)


def test_build_forecaster_unknown_raises():
    with pytest.raises(KeyError):
        build_forecaster("lightgbm_v2")

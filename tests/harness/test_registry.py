import pytest
from bakery.harness.registry import ForecasterKind, kind_of, LAYER_NAMES, is_supported_phase1


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


def test_phase1_supports_category_total_only():
    assert is_supported_phase1("category_total") is True
    assert is_supported_phase1("distributional_total") is False

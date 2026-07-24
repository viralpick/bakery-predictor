import pytest, yaml
from bakery.harness.config import (
    ExperimentSpec, load_spec, SpecError,
    DEFAULT_FORECASTERS, DEFAULT_LAYERS, DEFAULT_METRICS,
)


def _write(tmp_path, body):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


def test_defaults_are_category_stack(tmp_path):
    spec = load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"}}))
    assert spec.forecaster == DEFAULT_FORECASTERS       # [category_total, distributional_total]
    assert spec.layers == DEFAULT_LAYERS                # [event_prior]
    assert spec.target == "adjusted_demand_unit"
    assert spec.data.store == "store_gw01"
    assert spec.window.n_folds == 52
    assert spec.window.window_days == 730
    assert spec.alpha == 0.8
    assert spec.event_priors == "gwangyo"


def test_potential_demand_rejected(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "target": "potential_demand"}
    with pytest.raises(SpecError, match="potential_demand"):
        load_spec(_write(tmp_path, body))


def test_random_split_rejected(tmp_path):
    with pytest.raises(SpecError):
        load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"},
                                    "window": {"scheme": "random"}}))


def test_mape_only_warns(tmp_path):
    with pytest.warns(UserWarning, match="MAPE"):
        load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"}, "metrics": ["mape"]}))


def test_event_prior_without_preset_warns(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "layers": ["event_prior"], "event_priors": None}
    with pytest.warns(UserWarning, match="event_prior"):
        load_spec(_write(tmp_path, body))


def test_single_forecaster_string_wrapped(tmp_path):
    spec = load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"},
                                       "forecaster": "category_total"}))
    assert spec.forecaster == ["category_total"]

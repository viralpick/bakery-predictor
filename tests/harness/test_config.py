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
    assert spec.metrics == ["wape", "wpe", "stockout_risk", "surplus_mean_units", "surplus_rate"]


def test_operational_window_defaults_are_current_behavior(tmp_path):
    """lead_days/anchor_dow 기본값 = 현 헤드라인 동작(리드타임 0, 인덱스 기반 fold)."""
    spec = load_spec(_write(tmp_path, {"name": "x", "data": {"source": "real"}}))
    assert spec.window.lead_days == 0
    assert spec.window.anchor_dow is None


def test_operational_window_opt_in(tmp_path):
    body = {"name": "x", "data": {"source": "real"},
            "window": {"lead_days": 5, "anchor_dow": 0}}
    spec = load_spec(_write(tmp_path, body))
    assert spec.window.lead_days == 5
    assert spec.window.anchor_dow == 0


@pytest.mark.parametrize("bad", [-1, 7])
def test_anchor_dow_out_of_range_rejected(tmp_path, bad):
    body = {"name": "x", "data": {"source": "real"}, "window": {"anchor_dow": bad}}
    with pytest.raises(SpecError, match="anchor_dow"):
        load_spec(_write(tmp_path, body))


def test_negative_lead_days_rejected(tmp_path):
    body = {"name": "x", "data": {"source": "real"}, "window": {"lead_days": -1}}
    with pytest.raises(SpecError, match="lead_days"):
        load_spec(_write(tmp_path, body))


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

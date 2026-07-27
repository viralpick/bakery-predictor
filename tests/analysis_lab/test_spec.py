from pathlib import Path

import pytest
import yaml

from bakery.analysis.lab.spec import (
    DEFAULT_ALPHA,
    DEPRECATED_ANALYSES,
    MULTISTORE,
    AnalysisSpec,
    AnalysisSpecError,
    load_analysis_spec,
)


def _write(tmp_path, body):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def _base(**over):
    body = {"name": "analysis_gwangyo", "data": {"source": "real"}}
    body.update(over)
    return body


def test_defaults_are_gwangyo_alpha_08(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base()))
    assert spec.data.store == "store_gw01"
    assert spec.alpha == DEFAULT_ALPHA == 0.8
    assert spec.predictions is None
    assert spec.data_analyses == {}
    assert spec.hypotheses == {}
    assert spec.params == {}


def test_enabled_returns_only_true_keys(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base(
        data_analyses={"category_mix": True, "waste_rate": False},
        hypotheses={"demand_absorption": True, "substitution": False},
    )))
    assert spec.enabled("data_analyses") == ["category_mix"]
    assert spec.enabled("hypotheses") == ["demand_absorption"]


def test_deprecated_conformal_names_rejected(tmp_path):
    # v5 conformal 구간예측은 점추정+위험수치 전환으로 폐기 — 실수 이식 차단
    assert frozenset(
        {"diag_anchor_gh", "diag_chuseok_gh", "diagnose_conformal_residual"}) == DEPRECATED_ANALYSES
    path = _write(tmp_path, _base(hypotheses={"diag_anchor_gh": True}))
    with pytest.raises(AnalysisSpecError, match="diag_anchor_gh"):
        load_analysis_spec(path)


def test_deprecated_name_rejected_even_when_off(tmp_path):
    # off여도 spec에 남아 있으면 "이식 대상"으로 오해되므로 거부한다
    path = _write(tmp_path, _base(hypotheses={"diagnose_conformal_residual": False}))
    with pytest.raises(AnalysisSpecError, match="diagnose_conformal_residual"):
        load_analysis_spec(path)


def test_potential_demand_target_is_not_configurable(tmp_path):
    # 오염 소스 차단: target 키 자체를 받지 않는다(입력 데이터 평면은 target이 없음)
    with pytest.raises(AnalysisSpecError, match="target"):
        load_analysis_spec(_write(tmp_path, _base(target="potential_demand")))


def test_unknown_name_rejected_when_known_names_given(tmp_path):
    path = _write(tmp_path, _base(hypotheses={"demand_absorbtion": True}))   # 오타
    with pytest.raises(AnalysisSpecError, match="demand_absorbtion"):
        load_analysis_spec(path, known_names=frozenset({"demand_absorption"}))


def test_multistore_store_value_accepted(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base(data={"source": "real", "store": MULTISTORE})))
    assert spec.data.store == "multistore"


def test_synthetic_source_rejected(tmp_path):
    # 이 레이어는 실측 데이터/가설 검증 전용 — synthetic은 의미 없음
    with pytest.raises(AnalysisSpecError):
        load_analysis_spec(_write(tmp_path, _base(data={"source": "synthetic"})))


def test_predictions_path_is_parsed_as_path(tmp_path):
    spec = load_analysis_spec(_write(tmp_path, _base(
        predictions="reports/gwangyo_default/category_total/predictions.csv")))
    assert spec.predictions == Path("reports/gwangyo_default/category_total/predictions.csv")


def test_alpha_out_of_range_rejected(tmp_path):
    with pytest.raises(AnalysisSpecError):
        load_analysis_spec(_write(tmp_path, _base(alpha=1.4)))


def test_spec_constructed_directly_has_same_defaults():
    spec = AnalysisSpec(name="x", data={"source": "real"})
    assert spec.alpha == 0.8
    assert spec.data.store == "store_gw01"

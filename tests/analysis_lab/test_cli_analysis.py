import yaml
from typer.testing import CliRunner

from bakery.cli import app

runner = CliRunner()


def _yaml(tmp_path, body):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def test_analysis_run_all_off_produces_html(tmp_path):
    # 모든 항목 off → 계산 없이 HTML만. 실데이터 IO 없이 CLI 배선을 검증한다.
    config = _yaml(tmp_path, {"name": "analysis_smoke", "data": {"source": "real"}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 0
    html = (tmp_path / "analysis_smoke" / "analysis_report.html").read_text(encoding="utf-8")
    assert "섹션 A — 입력 데이터 분석" in html
    assert "섹션 B — 가설 검증" in html


def test_off_items_are_labelled_in_html(tmp_path):
    config = _yaml(tmp_path, {"name": "analysis_smoke", "data": {"source": "real"}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 0
    html = (tmp_path / "analysis_smoke" / "analysis_report.html").read_text(encoding="utf-8")
    assert "(off)" in html


def test_analysis_run_rejects_deprecated_name(tmp_path):
    config = _yaml(tmp_path, {"name": "x", "data": {"source": "real"},
                              "hypotheses": {"diag_anchor_gh": True}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 1


def test_analysis_run_rejects_unknown_name(tmp_path):
    config = _yaml(tmp_path, {"name": "x", "data": {"source": "real"},
                              "hypotheses": {"demand_absorbtion": True}})
    result = runner.invoke(app, ["analysis-run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 1


def test_shipped_gwangyo_yaml_loads_with_registry_names():
    # Task 17에서 seasonal_bias/weather_bias/event_prior_validation 등록 완료 —
    # experiments/analysis_gwangyo.yaml의 모든 키가 registry에 실제 등록돼 있는지
    from bakery.analysis.lab.registry import all_names
    from bakery.analysis.lab.spec import load_analysis_spec

    spec = load_analysis_spec("experiments/analysis_gwangyo.yaml", known_names=all_names())
    assert spec.name == "analysis_gwangyo"
    assert spec.data.store == "store_gw01"


def test_shipped_multistore_yaml_is_multistore():
    # Task 13에서 stockout_lost_demand 등록으로 이 YAML의 이름이 전부 커버됨 — xfail 해제.
    from bakery.analysis.lab.registry import all_names
    from bakery.analysis.lab.spec import load_analysis_spec

    spec = load_analysis_spec("experiments/analysis_multistore.yaml", known_names=all_names())
    assert spec.data.store == "multistore"

import pandas as pd
import pytest
import yaml

from bakery.analysis.lab import registry
from bakery.analysis.lab.result import (
    KIND_DATA,
    KIND_HYPOTHESIS,
    REASON_MULTISTORE_REQUIRED,
    REASON_OFF,
    REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED,
    AnalysisResult,
)
from bakery.analysis.lab.runner import run_analysis
from bakery.analysis.lab.spec import AnalysisSpec


def _fake_result(name, kind):
    return AnalysisResult(name=name, kind=kind, title="t",
                          tables=[("x", pd.DataFrame({"v": [1]}))], figures=[],
                          verdict="지지" if kind == KIND_HYPOTHESIS else None)


@pytest.fixture
def fake_registry(monkeypatch):
    """실 핸들러 대신 계산 없는 스텁 5개(data 1 + hypothesis 4)를 등록해 runner 배선만 검증한다."""
    calls: list[str] = []

    def _make(name, kind):
        def fn(inputs):
            calls.append(name)
            return _fake_result(name, kind)
        return fn

    data = {"stub_data": registry.Handler("stub_data", KIND_DATA, "스텁 데이터",
                                         _make("stub_data", KIND_DATA))}
    hypo = {
        "stub_hypo": registry.Handler("stub_hypo", KIND_HYPOTHESIS, "스텁 가설",
                                      _make("stub_hypo", KIND_HYPOTHESIS)),
        "stub_preds": registry.Handler("stub_preds", KIND_HYPOTHESIS, "스텁 preds",
                                       _make("stub_preds", KIND_HYPOTHESIS),
                                       needs_predictions=True),
        "stub_ms": registry.Handler("stub_ms", KIND_HYPOTHESIS, "스텁 다매장",
                                    _make("stub_ms", KIND_HYPOTHESIS),
                                    needs_multistore=True),
        "stub_single": registry.Handler("stub_single", KIND_HYPOTHESIS, "스텁 단매장",
                                        _make("stub_single", KIND_HYPOTHESIS),
                                        needs_single_store=True),
    }
    monkeypatch.setattr(registry, "DATA_ANALYSES", data)
    monkeypatch.setattr(registry, "HYPOTHESES", hypo)
    monkeypatch.setattr(registry, "load_handlers", lambda: None)
    return calls


def _spec(**over):
    body = {"name": "t", "data": {"source": "real"}}
    body.update(over)
    return AnalysisSpec(**body)


def test_only_enabled_items_run(fake_registry, tmp_path):
    report = run_analysis(_spec(data_analyses={"stub_data": True},
                                hypotheses={"stub_hypo": False}), out_dir=tmp_path)
    assert fake_registry == ["stub_data"]
    assert [r.name for r in report.results] == ["stub_data"]


def test_off_item_is_recorded_with_reason_off(fake_registry, tmp_path):
    # data_analyses에 stub_data를 명시적으로 켜 둔다. "off도 남긴다" 핵심 계약
    # (아래 test_unrequested_registry_items_are_also_listed_as_off)상 stub_preds/
    # stub_ms/stub_single도 미요청이라 off로 잡히므로, stub_hypo 항목만 골라
    # 비교한다(다른 테스트들의 name== 필터 패턴과 동일).
    report = run_analysis(_spec(data_analyses={"stub_data": True},
                                hypotheses={"stub_hypo": False}), out_dir=tmp_path)
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_hypo"] == [("stub_hypo", REASON_OFF)]


def test_unrequested_registry_items_are_also_listed_as_off(fake_registry, tmp_path):
    # 은폐 방지: spec에 없는 항목도 off로 리포트에 남는다
    report = run_analysis(_spec(data_analyses={"stub_data": True}), out_dir=tmp_path)
    assert {s.name for s in report.skipped} == {"stub_hypo", "stub_preds", "stub_ms",
                                               "stub_single"}
    assert {s.reason for s in report.skipped} == {REASON_OFF}


def test_preds_required_item_skipped_without_artifact(fake_registry, tmp_path):
    report = run_analysis(_spec(hypotheses={"stub_preds": True}), out_dir=tmp_path)
    assert fake_registry == []
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_preds"] == [("stub_preds", REASON_PREDS_REQUIRED)]


def test_preds_required_item_runs_with_artifact(fake_registry, tmp_path):
    preds = tmp_path / "predictions.csv"
    preds.write_text("date,fold,actual,expected,production\n2025-01-01,0,1,1,1\n",
                     encoding="utf-8")
    report = run_analysis(_spec(hypotheses={"stub_preds": True}, predictions=preds),
                          out_dir=tmp_path)
    assert fake_registry == ["stub_preds"]
    assert [r.name for r in report.results] == ["stub_preds"]


def test_multistore_item_skipped_on_single_store_spec(fake_registry, tmp_path):
    report = run_analysis(_spec(hypotheses={"stub_ms": True}), out_dir=tmp_path)
    assert fake_registry == []
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_ms"] == [("stub_ms", REASON_MULTISTORE_REQUIRED)]


def test_multistore_item_runs_on_multistore_spec(fake_registry, tmp_path):
    run_analysis(_spec(data={"source": "real", "store": "multistore"},
                       hypotheses={"stub_ms": True}), out_dir=tmp_path)
    assert fake_registry == ["stub_ms"]


def test_single_store_item_skipped_on_multistore_spec(fake_registry, tmp_path):
    # 광교 전용 소스를 4매장 라벨로 내보내는 조용한 오데이터 차단
    report = run_analysis(_spec(data={"source": "real", "store": "multistore"},
                                hypotheses={"stub_single": True}), out_dir=tmp_path)
    assert fake_registry == []
    assert [(s.name, s.reason) for s in report.skipped
            if s.name == "stub_single"] == [("stub_single", REASON_SINGLE_STORE_REQUIRED)]


def test_single_store_item_runs_on_single_store_spec(fake_registry, tmp_path):
    run_analysis(_spec(hypotheses={"stub_single": True}), out_dir=tmp_path)
    assert fake_registry == ["stub_single"]


def test_resolved_config_written_to_out_dir(fake_registry, tmp_path):
    run_analysis(_spec(name="analysis_x", data_analyses={"stub_data": True}), out_dir=tmp_path)
    written = yaml.safe_load((tmp_path / "analysis_x" / "config_resolved.yaml").read_text())
    assert written["name"] == "analysis_x"
    assert written["alpha"] == 0.8


def test_tables_written_as_csv(fake_registry, tmp_path):
    run_analysis(_spec(name="analysis_x", data_analyses={"stub_data": True}), out_dir=tmp_path)
    csv = pd.read_csv(tmp_path / "analysis_x" / "stub_data__x.csv")
    assert csv["v"].tolist() == [1]


def test_handler_exception_becomes_skip_not_crash(fake_registry, tmp_path, monkeypatch):
    def boom(inputs):
        raise ValueError("데이터 부족")
    monkeypatch.setitem(registry.DATA_ANALYSES, "stub_data",
                        registry.Handler("stub_data", KIND_DATA, "스텁 데이터", boom))
    report = run_analysis(_spec(data_analyses={"stub_data": True}), out_dir=tmp_path)
    assert report.results == []
    assert [s.reason for s in report.skipped if s.name == "stub_data"] == ["error: 데이터 부족"]


def test_table_serialization_failure_degrades_to_error_skip_not_crash(
        fake_registry, tmp_path, monkeypatch):
    """fix(final review 6): _write_tables/results.append이 try 밖에 있으면 to_csv 실패
    (여기서는 path-hostile 테이블 라벨이 존재하지 않는 하위디렉터리를 가리켜 발생)가
    이 항목만이 아니라 run_analysis 전체를 죽였다 — 다른 항목(stub_hypo)은 계속
    실행되고 리포트가 나와야 한다는 "항목 단위 격리" 계약을 지킨다.
    """
    def bad_table(inputs):
        return AnalysisResult(name="stub_data", kind=KIND_DATA, title="스텁 데이터",
                              tables=[("bad/label", pd.DataFrame({"v": [1]}))], figures=[])

    monkeypatch.setitem(registry.DATA_ANALYSES, "stub_data",
                        registry.Handler("stub_data", KIND_DATA, "스텁 데이터", bad_table))
    report = run_analysis(_spec(data_analyses={"stub_data": True},
                                hypotheses={"stub_hypo": True}), out_dir=tmp_path)
    assert [r.name for r in report.results] == ["stub_hypo"]
    bad_skips = [s for s in report.skipped if s.name == "stub_data"]
    assert len(bad_skips) == 1
    assert bad_skips[0].reason.startswith("error: ")


def test_report_carries_spec_resolved(fake_registry, tmp_path):
    report = run_analysis(_spec(name="analysis_x"), out_dir=tmp_path)
    assert report.name == "analysis_x"
    assert report.spec_resolved["data"]["store"] == "store_gw01"

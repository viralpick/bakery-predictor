import pandas as pd
import pytest

from bakery.analysis.lab.result import (
    KIND_DATA,
    KIND_HYPOTHESIS,
    REASON_OFF,
    REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED,
    AnalysisReport,
    AnalysisResult,
    SkippedResult,
)


def _result(name="category_mix", kind=KIND_DATA):
    return AnalysisResult(
        name=name, kind=kind, title="카테고리 매출 비중",
        tables=[("share", pd.DataFrame({"category_id": ["bread"], "share": [1.0]}))],
        figures=[],
    )


def test_verdict_defaults_to_none_and_notes_empty():
    r = _result()
    assert r.verdict is None
    assert r.notes == []


def test_notes_are_not_shared_between_instances():
    a, b = _result(), _result()
    a.notes.append("주의")
    assert b.notes == []          # 가변 기본값 공유 버그 방지


def test_kind_and_reason_constants_are_exact():
    assert KIND_DATA == "data"
    assert KIND_HYPOTHESIS == "hypothesis"
    assert REASON_OFF == "off"
    assert REASON_PREDS_REQUIRED == "preds_required"
    assert REASON_SINGLE_STORE_REQUIRED == "single_store_required"


def test_report_table_of_returns_the_named_table():
    report = AnalysisReport(
        name="analysis_gwangyo", spec_resolved={"name": "analysis_gwangyo"},
        results=[_result()],
        skipped=[SkippedResult(name="substitution", kind=KIND_HYPOTHESIS,
                               title="수요 대체", reason=REASON_OFF)],
    )
    table = report.table_of("category_mix", "share")
    assert table["share"].tolist() == [1.0]


def test_report_table_of_raises_on_unknown_name():
    report = AnalysisReport(name="x", spec_resolved={}, results=[_result()], skipped=[])
    with pytest.raises(KeyError, match="waste_rate"):
        report.table_of("waste_rate", "share")

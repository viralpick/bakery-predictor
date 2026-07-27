import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.report import build_analysis_report, fig_to_div
from bakery.analysis.lab.result import (
    KIND_DATA,
    KIND_HYPOTHESIS,
    REASON_OFF,
    REASON_PREDS_REQUIRED,
    AnalysisReport,
    AnalysisResult,
    SkippedResult,
)


def _report():
    share = pd.DataFrame({"category_id": ["bread", "pastry"], "share": [0.7, 0.3]})
    absorb = pd.DataFrame({"category_id": ["bread"], "beta": [-0.01], "verdict": ["absorb"]})
    return AnalysisReport(
        name="analysis_gwangyo",
        spec_resolved={"name": "analysis_gwangyo", "data": {"source": "real",
                                                            "store": "store_gw01"}},
        results=[
            AnalysisResult(name="category_mix", kind=KIND_DATA, title="카테고리 매출 비중",
                           tables=[("share", share)],
                           figures=[go.Figure(go.Bar(x=["bread"], y=[0.7]))]),
            AnalysisResult(name="demand_absorption", kind=KIND_HYPOTHESIS,
                           title="카테고리 총량 수요이전 흡수",
                           tables=[("results", absorb)],
                           figures=[go.Figure(go.Bar(x=["bread"], y=[-0.01]))],
                           verdict="지지 — 일반 카테고리 walk-away 0건",
                           notes=["censoring 무시 가정(측정 헌장)"]),
        ],
        skipped=[
            SkippedResult(name="substitution", kind=KIND_HYPOTHESIS, title="수요 대체",
                          reason=REASON_OFF),
            SkippedResult(name="weekday_bias", kind=KIND_HYPOTHESIS, title="평일 과대예측",
                          reason=REASON_PREDS_REQUIRED),
        ],
    )


def test_html_written_and_returns_path(tmp_path):
    out = tmp_path / "analysis_report.html"
    assert build_analysis_report(_report(), out_path=out) == out
    assert out.exists()


def test_both_sections_and_titles_present(tmp_path):
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "입력 데이터 분석" in html
    assert "가설 검증" in html
    assert "카테고리 매출 비중" in html
    assert "카테고리 총량 수요이전 흡수" in html


def test_verdict_and_notes_rendered(tmp_path):
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "지지 — 일반 카테고리 walk-away 0건" in html
    assert "censoring 무시 가정(측정 헌장)" in html


def test_skipped_items_are_disclosed_with_reasons(tmp_path):
    # 은폐 방지 = 성공기준. off와 preds 부재는 서로 다른 라벨로 나와야 한다.
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "substitution" in html
    assert "(off)" in html
    assert "weekday_bias" in html
    assert "(preds 필요 — 미실행)" in html


def test_plotly_js_embedded_exactly_once(tmp_path):
    # stateless fig_to_div 불변: fig가 여러 개여도 plotly.js는 1회만
    out = tmp_path / "r.html"
    build_analysis_report(_report(), out_path=out)
    html = out.read_text(encoding="utf-8")
    assert html.count("cdn.plot.ly") == 1


def test_report_without_any_figure_still_embeds_js(tmp_path):
    report = _report()
    for result in report.results:
        result.figures = []
    out = tmp_path / "r.html"
    build_analysis_report(report, out_path=out)
    html = out.read_text(encoding="utf-8")
    assert html.count("cdn.plot.ly") == 1        # on/off 요약표가 첫 fig 역할


def test_fig_to_div_include_js_toggle():
    fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
    assert "cdn.plot.ly" in fig_to_div(fig, "d1", include_js=True)
    assert "cdn.plot.ly" not in fig_to_div(fig, "d2", include_js=False)

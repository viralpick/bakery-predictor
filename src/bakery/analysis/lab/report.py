"""AnalysisReport → 자기포함 HTML. harness report.py의 stateless fig_to_div 패턴 재사용.

섹션 A(입력 데이터 분석) / B(가설 검증) + 실행 요약표. 끈 항목·못 돌린 항목은
사유와 함께 반드시 표기한다(은폐 방지 = 성공기준).
"""
from __future__ import annotations

import html as html_lib
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from bakery.analysis.lab.result import (
    KIND_DATA,
    KIND_HYPOTHESIS,
    REASON_MULTISTORE_REQUIRED,
    REASON_OFF,
    REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED,
    AnalysisReport,
    AnalysisResult,
)

SKIP_LABELS: dict[str, str] = {
    REASON_OFF: "(off)",
    REASON_PREDS_REQUIRED: "(preds 필요 — 미실행)",
    REASON_MULTISTORE_REQUIRED: "(multistore spec 필요 — 미실행)",
    REASON_SINGLE_STORE_REQUIRED: "(단매장 spec 필요 — 미실행)",
}
_TABLE_ROW_LIMIT = 200        # HTML 비대 방지. 전체는 out_dir CSV에 있다.

_HTML_SHELL = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>analysis report — {name}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}}
h1{{border-bottom:2px solid #2c3e50}} h2{{margin-top:32px;color:#2c3e50}}
h3{{margin-top:24px}} .verdict{{background:#eef6ff;padding:8px 12px;border-left:4px solid #2c3e50}}
.note{{color:#666;font-size:0.9em}} .skip{{color:#999}}
table{{border-collapse:collapse;font-size:0.9em}} td,th{{border:1px solid #ddd;padding:4px 8px}}
</style></head><body><h1>데이터분석 + 가설검증 report — {name}</h1>{body}</body></html>"""


def fig_to_div(fig: go.Figure, div_id: str, *, include_js: bool, height: int = 450) -> str:
    """Plotly fig → HTML div. include_js=True인 첫 호출만 plotly.js를 cdn으로 임베드."""
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=50), height=height,
                      autosize=True, hovermode="x unified")
    return pio.to_html(fig, include_plotlyjs=("cdn" if include_js else False),
                       div_id=div_id, full_html=False)


def _summary_fig(report: AnalysisReport) -> go.Figure:
    rows = [{"항목": r.name, "구분": r.kind, "상태": "실행"} for r in report.results]
    rows += [{"항목": s.name, "구분": s.kind, "상태": SKIP_LABELS.get(s.reason, s.reason)}
             for s in report.skipped]
    frame = pd.DataFrame(rows)
    fig = go.Figure(go.Table(
        header=dict(values=list(frame.columns), fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=[frame[c].tolist() for c in frame.columns]),
    ))
    fig.update_layout(title="실행 항목 on/off 요약")
    return fig


def _table_html(label: str, table: pd.DataFrame) -> str:
    shown = table.head(_TABLE_ROW_LIMIT)
    suffix = (f"<p class='note'>상위 {_TABLE_ROW_LIMIT}행만 표시 "
              f"(전체 {len(table)}행은 CSV 참조)</p>") if len(table) > _TABLE_ROW_LIMIT else ""
    return (f"<p><b>{html_lib.escape(label)}</b></p>"
            f"{shown.to_html(index=False, border=0)}{suffix}")


def _result_html(result: AnalysisResult, figures: list[Any], js_used: bool) -> tuple[str, bool]:
    parts = [f"<h3>{html_lib.escape(result.title)} "
             f"<span class='note'>({html_lib.escape(result.name)})</span></h3>"]
    if result.verdict is not None:
        parts.append(f"<p class='verdict'><b>판정</b>: {html_lib.escape(result.verdict)}</p>")
    for index, fig in enumerate(figures):
        parts.append(fig_to_div(fig, f"{result.name}_{index}", include_js=not js_used))
        js_used = True
    for label, table in result.tables:
        parts.append(_table_html(label, table))
    for note in result.notes:
        parts.append(f"<p class='note'>⚠️ {html_lib.escape(note)}</p>")
    return "\n".join(parts), js_used


def _section_html(report: AnalysisReport, kind: str, heading: str,
                  js_used: bool) -> tuple[str, bool]:
    parts = [f"<h2>{heading}</h2>"]
    results = [r for r in report.results if r.kind == kind]
    if not results:
        parts.append("<p class='skip'>실행된 항목 없음</p>")
    for result in results:
        body, js_used = _result_html(result, result.figures, js_used)
        parts.append(body)
    skipped = [s for s in report.skipped if s.kind == kind]
    if skipped:
        items = "".join(f"<li class='skip'>{html_lib.escape(s.name)} — "
                        f"{html_lib.escape(s.title)} "
                        f"{html_lib.escape(SKIP_LABELS.get(s.reason, s.reason))}</li>"
                        for s in skipped)
        parts.append(f"<p class='note'>미실행 항목:</p><ul>{items}</ul>")
    return "\n".join(parts), js_used


def build_analysis_report(report: AnalysisReport, *, out_path: Path) -> Path:
    """AnalysisReport → 자기포함 HTML 1개."""
    store = report.spec_resolved.get("data", {}).get("store", "?")
    divs = [f"<p>데이터 소스: real / 매장: {html_lib.escape(str(store))}</p>",
            fig_to_div(_summary_fig(report), "summary", include_js=True, height=260)]
    body_a, js_used = _section_html(report, KIND_DATA, "섹션 A — 입력 데이터 분석", True)
    divs.append(body_a)
    body_b, _ = _section_html(report, KIND_HYPOTHESIS, "섹션 B — 가설 검증", js_used)
    divs.append(body_b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_HTML_SHELL.format(name=report.name, body="\n".join(divs)),
                        encoding="utf-8")
    return out_path

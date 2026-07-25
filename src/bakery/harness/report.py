"""harness 실험 결과 report — ExperimentResult → 자기포함 HTML(plotly).

plotly 의존은 이 모듈에만(runner/backtest_core 코어는 viz 무의존).
섹션: (1) forecaster 비교표+전체매진 (2) fold WAPE (3) 예측 오버레이 (4) 품목별 매진 실측.
"""
from __future__ import annotations

import html as html_lib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from bakery.harness.runner import ExperimentResult

_STOCKOUT_LABEL = "전체매진 위험(발주<실수요)"


def fig_to_div(fig: go.Figure, div_id: str, *, include_js: bool, height: int = 450) -> str:
    """Plotly fig → HTML div. include_js=True인 첫 호출만 plotly.js를 cdn으로 임베드."""
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=50), height=height,
                      autosize=True, hovermode="x unified")
    js = "cdn" if include_js else False
    return pio.to_html(fig, include_plotlyjs=js, div_id=div_id, full_html=False)


def _comparison_table_div(comparison: pd.DataFrame, *, include_js: bool) -> str:
    disp = comparison.rename(columns={"stockout_risk": _STOCKOUT_LABEL})
    header = list(disp.columns)
    cells = [disp[c].tolist() for c in disp.columns]
    fig = go.Figure(go.Table(
        header=dict(values=header, fill_color="#2c3e50", font=dict(color="white")),
        cells=dict(values=cells),
    ))
    fig.update_layout(title="forecaster 비교")
    return fig_to_div(fig, "cmp_table", include_js=include_js, height=200)


def _fold_wape_div(runs: dict, *, include_js: bool) -> str:
    fig = go.Figure()
    for name, rr in runs.items():
        fm = rr.fold_metrics.sort_values("fold")
        fig.add_trace(go.Scatter(x=fm["fold"], y=fm["wape"], mode="lines+markers", name=name))
    fig.update_layout(title="fold별 WAPE", xaxis_title="fold", yaxis_title="WAPE")
    return fig_to_div(fig, "fold_wape", include_js=include_js)


def _overlay_div(name: str, preds: pd.DataFrame, *, include_js: bool) -> str:
    p = preds.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["date"], y=p["actual"], mode="lines", name="actual"))
    fig.add_trace(go.Scatter(x=p["date"], y=p["expected"], mode="lines", name="expected"))
    fig.add_trace(go.Scatter(x=p["date"], y=p["production"], mode="lines", name="production"))
    fig.update_layout(title=f"예측 오버레이 — {name}", xaxis_title="date", yaxis_title="units")
    return fig_to_div(fig, f"overlay_{name}", include_js=include_js)


_HTML_SHELL = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>harness report — {name}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}}
h1{{border-bottom:2px solid #2c3e50}} h2{{margin-top:32px;color:#2c3e50}}</style>
</head><body><h1>harness 실험 report — {name}</h1>{body}</body></html>"""


def build_report(result: ExperimentResult, *, out_path: Path, store: str | None = None) -> Path:
    divs: list[str] = []
    divs.append("<h2>1. forecaster 비교 + 전체매진</h2>")
    # note: plotly json 인코더가 비-ascii를 \uXXXX로 이스케이프하므로(테이블 렌더 자체는
    # 브라우저 JS가 정상 복원) 라벨을 원문 HTML로도 노출해 가독성/검색성을 보존한다.
    divs.append(f"<p class='note'>{html_lib.escape(_STOCKOUT_LABEL)} 컬럼 포함</p>")
    divs.append(_comparison_table_div(result.comparison, include_js=True))   # 첫 fig = js 임베드
    divs.append("<h2>2. fold별 WAPE</h2>")
    divs.append(_fold_wape_div(result.runs, include_js=False))
    divs.append("<h2>3. 예측 오버레이</h2>")
    for name, rr in result.runs.items():
        divs.append(_overlay_div(name, rr.predictions, include_js=False))
    # 섹션 4(품목별 매진 실측)는 Task 3에서 store 인자로 추가
    html = _HTML_SHELL.format(name=result.name, body="\n".join(divs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path

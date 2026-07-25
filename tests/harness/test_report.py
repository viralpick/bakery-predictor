import pandas as pd

from bakery.harness.report import build_report, fig_to_div
from bakery.harness.runner import ExperimentResult, RunResult


def _dummy_result():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    def _rr(name, wape):
        preds = pd.DataFrame({
            "date": dates, "fold": [0, 0, 0],
            "actual": [100.0, 110.0, 90.0],
            "expected": [102.0, 108.0, 95.0],
            "production": [120.0, 125.0, 110.0],
        })
        folds = pd.DataFrame({"fold": [0], "wape": [wape], "wpe": [0.01]})
        metrics = {"n_test": 3, "wape": wape, "wpe": 0.01,
                   "stockout_risk": 0.0, "surplus_mean_units": 15.0, "surplus_rate": 0.15}
        return RunResult(name=name, predictions=preds, fold_metrics=folds,
                         metrics=metrics, resolved={})
    comparison = pd.DataFrame([
        {"forecaster": "category_total", "n_test": 3, "wape": 0.08, "wpe": 0.01,
         "stockout_risk": 0.0, "surplus_mean_units": 15.0, "surplus_rate": 0.15},
        {"forecaster": "distributional_total", "n_test": 3, "wape": 0.09, "wpe": 0.02,
         "stockout_risk": 0.0, "surplus_mean_units": 16.0, "surplus_rate": 0.16},
    ])
    return ExperimentResult(
        name="exp_x",
        runs={"category_total": _rr("category_total", 0.08),
              "distributional_total": _rr("distributional_total", 0.09)},
        comparison=comparison,
    )


def test_build_report_creates_html(tmp_path):
    out = tmp_path / "report.html"
    path = build_report(_dummy_result(), out_path=out, store=None)
    assert path == out and out.exists()
    html = out.read_text(encoding="utf-8")
    # forecaster명·전체매진 라벨·plotly div 포함
    assert "category_total" in html
    assert "distributional_total" in html
    assert "전체매진 위험" in html          # stockout_risk relabel
    assert "plotly" in html.lower()          # plotly.js 포함(첫 fig include_js)
    assert "exp_x" in html                   # 실험명


def test_fig_to_div_include_js_toggle():
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
    with_js = fig_to_div(fig, "d1", include_js=True)
    without_js = fig_to_div(fig, "d2", include_js=False)
    assert "cdn.plot.ly" in with_js or "Plotly" in with_js   # js 임베드/로더
    assert "cdn.plot.ly" not in without_js

import pandas as pd

from bakery.harness.report import _soldout_stats, build_report, fig_to_div
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
    # ★stateless fig_to_div 핵심 불변: 여러 fig여도 plotly.js는 정확히 1회만 임베드
    # ("cdn.plot.ly" 마커는 test_fig_to_div_include_js_toggle서 검증된 리터럴)
    assert html.count("cdn.plot.ly") == 1
    assert "exp_x" in html                   # 실험명


def test_fig_to_div_include_js_toggle():
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
    with_js = fig_to_div(fig, "d1", include_js=True)
    without_js = fig_to_div(fig, "d2", include_js=False)
    assert "cdn.plot.ly" in with_js or "Plotly" in with_js   # js 임베드/로더
    assert "cdn.plot.ly" not in without_js


def _soldout_daily():
    # item A: 3일 활성, 2일 완판(20시/18시) → 매진률 2/3. item B: 2일 활성 0완판.
    rows = []
    for d, so, hr in [("2025-01-01", True, 20), ("2025-01-02", True, 18), ("2025-01-03", False, None)]:
        rows.append({"date": pd.Timestamp(d), "item_id": "A", "sold_units": 10,
                     "store_id": "s", "category_id": "bread", "is_stockout": so,
                     "stockout_time": pd.Timestamp(f"{d} {hr}:00") if so else pd.NaT})
    for d in ["2025-01-01", "2025-01-02"]:
        rows.append({"date": pd.Timestamp(d), "item_id": "B", "sold_units": 5,
                     "store_id": "s", "category_id": "bread", "is_stockout": False,
                     "stockout_time": pd.NaT})
    return pd.DataFrame(rows)


def test_soldout_stats_median_and_rate():
    stats = _soldout_stats(_soldout_daily(), min_active_days=1)
    # 완판 매진시각 hour = [20, 18] → median 19.0
    assert stats["median_hour"] == 19.0
    assert stats["n_soldout"] == 2
    # per_item_rate: A=2/3, B=0/2 (min_active_days=1이라 둘 다 포함)
    assert stats["per_item_rate"]["A"] == 2 / 3
    assert stats["per_item_rate"]["B"] == 0.0


def test_soldout_stats_active_filter():
    # min_active_days=3이면 B(2일)는 제외
    stats = _soldout_stats(_soldout_daily(), min_active_days=3)
    assert "A" in stats["per_item_rate"].index
    assert "B" not in stats["per_item_rate"].index


def test_build_report_with_store_includes_soldout(tmp_path, monkeypatch):
    import bakery.harness.report as rp
    monkeypatch.setattr(rp, "_store_daily_for", lambda store_id: _soldout_daily())
    out = tmp_path / "r.html"
    build_report(_dummy_result(), out_path=out, store="store_gw01")
    html = out.read_text(encoding="utf-8")
    assert "품목별 매진" in html
    assert "매진 median" in html

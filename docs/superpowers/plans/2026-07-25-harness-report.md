# Harness 실험 report 표면 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** harness `ExperimentResult`를 자기포함 HTML report(plotly)로 만들고 — forecaster 비교·예측 오버레이·fold WAPE + 품목별 매진 실측(매진률·median t) — `harness-run`이 자동 생성한다.

**Architecture:** `src/bakery/harness/report.py`가 `ExperimentResult`(in-memory)와 store_daily(실측)를 소비해 HTML을 조립한다. plotly는 report.py에만 의존(runner/backtest_core 코어는 viz 무의존 유지). 매진 통계는 순수 함수(`_soldout_stats`, DataFrame-in)로 분리해 단위 테스트 가능하게 한다.

**Tech Stack:** Python 3.12, pandas, numpy, plotly 6.7.0(정식 의존성), typer, rich, pytest.

## Global Constraints

- **매진 3층위(소스·층위 구분)**:
  - 전체매진 위험 = `comparison["stockout_risk"]`(발주<실수요, forecaster별 다름, 이미 계산됨) — report에서 "전체매진 위험(발주<실수요)"로 relabel.
  - 품목별 매진률 = store_daily item-day `is_stockout` 품목별 평균(관측, 실험 무관).
  - 매진 median t = 완판 item-day `stockout_time` hour의 median(관측, 헌장 KPI).
- **품목별 매진률 raw median=0.0**(1150품목 다수 희소) → 활성일 ≥ `min_active_days`(기본 30) 필터 후 분포+top 표시.
- **검증된 실측값**(2026-07-25): 매진시각 median **18시**, 완판행 stockout_time 100% 채워짐/비완판 100% NaN. ⚠️ 매진률 0.151은 **완제품(etc 968개) 희석 아티팩트** — 생산품목 기준 진짜 매진율은 **0.605**. 아이템 스코프/희석 수정은 **데이터 진입점 문제로 로드맵 3/4단계 이연**(사용자 결정). 이 스텝은 build_store_daily의 매진 정의를 건드리지 않고 report만 만든다.
- **plotly stateless**: build_dashboard의 `_PLOTLY_INCLUDED` 전역 플래그 패턴 금지(여러 호출 시 깨짐). `fig_to_div(fig, div_id, *, include_js: bool, height=450)`로 명시 인자화, 첫 fig만 `include_js=True`.
- **재구현 금지**: report는 `src/bakery` 심볼 호출만. store_daily는 `scripts/store_daily.py` `build_store_daily`(sys.path.insert 패턴).
- **테스트 단언**: 기대값 아는 것은 정확값 `==`(매진 median t, 비교표 값, 라벨 문자열). 부동소수는 근사.
- **pytest 실행**: 이 repo addopts에 `-q` 있음. 카운트 필요 시 `uv run pytest --color=no`(추가 `-q` 금지).
- **DataFrame 컬럼 계약**(참조):
  - `ExperimentResult`: `name: str`, `runs: dict[str, RunResult]`, `comparison: pd.DataFrame`.
  - `RunResult`: `name`, `predictions`(cols: date, fold, actual, expected, production), `fold_metrics`(cols: fold, n_train, n_test, test_start, test_end, wape, wpe, prod_pct_under), `metrics: dict`, `resolved: dict`.
  - `comparison` cols: `forecaster, n_test, wape, wpe, stockout_risk, surplus_mean_units, surplus_rate`.
  - store_daily cols: `date, item_id, sold_units, store_id, category_id, is_stockout, stockout_time`.

---

## File Structure

- **Modify** `src/bakery/harness/config.py` — `DEFAULT_METRICS` 실산출 5종으로 정합.
- (제외) ~~`tests/test_store_daily_redefine.py` 재baseline~~ — 취소, canary로 유지.
- **Create** `src/bakery/harness/report.py` — `fig_to_div` + `_soldout_stats` + `_soldout_view` + `_store_daily_for` + `build_report`.
- **Modify** `src/bakery/harness/__init__.py` — `build_report` re-export.
- **Modify** `src/bakery/cli.py` — `cmd_harness_run` report 자동생성.
- **Tests**: modify `test_config.py`; create `test_report.py`; modify `test_cli_harness.py`.

---

## Task 1: DEFAULT_METRICS 정합

**Files:**
- Modify: `src/bakery/harness/config.py`, `tests/harness/test_config.py`

**Interfaces:**
- Produces: `DEFAULT_METRICS = ["wape", "wpe", "stockout_risk", "surplus_mean_units", "surplus_rate"]`.

**배경:** `metrics_from_preds` 실산출은 5종(+n_test). 기존 `DEFAULT_METRICS` 6종 중 3종(soldout_median/stockout_item_rate/shortfall_day_rate)은 카테고리 총량 레벨에 없는 데이터(매진시각/item-level) 요구 → 제거(decorative→honest).

⚠️ **매진 재baseline 안 함**: 초기 계획의 0.60→0.151 재baseline은 **폐기**(0.151은 완제품 희석 버그값이라 고착 금지). `test_store_daily_redefine`은 사전존재 실패(canary)로 그대로 둔다 — 이 Task는 그 파일을 건드리지 않는다.

- [ ] **Step 1: Update test (fail 유도)**

`tests/harness/test_config.py`의 `test_defaults_are_category_stack`에 metrics 내용 단언 추가(함수 마지막 줄 뒤):
```python
    assert spec.metrics == ["wape", "wpe", "stockout_risk", "surplus_mean_units", "surplus_rate"]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/harness/test_config.py::test_defaults_are_category_stack --color=no`
Expected: FAIL — `assert [...6종...] == [...5종...]`.

- [ ] **Step 3: Update DEFAULT_METRICS**

`src/bakery/harness/config.py`의 `DEFAULT_METRICS` 정의 교체:
```python
DEFAULT_METRICS: list[str] = ["wape", "wpe", "stockout_risk", "surplus_mean_units", "surplus_rate"]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/harness/test_config.py --color=no`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/harness/config.py tests/harness/test_config.py
git commit -m "fix(harness): DEFAULT_METRICS 실산출 5종 정합(계산불가 3종 제거)"
```

---

## Task 2: report.py — backtest 섹션 (비교표 + fold WAPE + 예측 오버레이)

**Files:**
- Create: `src/bakery/harness/report.py`
- Modify: `src/bakery/harness/__init__.py`
- Test: `tests/harness/test_report.py`

**Interfaces:**
- Consumes: `ExperimentResult`(runner), plotly.
- Produces:
  - `fig_to_div(fig, div_id: str, *, include_js: bool, height: int = 450) -> str`
  - `build_report(result: ExperimentResult, *, out_path: Path, store: str | None = None) -> Path` (이 태스크는 store 무시; 섹션 4는 Task 3서 추가)

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_report.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_report.py --color=no`
Expected: FAIL — `ModuleNotFoundError: bakery.harness.report`

- [ ] **Step 3: Write implementation**

```python
# src/bakery/harness/report.py
"""harness 실험 결과 report — ExperimentResult → 자기포함 HTML(plotly).

plotly 의존은 이 모듈에만(runner/backtest_core 코어는 viz 무의존).
섹션: (1) forecaster 비교표+전체매진 (2) fold WAPE (3) 예측 오버레이 (4) 품목별 매진 실측.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_report.py --color=no`
Expected: PASS (2 passed).

- [ ] **Step 5: Update __init__ + commit**

`__init__.py`에 추가:
```python
from bakery.harness.report import build_report
```
`__all__`에 `"build_report"` 추가.

```bash
git add src/bakery/harness/report.py src/bakery/harness/__init__.py tests/harness/test_report.py
git commit -m "feat(harness): report.py — backtest 섹션(비교표+fold WAPE+예측 오버레이)"
```

---

## Task 3: report.py — 품목별 매진 실측 섹션

**Files:**
- Modify: `src/bakery/harness/report.py`
- Test: `tests/harness/test_report.py`

**Interfaces:**
- Consumes: `scripts/store_daily.build_store_daily`(sys.path 패턴), `STORE_MAP`.
- Produces:
  - `_soldout_stats(daily: pd.DataFrame, *, min_active_days: int = 30) -> dict` — 순수 함수(테스트 대상). 반환: `{"median_hour": int | None, "n_soldout": int, "rate_overall": float, "hour_counts": pd.Series, "per_item_rate": pd.Series, "top_items": pd.DataFrame}`.
  - `_soldout_view(daily: pd.DataFrame, *, include_js: bool) -> str` — stats → HTML divs.
  - `_store_daily_for(store_id: str) -> pd.DataFrame | None` — store_id 역참조 후 build_store_daily. 미등록/실패 → None.
  - `build_report`에 섹션 4 배선(store 있고 데이터 있으면).

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/test_report.py 에 추가
import numpy as np
from bakery.harness.report import _soldout_stats


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_report.py --color=no`
Expected: FAIL — `ImportError: cannot import name '_soldout_stats'`

- [ ] **Step 3: Write implementation** (report.py에 추가 + build_report 섹션 4 배선)

report.py 상단에 `import numpy as np` 추가. 아래 함수들 추가:
```python
def _soldout_stats(daily: pd.DataFrame, *, min_active_days: int = 30) -> dict:
    so = daily[daily["is_stockout"]].copy()
    hours = pd.to_datetime(so["stockout_time"]).dt.hour if len(so) else pd.Series(dtype=float)
    median_hour = float(hours.median()) if len(hours) else None
    active = daily.groupby("item_id").size()
    keep = active[active >= min_active_days].index
    per_item_rate = daily[daily["item_id"].isin(keep)].groupby("item_id")["is_stockout"].mean()
    top_items = (per_item_rate.sort_values(ascending=False).head(20)
                 .rename("soldout_rate").reset_index())
    return {
        "median_hour": median_hour,
        "n_soldout": int(len(so)),
        "rate_overall": float(daily["is_stockout"].mean()) if len(daily) else 0.0,
        "hour_counts": hours.value_counts().sort_index(),
        "per_item_rate": per_item_rate,
        "top_items": top_items,
    }


def _soldout_view(daily: pd.DataFrame, *, include_js: bool) -> str:
    stats = _soldout_stats(daily)
    parts = [f"<p><b>매진 median t</b>: {stats['median_hour']}시 "
             f"| 전체 매진율 {stats['rate_overall']:.3f} (n_soldout={stats['n_soldout']})</p>"]
    hc = stats["hour_counts"]
    fig_h = go.Figure(go.Bar(x=list(hc.index), y=list(hc.values)))
    fig_h.update_layout(title="매진시각 hour 분포", xaxis_title="hour", yaxis_title="완판 item-day 수")
    parts.append(fig_to_div(fig_h, "soldout_hour", include_js=include_js))
    ti = stats["top_items"]
    fig_t = go.Figure(go.Bar(x=ti["item_id"].astype(str), y=ti["soldout_rate"]))
    fig_t.update_layout(title="품목별 매진률 top 20", xaxis_title="item_id", yaxis_title="매진률")
    parts.append(fig_to_div(fig_t, "soldout_top", include_js=False))
    return "\n".join(parts)


def _store_daily_for(store_id: str) -> pd.DataFrame | None:
    import sys
    sys.path.insert(0, "scripts")
    try:
        from store_daily import STORE_MAP, build_store_daily
    except ImportError:
        return None
    cd = next((code for code, (_, sid) in STORE_MAP.items() if sid == store_id), None)
    if cd is None:
        return None
    return build_store_daily(cd, store_id, exclude_bulk=True)
```

build_report의 섹션 4 주석 줄을 교체:
```python
    if store is not None:
        daily = _store_daily_for(store)
        if daily is not None and not daily.empty:
            divs.append("<h2>4. 품목별 매진 실측 (관측, forecaster 무관)</h2>")
            divs.append(_soldout_view(daily, include_js=False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/harness/test_report.py --color=no`
Expected: PASS (5 passed — Task 2의 2 + Task 3의 3).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/harness/report.py tests/harness/test_report.py
git commit -m "feat(harness): report 품목별 매진 실측 섹션(매진률·median t, 순수 stats 분리)"
```

---

## Task 4: CLI 자동 생성 + 전체 스위트

**Files:**
- Modify: `src/bakery/cli.py`
- Test: `tests/harness/test_cli_harness.py`

**Interfaces:**
- Consumes: `build_report`(Task 2), `run_experiment`.

- [ ] **Step 1: Update the test**

`tests/harness/test_cli_harness.py`의 `test_harness_run_default_config`에 단언 추가(기존 단언 뒤):
```python
    report = tmp_path / "out" / "gwangyo_default" / "report.html"
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "전체매진 위험" in html
    assert "품목별 매진" in html          # gwangyo_default는 store_gw01 → 매진 섹션 포함
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: FAIL — `assert report.exists()` (아직 report 생성 안 함). build_store_daily 실행 포함 ~1-2분.

- [ ] **Step 3: Modify cmd_harness_run**

`src/bakery/cli.py` 상단 harness import에 `build_report` 추가:
```python
from .harness import load_spec, run_experiment, build_report
```
`cmd_harness_run` 본문의 comparison 표 출력 뒤(함수 끝)에 추가:
```python
    report_path = build_report(result, out_path=out / result.name / "report.html", store=spec.data.store)
    console.print(f"[green]report[/] {report_path}")
```

- [ ] **Step 4: Run test + 전체 스위트**

Run: `uv run pytest tests/harness/test_cli_harness.py --color=no`
Expected: PASS (1 passed).

Run(전체, distributional/equivalence 포함 ~20분): `uv run pytest --color=no`
Expected: **사전존재 `test_store_daily_redefine` 1건만 실패**(매진률 희석 canary, 데이터 진입점 버그 — 로드맵 3/4서 해소, 이 스텝 범위 밖), 나머지 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py tests/harness/test_cli_harness.py
git commit -m "feat(cli): harness-run report.html 자동 생성"
```

---

## Self-Review 결과

**Spec coverage:**
- §아키텍처 1(report.py, 섹션 1-3) → Task 2. ✅
- §아키텍처 1 섹션 4(품목별 매진 실측) + §2(store 해석) → Task 3. ✅
- §아키텍처 3(CLI 자동생성) → Task 4. ✅
- §아키텍처 4(metrics 정합) → Task 1. §5(redefine 재baseline 취소, canary 유지) → 어느 Task도 test_store_daily_redefine 안 건드림. ✅
- §Acceptance 1-2(build_report/soldout 단위 test) → Task 2/3. §3(CLI) → Task 4. §4(config) → Task 1. §5(redefine 유지=사전존재 1건 실패 예상) → Task 4 전체 스위트 expectation. ✅

**Placeholder scan:** 모든 코드 스텝 완전(report.py 전체·stats·view·store 해석·CLI). Task 2의 "섹션 4는 Task 3서 추가" 주석은 의도적 순차(Task 3가 교체). ✅

**Type consistency:** `fig_to_div(fig, div_id, *, include_js, height)` Task 2 정의 = Task 3 호출 일치. `build_report(result, *, out_path, store)` Task 2 정의 = Task 3 섹션4 배선·Task 4 CLI 호출 일치. `_soldout_stats` 반환 dict 키(median_hour/n_soldout/rate_overall/hour_counts/per_item_rate/top_items) Task 3 정의 = test 단언·`_soldout_view` 소비 일치. `_store_daily_for(store_id)→df|None` Task 3 정의 = build_report·test monkeypatch 일치. ✅

**VERIFY 완료(실행 전, 2026-07-25):**
1. ✅ plotly 6.7.0 정식 의존성. pio/go/make_subplots import 가능(build_dashboard 사용).
2. ✅ STORE_MAP dict `scripts/store_daily.py`(store_gw01→1000000047), build_store_daily(cd, store_id, exclude_bulk) 시그니처.
3. ✅ ExperimentResult/RunResult/comparison/predictions/fold_metrics 컬럼 계약 확인.
4. ✅ 매진 실측: median 18시·완판행 stockout_time 100%. 매진률 0.151=완제품 희석 아티팩트(생산품목 진짜 0.605), 스코프 수정은 데이터 진입점 문제로 로드맵 3/4 이연(사용자 결정) — 이 스텝은 정의 안 건드림.
5. ✅ DEFAULT_METRICS 소비처 = config+__init__+test_config(내용 미단언). 후방호환.
6. ✅ fig_to_div stateless 재설계(전역 플래그 제거).

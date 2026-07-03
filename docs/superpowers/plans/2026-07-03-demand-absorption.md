# 카테고리 총량 수요이전 흡수 검증 (W0 게이트) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 카테고리 내 품목 조기품절이 같은 카테고리 총 판매량을 떨어뜨리는지(walk-away) 아니면 흡수되는지를 leave-one-out 총량보존 계수 β로 직접 검정해 W0 게이트를 판정한다.

**Architecture:** `analysis/demand_absorption.py`에 (1) leakage-safe 패널 빌더, (2) numpy 손코딩 OLS + HC3 robust SE + TOST equivalence, (3) 매장×카테고리 오케스트레이터를 넣고 CLI로 노출한다. 회귀는 statsmodels 없이 numpy/scipy만 사용(기존 `substitution_did._ols_fe` 패턴 계승). 타깃은 raw sold_units.

**Tech Stack:** Python 3.12, pandas, numpy, scipy.stats, pytest, uv. 스펙: `docs/superpowers/specs/2026-07-03-demand-absorption-design.md`

## Global Constraints

- 타깃·통제 모두 raw `sold_units` (potential_demand 파생 일절 금지 — 순환; 스펙 D4)
- confound 이중 통제: OtherCatSold(c 제외 매장 총 sold) + c_baseline(c의 최근 4주 동일요일 평균, **lag=그날 이전만**, time leakage 절대금지; 스펙 D2)
- 판정 = TOST equivalence, δ = 품절강도 IQR 변화가 카테고리 총량의 5%에 해당하는 β; β의 90% CI ⊂ [−δ,+δ] → 흡수(통과) (스펙 D3)
- 처치 T = 카테고리 내 품목 조기품절 손실영업시간 합 `Σ max(close_hour − stockout_time, 0)`, 매진 없는 품목 0, `close_hour=22` (스펙 모델)
- 회귀는 numpy/scipy만 (statsmodels 미설치 — DiD `_ols_fe` 패턴 계승), HC3 robust SE
- 실데이터 loader `load_dataset('real')` = 광교 단일 매장(store_gw01, cats: bread/cake/pastry/sandwich). 게이트 판정은 일반 카테고리(bread/pastry/sandwich), cake는 별도 리포트 (스펙 D5). 4매장 확장은 범위 밖(후속)
- 함수 30줄 이내, guard clause 우선, 매직값 상수화 (code-quality 규칙)

---

### Task 1: 패널 빌더 `build_absorption_panel` (leakage-safe)

**Files:**
- Create: `src/bakery/analysis/demand_absorption.py`
- Test: `tests/test_demand_absorption.py`

**Interfaces:**
- Consumes: daily DataFrame (cols: store_id, item_id, category_id, date, sold_units, is_stockout, stockout_time)
- Produces: `build_absorption_panel(daily, *, close_hour=DEFAULT_CLOSE_HOUR, baseline_weeks=4) -> pd.DataFrame` with columns `[store_id, category_id, date, cat_sold, stockout_hours, other_cat_sold, cat_baseline, dow, month, trend]`; rows with insufficient baseline dropped. `DEFAULT_CLOSE_HOUR = 22`, `BASELINE_WEEKS = 4`. Task 2/3가 소비.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_demand_absorption.py` 생성:

```python
"""수요이전 흡수 검증 (W0 게이트) — 패널 빌더 + 회귀/TOST 테스트.

합성 fixture로 완전흡수(β≈0)와 walk-away(β<0) 시나리오를 심어 회귀가
부호를 회복하는지 검증한다. leakage-safe baseline은 미래 미참조를 확인."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakery.analysis.demand_absorption import (
    DEFAULT_CLOSE_HOUR, BASELINE_WEEKS, build_absorption_panel,
)


def _daily_two_items_one_cat(n_weeks: int = 12, seed: int = 0) -> pd.DataFrame:
    """1 store, 1 category, 2 items, daily rows over n_weeks. No stockouts (base)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks * 7, freq="D")
    rows = []
    for d in dates:
        for item in ("i1", "i2"):
            rows.append({
                "store_id": "s1", "item_id": item, "category_id": "bread",
                "date": d, "sold_units": float(rng.integers(8, 12)),
                "is_stockout": False, "stockout_time": pd.NaT,
            })
    return pd.DataFrame(rows)


def test_panel_has_expected_columns_and_grain():
    panel = build_absorption_panel(_daily_two_items_one_cat())
    assert list(panel.columns) == [
        "store_id", "category_id", "date", "cat_sold", "stockout_hours",
        "other_cat_sold", "cat_baseline", "dow", "month", "trend"]
    # one row per (store, category, date) that survives baseline warmup
    assert (panel.groupby(["store_id", "category_id", "date"]).size() == 1).all()


def test_panel_stockout_hours_from_earliest():
    daily = _daily_two_items_one_cat(n_weeks=12)
    # inject a stockout: i1 sells out at 14:00 on one date
    d0 = daily["date"].max()
    mask = (daily["item_id"] == "i1") & (daily["date"] == d0)
    daily.loc[mask, "is_stockout"] = True
    daily.loc[mask, "stockout_time"] = pd.Timestamp(f"{d0.date()} 14:00")
    panel = build_absorption_panel(daily)
    row = panel[(panel["category_id"] == "bread") & (panel["date"] == d0)].iloc[0]
    assert row["stockout_hours"] == pytest.approx(DEFAULT_CLOSE_HOUR - 14.0)  # 8.0


def test_panel_baseline_is_leakage_safe():
    """cat_baseline for date d must use only same-dow rows strictly before d."""
    daily = _daily_two_items_one_cat(n_weeks=12)
    panel = build_absorption_panel(daily).sort_values("date")
    # first BASELINE_WEEKS of each dow are dropped (no prior window)
    first_date = daily["date"].min()
    assert (panel["date"] > first_date).all()
    # baseline of a row equals mean of same-dow cat_sold on strictly-earlier dates
    r = panel.iloc[-1]
    same_dow_earlier = panel[(panel["dow"] == r["dow"]) & (panel["date"] < r["date"])]
    expected = same_dow_earlier["cat_sold"].tail(BASELINE_WEEKS).mean()
    assert r["cat_baseline"] == pytest.approx(expected)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_demand_absorption.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.analysis.demand_absorption'`

- [ ] **Step 3: 구현**

`src/bakery/analysis/demand_absorption.py` 생성:

```python
"""카테고리 총량 수요이전 흡수 검증 (W0 게이트).

leave-one-out 총량보존 계수 β: 카테고리 내 품목 조기품절(품절강도 T)이
같은 카테고리 총 sold(Y)를 떨어뜨리는가. β≈0 = 흡수(총량 보존), β<0 = walk-away.
confound(고수요일=품절많은날)는 OtherCatSold(그날 전반 traffic) + cat_baseline
(c의 최근 4주 동일요일 평균, lag)로 이중 통제. 타깃은 raw sold_units(순환 회피).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CLOSE_HOUR = 22
BASELINE_WEEKS = 4
PANEL_COLUMNS = [
    "store_id", "category_id", "date", "cat_sold", "stockout_hours",
    "other_cat_sold", "cat_baseline", "dow", "month", "trend",
]


def _stockout_hours(sub: pd.DataFrame, close_hour: int) -> float:
    """Category-day stockout intensity = Σ max(close_hour − stockout time-of-day, 0)."""
    so = pd.to_datetime(sub["stockout_time"])
    tod = so.dt.hour + so.dt.minute / 60.0
    return float((close_hour - tod).clip(lower=0.0).fillna(0.0).sum())


def _category_day_frame(daily: pd.DataFrame, close_hour: int) -> pd.DataFrame:
    """Aggregate item×day → (store, category, date) with cat_sold + stockout_hours."""
    grp = daily.groupby(["store_id", "category_id", "date"], observed=True)
    agg = grp.agg(cat_sold=("sold_units", "sum")).reset_index()
    hours = (grp.apply(lambda s: _stockout_hours(s, close_hour), include_groups=False)
             .rename("stockout_hours").reset_index())
    return agg.merge(hours, on=["store_id", "category_id", "date"])


def _add_other_cat_sold(cat_day: pd.DataFrame) -> pd.DataFrame:
    """OtherCatSold = same store-day total sold across all OTHER categories."""
    store_day = (cat_day.groupby(["store_id", "date"], observed=True)["cat_sold"]
                 .sum().rename("store_total").reset_index())
    out = cat_day.merge(store_day, on=["store_id", "date"])
    out["other_cat_sold"] = out["store_total"] - out["cat_sold"]
    return out.drop(columns=["store_total"])


def _add_leakage_safe_baseline(cat_day: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """cat_baseline = mean of same (store,category,dow) cat_sold over the prior
    `weeks` occurrences, strictly before the row's date (no leakage)."""
    df = cat_day.sort_values("date").copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    def _roll(g: pd.DataFrame) -> pd.Series:
        return g["cat_sold"].shift(1).rolling(weeks, min_periods=weeks).mean()
    df["cat_baseline"] = (df.groupby(["store_id", "category_id", "dow"], observed=True,
                                     group_keys=False).apply(_roll, include_groups=False))
    return df


def build_absorption_panel(daily: pd.DataFrame, *, close_hour: int = DEFAULT_CLOSE_HOUR,
                           baseline_weeks: int = BASELINE_WEEKS) -> pd.DataFrame:
    """Build the (store, category, date) regression panel. Rows without a full
    baseline window are dropped. Target/controls are all raw sold_units."""
    cat_day = _category_day_frame(daily, close_hour)
    cat_day = _add_other_cat_sold(cat_day)
    cat_day = _add_leakage_safe_baseline(cat_day, baseline_weeks)
    cat_day = cat_day.dropna(subset=["cat_baseline"]).copy()
    dt = pd.to_datetime(cat_day["date"])
    cat_day["month"] = dt.dt.month
    cat_day["trend"] = (dt - dt.min()).dt.days.astype(float)
    return cat_day[PANEL_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_demand_absorption.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/bakery/analysis/demand_absorption.py tests/test_demand_absorption.py
git commit -m "feat: 흡수검증 패널 빌더 — leakage-safe baseline + 품절강도/traffic 이중통제 재료"
```

---

### Task 2: 회귀 + HC3 + TOST 판정 `fit_absorption`

**Files:**
- Modify: `src/bakery/analysis/demand_absorption.py`
- Test: `tests/test_demand_absorption.py` (append)

**Interfaces:**
- Consumes: `build_absorption_panel` 출력 (Task 1)
- Produces: `AbsorptionResult` (frozen dataclass: store_id, category_id, n, beta, se, ci_low, ci_high, delta, verdict) + `fit_absorption(panel, store_id, category_id, *, equiv_frac=EQUIV_FRAC) -> AbsorptionResult | None`. `EQUIV_FRAC = 0.05`. verdict ∈ {"absorb","walkaway","inconclusive"}.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_demand_absorption.py`에 추가. 합성 시나리오로 β 부호 회복 검정:

```python
from bakery.analysis.demand_absorption import AbsorptionResult, EQUIV_FRAC, fit_absorption


def _panel_with_effect(beta_true: float, n_weeks: int = 40, seed: int = 1):
    """Synthetic (store,cat,date) panel: cat_sold = base + beta_true*T + traffic + noise.
    beta_true=0 → absorption; beta_true<0 → walk-away. T correlated with traffic to
    stress the confound control."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks * 7, freq="D")
    demand_level = rng.normal(100, 15, len(dates))          # daily category demand
    traffic = demand_level + rng.normal(0, 5, len(dates))   # other-cat proxy (correlated)
    # high-demand days → more stockout hours (the confound)
    stockout_hours = np.clip((demand_level - 100) * 0.3 + rng.normal(0, 1, len(dates)), 0, None)
    cat_sold = demand_level + beta_true * stockout_hours + rng.normal(0, 3, len(dates))
    daily = pd.DataFrame({
        "store_id": "s1", "category_id": "bread", "date": dates,
        "cat_sold": cat_sold, "stockout_hours": stockout_hours,
        "other_cat_sold": traffic,
        "dow": dates.dayofweek, "month": dates.month,
        "trend": (dates - dates.min()).days.astype(float),
    })
    # leakage-safe baseline on this pre-aggregated frame
    daily = daily.sort_values("date")
    daily["cat_baseline"] = (daily.groupby("dow")["cat_sold"]
                             .shift(1).rolling(4, min_periods=4).mean())
    return daily.dropna(subset=["cat_baseline"]).reset_index(drop=True)


def test_fit_recovers_absorption_zero_beta():
    panel = _panel_with_effect(beta_true=0.0)
    res = fit_absorption(panel, "s1", "bread")
    assert res.verdict == "absorb"
    assert abs(res.beta) < res.delta            # inside equivalence band


def test_fit_recovers_walkaway_negative_beta():
    # strong negative: each stockout-hour loses ~4 units, no absorption
    panel = _panel_with_effect(beta_true=-4.0)
    res = fit_absorption(panel, "s1", "bread")
    assert res.beta < 0
    assert res.verdict == "walkaway"


def test_fit_returns_none_on_tiny_panel():
    panel = _panel_with_effect(beta_true=0.0).head(10)
    assert fit_absorption(panel, "s1", "bread") is None
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_demand_absorption.py -q -k fit`
Expected: FAIL — `ImportError: cannot import name 'fit_absorption'`

- [ ] **Step 3: 구현**

`src/bakery/analysis/demand_absorption.py`에 추가 (상단 import에 `from dataclasses import dataclass` 추가):

```python
from dataclasses import dataclass

EQUIV_FRAC = 0.05            # δ = 5% of mean category sold, mapped via T IQR
MIN_PANEL_ROWS = 30
MAX_CONDITION_NUMBER = 1e10


@dataclass(frozen=True)
class AbsorptionResult:
    store_id: str
    category_id: str
    n: int
    beta: float              # effect of 1 stockout-hour on category total sold
    se: float
    ci_low: float            # 90% CI
    ci_high: float
    delta: float             # equivalence bound (in sold units per stockout-hour)
    verdict: str             # "absorb" | "walkaway" | "inconclusive"


def _design_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Y and X=[const, T, other_cat_sold, cat_baseline, dow dummies, month dummies, trend]."""
    y = panel["cat_sold"].to_numpy(dtype=float)
    n = len(y)
    const = np.ones((n, 1))
    cols = [const,
            panel["stockout_hours"].to_numpy(float).reshape(-1, 1),      # T = index 1
            panel["other_cat_sold"].to_numpy(float).reshape(-1, 1),
            panel["cat_baseline"].to_numpy(float).reshape(-1, 1),
            pd.get_dummies(panel["dow"], drop_first=True).to_numpy(float),
            pd.get_dummies(panel["month"], drop_first=True).to_numpy(float),
            panel["trend"].to_numpy(float).reshape(-1, 1)]
    X = np.hstack(cols)
    keep = X.std(axis=0) > 1e-12
    keep[0] = True                       # keep constant
    keep[1] = True                       # keep treatment even if degenerate → caller guards
    return y, X[:, keep], 1              # treatment is column index 1 after keep (const stays 0)


def _ols_hc3(y: np.ndarray, X: np.ndarray, treat_idx: int) -> tuple[float, float] | None:
    """OLS β and HC3 robust SE for the treatment column. numpy only. None if ill-posed."""
    n, k = X.shape
    if n - k < 5:
        return None
    XtX = X.T @ X
    if not np.isfinite(np.linalg.cond(XtX)) or np.linalg.cond(XtX) > MAX_CONDITION_NUMBER:
        return None
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverages
    denom = np.clip((1.0 - h) ** 2, 1e-8, None)
    meat = X.T @ (X * (resid ** 2 / denom)[:, None])     # HC3 sandwich meat
    cov = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(cov[treat_idx, treat_idx]))
    return float(beta[treat_idx]), se


def fit_absorption(panel: pd.DataFrame, store_id: str, category_id: str, *,
                   equiv_frac: float = EQUIV_FRAC) -> AbsorptionResult | None:
    """Regress category total sold on stockout intensity (dual-controlled) and
    judge absorption via TOST. Returns None on an unusable panel."""
    from scipy.stats import norm
    sub = panel[(panel["store_id"] == store_id)
                & (panel["category_id"] == category_id)]
    if len(sub) < MIN_PANEL_ROWS:
        return None
    y, X, treat_idx = _design_matrix(sub)
    fit = _ols_hc3(y, X, treat_idx)
    if fit is None:
        return None
    beta, se = fit
    z = norm.ppf(0.95)                                   # 90% CI (two-sided)
    ci_low, ci_high = beta - z * se, beta + z * se
    t_iqr = np.subtract(*np.percentile(sub["stockout_hours"], [75, 25]))
    mean_y = float(sub["cat_sold"].mean())
    delta = (equiv_frac * mean_y / t_iqr) if t_iqr > 1e-9 else float("inf")
    if ci_low > -delta and ci_high < delta:
        verdict = "absorb"
    elif ci_high < 0:
        verdict = "walkaway"
    else:
        verdict = "inconclusive"
    return AbsorptionResult(store_id, category_id, len(sub), beta, se,
                            ci_low, ci_high, delta, verdict)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_demand_absorption.py -q`
Expected: 전부 PASS (Task 1 3개 + Task 2 3개)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/analysis/demand_absorption.py tests/test_demand_absorption.py
git commit -m "feat: 흡수검증 회귀 — numpy OLS + HC3 robust SE + TOST equivalence 판정"
```

---

### Task 3: 오케스트레이션 + CLI `demand-absorption`

**Files:**
- Modify: `src/bakery/analysis/demand_absorption.py` (`run_absorption`)
- Modify: `src/bakery/cli.py` (커맨드 추가)
- Test: `tests/test_demand_absorption.py` (append) + `tests/test_cli_helpers.py` (커맨드 등록)

**Interfaces:**
- Consumes: `build_absorption_panel`, `fit_absorption` (Task 1/2), `load_dataset`
- Produces: `run_absorption(daily, *, close_hour=DEFAULT_CLOSE_HOUR, baseline_weeks=BASELINE_WEEKS) -> list[AbsorptionResult]` (매장×카테고리 전체, None 제외); CLI `demand-absorption --source real`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_demand_absorption.py`에 추가:

```python
from bakery.analysis.demand_absorption import run_absorption


def test_run_absorption_covers_store_categories():
    daily = _daily_two_items_one_cat(n_weeks=40)
    # add a second category so other_cat_sold is non-trivial
    d2 = _daily_two_items_one_cat(n_weeks=40, seed=9)
    d2["category_id"] = "pastry"
    d2["item_id"] = d2["item_id"] + "_p"
    results = run_absorption(pd.concat([daily, d2], ignore_index=True))
    cats = {r.category_id for r in results}
    assert cats == {"bread", "pastry"}
    assert all(r.verdict in {"absorb", "walkaway", "inconclusive"} for r in results)
```

`tests/test_cli_helpers.py`에 추가:

```python
def test_demand_absorption_command_registered():
    import typer
    import bakery.cli as c
    group = typer.main.get_group(c.app)
    cmd = group.get_command(None, "demand-absorption")
    assert cmd is not None
    opts = [p.name for p in cmd.params]
    assert "source" in opts
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_demand_absorption.py tests/test_cli_helpers.py -q -k "run_absorption or demand_absorption"`
Expected: FAIL — `ImportError: run_absorption` / 커맨드 미등록

- [ ] **Step 3: 구현**

`src/bakery/analysis/demand_absorption.py`에 추가:

```python
def run_absorption(daily: pd.DataFrame, *, close_hour: int = DEFAULT_CLOSE_HOUR,
                   baseline_weeks: int = BASELINE_WEEKS) -> list[AbsorptionResult]:
    """Fit absorption for every (store, category) with enough data. Skips None."""
    panel = build_absorption_panel(daily, close_hour=close_hour, baseline_weeks=baseline_weeks)
    out: list[AbsorptionResult] = []
    pairs = panel[["store_id", "category_id"]].drop_duplicates().itertuples(index=False)
    for store_id, category_id in pairs:
        res = fit_absorption(panel, store_id, category_id)
        if res is not None:
            out.append(res)
    return out
```

`src/bakery/cli.py`에 커맨드 추가 (기존 `console`/`app`/`REPORTS_DIR` 재사용, 다른 `@app.command` 뒤):

```python
@app.command("demand-absorption")
def cmd_demand_absorption(
    source: str = "real",
    data_dir: Path | None = None,
    close_hour: int = 22,
    out_dir: Path = REPORTS_DIR / "demand_absorption",
) -> None:
    """W0 게이트: 카테고리 총량 수요이전 흡수 검정 (leave-one-out 총량보존 β + TOST).

    β≈0(TOST 통과)=흡수→Stage 2 진입 허가, β<0=walk-away. raw sold 타깃.
    """
    from .analysis.demand_absorption import build_absorption_panel, run_absorption

    ds = _load_dataset(source, data_dir)
    panel = build_absorption_panel(ds.daily, close_hour=close_hour)
    results = run_absorption(ds.daily, close_hour=close_hour)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    rows = pd.DataFrame([r.__dict__ for r in results])
    rows.to_csv(out_dir / "results.csv", index=False)

    console.print(f"[bold]demand-absorption[/] source={source} close_hour={close_hour}")
    for r in results:
        color = {"absorb": "green", "walkaway": "red"}.get(r.verdict, "yellow")
        console.print(f"  {r.store_id}/{r.category_id}: β={r.beta:+.3f} "
                      f"CI90[{r.ci_low:+.3f},{r.ci_high:+.3f}] δ={r.delta:.3f} "
                      f"[{color}]{r.verdict}[/] (n={r.n})")
    console.print(f"[green]wrote[/] {out_dir}/panel.parquet, results.csv")
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_demand_absorption.py tests/test_cli_helpers.py -q`
Expected: 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/analysis/demand_absorption.py src/bakery/cli.py tests/test_demand_absorption.py tests/test_cli_helpers.py
git commit -m "feat: demand-absorption 오케스트레이터 + CLI (매장×카테고리 β/TOST 판정)"
```

---

### Task 4: 광교 실데이터 실행 + 판정 + 문서 + TODO

**Files:**
- Create: `reports/demand_absorption/verdict.md` (gitignored reports지만 판정 요약은 문서로)
- Create: `docs/w0_demand_absorption_result.md`
- Modify: `TODO.md`
- Test: 전체 스위트 + 실데이터 실행

**Interfaces:**
- Consumes: Task 1~3 전체

- [ ] **Step 1: 전체 테스트**

Run: `uv run pytest -q`
Expected: 전부 PASS (기준선 + 신규 ~8).

- [ ] **Step 2: 광교 실데이터 실행**

Run: `uv run bakery demand-absorption --source real`
Expected: store_gw01의 bread/pastry/sandwich/cake 각각 β·CI90·δ·verdict 출력, `reports/demand_absorption/{panel.parquet,results.csv}` 생성. 에러 없음.

- [ ] **Step 3: placebo 강건성 확인 (수동)**

placebo: 처치를 미래(d+7)로 shift한 패널로 재적합해 β가 유의하지 않은지 확인. 아래 일회성 스크립트를 repo root에서 실행하고 결과를 verdict.md에 기록:

```bash
uv run python -c "
import pandas as pd
from bakery.data.loader import load_dataset
from bakery.analysis.demand_absorption import build_absorption_panel, fit_absorption
p = build_absorption_panel(load_dataset('real').daily)
# placebo: shift treatment forward within (store,cat) by 7 rows
p = p.sort_values('date')
p['stockout_hours'] = p.groupby(['store_id','category_id'])['stockout_hours'].shift(-7)
p = p.dropna(subset=['stockout_hours'])
for cat in sorted(p['category_id'].unique()):
    r = fit_absorption(p, 'store_gw01', cat)
    if r: print(f'placebo {cat}: beta={r.beta:+.3f} verdict={r.verdict}')
"
```
Expected: placebo β가 실제 β보다 0에 가깝거나 verdict가 흡수/무의미 (허위상관 없음 확인). walkaway가 placebo에서도 강하게 나오면 경보 → verdict.md에 명시.

**조건부 추가 강건성** (게이트 판정이 walkaway 또는 inconclusive로 나온 카테고리에 한해서만 — absorb면 생략):
- **통제 ablation**: `other_cat_sold`만 / `cat_baseline`만 / 둘 다로 β 이동 방향 비교 (이중 통제가 confound를 실제로 끊는지 입증). `_design_matrix`에서 해당 컬럼을 뺀 패널로 `fit_absorption` 재호출.
- **처치 sensitivity**: `stockout_hours`(lost-hours) 대신 카테고리 내 조기품절 품목 수(count)로 T 재정의 후 β 부호 재현성 확인.
두 결과를 verdict.md에 첨부해 판정을 방어한다. absorb 판정은 placebo만으로 충분.

- [ ] **Step 4: 판정 문서 작성**

실제 결과 수치를 넣어 `docs/w0_demand_absorption_result.md` 작성 — 각 카테고리 β/CI/verdict 표, cake 별도, placebo 결과, **W0 게이트 통과/실패 판정과 그에 따른 모델링 확정**(통과→v4 Stage 1→2 정당 / 실패→품목단위 유지 or 흡수 부분반영). `reports/demand_absorption/verdict.md`에도 동일 요약.

- [ ] **Step 5: TODO 갱신 + Commit**

`TODO.md`의 W0 항목을 결과에 맞게:
```markdown
- [x] W0 게이트 = 수요이전(흡수) 검증 — 카테고리 총량보존 β/TOST 직접검정. 결과 docs/w0_demand_absorption_result.md.
```

```bash
git add docs/w0_demand_absorption_result.md TODO.md
git commit -m "docs: W0 수요이전 흡수 검증 결과 + 게이트 판정"
```

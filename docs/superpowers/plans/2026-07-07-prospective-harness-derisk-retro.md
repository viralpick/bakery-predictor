# 전향 KPI harness de-risk + full-window 회고 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 신규 데이터 없이 광교 5년 실데이터로 전향 KPI harness를 안정화(음수폐기·baseline proxy·calibration)하고, full-window rolling 회고로 Δ KPI를 신뢰구간과 함께 확정한다.

**Architecture:** 기존 단일창 `prospective-eval --source real` 파이프라인(cli.py)을 재사용하며, (Phase 1) 진단·정합 함수를 얇게 추가하고, (Phase 2) `_quantile_backtest_predictions`를 `n_folds`로 열어 fold별 KPI를 집계한다. leakage 방어(`generate_time_splits` expanding + `assert_no_leakage`)는 손대지 않는다.

**Tech Stack:** Python, pandas, numpy, LightGBM(GlobalLGBM v2), typer CLI, pytest. 신규 의존성 없음.

## Global Constraints

- **Time leakage 금지** — lag/rolling feature는 split 이후. `test_split_leakage.py`/`test_features_leakage.py` + `assert_no_leakage`가 착수·완료 조건으로 그린이어야 한다. backtest 경로(`generate_time_splits` expanding + `run_backtest`)를 직접 우회하지 않는다.
- **품절 데이터는 censored** — `is_stockout` 보존. 도착곡선(`build_arrival_profile`)은 품절 item-day를 exclude한다(기존 계약 유지).
- **Random split 금지** — 시간순 rolling/expanding만.
- **MAPE 단독 금지** — 메인 지표 WAPE, calibration은 WAPE/WPE 짝으로 본다.
- **Synthetic↔Real 경계** — 신규 함수는 DataFrame을 받아 테스트는 합성 fixture로만 한다(실 parquet/xlsx 의존 금지).
- **코드 품질** — 함수 30줄 이내, 테스트 단언은 정확값 `==`(비결정값만 느슨, 이유 주석).
- **단일매장 가드 유지** — `_load_real_daily`의 `n_stores != 1` loud-fail 보존.

---

## File Structure

- `src/bakery/ingest/inventory.py` — 음수폐기 처리 함수 추가 (T2 지점: `waste_qty` 생성처)
- `src/bakery/evaluation/prospective.py` — actual-vs-simulated waste 대조 + fold별 KPI 집계 함수 추가
- `src/bakery/evaluation/metrics.py` — quantile 초과율(calibration 진단) 추가
- `src/bakery/evaluation/backtest.py` — 변경 없음 (pred_df에 이미 `fold` 존재, 재사용)
- `src/bakery/cli.py` — `_quantile_backtest_predictions`/`_our_order_predictions`/`_fill_our_order`/`cmd_prospective_eval` multi-fold 확장 + 진단 wiring + baseline proxy 특성화
- `tests/ingest/test_inventory_negative_waste.py` — 신규
- `tests/evaluation/test_waste_reconciliation.py` — 신규
- `tests/evaluation/test_calibration_diagnostic.py` — 신규
- `tests/evaluation/test_baseline_proxy.py` — 신규
- `tests/evaluation/test_fold_aggregation.py` — 신규
- `docs/retro_harness_result.md` — Phase 2 결과로 갱신

---

## Phase 1 — de-risk

### Task 1: 음수 폐기량 처리 (T2)

**Files:**
- Modify: `src/bakery/ingest/inventory.py` (add `handle_negative_waste`)
- Test: `tests/ingest/test_inventory_negative_waste.py`

**Interfaces:**
- Produces: `handle_negative_waste(inv: pd.DataFrame, *, policy: str = "clip") -> tuple[pd.DataFrame, dict]` — `policy="clip"`이면 음수 `waste_qty`를 0으로 clip; report = `{"policy": str, "n_negative": int, "min_value": float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_inventory_negative_waste.py
import pandas as pd
from bakery.ingest.inventory import handle_negative_waste


def test_clip_negative_waste_clips_and_reports():
    inv = pd.DataFrame({
        "date": ["20240101", "20240102", "20240103"],
        "item_id": ["a", "b", "c"],
        "production_qty": [10, 5, 8],
        "waste_qty": [-3, 2, -1],
    })
    cleaned, report = handle_negative_waste(inv, policy="clip")
    assert cleaned["waste_qty"].tolist() == [0, 2, 0]
    assert report == {"policy": "clip", "n_negative": 2, "min_value": -3.0}


def test_clip_no_negatives_reports_zero():
    inv = pd.DataFrame({"waste_qty": [0, 2, 5]})
    cleaned, report = handle_negative_waste(inv, policy="clip")
    assert cleaned["waste_qty"].tolist() == [0, 2, 5]
    assert report == {"policy": "clip", "n_negative": 0, "min_value": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingest/test_inventory_negative_waste.py -v`
Expected: FAIL with `ImportError: cannot import name 'handle_negative_waste'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/ingest/inventory.py — module-level function 추가
def handle_negative_waste(
    inv: pd.DataFrame, *, policy: str = "clip"
) -> tuple[pd.DataFrame, dict]:
    """재고정보 폐기량 음수(반품/보정 추정) 처리 + 리포트.

    광교 실데이터에 음수 ~3.3%(min −31) 관측. actual-waste sanity(Task 2) 전에
    반드시 통과시킨다. 현재 policy는 clip-at-0만 지원(음수를 반품으로 보고 폐기 0 처리).
    """
    if policy != "clip":
        raise ValueError(f"unsupported policy: {policy!r} (only 'clip')")
    w = pd.to_numeric(inv["waste_qty"], errors="coerce")
    report = {
        "policy": policy,
        "n_negative": int((w < 0).sum()),
        "min_value": float(w.min()) if len(w) else 0.0,
    }
    out = inv.copy()
    out["waste_qty"] = w.clip(lower=0)
    return out, report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingest/test_inventory_negative_waste.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ingest/inventory.py tests/ingest/test_inventory_negative_waste.py
git commit -m "feat: 재고정보 음수 폐기량 clip 처리 + 리포트 (T2)"
```

---

### Task 2: actual vs simulated 폐기 대조 (T2)

**Files:**
- Modify: `src/bakery/evaluation/prospective.py` (add `compare_actual_vs_simulated_waste`)
- Test: `tests/evaluation/test_waste_reconciliation.py`

**Interfaces:**
- Consumes: baseline KPI 프레임(`simulate_item_day_kpis` 출력, `waste_units` 컬럼) + rows(`waste_qty` 실측, item_id/date 키).
- Produces: `compare_actual_vs_simulated_waste(rows: pd.DataFrame, base_kpis: pd.DataFrame) -> dict` → `{"actual_total": float, "simulated_total": float, "ratio": float, "n_rows": int}`. ratio = simulated/actual(actual==0이면 nan).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_waste_reconciliation.py
import numpy as np
import pandas as pd
from bakery.evaluation.prospective import compare_actual_vs_simulated_waste


def test_actual_vs_simulated_waste_totals_and_ratio():
    rows = pd.DataFrame({
        "item_id": ["a", "b"], "date": ["2024-01-01", "2024-01-01"],
        "waste_qty": [4.0, 6.0],  # actual total = 10
    })
    base_kpis = pd.DataFrame({
        "item_id": ["a", "b"], "date": ["2024-01-01", "2024-01-01"],
        "waste_units": [3.0, 2.0],  # simulated total = 5
    })
    result = compare_actual_vs_simulated_waste(rows, base_kpis)
    assert result == {"actual_total": 10.0, "simulated_total": 5.0,
                      "ratio": 0.5, "n_rows": 2}


def test_actual_zero_gives_nan_ratio():
    rows = pd.DataFrame({"item_id": ["a"], "date": ["2024-01-01"], "waste_qty": [0.0]})
    base_kpis = pd.DataFrame({"item_id": ["a"], "date": ["2024-01-01"], "waste_units": [2.0]})
    result = compare_actual_vs_simulated_waste(rows, base_kpis)
    assert np.isnan(result["ratio"])
    assert result["actual_total"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_waste_reconciliation.py -v`
Expected: FAIL with `ImportError: cannot import name 'compare_actual_vs_simulated_waste'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py — module-level function 추가
def compare_actual_vs_simulated_waste(
    rows: pd.DataFrame, base_kpis: pd.DataFrame
) -> dict:
    """실측 폐기량(rows.waste_qty) 총합 vs 시뮬 폐기(base_kpis.waste_units) 총합 대조.

    baseline 발주=생산량이므로 시뮬 폐기(생산−복원수요)와 실측 폐기(생산−판매)는
    복원분만큼 구조적으로 다르다. ratio가 1에서 크게 벗어나면 시뮬/복원 가정 재점검 신호.
    """
    actual = float(pd.to_numeric(rows["waste_qty"], errors="coerce").fillna(0.0).sum())
    simulated = float(base_kpis["waste_units"].sum())
    ratio = simulated / actual if actual != 0 else float("nan")
    return {"actual_total": actual, "simulated_total": simulated,
            "ratio": ratio, "n_rows": int(len(rows))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_waste_reconciliation.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/evaluation/test_waste_reconciliation.py
git commit -m "feat: actual vs simulated 폐기 대조 진단 (T2)"
```

---

### Task 3: quantile calibration 초과율 진단 (T3)

**Files:**
- Modify: `src/bakery/evaluation/metrics.py` (add `quantile_exceedance_rate`)
- Test: `tests/evaluation/test_calibration_diagnostic.py`

**Interfaces:**
- Produces: `quantile_exceedance_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float` = `mean(y_true > y_pred)`. q_α 발주가 보정됐으면 초과율 ≈ 1−α (예: q0.85 → ≈0.15). shape 불일치는 ValueError.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_calibration_diagnostic.py
import numpy as np
import pytest
from bakery.evaluation.metrics import quantile_exceedance_rate


def test_exceedance_rate_counts_strict_exceed():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([15.0, 15.0, 35.0, 35.0])  # exceed at idx1, idx3
    assert quantile_exceedance_rate(y_true, y_pred) == 0.5


def test_exceedance_all_covered_is_zero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    assert quantile_exceedance_rate(y_true, y_pred) == 0.0


def test_exceedance_shape_mismatch_raises():
    with pytest.raises(ValueError):
        quantile_exceedance_rate(np.array([1.0, 2.0]), np.array([1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_calibration_diagnostic.py -v`
Expected: FAIL with `ImportError: cannot import name 'quantile_exceedance_rate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/metrics.py — wpe 아래에 추가
def quantile_exceedance_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """실측이 예측(q_α 발주)을 초과한 비율 = P(y_true > y_pred).

    보정된 q_α 발주면 ≈ 1−α (q0.85 → ≈0.15). 이보다 크게 높으면 과소발주(매진↑),
    낮으면 과대발주(폐기↑). WPE(부호 편향)와 짝으로 분포 편중을 진단한다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true > y_pred))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_calibration_diagnostic.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/metrics.py tests/evaluation/test_calibration_diagnostic.py
git commit -m "feat: quantile calibration 초과율 진단 (T3)"
```

---

### Task 4: baseline proxy 특성화 + swap 지점 정비 (T1)

**Files:**
- Modify: `src/bakery/evaluation/prospective.py` (add `characterize_baseline_proxy`)
- Modify: `src/bakery/cli.py` (`_assemble_real_rows`에서 base_order 선택을 `select_base_order`로 추출)
- Test: `tests/evaluation/test_baseline_proxy.py`

**Interfaces:**
- Produces:
  - `characterize_baseline_proxy(rows: pd.DataFrame, waste_report: dict) -> dict` → `{"n_item_days": int, "stockout_share": float, "negative_waste_share": float, "carryover_note": str}`. `stockout_share`=is_stockout 비율(생산=판매+폐기 항등식이 깨지는 대표 요인), `negative_waste_share`=waste_report의 n_negative/n_item_days.
  - `select_base_order(merged: pd.DataFrame, *, source: str = "production") -> pd.Series` (cli.py) — 현재 `source="production"`은 `production_qty` 반환. 전향 실발주 수령 시 `source="order_feed"` 분기를 여기 한 곳에만 추가(swap 지점).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_baseline_proxy.py
import pandas as pd
from bakery.evaluation.prospective import characterize_baseline_proxy


def test_characterize_baseline_proxy_shares():
    rows = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "date": ["2024-01-01"] * 4,
        "is_stockout": [True, False, True, False],  # 2/4 = 0.5
    })
    waste_report = {"policy": "clip", "n_negative": 1, "min_value": -3.0}
    result = characterize_baseline_proxy(rows, waste_report)
    assert result["n_item_days"] == 4
    assert result["stockout_share"] == 0.5
    assert result["negative_waste_share"] == 0.25  # 1/4
    assert isinstance(result["carryover_note"], str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_baseline_proxy.py -v`
Expected: FAIL with `ImportError: cannot import name 'characterize_baseline_proxy'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py — module-level function 추가
def characterize_baseline_proxy(rows: pd.DataFrame, waste_report: dict) -> dict:
    """'생산량 ≈ 발주' proxy를 깨는 요인 정량화(실발주 대조는 전향 단계).

    - stockout_share: 생산=판매+폐기 항등식이 복원분만큼 깨지는 item-day 비율
    - negative_waste_share: 반품/보정으로 clip된 비율(Task 1 report)
    """
    n = int(len(rows))
    stockout_share = float(rows["is_stockout"].astype(bool).mean()) if n else float("nan")
    neg_share = waste_report["n_negative"] / n if n else float("nan")
    return {
        "n_item_days": n,
        "stockout_share": stockout_share,
        "negative_waste_share": float(neg_share),
        "carryover_note": (
            "base_order=생산량 proxy. 당일폐기 N 품목 이월·당일 재생산은 미분리 — "
            "전향 실발주 피드 수령 시 select_base_order(source='order_feed')로 교체."
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_baseline_proxy.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: swap 지점 추출 — `select_base_order` (cli.py)**

`_assemble_real_rows`의 `merged.rename(columns={"production_qty": "base_order"})`를 아래로 교체:

```python
# src/bakery/cli.py — _assemble_real_rows 위에 추가
def select_base_order(merged: pd.DataFrame, *, source: str = "production") -> pd.Series:
    """현행 발주 proxy 선택. 지금은 생산량만. 전향 실발주 수령 시 여기만 확장(swap 지점)."""
    if source == "production":
        return merged["production_qty"].astype(float)
    raise ValueError(f"unsupported base_order source: {source!r} (only 'production' until 실발주 수령)")
```

`_assemble_real_rows` 내부 수정 (rename 대신):

```python
    merged["base_order"] = select_base_order(merged, source="production")
    return merged[REAL_ROWS_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 6: Run leakage + prospective tests to verify no regression**

Run: `uv run pytest tests/ -k "leakage or prospective" -v`
Expected: PASS (기존 그린 유지)

- [ ] **Step 7: Commit**

```bash
git add src/bakery/evaluation/prospective.py src/bakery/cli.py tests/evaluation/test_baseline_proxy.py
git commit -m "feat: baseline proxy 특성화 + base_order swap 지점 추출 (T1)"
```

---

## Phase 2 — full-window rolling 회고

### Task 5: multi-fold 예측 + fold별 KPI 집계 (T4+T5)

**Files:**
- Modify: `src/bakery/cli.py` (`_quantile_backtest_predictions`, `_our_order_predictions`, `_fill_our_order`, `cmd_prospective_eval`)
- Modify: `src/bakery/evaluation/prospective.py` (add `compare_policies_by_fold`, `aggregate_fold_kpis`)
- Test: `tests/evaluation/test_fold_aggregation.py`

**Interfaces:**
- Consumes: `simulate_item_day_kpis` 출력(우리/baseline), 각 행에 `fold` 컬럼 존재.
- Produces:
  - `compare_policies_by_fold(our_kpis, base_kpis) -> pd.DataFrame` — fold별 Δ(우리−baseline) 1행씩. 컬럼: `fold, waste_cost_krw, lost_margin_krw, stockout_rate, soldout_median_h`.
  - `aggregate_fold_kpis(per_fold: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame` — metric별 `{metric, mean, std, sem, n, ci95_low, ci95_high}`. CI는 정규근사 `mean ± 1.96*sem`(fold 수 적음 — 문서에 caveat).
- `_quantile_backtest_predictions(daily, *, val_weeks, production_quantile, n_folds=1)`가 preds에 `fold` 컬럼을 포함해 반환하도록 확장.

- [ ] **Step 1: Write the failing test (집계 함수)**

```python
# tests/evaluation/test_fold_aggregation.py
import numpy as np
import pandas as pd
from bakery.evaluation.prospective import compare_policies_by_fold, aggregate_fold_kpis


def test_compare_policies_by_fold_computes_delta_per_fold():
    our = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "waste_cost_krw": [10.0, 10.0, 20.0, 20.0],
        "lost_margin_krw": [1.0, 1.0, 2.0, 2.0],
        "is_stockout": [True, False, True, True],
        "soldout_hour": [15.0, np.nan, 16.0, 14.0],
    })
    base = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "waste_cost_krw": [4.0, 4.0, 5.0, 5.0],
        "lost_margin_krw": [0.5, 0.5, 1.0, 1.0],
        "is_stockout": [True, True, True, True],
        "soldout_hour": [16.0, 17.0, 18.0, 18.0],
    })
    out = compare_policies_by_fold(our, base)
    row0 = out[out["fold"] == 0].iloc[0]
    # fold0: waste Δ = (10+10) - (4+4) = 12 ; stockout_rate Δ = 0.5 - 1.0 = -0.5
    assert row0["waste_cost_krw"] == 12.0
    assert row0["stockout_rate"] == -0.5


def test_aggregate_fold_kpis_mean_and_ci():
    per_fold = pd.DataFrame({"fold": [0, 1, 2], "waste_cost_krw": [10.0, 20.0, 30.0]})
    agg = aggregate_fold_kpis(per_fold, ["waste_cost_krw"])
    r = agg.iloc[0]
    assert r["metric"] == "waste_cost_krw"
    assert r["mean"] == 20.0
    assert r["n"] == 3
    # std(ddof=1)=10, sem=10/sqrt(3)
    assert abs(r["sem"] - (10.0 / np.sqrt(3))) < 1e-9
    assert abs(r["ci95_low"] - (20.0 - 1.96 * 10.0 / np.sqrt(3))) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_fold_aggregation.py -v`
Expected: FAIL with `ImportError: cannot import name 'compare_policies_by_fold'`

- [ ] **Step 3: Write implementation (prospective.py)**

```python
# src/bakery/evaluation/prospective.py — compare_policies 아래에 추가
def compare_policies_by_fold(
    our_kpis: pd.DataFrame, base_kpis: pd.DataFrame
) -> pd.DataFrame:
    """fold별 Δ(우리−baseline) KPI. 각 프레임은 fold 컬럼을 가져야 한다."""
    rows = []
    for fold in sorted(our_kpis["fold"].unique()):
        our = _summarize_policy(our_kpis[our_kpis["fold"] == fold])
        base = _summarize_policy(base_kpis[base_kpis["fold"] == fold])
        rows.append({"fold": int(fold), **{k: our[k] - base[k] for k in our}})
    return pd.DataFrame(rows)


def aggregate_fold_kpis(per_fold: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """fold별 Δ를 metric별 mean ± 95%CI(정규근사)로 집계. fold 수 적음 — caveat 문서화."""
    out = []
    for col in metric_cols:
        vals = per_fold[col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        n = int(len(vals))
        mean = float(np.mean(vals)) if n else float("nan")
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 else float("nan")
        half = 1.96 * sem if n > 1 else float("nan")
        out.append({"metric": col, "mean": mean, "std": std, "sem": sem, "n": n,
                    "ci95_low": mean - half, "ci95_high": mean + half})
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_fold_aggregation.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: CLI multi-fold wiring — `_quantile_backtest_predictions`에 `fold` 보존**

`src/bakery/cli.py`의 `_quantile_backtest_predictions`를 수정 (n_folds 파라미터 + fold 컬럼 보존):

```python
def _quantile_backtest_predictions(
    daily: pd.DataFrame, *, val_weeks: int, production_quantile: float, n_folds: int = 1,
) -> tuple[pd.DataFrame, list]:
    """최근 n_folds개 non-overlapping val 창(각 val_weeks)에서 q{α} v2 예측.
    Leakage 없음(expanding + assert_no_leakage). fold별 KPI 집계를 위해 fold 컬럼 보존."""
    windows = generate_time_splits(
        daily["date"], n_splits=n_folds,
        val_horizon_days=val_weeks * 7, step_days=val_weeks * 7,
    )
    forecaster = GlobalLGBM(
        feature_set="v2", params=LGBMParams(objective="quantile", alpha=production_quantile)
    )
    _, pred_df = run_backtest(daily, [forecaster], windows, y_col="potential_demand")
    preds = pred_df[["item_id", "date", "fold", "yhat"]].rename(columns={"yhat": "our_order"})
    return preds, windows
```

- [ ] **Step 6: `_our_order_predictions` / `_fill_our_order`에 n_folds·fold 전파**

`_our_order_predictions` 시그니처에 `n_folds: int = 1` 추가, 내부 호출에 전달, 로그에 `n_folds`/fold별 val 기간 출력. `_fill_our_order`는 `preds`에 `fold`가 있으면 merge 후 유지(현재 `["item_id","date"]` inner merge에 fold가 preds에서 딸려옴 — 코드 변경 없이 유지되나, 명시적으로 fold 컬럼 존재 확인 로그 추가). `_real_prospective_inputs`/`_load_prospective_inputs`에도 `n_folds` 전파.

```python
# _our_order_predictions 시그니처
def _our_order_predictions(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
) -> pd.DataFrame:
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    preds, windows = _quantile_backtest_predictions(
        daily, val_weeks=val_weeks, production_quantile=production_quantile, n_folds=n_folds
    )
    console.print(
        f"[cyan]our_order[/] {len(windows)} fold(s), each {val_weeks}주 "
        f"(store={store_id}, quantile α={production_quantile})"
    )
    return preds
```

- [ ] **Step 7: `cmd_prospective_eval`에 `--n-folds` 옵션 + fold별 집계 출력**

`cmd_prospective_eval`에 옵션 추가 및 집계 분기:

```python
    n_folds: int = typer.Option(1, help="full-window 회고 fold 수(real 소스). 1=단일창(기존)"),
```

KPI 계산부(`our`/`base` 생성 이후) 뒤에 추가:

```python
    from ..evaluation.prospective import (
        compare_policies_by_fold, aggregate_fold_kpis,
        compare_actual_vs_simulated_waste, characterize_baseline_proxy,
    )
    from ..evaluation.metrics import quantile_exceedance_rate

    if source == "real" and n_folds > 1 and "fold" in our.columns:
        per_fold = compare_policies_by_fold(our, base)
        metric_cols = ["waste_cost_krw", "lost_margin_krw", "stockout_rate", "soldout_median_h"]
        agg = aggregate_fold_kpis(per_fold, metric_cols)
        console.print(per_fold.to_string(index=False))
        console.print(agg.to_string(index=False))
        per_fold.to_csv(out_path.with_name("prospective_kpi_per_fold.csv"), index=False)
        agg.to_csv(out_path.with_name("prospective_kpi_agg.csv"), index=False)
```

Phase 1 진단도 real 경로에서 출력(초과율/waste 대조/proxy 특성화):

```python
    if source == "real":
        exceed = quantile_exceedance_rate(
            rows["potential_demand"].to_numpy(), rows["our_order"].to_numpy()
        )
        console.print(f"[cyan]calibration[/] 초과율 P(demand>order)={exceed:.3f} "
                      f"(nominal 1−α={1 - production_quantile:.2f})")
        console.print(f"[cyan]waste sanity[/] {compare_actual_vs_simulated_waste(rows, base)}")
```

- [ ] **Step 8: Run leakage tests + full test suite**

Run: `uv run pytest tests/ -k "leakage" -v && uv run pytest tests/evaluation -v`
Expected: PASS (leakage 그린 + 신규 evaluation 테스트 그린)

- [ ] **Step 9: Commit**

```bash
git add src/bakery/cli.py src/bakery/evaluation/prospective.py tests/evaluation/test_fold_aggregation.py
git commit -m "feat: multi-fold rolling 회고 + fold별 KPI 집계(CI) + real 경로 진단 wiring (T4/T5/T3/T2)"
```

---

### Task 6: full-window 회고 실행 + 문서 갱신 (T6)

**Files:**
- Modify: `docs/retro_harness_result.md`

- [ ] **Step 1: full-window 회고 실행 (최근 1~1.5년, 8주 step)**

Run:
```bash
uv run bakery prospective-eval --source real --store-id store_gw01 \
  --production-quantile 0.85 --our-order-val-weeks 8 --n-folds 8 \
  --out-csv reports/prospective_kpi.csv
```
Expected: 콘솔에 fold별 Δ 표 + 집계(mean±CI) 표 + calibration 초과율 + waste sanity 출력. `reports/prospective_kpi_per_fold.csv`, `reports/prospective_kpi_agg.csv` 생성.

- [ ] **Step 2: 결과 확인 (무언 축소 금지)**

Run: `cat reports/prospective_kpi_agg.csv`
확인: fold 수(n)가 실제 채점된 fold와 일치하는지, 창 밖 drop 로그가 출력됐는지. n_folds=8이 데이터 길이 초과로 빈 fold를 냈으면 로그로 드러나야 한다(그러면 n_folds를 실제 채점 가능 수로 낮춰 재실행하고 그 수를 문서에 기록).

- [ ] **Step 3: `docs/retro_harness_result.md` 갱신**

기존 "## 결과 (8주 창, 단일 fold ...)" 절 아래에 신규 절 추가 (실제 실행 수치로 채움 — 아래는 형식):

```markdown
## Full-window rolling 회고 (최근 1~1.5년, 8주 × N fold)

재현: `uv run bakery prospective-eval --source real --store-id store_gw01 --n-folds N`

| metric (Δ 우리−아티제) | mean | 95% CI | n(fold) |
|---|---|---|---|
| waste_cost_krw | … | […, …] | N |
| stockout_rate | … | […, …] | N |
| soldout_median_h | … | […, …] | N |

- CI는 정규근사(mean ± 1.96·sem), fold 수 적어 넓다 — 방향성 판정용.
- calibration 초과율 P(demand>order)=… (nominal 0.15): …로 해석(과소/과대발주).
- waste sanity ratio(simulated/actual)=…: baseline proxy 신뢰도 …
- baseline proxy 특성화: stockout_share=…, negative_waste_share=…

### Phase 1 de-risk 반영 caveat
- 음수 폐기 clip 처리(n_negative=…, min=…) 후 수치.
- baseline=생산량 proxy, 실발주 대조는 전향 단계(select_base_order swap 지점 준비됨).
- calibration은 진단만(conformal 구현은 별도 spec).
```

- [ ] **Step 4: Commit**

```bash
git add docs/retro_harness_result.md reports/prospective_kpi_agg.csv reports/prospective_kpi_per_fold.csv
git commit -m "docs: full-window rolling 회고 결과 + Phase 1 de-risk caveat (T6)"
```

> 주의: `reports/`는 gitignored일 수 있다(.claude/CLAUDE.md). `git add -f` 필요 여부 확인하고, ignored면 CSV는 커밋에서 빼고 문서에 수치만 반영한다.

---

## Self-Review

**Spec coverage:**
- T1(baseline proxy 특성화+swap) → Task 4 ✓
- T2(음수폐기+actual sanity) → Task 1 + Task 2 ✓
- T3(calibration 진단, conformal 없음) → Task 3 + Task 5 Step 7 wiring ✓
- T4(rolling multi-fold) → Task 5 Step 5–6 ✓
- T5(fold 집계 mean±CI) → Task 5 Step 3 ✓
- T6(문서 갱신) → Task 6 ✓
- 비범위(conformal/실발주/다매장/FreshRetailNet/classification) → 계획에 없음 ✓

**Placeholder scan:** Task 6 문서 절은 "실제 실행 수치로 채움"이 명시된 형식 템플릿(실행 결과 의존이라 불가피) — 그 외 code step은 모두 완전 코드. ✓

**Type consistency:** `handle_negative_waste`→report dict 키(`policy/n_negative/min_value`)가 Task 4 `characterize_baseline_proxy`의 `waste_report["n_negative"]` 소비와 일치 ✓. `compare_policies_by_fold` 출력 컬럼(`waste_cost_krw` 등)이 `aggregate_fold_kpis`의 metric_cols와 일치 ✓. `_summarize_policy` 키(`waste_cost_krw/lost_margin_krw/stockout_rate/soldout_median_h`) 재사용 ✓. preds `fold` 컬럼은 `run_backtest`가 실제 생성(`assign(fold=w.fold_index)`) ✓.

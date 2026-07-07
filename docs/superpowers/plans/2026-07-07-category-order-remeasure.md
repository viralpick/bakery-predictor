# 카테고리-레벨 발주 재측정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 전향 retro harness의 `our_order`를 item-level v2 LGBM → v4 카테고리 스택(Stage1 통합 총합 q0.85 예측 + Stage2 품목 배분)으로 교체하고, full-window 회고로 calibration/KPI를 재측정해 item-level 결과와 side-by-side 비교한다.

**Architecture:** 기존 `category_total.expanding_window_backtest`의 leakage-safe fold 패턴(train=이전, test=이후, `fit_category_total`→`predict_production`)을 그대로 따르는 fold-예측 헬퍼를 만들고, `item_proportion.distribute_total`(비율은 `history<target_date`만 사용)로 item 발주로 배분한다. `prospective-eval --order-level {item|category}`로 분기(기본 item = 기존 동작 보존). KPI 집계·진단은 PR#26 harness 재사용.

**Tech Stack:** Python, pandas, numpy, LightGBM, typer, pytest. 신규 의존성 없음. 기존 모듈만 배선.

## Global Constraints

- **Time leakage 금지**: (a) 카테고리 총합 예측은 `expanding_window_backtest`식 train=이전/test=이후만. (b) 품목 비율은 `compute_proportions`가 `history < target_date`만 사용 — `_per_item_signals`가 target_date 이후를 참조 안 함을 테스트로 고정. `test_split_leakage`/`test_features_leakage` 그린 유지.
- **품절 데이터 censored**: 도착곡선/KPI는 기존 계약(품절 item-day exclude) 유지.
- **Random split 금지**: 시간순 rolling만.
- **MAPE 단독 금지**: calibration은 초과율 + WAPE/WPE 짝.
- **Synthetic↔Real**: 신규 헬퍼 테스트는 합성 fixture로만(실 parquet/xlsx 의존 금지).
- **함수 30줄 이내**; **테스트 단언 정확값 `==`** (LGBM 예측처럼 비결정/계약형만 느슨, 이유 주석).
- **`--order-level` 기본 `item`** = 기존 동작 100% 보존.
- **단일매장 가드**(`_load_real_daily`) 유지.
- **Target confound (문서화)**: 발주는 `adjusted_demand_unit` 학습, calibration/KPI 실현수요는 `potential_demand`. 버그 아님 — caveat.

---

## File Structure

- `src/bakery/cli.py` — 신규 `_category_total_fold_predictions`, `_category_order_predictions`; `_real_prospective_inputs`/`_load_prospective_inputs`/`cmd_prospective_eval`에 `order_level` 분기 + 카테고리 calibration 진단. (import 추가: `build_category_daily`, `build_features`, `fit_category_total`, `distribute_total`)
- `tests/evaluation/test_category_order.py` — 신규 (fold 예측 contract + 배분 보존)
- `tests/models/test_item_proportion_leakage.py` — 신규 (비율 pre-cutoff 누수 가드)
- `docs/retro_harness_result.md` — 카테고리 재측정 절 추가 (item vs category)

---

### Task 1: 카테고리 총합 fold 예측 헬퍼

**Files:**
- Modify: `src/bakery/cli.py` (add `_category_total_fold_predictions`)
- Test: `tests/evaluation/test_category_order.py`

**Interfaces:**
- Produces: `_category_total_fold_predictions(features: pd.DataFrame, *, production_quantile: float, horizon_days: int, n_folds: int, target_col: str = "adjusted_demand_unit", min_train_days: int = 365) -> pd.DataFrame` → columns `[date, fold, total_order]` (fold 0 = 최신, total_order ≥ 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_category_order.py
import numpy as np
import pandas as pd
from bakery.cli import _category_total_fold_predictions


def _synth_category_features(n_days: int) -> pd.DataFrame:
    """합성 카테고리-일별 프레임: date + target + 숫자 feature 2개. lag/holiday 없이
    fit_category_total의 select_feature_cols가 쓸 수 있는 최소 형태."""
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    rng = np.arange(n_days, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "adjusted_demand_unit": 100.0 + 10.0 * np.sin(rng / 7.0) + rng * 0.1,
        "dow": dates.dayofweek.astype(float),
        "trend": rng,
    })


def test_fold_predictions_shape_and_folds():
    feats = _synth_category_features(365 + 2 * 30)
    out = _category_total_fold_predictions(
        feats, production_quantile=0.85, horizon_days=30, n_folds=2,
    )
    # 계약: fold별 horizon_days개 test date, fold 라벨 {0,1}, 발주 비음수.
    assert set(out["fold"].unique()) == {0, 1}
    assert len(out) == 2 * 30
    assert (out["total_order"] >= 0).all()


def test_fold_predictions_raises_when_insufficient_days():
    feats = _synth_category_features(100)
    try:
        _category_total_fold_predictions(feats, production_quantile=0.85, horizon_days=30, n_folds=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_category_order.py -v`
Expected: FAIL with `ImportError: cannot import name '_category_total_fold_predictions'`

- [ ] **Step 3: Write minimal implementation**

Add imports near cli.py's other model imports:
```python
from .features.category_aggregate import TARGET_CATEGORIES, build_category_daily, build_features
from .models.category_total import fit_category_total
from .models.item_proportion import distribute_total
```
(the `TARGET_CATEGORIES` import line already exists — extend it.)

Add the function:
```python
def _category_total_fold_predictions(
    features: pd.DataFrame, *, production_quantile: float, horizon_days: int,
    n_folds: int, target_col: str = "adjusted_demand_unit", min_train_days: int = 365,
) -> pd.DataFrame:
    """expanding-window fold별 q{production_quantile} 카테고리 총합 발주.

    category_total.expanding_window_backtest의 leakage-safe 패턴(train=이전/test=이후
    iloc 분할, sorted date 1행/1일)을 따르되 production 예측을 test date별로 반환한다.
    """
    df = features.sort_values("date").dropna().reset_index(drop=True)
    total = len(df)
    if total < min_train_days + n_folds * horizon_days:
        raise ValueError(f"not enough category-days: {total} < {min_train_days + n_folds * horizon_days}")
    chunks = []
    for k in range(n_folds):
        test_end = total - k * horizon_days
        test_start = test_end - horizon_days
        model = fit_category_total(
            df.iloc[:test_start], target_col=target_col, production_q=production_quantile,
        )
        test_df = df.iloc[test_start:test_end]
        order = np.clip(model.predict_production(test_df), 0.0, None)
        chunks.append(pd.DataFrame({"date": test_df["date"].to_numpy(), "fold": k, "total_order": order}))
    return pd.concat(chunks, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_category_order.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py tests/evaluation/test_category_order.py
git commit -m "feat: 카테고리 총합 fold 예측 헬퍼 (v4 Stage1 expanding q)"
```

---

### Task 2: 품목 비율 pre-cutoff 누수 가드 테스트

**Files:**
- Test: `tests/models/test_item_proportion_leakage.py`

**Interfaces:**
- Consumes: `bakery.models.item_proportion.compute_proportions(history, target_date)` (기존).

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_item_proportion_leakage.py
import pandas as pd
from bakery.models.item_proportion import compute_proportions


def _hist(rows):
    return pd.DataFrame(rows, columns=["date", "item_id", "category_id", "sold_units", "is_stockout", "stockout_time"])


def test_compute_proportions_ignores_rows_at_or_after_cutoff():
    cutoff = pd.Timestamp("2024-02-01")
    base = [
        ["2024-01-10", "a", "bread", 10, False, pd.NaT],
        ["2024-01-10", "b", "bread", 30, False, pd.NaT],
        ["2024-01-20", "a", "bread", 10, False, pd.NaT],
        ["2024-01-20", "b", "bread", 30, False, pd.NaT],
    ]
    hist1 = _hist([[pd.Timestamp(d), i, c, s, so, t] for d, i, c, s, so, t in base])
    # 미래(>= cutoff)에 극단 판매를 넣어도 비율이 바뀌면 안 된다 (누수 검출).
    future = [[pd.Timestamp("2024-02-05"), "a", "bread", 9999, False, pd.NaT]]
    hist2 = pd.concat([hist1, _hist(future)], ignore_index=True)

    p1 = compute_proportions(hist1, cutoff).set_index("item_id")["proportion"].sort_index()
    p2 = compute_proportions(hist2, cutoff).set_index("item_id")["proportion"].sort_index()
    assert p1.round(9).equals(p2.round(9))
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/models/test_item_proportion_leakage.py -v`
Expected: PASS immediately (this guards existing correct behavior — `_per_item_signals` filters `< cutoff_date`). If it FAILS, STOP and report: `compute_proportions` leaks future data — a real leakage bug that blocks the whole category path.

- [ ] **Step 3: (no implementation — regression guard for existing code)**

If Step 2 passed, no code change. If it failed, do NOT patch silently — report BLOCKED to the controller (this is a Global Constraint violation in existing code that must be triaged).

- [ ] **Step 4: Commit**

```bash
git add tests/models/test_item_proportion_leakage.py
git commit -m "test: 품목 비율 pre-cutoff 누수 가드 (compute_proportions)"
```

---

### Task 3: 카테고리 발주 예측 (실경로 조립 + 배분)

**Files:**
- Modify: `src/bakery/cli.py` (add `_category_order_predictions`)
- Test: `tests/evaluation/test_category_order.py` (배분 보존 추가)

**Interfaces:**
- Consumes: `_category_total_fold_predictions` (Task 1); `distribute_total(history, total_by_date) -> ItemProportionResult` (`.quantities` = [date, item_id, qty]); `build_category_daily()`, `build_features(cd, target_col)`, `_load_real_daily(store_id)`.
- Produces: `_category_order_predictions(store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1) -> pd.DataFrame` → `[item_id, date, fold, our_order]` (item 경로 `_our_order_predictions`와 동일 스키마).

- [ ] **Step 1: Write the failing test (배분 보존 — distribute_total 계약)**

```python
# tests/evaluation/test_category_order.py 에 추가
import pandas as pd
from bakery.models.item_proportion import distribute_total


def test_distribute_total_preserves_category_sum():
    # history: 두 품목, cutoff 이전 판매로 비율 형성
    hist = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-20", "2024-01-20"]),
        "item_id": ["a", "b", "a", "b"],
        "category_id": ["bread", "bread", "bread", "bread"],
        "sold_units": [10, 30, 10, 30],
        "is_stockout": [False, False, False, False],
        "stockout_time": [pd.NaT] * 4,
    })
    totals = pd.Series({pd.Timestamp("2024-02-01"): 100.0})
    res = distribute_total(hist, totals)
    # 배분 보존: 그 날 품목 발주 합 == 카테고리 총합.
    day_sum = res.quantities.groupby("date")["qty"].sum().iloc[0]
    assert round(day_sum, 6) == 100.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/evaluation/test_category_order.py::test_distribute_total_preserves_category_sum -v`
Expected: PASS (guards the allocation invariant we rely on). If it FAILS, STOP — allocation doesn't conserve the total, category orders can't be trusted.

- [ ] **Step 3: Write `_category_order_predictions`**

```python
def _category_order_predictions(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
) -> pd.DataFrame:
    """v4 카테고리 스택: build_category_daily → fold별 q총합(Task1) → distribute_total 배분
    → item별 our_order. item 경로(_our_order_predictions)와 동일 [item_id,date,fold,our_order]."""
    features = build_features(build_category_daily(), target_col="adjusted_demand_unit")
    totals = _category_total_fold_predictions(
        features, production_quantile=production_quantile,
        horizon_days=val_weeks * 7, n_folds=n_folds,
    )
    daily = _load_real_daily(store_id)          # 배분 비율 history (compute_proportions가 <date만 사용)
    chunks = []
    for fold, g in totals.groupby("fold"):
        res = distribute_total(daily, g.set_index("date")["total_order"])
        q = res.quantities.rename(columns={"qty": "our_order"})
        q["fold"] = int(fold)
        chunks.append(q[["item_id", "date", "fold", "our_order"]])
    preds = pd.concat(chunks, ignore_index=True)
    preds["item_id"] = preds["item_id"].astype(str)
    console.print(
        f"[cyan]category our_order[/] {n_folds} fold(s) × {val_weeks}주, q={production_quantile}, "
        f"{preds['date'].nunique()} dates × {preds['item_id'].nunique()} items"
    )
    return preds
```

- [ ] **Step 4: Verify import + no regression**

Run: `uv run python -c "import bakery.cli"` (exit 0) then `uv run pytest tests/evaluation/test_category_order.py -v` (all pass).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py tests/evaluation/test_category_order.py
git commit -m "feat: 카테고리 발주 예측 실경로 조립 + 품목 배분 (v4 Stage1+Stage2 wiring)"
```

---

### Task 4: CLI `--order-level` 분기 + 카테고리 calibration 진단

**Files:**
- Modify: `src/bakery/cli.py` (`_real_prospective_inputs`, `_load_prospective_inputs`, `cmd_prospective_eval`)

**Interfaces:**
- Consumes: `_category_order_predictions` (Task 3), `_our_order_predictions` (기존).
- Produces: `prospective-eval --order-level {item|category}` (기본 `item`). category일 때 `_category_order_predictions` 사용 + (date별 Σpotential_demand vs Σour_order) 초과율 출력.

- [ ] **Step 1: `_real_prospective_inputs`에 order_level 분기**

`_real_prospective_inputs` 시그니처에 `order_level: str = "item"` 추가. 예측 생성부를 분기:
```python
    if order_level == "category":
        predictions = _category_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks, n_folds=n_folds
        )
    else:
        predictions = _our_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks, n_folds=n_folds
        )
```
(나머지 `rows = _fill_our_order(rows, predictions)` 등은 동일 — 두 경로 반환 스키마가 같다.)

- [ ] **Step 2: `_load_prospective_inputs`에 order_level 전파**

시그니처에 `order_level: str = "item"` 추가하고 real 분기에 전달:
```python
    if source == "real":
        return _real_prospective_inputs(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, order_level=order_level,
        )
```

- [ ] **Step 3: `cmd_prospective_eval`에 옵션 + 진단**

typer 옵션 추가:
```python
    order_level: str = typer.Option("item", help="item(기존 v2 LGBM) | category(v4 총합→배분)"),
```
`_load_prospective_inputs(...)` 호출에 `order_level=order_level` 전달. 그리고 real 경로 진단 블록(초과율 출력부) 근처에 카테고리-레벨 초과율 추가:
```python
    if source == "real" and order_level == "category":
        by_date = rows.groupby("date").agg(
            pd_sum=("potential_demand", "sum"), order_sum=("our_order", "sum"),
        )
        cat_exceed = float((by_date["pd_sum"] > by_date["order_sum"]).mean())
        console.print(
            f"[cyan]category calibration[/] 초과율 P(Σdemand>Σorder)={cat_exceed:.3f} "
            f"(nominal 1−q={1 - production_quantile:.2f}), {len(by_date)} dates"
        )
```

- [ ] **Step 4: Verify item-level 회귀 + category 스모크**

Run: `uv run python -c "import bakery.cli"` then `uv run pytest tests/ -k "prospective or leakage or category" -v`
Expected: PASS (기존 prospective/leakage 그린 + 신규 category 테스트 그린). item 기본 경로 불변.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py
git commit -m "feat: prospective-eval --order-level 분기 + 카테고리 calibration 진단"
```

---

### Task 5: full-window 재측정 실행 + 문서 (item vs category)

**Files:**
- Modify: `docs/retro_harness_result.md`

- [ ] **Step 1: 카테고리 경로 full-window 실행 (IN-TURN 대기)**

Run (LGBM fold당 fit — 수 분, 완료까지 대기):
```bash
uv run bakery prospective-eval --source real --store-id store_gw01 \
  --production-quantile 0.85 --our-order-val-weeks 8 --n-folds 8 \
  --order-level category --out-csv reports/prospective_kpi_category.csv
```
Capture verbatim: 카테고리 calibration 초과율, per-fold Δ 표, 집계 mean±CI, waste sanity, dropped/fold 로그.

- [ ] **Step 2: 결과 검증 (무언 축소 금지)**

카테고리 초과율이 item-level 0.636 대비 어떤지, n_folds가 실제 채점됐는지, drop 로그 확인. n_folds=8이 카테고리-일수(min_train 365 + 8×56) 부족으로 실패하면 로그로 드러남 → n_folds를 가능한 수로 낮춰 재실행하고 그 수를 기록.

- [ ] **Step 3: `docs/retro_harness_result.md`에 "카테고리-레벨 재측정" 절 추가**

item-level 절 뒤에 신규 절(실제 수치로 채움 — 형식):
```markdown
## 카테고리-레벨 재측정 (v4 통합 총합 q0.85 → 품목 배분)

재현: `prospective-eval --source real --order-level category --n-folds N`

| Δ (우리−아티제) | item-level (기존) | category-level | 
|---|---|---|
| stockout_rate | −0.015 [CI] | … |
| waste_cost_krw | +0.48M [CI] | … |
| soldout_median_h | −2.03 [CI] | … |
| lost_margin_krw | +[CI] | … |
| **calibration 초과율** | **0.636** | **…** (nominal 0.15) |

- 해석: 카테고리 결정이 초과율을 0.15에 근접시키는가 / 아티제를 이기는가.
- caveat: 발주 target=adjusted_demand vs 실현수요=potential_demand(운영 현실, 버그 아님) / 배분오차 / baseline=생산량 proxy / v4 카테고리=bread·pastry·sandwich 통합 총합(한 묶음 수요).
```

- [ ] **Step 4: Commit**

`git check-ignore reports/prospective_kpi_category.csv` — ignored면 doc만 커밋.
```bash
git add docs/retro_harness_result.md
git commit -m "docs: 카테고리-레벨 재측정 결과 (item vs category side-by-side)"
```

---

## Self-Review

**Spec coverage:**
- 엔진=v4 category_total q0.85 → Task 1 + Task 3 ✓
- item_proportion 배분 → Task 3 ✓
- `--order-level` 분기(기본 item 보존) → Task 4 ✓
- 카테고리 calibration 초과율 (vs 0.636) → Task 4 ✓
- item-level KPI side-by-side + 문서 → Task 5 ✓
- 누수 2겹: (a) expanding fit → Task 1(expanding_window_backtest 패턴) / (b) 비율 pre-cutoff → Task 2 ✓
- 배분 보존 sanity → Task 3 ✓
- target confound caveat → Task 5 문서 ✓
- 비범위(Phase B/q스윕/conformal/per-category/배분개선) → 계획에 없음 ✓

**Placeholder scan:** Task 5 문서 절만 "실제 수치로 채움"(실행 의존, 불가피). 그 외 code step 완전 코드. ✓

**Type consistency:** `_category_total_fold_predictions` 반환 `[date,fold,total_order]`가 Task 3의 `g.set_index("date")["total_order"]` 소비와 일치 ✓. `_category_order_predictions` 반환 `[item_id,date,fold,our_order]`가 item 경로 `_our_order_predictions`와 동일 → `_fill_our_order` 무변경 재사용 ✓. `distribute_total(...).quantities`가 `[date,item_id,qty]`(→our_order 리네임) ✓. `order_level` 파라미터가 cmd→_load→_real 체인 일관 ✓.

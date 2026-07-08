# Target 통일(adjusted_demand) + PR#26·#27 재측정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 전향 retro harness(`prospective-eval`)의 발주 target과 평가 잣대를 `potential_demand`에서 `adjusted_demand`로 통일해 confound/오염을 제거하고, PR#26(item)·#27(category) 결론을 재측정한다.

**Architecture:** item-level `adjusted_demand = sold_units − closing_qty×(1−α)` 필드를 신설하고, (1) item 발주 경로의 v2 LGBM을 이 target으로 재학습, (2) 평가 잣대(초과율/WPE/ρ_DS/KPI 시뮬)를 이 필드로 교체한다. category 경로는 이미 `adjusted_demand_unit` 학습이라 잣대만 교체. 발주·평가가 같은 α를 쓰므로 apples-to-apples.

**Tech Stack:** Python, pandas, LightGBM(quantile), typer CLI, pytest, uv.

## Global Constraints

- **Time leakage 금지**: adjusted_demand는 당일 관측 label로만 사용. lag/rolling은 split 이후 계산(기존 GlobalLGBM 계약 준수). `test_split_leakage.py`/`test_features_leakage.py` 반드시 통과.
- **테스트 단언 강도**: 기대값 아는 단언은 정확값 비교(`==` 또는 `pytest.approx`). truthy/substring 금지.
- **광교 단독**: harness는 단일매장 전제(`_load_real_daily`가 n_stores≠1이면 raise). 4매장 확장은 비목표.
- **하위호환**: synthetic 경로·기존 backtest 호출은 default 인자로 불변 유지.
- **매직값 금지**: α 기본값은 기존 `DEFAULT_ALPHA=0.5`(features/category_aggregate.py) 재사용.
- **커밋 메시지 꼬리말**:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD
  ```

---

## File Structure

- `src/bakery/features/category_aggregate.py` — `build_item_adjusted_demand()` 추가(item 레벨 헬퍼; 카테고리 합 로직과 동일 소스 재사용).
- `src/bakery/evaluation/backtest.py` — `_clone()`이 `y_col` 보존하도록 수정.
- `src/bakery/evaluation/prospective.py` — `simulate_item_day_kpis()`에 `demand_col` 파라미터 추가(기본 `"potential_demand"`).
- `src/bakery/cli.py` — 발주 경로(`_quantile_backtest_predictions`/`_our_order_predictions`/`_category_order_predictions`)와 평가 경로(`_assemble_real_rows`/`REAL_ROWS_COLUMNS`/`cmd_prospective_eval` 본문)에 α 배선 + adjusted_demand 잣대 스왑.
- `tests/test_item_adjusted_demand.py` — 신규 feature 단위 테스트.
- `tests/test_backtest_clone.py` — `_clone` y_col 보존 회귀 테스트.
- `tests/test_prospective.py` — `simulate_item_day_kpis` demand_col 파라미터 테스트 추가.
- `docs/target_unification_remeasure_result.md` — 재측정 결과 문서(Task 6).

---

### Task 1: item-level adjusted_demand feature

**Files:**
- Modify: `src/bakery/features/category_aggregate.py` (신규 함수 추가; 파일 끝 `build_category_daily` 근처)
- Test: `tests/test_item_adjusted_demand.py` (create)

**Interfaces:**
- Consumes: `load_sales_with_discount()`(analysis/discount.py) → `.closing_discount()` (columns: `item_id`, `date`[datetime64], `qty`), `DEFAULT_ALPHA`(=0.5).
- Produces: `build_item_adjusted_demand(daily: pd.DataFrame, discount_rows: pd.DataFrame | None = None, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame` — 입력 daily(요구 컬럼 `item_id`, `date`, `sold_units`)에 `adjusted_demand` 컬럼을 추가한 복사본 반환.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_item_adjusted_demand.py
import pandas as pd
import pytest

from bakery.features.category_aggregate import build_item_adjusted_demand


def _daily(sold):
    return pd.DataFrame({
        "item_id": ["A", "B"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        "sold_units": sold,
    })


def _closing(qty_by_item):
    return pd.DataFrame({
        "item_id": list(qty_by_item.keys()),
        "date": pd.to_datetime(["2021-01-01"] * len(qty_by_item)),
        "qty": list(qty_by_item.values()),
    })


def test_alpha_half_discounts_closing():
    # A: sold=10, closing=4 → adjusted = 10 - 4*(1-0.5) = 8
    daily = _daily([10, 20])
    closing = _closing({"A": 4})  # B has no closing
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=0.5)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(8.0)
    assert got["B"] == pytest.approx(20.0)  # no closing → adjusted == sold


def test_alpha_one_equals_sold():
    daily = _daily([10, 20])
    closing = _closing({"A": 4, "B": 5})
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=1.0)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(10.0)  # adjusted == sold
    assert got["B"] == pytest.approx(20.0)


def test_alpha_zero_equals_normal():
    # adjusted = sold - closing (all closing removed)
    daily = _daily([10, 20])
    closing = _closing({"A": 4, "B": 5})
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=0.0)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(6.0)
    assert got["B"] == pytest.approx(15.0)


def test_input_not_mutated():
    daily = _daily([10, 20])
    build_item_adjusted_demand(daily, discount_rows=_closing({"A": 4}), alpha=0.5)
    assert "adjusted_demand" not in daily.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_item_adjusted_demand.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_item_adjusted_demand'`

- [ ] **Step 3: Write minimal implementation**

`src/bakery/features/category_aggregate.py`의 `build_category_daily` 함수 정의 바로 뒤(약 130행)에 추가:

```python
def build_item_adjusted_demand(
    daily: pd.DataFrame,
    discount_rows: pd.DataFrame | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """item-day별 adjusted_demand = sold_units − closing_qty × (1 − α) 를 추가.

    adjusted = normal + closing×α = (sold − closing) + closing×α = sold − closing×(1−α).
    closing 매칭 없는 item-day는 closing=0 → adjusted == sold_units.
    당일 관측 label(leakage-safe). 입력은 변형하지 않는다.
    """
    if discount_rows is None:
        discount_rows = load_sales_with_discount().closing_discount()
    out = daily.copy()
    out["item_id"] = out["item_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    cd = discount_rows.copy()
    cd["item_id"] = cd["item_id"].astype(str)
    cd["date"] = pd.to_datetime(cd["date"])
    closing_qty = (
        cd.groupby(["item_id", "date"])["qty"].sum().rename("closing_qty").reset_index()
    )
    out = out.merge(closing_qty, on=["item_id", "date"], how="left")
    out["closing_qty"] = out["closing_qty"].fillna(0.0)
    out["adjusted_demand"] = out["sold_units"] - out["closing_qty"] * (1.0 - alpha)
    return out.drop(columns=["closing_qty"])
```

`load_sales_with_discount`는 이미 import돼 있음(`category_aggregate.py:20` `from bakery.analysis.discount import load_sales_with_discount`, `build_category_daily`가 사용 중) — 추가 import 불필요.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_item_adjusted_demand.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/features/category_aggregate.py tests/test_item_adjusted_demand.py
git commit -m "feat: item-level adjusted_demand 필드 (sold − closing×(1−α))

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 2: run_backtest._clone y_col 보존 수정

**Files:**
- Modify: `src/bakery/evaluation/backtest.py:111-125` (`_clone`)
- Test: `tests/test_backtest_clone.py` (create)

**Interfaces:**
- Consumes: `GlobalLGBM(params, y_col, feature_set)` — `self.y_col` 속성 보유(lightgbm_regressor.py:126-132).
- Produces: `_clone(forecaster)`가 LightGBM 클론 시 `forecaster.y_col`을 보존.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_clone.py
from bakery.evaluation.backtest import _clone
from bakery.models.lightgbm_regressor import GlobalLGBM


def test_clone_preserves_non_default_y_col():
    original = GlobalLGBM(feature_set="v2", y_col="adjusted_demand")
    clone = _clone(original)
    assert clone.y_col == "adjusted_demand"


def test_clone_preserves_feature_set_and_default_y_col():
    original = GlobalLGBM(feature_set="v2")  # default y_col = potential_demand
    clone = _clone(original)
    assert clone.feature_set == "v2"
    assert clone.y_col == "potential_demand"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_clone.py -v`
Expected: `test_clone_preserves_non_default_y_col` FAIL — clone.y_col == "potential_demand" (default로 되돌아감), assert 실패.
(두 번째 테스트는 우연히 통과할 수 있으나 첫 테스트가 버그를 잡음.)

- [ ] **Step 3: Write minimal implementation**

`src/bakery/evaluation/backtest.py`의 `_clone` LightGBM 분기 수정:

```python
    if hasattr(forecaster, "params"):  # LightGBM
        kwargs = {"params": forecaster.params}
        if hasattr(forecaster, "feature_set"):
            kwargs["feature_set"] = forecaster.feature_set
        if hasattr(forecaster, "y_col"):
            kwargs["y_col"] = forecaster.y_col
        return cls(**kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest_clone.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run backtest regression to confirm no break**

Run: `uv run pytest tests/ -k "backtest or leakage" --color=no`
Expected: 기존 통과 유지 (leakage 테스트 포함).

- [ ] **Step 6: Commit**

```bash
git add src/bakery/evaluation/backtest.py tests/test_backtest_clone.py
git commit -m "fix: run_backtest._clone이 LGBM y_col 보존 (adjusted_demand 재학습 전제)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 3: simulate_item_day_kpis demand_col 파라미터화

**Files:**
- Modify: `src/bakery/evaluation/prospective.py:80-121` (`simulate_item_day_kpis`)
- Test: `tests/test_prospective.py` (기존 파일에 테스트 추가)

**Interfaces:**
- Consumes: `rows` DataFrame(demand 컬럼 보유).
- Produces: `simulate_item_day_kpis(rows, profiles, *, order_col, store_hours, group_cols, params=None, unit_prices=None, demand_col="potential_demand")` — 실현수요 잣대 컬럼명을 파라미터화. 기본값으로 기존 동작(synthetic) 불변.

- [ ] **Step 1: Write the failing test**

`tests/test_prospective.py`에 추가. 파일 상단 기존 import·헬퍼를 재사용한다고 가정하고, 없으면 아래 자족 테스트 사용:

```python
# tests/test_prospective.py (추가)
import numpy as np
import pandas as pd

from bakery.evaluation.prospective import simulate_item_day_kpis
from bakery.features.potential_demand import StoreHours


def test_simulate_uses_named_demand_col():
    rows = pd.DataFrame({
        "item_id": ["A"],
        "date": pd.to_datetime(["2021-01-01"]),
        "adjusted_demand": [10.0],
        "potential_demand": [999.0],  # 잘못된 잣대 — 선택되면 결과가 달라짐
        "our_order": [10.0],
    })
    sh = StoreHours("store_gw01", 8, 22)
    out = simulate_item_day_kpis(
        rows, profiles={}, order_col="our_order", store_hours=sh,
        group_cols=["item_id"], demand_col="adjusted_demand",
    )
    # order == adjusted_demand == 10 → 폐기 0, lost 0
    assert float(out["waste_units"].iloc[0]) == 0.0
    assert float(out["lost_sale_units"].iloc[0]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective.py::test_simulate_uses_named_demand_col -v`
Expected: FAIL — `TypeError: simulate_item_day_kpis() got an unexpected keyword argument 'demand_col'`

- [ ] **Step 3: Write minimal implementation**

`simulate_item_day_kpis` 시그니처·본문 수정. 3곳의 `potential_demand` 하드코딩(94, 97, 109행)을 `demand_col`로:

```python
def simulate_item_day_kpis(
    rows: pd.DataFrame,
    profiles: dict[tuple, np.ndarray],
    *,
    order_col: str,
    store_hours: StoreHours,
    group_cols: list[str],
    params: CostParams | None = None,
    unit_prices=None,
    demand_col: str = "potential_demand",
) -> pd.DataFrame:
    """item-day별 폐기/lost 비용(business_metrics) + 매진시각/매진여부.

    demand_col: 실현수요 잣대 컬럼명(기본 potential_demand; adjusted_demand로 교체 가능).
    """
    params = params or CostParams()
    prof_in = rows.rename(columns={order_col: "yhat"}).copy()
    prof_in["sold_units"] = prof_in[demand_col]
    costed = simulate_profit(
        prof_in, unit_prices=unit_prices, params=params,
        yhat_col="yhat", sold_col="sold_units", potential_col=demand_col,
    )
    soldout_hours, stockouts = [], []
    for _, r in rows.iterrows():
        gkey = tuple(str(r[c]) for c in group_cols)
        raw = profiles.get(gkey)
        prof = bakery_hour_profile(
            store_hours.open_hour, store_hours.close_hour,
            measured=raw if raw is not None else None,
        )
        t, is_so = simulate_soldout(
            float(r[order_col]), float(r[demand_col]), prof,
            open_hour=store_hours.open_hour, close_hour=store_hours.close_hour,
        )
        soldout_hours.append(t if t is not None else np.nan)
        stockouts.append(is_so)
    out = rows.copy()
    out["waste_units"] = costed["waste_units"].to_numpy()
    out["lost_sale_units"] = costed["lost_sale_units"].to_numpy()
    out["waste_cost_krw"] = costed["waste_cost_krw"].to_numpy()
    out["lost_margin_krw"] = costed["lost_margin_krw"].to_numpy()
    out["soldout_hour"] = soldout_hours
    out["is_stockout"] = stockouts
    return out
```

주의: `simulate_profit`의 `potential_col`은 컬럼명이 실제 존재해야 함 — `demand_col`이 rows에 있으므로 OK. `prof_in`에 `sold_units`를 `demand_col` 값으로 덮어쓰는 기존 동작 유지.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: PASS (신규 + 기존 test_prospective 전부).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/test_prospective.py
git commit -m "refactor: simulate_item_day_kpis에 demand_col 파라미터 (기본 potential_demand 하위호환)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 4: α 배선 + item/category 발주 target을 adjusted_demand로 통일

**Files:**
- Modify: `src/bakery/cli.py`
  - `_quantile_backtest_predictions` (1879-1898): forecaster y_col + run_backtest y_col
  - `_our_order_predictions` (1901-1916): alpha 인자 + daily에 adjusted 주입
  - `_category_order_predictions` (1852~): alpha 인자 → `build_category_daily(alpha=...)`
  - `_real_prospective_inputs` (1938-1961) / `_load_prospective_inputs` (1964-1977): alpha thread
  - `cmd_prospective_eval` (2003~): `--alpha` 옵션 추가

**Interfaces:**
- Consumes: `build_item_adjusted_demand`(Task 1), `_clone` y_col 보존(Task 2), `DEFAULT_ALPHA`.
- Produces: `our_order` 예측이 adjusted_demand로 학습됨(item·category 공통 target 정의). alpha가 CLI→입력조립까지 흐름.

- [ ] **Step 1: import 추가 (cli.py:34 기존 category_aggregate import 확장)**

기존:
```python
from .features.category_aggregate import TARGET_CATEGORIES, build_category_daily, build_features
```
수정(같은 줄에 `build_item_adjusted_demand`, `DEFAULT_ALPHA` 추가):
```python
from .features.category_aggregate import (
    DEFAULT_ALPHA, TARGET_CATEGORIES, build_category_daily,
    build_features, build_item_adjusted_demand,
)
```

- [ ] **Step 2: `_quantile_backtest_predictions`에 target_col 인자 추가**

기존(1879-1898):
```python
def _quantile_backtest_predictions(
    daily: pd.DataFrame, *, val_weeks: int, production_quantile: float, n_folds: int = 1,
) -> tuple[pd.DataFrame, list[SplitWindow]]:
    ...
    forecaster = GlobalLGBM(
        feature_set="v2", params=LGBMParams(objective="quantile", alpha=production_quantile)
    )
    _, pred_df = run_backtest(daily, [forecaster], windows, y_col="potential_demand")
```
수정:
```python
def _quantile_backtest_predictions(
    daily: pd.DataFrame, *, val_weeks: int, production_quantile: float, n_folds: int = 1,
    target_col: str = "adjusted_demand",
) -> tuple[pd.DataFrame, list[SplitWindow]]:
    ...
    forecaster = GlobalLGBM(
        feature_set="v2", y_col=target_col,
        params=LGBMParams(objective="quantile", alpha=production_quantile),
    )
    _, pred_df = run_backtest(daily, [forecaster], windows, y_col=target_col)
```
(docstring의 "q{α} v2 예측" 유지. `daily`가 `target_col` 컬럼을 반드시 보유하도록 호출자 Step 3에서 보장.)

- [ ] **Step 3: `_our_order_predictions`에 alpha 인자 + adjusted 주입**

기존(1901-1916):
```python
def _our_order_predictions(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
) -> pd.DataFrame:
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    preds, windows = _quantile_backtest_predictions(
        daily, val_weeks=val_weeks, production_quantile=production_quantile, n_folds=n_folds
    )
```
수정:
```python
def _our_order_predictions(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    daily = build_item_adjusted_demand(daily, alpha=alpha)
    preds, windows = _quantile_backtest_predictions(
        daily, val_weeks=val_weeks, production_quantile=production_quantile, n_folds=n_folds,
        target_col="adjusted_demand",
    )
```

- [ ] **Step 4: `_category_order_predictions`에 alpha 전달**

`_category_order_predictions` 시그니처에 `alpha: float = DEFAULT_ALPHA` 추가하고, 내부 `build_features(build_category_daily(), target_col="adjusted_demand_unit")` 호출을 `build_features(build_category_daily(alpha=alpha), target_col="adjusted_demand_unit")`로 수정. (다른 인자는 불변.)

- [ ] **Step 5: alpha를 입력 조립까지 thread**

`_real_prospective_inputs` 시그니처에 `alpha: float = DEFAULT_ALPHA` 추가. 내부 두 예측 호출에 `alpha=alpha` 전달:
```python
    if order_level == "category":
        predictions = _category_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, alpha=alpha,
        )
    else:
        predictions = _our_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, alpha=alpha,
        )
```
`_load_prospective_inputs`에도 `alpha: float = DEFAULT_ALPHA` 추가하고 `_real_prospective_inputs(..., alpha=alpha)` 전달.

- [ ] **Step 6: `cmd_prospective_eval`에 `--alpha` 옵션 + 전달**

옵션 추가(other typer.Option들 사이):
```python
    alpha: float = typer.Option(
        DEFAULT_ALPHA, help="adjusted_demand의 마감할인 실수요 비율 α (real 소스만 사용)"
    ),
```
그리고 `_load_prospective_inputs(...)` 호출에 `alpha=alpha` 추가.

- [ ] **Step 7: Smoke test (item 발주가 adjusted로 학습되는지)**

Run: `uv run bakery prospective-eval --source real --order-level item --n-folds 1 --our-order-val-weeks 4`
Expected: 크래시 없이 완주. 콘솔에 `our_order 1 fold(s)` 로그. (이 단계에선 rows 잣대가 아직 potential일 수 있어 WPE/초과율 값은 Task 5에서 확정. 여기선 실행 성공만 확인.)

- [ ] **Step 8: 전체 테스트 회귀**

Run: `uv run pytest tests/ --color=no`
Expected: 전부 통과.

- [ ] **Step 9: Commit**

```bash
git add src/bakery/cli.py
git commit -m "feat: item/category 발주 target을 adjusted_demand로 통일 + α CLI 배선

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 5: 평가 잣대(rows)를 adjusted_demand로 교체

**Files:**
- Modify: `src/bakery/cli.py`
  - `REAL_ROWS_COLUMNS` (1756): `adjusted_demand` 포함
  - `_assemble_real_rows` (1768-1786): adjusted 주입
  - `_real_prospective_inputs` (1945~): daily에 build_item_adjusted_demand 적용
  - `_decoupling_by_category` (1988-2000): potential_demand → adjusted_demand
  - `cmd_prospective_eval` 본문: WPE(2051), exceedance(2069), simulate 호출(2038-2043), category calibration(2074-2085) 잣대 교체

**Interfaces:**
- Consumes: `build_item_adjusted_demand`(Task 1), `simulate_item_day_kpis(..., demand_col=)`(Task 3), alpha(Task 4).
- Produces: 평가 전 지표(초과율/WPE/ρ_DS/KPI 시뮬/category calibration)가 adjusted_demand 잣대로 산출.

- [ ] **Step 1: REAL_ROWS_COLUMNS에 adjusted_demand 추가**

기존(1756):
```python
REAL_ROWS_COLUMNS = [
    "item_id", "date", "category_id", "potential_demand",
    "sold_units", "is_stockout", "base_order", "waste_qty",
]
```
수정:
```python
REAL_ROWS_COLUMNS = [
    "item_id", "date", "category_id", "potential_demand", "adjusted_demand",
    "sold_units", "is_stockout", "base_order", "waste_qty",
]
```
(potential_demand는 진단·비교용으로 잠정 유지.)

- [ ] **Step 2: `_real_prospective_inputs`에서 rows daily에 adjusted 주입**

`daily = _load_real_daily(store_id)` 다음 줄에 추가:
```python
    daily = _load_real_daily(store_id)
    daily = build_item_adjusted_demand(daily, alpha=alpha)
```
(`_assemble_real_rows(daily, inventory)`가 `REAL_ROWS_COLUMNS` select 시 adjusted_demand를 그대로 보존.)

- [ ] **Step 3: KPI 시뮬 호출을 adjusted 잣대로**

`cmd_prospective_eval` 본문의 두 `simulate_item_day_kpis` 호출(our/base)에 `demand_col` 인자 추가. 단 real일 때만 adjusted, synthetic은 potential 유지:
```python
    demand_col = "adjusted_demand" if source == "real" else "potential_demand"
    our = simulate_item_day_kpis(rows, profiles, order_col="our_order",
                                 store_hours=sh, group_cols=["item_id"],
                                 unit_prices=unit_prices, demand_col=demand_col)
    base = simulate_item_day_kpis(rows, profiles, order_col="base_order",
                                  store_hours=sh, group_cols=["item_id"],
                                  unit_prices=unit_prices, demand_col=demand_col)
```

- [ ] **Step 4: WPE·exceedance·ρ_DS·category calibration 잣대 교체**

WPE(2051): real에서 adjusted 사용. `demand_col` 재사용:
```python
    console.print(
        f"[cyan]예측 편향 WPE="
        f"{wpe(rows[demand_col].to_numpy(), rows['our_order'].to_numpy()):.3f}[/]"
    )
```
`_decoupling_by_category`(1996): `group["potential_demand"]` → `group["adjusted_demand"]`.
exceedance(2067-2072):
```python
    if source == "real":
        exceed = quantile_exceedance_rate(
            rows["adjusted_demand"].to_numpy(), rows["our_order"].to_numpy()
        )
```
category calibration(2074-2080): `pd_sum=("potential_demand","sum")` → `pd_sum=("adjusted_demand","sum")` (변수명 pd_sum 유지 가능하나 주석/로그 문구 "Σdemand" 유지).

- [ ] **Step 5: Smoke test (초과율이 adjusted 잣대로 바뀌는지)**

Run(item): `uv run bakery prospective-eval --source real --order-level item --n-folds 1 --our-order-val-weeks 4`
Expected: `calibration 초과율 P(demand>order)=...` 출력. 발주·평가 모두 adjusted라 초과율이 nominal(1−0.85=0.15)에 근접(정확값은 데이터 의존, Task 6에서 판정).

Run(category): `uv run bakery prospective-eval --source real --order-level category --n-folds 1 --our-order-val-weeks 4`
Expected: `category calibration 초과율 ...` 출력, 크래시 없음.

Run(synthetic 회귀): `uv run bakery prospective-eval --source synthetic`
Expected: 기존과 동일 동작(potential 잣대 유지).

- [ ] **Step 6: 전체 테스트 + leakage 회귀**

Run: `uv run pytest tests/ --color=no`
Expected: 전부 통과(특히 test_prospective_cli, leakage).

- [ ] **Step 7: Commit**

```bash
git add src/bakery/cli.py
git commit -m "feat: 평가 잣대(초과율/WPE/ρ_DS/KPI 시뮬)를 adjusted_demand로 교체 (real)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 6: 재측정 실행 + 결과 문서화

**Files:**
- Create: `docs/target_unification_remeasure_result.md`
- (읽기) `reports/prospective_kpi*.csv` 산출물

**Interfaces:**
- Consumes: Task 4·5 완료된 harness.
- Produces: PR#26·#27 대비 재측정 표 + 판정.

- [ ] **Step 1: 헤드라인 재측정 (α=0.5, n_folds=8, item·category)**

Run(item):
```bash
uv run bakery prospective-eval --source real --order-level item \
  --n-folds 8 --production-quantile 0.85 --alpha 0.5 \
  --out-csv reports/remeasure_item_a0.5.csv
```
Run(category):
```bash
uv run bakery prospective-eval --source real --order-level category \
  --n-folds 8 --production-quantile 0.85 --alpha 0.5 \
  --out-csv reports/remeasure_cat_a0.5.csv
```
각 실행의 콘솔 출력(초과율, WPE, Δ KPI + CI, category calibration)을 기록.

- [ ] **Step 2: α 민감도 스윙**

item·category 각각 `--alpha 0.3`, `--alpha 0.7`, `--alpha 1.0`으로 반복 실행(out-csv 이름에 α 표기). 초과율·waste_cost Δ·lost_margin Δ·soldout_median_h Δ를 α별로 표로 수집.

- [ ] **Step 3: 결과 문서 작성**

`docs/target_unification_remeasure_result.md`에:
- 재측정 배경(potential_demand 오염 + target confound 요약, spec 링크).
- **표 1**: PR#26·#27 원본(potential 잣대) vs 재측정(adjusted 잣대) — 초과율/WPE/stockout_rate Δ/waste_cost Δ/lost_margin Δ/soldout_median_h Δ, item·category 각각.
- **표 2**: α 민감도(0.3/0.5/0.7/1.0) — 초과율(α-불변 확인)과 KPI 원화 Δ(α-민감).
- **판정**: 초과율이 nominal(0.15) 근처로 회복했으면 "PR#26·#27 negative는 데이터 아티팩트, adjusted target 전환 정당". 회복 안 했으면 "v2 quantile 모델 자체 under-calibration" 정직 보고.
- 캐비엣: 광교 단독, α 미확정, potential 잔존 참조.

- [ ] **Step 4: PR#26·#27 결과 문서·메모리 헤드라인 정정 표기**

`docs/prospective_harness_derisk_retro` 관련 문서(있으면)와 결과 md 상단에 "⚠️ 재측정으로 갱신됨 — docs/target_unification_remeasure_result.md 참조" 배너 추가.

- [ ] **Step 5: Commit**

```bash
git add docs/target_unification_remeasure_result.md docs/*derisk*.md 2>/dev/null; git add docs/
git commit -m "docs: target 통일 재측정 결과 — PR#26·#27 재판정 (adjusted_demand 잣대)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

## Self-Review (작성자 체크)

- **Spec coverage**: 컴포넌트1→Task1, 컴포넌트2(발주+_clone)→Task2·4, 컴포넌트3(평가)→Task3·5, 컴포넌트4(재측정+문서)→Task6. α 스윙→Task6 Step2. 4매장 비목표→계획 없음(의도적). ✅
- **Placeholder scan**: 모든 코드 스텝에 실제 코드/명령 포함. TBD 없음. ✅
- **Type consistency**: `build_item_adjusted_demand(daily, discount_rows, alpha)` (Task1↔4↔5 동일), `demand_col`(Task3↔5 동일), `target_col="adjusted_demand"`(Task4), `y_col` 보존(Task2↔4). ✅

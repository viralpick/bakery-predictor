# 마감할인 실수요 검증 (Phase A + Phase B 스파이크) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 마감할인 물량 중 진짜 수요 비율 α를 데이터로 실증(Phase A, cost-free 3각 식별)하고, Phase B(원가율 파라미터 백테스트) 착수를 위한 counterfactual formulation을 스파이크로 확정한다.

**Architecture:** 신규 `analysis/closing_demand.py`에 category×day 패널 빌더 + A1(kink)/A2(depth)/A3(surplus) 세 추정기 + α_A 집계. OLS/HC3는 기존 `demand_absorption.py`에서 공유 유틸로 추출해 재사용. Phase B는 마지막 스파이크 태스크에서 salvage-newsvendor(B-1) vs 시뮬레이션(B-2)을 실데이터로 판정 후 후속 플랜.

**Tech Stack:** Python, pandas, numpy (statsmodels 무의존 — 기존 `_ols_hc3` HC3 패턴 재사용), pytest, 기존 CLI(typer) + loader/discount/waste 모듈.

## Global Constraints

- **Time leakage 금지**: 모든 lag/rolling·baseline은 해당 행 date 이전 데이터로만. 미래 sales/weather/폐기 관측을 feature로 금지. (`test_features_leakage.py` 정신 준수)
- **품절 데이터 censored**: 품절 flag(`is_stockout`, `stockout_time`) 보존. 판매모델·위험모델 분리.
- **Random split 금지**: Phase B 백테스트는 시간순 rolling/expanding window.
- **MAPE 단독 금지**: Phase B 성능지표 메인 = WAPE, MAE/RMSE 보조 + 운영 cost.
- **Synthetic↔Real 경계**: 실데이터는 `data/loader.py`/`analysis/discount.py` 진입점 경유. 동일 schema.
- **범위**: 광교 bread/pastry만. sandwich(단일품목)·cake(시즌) 제외 (W0와 동일).
- **레이어 분리**: Stage 1 카테고리 총량만. 품목 potential을 카테고리 합에 넣지 않음(double counting 회피).
- **테스트 단언**: 기대값 아는 단언은 정확값 `==`(또는 `pytest.approx` 수치오차). truthy/부분문자열 금지.
- **코드 품질**: 함수 30줄 이내, 인자 4개 초과 시 dict/dataclass, guard clause early-return, 매직값 상수화.

---

## File Structure

- `src/bakery/analysis/_ols.py` **(신규)** — 공유 OLS/HC3 유틸 (`_ols_hc3`, `_design_matrix`, `MAX_CONDITION_NUMBER`). demand_absorption에서 추출.
- `src/bakery/analysis/demand_absorption.py` **(수정)** — 위 유틸을 `_ols`에서 import (behavior-preserving).
- `src/bakery/analysis/closing_demand.py` **(신규)** — 패널 빌더 + A1/A2/A3 추정기 + α_A 집계.
- `src/bakery/cli.py` **(수정)** — `closing-demand` 명령 추가.
- `tests/test_closing_demand.py` **(신규)** — 항등식 + known-answer 추정 테스트.
- `docs/superpowers/spikes/2026-07-04-phaseB-counterfactual-formulation.md` **(신규, Task 8)** — Phase B formulation 결정 문서.

---

### Task 1: 공유 OLS/HC3 유틸 추출 (behavior-preserving)

DRY: A1/A2/A3 모두 HC3 로버스트 OLS가 필요하다. 기존 `demand_absorption._ols_hc3`/`_design_matrix`를 `_ols.py`로 옮기고 재import한다. 동작 불변 — 기존 흡수 테스트가 그대로 통과해야 한다.

**Files:**
- Create: `src/bakery/analysis/_ols.py`
- Modify: `src/bakery/analysis/demand_absorption.py` (정의 삭제 → import)
- Test: 기존 `tests/test_demand_absorption.py` (변경 없이 green 유지)

**Interfaces:**
- Produces: `_ols_hc3(y: np.ndarray, X: np.ndarray, treat_idx: int) -> tuple[float, float] | None` (β, HC3 SE; ill-posed 시 None), `_design_matrix(...)` (기존 시그니처 동일), `MAX_CONDITION_NUMBER: float`.

- [ ] **Step 1: 기존 흡수 테스트 먼저 실행 (green 기준선 확보)**

Run: `uv run pytest tests/test_demand_absorption.py -q`
Expected: PASS (현재 전부 통과 — 리팩토링 전 기준선)

- [ ] **Step 2: `_ols.py` 생성 — 기존 `_ols_hc3`/`_design_matrix`/`MAX_CONDITION_NUMBER` 본문 그대로 이동**

`demand_absorption.py`의 `_ols_hc3`(현 109–137행), `_design_matrix`, 상수 `MAX_CONDITION_NUMBER`를 잘라 `_ols.py`로 옮긴다. numpy import 포함. 본문은 한 글자도 바꾸지 않는다.

- [ ] **Step 3: `demand_absorption.py`에서 import로 교체**

```python
from ._ols import _ols_hc3, _design_matrix, MAX_CONDITION_NUMBER
```
옮긴 정의·중복 상수는 삭제. 나머지 코드는 그대로.

- [ ] **Step 4: 흡수 테스트 재실행 — 동작 불변 확인**

Run: `uv run pytest tests/test_demand_absorption.py -q`
Expected: PASS (Step 1과 동일 결과 — β/verdict 불변)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/_ols.py src/bakery/analysis/demand_absorption.py
git commit -m "refactor: OLS/HC3 유틸 _ols.py로 추출 (closing_demand 재사용 준비)"
```

---

### Task 2: category×day 패널 빌더 + 항등식 테스트

모든 방법의 공통 입력. discount 라인아이템 + 실측 폐기를 category×day로 집계한다. normal/closing(20·30 분리)/waste/surplus를 한 프레임에.

**Files:**
- Create: `src/bakery/analysis/closing_demand.py`
- Test: `tests/test_closing_demand.py`

**Interfaces:**
- Consumes: `DiscountSales.rows`(receipt_id,date,hour,minute,item_id,qty,unit_price,paid,discount_amt,discount_code,label,is_set), 폐기 DataFrame(date,item_id,waste_qty), `item_to_category: pd.Series` (item_id→category).
- Produces: `build_closing_panel(rows: pd.DataFrame, waste: pd.DataFrame, item_to_category: pd.Series) -> pd.DataFrame` — columns: `category_id, date, normal_qty, closing_qty, closing_qty_20, closing_qty_30, waste_qty, surplus, dow, month, trend`. `surplus == closing_qty + waste_qty`.

- [ ] **Step 1: 실패 테스트 작성 — 분해 항등식**

```python
# tests/test_closing_demand.py
import pandas as pd
import pytest
from bakery.analysis.closing_demand import build_closing_panel

def _rows():
    # 1 day, 1 category(bread) via item->cat map; 2 normal + 3 closing(30%) + 2 closing(20%)
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05"] * 7),
        "hour": [10, 11, 20, 20, 21, 20, 21],
        "minute": [0, 0, 5, 10, 0, 15, 30],
        "item_id": ["A"] * 7,
        "qty": [2, 3, 1, 1, 1, 1, 1],   # normal:5, closing30:3, closing20:2
        "label": ["none", "none", "closing", "closing", "closing", "closing", "closing"],
        "discount_code": ["", "", "0077", "0077", "0077", "0069", "0069"],
    })

def test_panel_decomposition_identity():
    rows = _rows()
    waste = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "item_id": ["A"], "waste_qty": [4]})
    itc = pd.Series({"A": "bread"})
    panel = build_closing_panel(rows, waste, itc)
    r = panel.iloc[0]
    assert r["normal_qty"] == 5
    assert r["closing_qty"] == 5           # 3 + 2
    assert r["closing_qty_30"] == 3
    assert r["closing_qty_20"] == 2
    assert r["waste_qty"] == 4
    assert r["surplus"] == 9               # closing 5 + waste 4

def test_surplus_equals_closing_plus_waste_all_rows():
    rows = _rows()
    waste = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "item_id": ["A"], "waste_qty": [4]})
    panel = build_closing_panel(rows, waste, pd.Series({"A": "bread"}))
    assert (panel["surplus"] == panel["closing_qty"] + panel["waste_qty"]).all()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closing_demand.py -q`
Expected: FAIL (`ModuleNotFoundError` / `build_closing_panel` 없음)

- [ ] **Step 3: `build_closing_panel` 구현**

```python
"""마감할인 실수요 검증 — α 실증 (Phase A: cost-free 3각 식별).

α = B/C: 마감할인 물량 C 중 진짜 수요 B의 비율. 유도분 I=C-B를 인과적으로
분리한다. A1 kink-in-time / A2 depth elasticity / A3 surplus counterfactual.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

CLOSING_DEPTH_30 = "0077"
CLOSING_DEPTH_20 = "0069"

def build_closing_panel(rows, waste, item_to_category):
    df = rows.copy()
    df["category_id"] = df["item_id"].map(item_to_category)
    df = df[df["category_id"].notna()]
    df["is_closing"] = df["label"] == "closing"
    df["is_c30"] = df["discount_code"] == CLOSING_DEPTH_30
    df["is_c20"] = df["discount_code"] == CLOSING_DEPTH_20
    df["normal_q"] = df["qty"].where(~df["is_closing"], 0)
    df["closing_q"] = df["qty"].where(df["is_closing"], 0)
    df["c30_q"] = df["qty"].where(df["is_closing"] & df["is_c30"], 0)
    df["c20_q"] = df["qty"].where(df["is_closing"] & df["is_c20"], 0)
    agg = df.groupby(["category_id", "date"], observed=True).agg(
        normal_qty=("normal_q", "sum"),
        closing_qty=("closing_q", "sum"),
        closing_qty_30=("c30_q", "sum"),
        closing_qty_20=("c20_q", "sum"),
    ).reset_index()
    w = waste.copy()
    w["category_id"] = w["item_id"].map(item_to_category)
    w = w[w["category_id"].notna()]
    w = w.groupby(["category_id", "date"], observed=True)["waste_qty"].sum().reset_index()
    panel = agg.merge(w, on=["category_id", "date"], how="left")
    panel["waste_qty"] = panel["waste_qty"].fillna(0)
    panel["surplus"] = panel["closing_qty"] + panel["waste_qty"]
    d = pd.to_datetime(panel["date"])
    panel["dow"] = d.dt.dayofweek
    panel["month"] = d.dt.month
    panel["trend"] = (d - d.min()).dt.days
    return panel.sort_values(["category_id", "date"]).reset_index(drop=True)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_closing_demand.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/analysis/closing_demand.py tests/test_closing_demand.py
git commit -m "feat: closing_demand 패널 빌더 (normal/closing20-30/waste/surplus 분해)"
```

---

### Task 3: A2 depth elasticity 추정기

두 depth(20%/30%)로 마감 qty의 가격반응 추정 → depth=0 외삽 = base B. α_A2 = B / mean(closing). depth 내생성(surplus 많은 날 깊게 할인)으로 elasticity 상향편향 → α_A2는 **하한**. surplus·dow·trend 통제.

**Files:**
- Modify: `src/bakery/analysis/closing_demand.py`
- Test: `tests/test_closing_demand.py`

**Interfaces:**
- Consumes: `build_closing_panel` 출력을 depth-long으로 편 프레임 (category-date-depth 관측; depth∈{0.2,0.3}, y=해당 depth qty).
- Produces: `@dataclass DepthResult(n:int, slope:float, se:float|None, base:float, alpha:float, note:str)`, `fit_depth_elasticity(panel: pd.DataFrame) -> DepthResult`.

- [ ] **Step 1: 실패 테스트 — known-answer(기울기·α 회복)**

```python
from bakery.analysis.closing_demand import fit_depth_elasticity
import numpy as np, pandas as pd

def _depth_panel(slope, base_per_depth, n=200):
    # closing_qty_d = base + slope*depth (+small noise-free); surplus constant → no confound
    rng = np.arange(n)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng, "D")
    c30 = base_per_depth + slope * 0.30
    c20 = base_per_depth + slope * 0.20
    return pd.DataFrame({
        "category_id": ["bread"] * n, "date": dates,
        "closing_qty_30": [c30] * n, "closing_qty_20": [c20] * n,
        "closing_qty": [c30 + c20] * n, "surplus": [100.0] * n,
        "dow": (rng % 7), "month": 1, "trend": rng,
    })

def test_depth_elasticity_recovers_slope():
    panel = _depth_panel(slope=50.0, base_per_depth=10.0)
    res = fit_depth_elasticity(panel)
    assert res.slope == pytest.approx(50.0, abs=1.0)
    # base at depth=0 == 10 per depth-observation; α = base / mean(observed depth qty)
    assert res.base == pytest.approx(10.0, abs=0.5)
    assert 0.0 <= res.alpha <= 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closing_demand.py::test_depth_elasticity_recovers_slope -q`
Expected: FAIL (`fit_depth_elasticity` 없음)

- [ ] **Step 3: 구현 — depth-long 변환 + `_ols_hc3` 회귀**

```python
from ._ols import _ols_hc3

@dataclass(frozen=True)
class DepthResult:
    n: int
    slope: float
    se: float | None
    base: float
    alpha: float
    note: str

def _depth_long(panel):
    a = panel[["closing_qty_30", "surplus", "dow", "trend"]].rename(columns={"closing_qty_30": "y"})
    a["depth"] = 0.30
    b = panel[["closing_qty_20", "surplus", "dow", "trend"]].rename(columns={"closing_qty_20": "y"})
    b["depth"] = 0.20
    return pd.concat([a, b], ignore_index=True)

def fit_depth_elasticity(panel):
    long = _depth_long(panel)
    long = long[long["y"].notna()]
    if long["depth"].nunique() < 2 or len(long) < 20:
        return DepthResult(len(long), float("nan"), None, float("nan"), float("nan"),
                           "insufficient depth variation")
    y = long["y"].to_numpy(float)
    # design: intercept, depth(treat), surplus, trend, dow one-hot(-1)
    dow = pd.get_dummies(long["dow"], prefix="dow", drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(long)), long["depth"], long["surplus"], long["trend"], dow])
    out = _ols_hc3(y, X, treat_idx=1)
    if out is None:
        return DepthResult(len(long), float("nan"), None, float("nan"), float("nan"), "ill-posed")
    slope, se = out
    base = float(np.clip(y.mean() - slope * long["depth"].mean(), 0.0, None))  # predicted at depth=0
    alpha = float(np.clip(base / y.mean(), 0.0, 1.0)) if y.mean() > 0 else float("nan")
    return DepthResult(len(long), slope, se, base, alpha, "lower-bound (depth endogeneity)")
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_closing_demand.py::test_depth_elasticity_recovers_slope -q`
Expected: PASS

- [ ] **Step 5: depth 시간-confound 진단 테스트 + 헬퍼 추가**

20% vs 30%가 시간대로 갈리면(20% 이른 저녁, 30% 마감임박) A1 confound와 겹친다. 진단만 산출(판정 아님).

```python
def test_depth_time_confound_flag():
    from bakery.analysis.closing_demand import depth_time_overlap
    rows = pd.DataFrame({
        "discount_code": ["0069","0069","0077","0077"],
        "label": ["closing"]*4, "hour": [20,20,21,21], "minute":[0,5,0,5],
    })
    ov = depth_time_overlap(rows)   # returns dict: median hour per depth + separated flag
    assert ov["median_hour_20"] == 20.0
    assert ov["median_hour_30"] == 21.0
    assert ov["time_separated"] is True   # medians differ ≥ 1h
```
구현:
```python
def depth_time_overlap(rows):
    c = rows[rows["label"] == "closing"].copy()
    c["tod"] = c["hour"] + c["minute"] / 60.0
    m20 = float(c.loc[c["discount_code"] == CLOSING_DEPTH_20, "tod"].median())
    m30 = float(c.loc[c["discount_code"] == CLOSING_DEPTH_30, "tod"].median())
    return {"median_hour_20": round(m20), "median_hour_30": round(m30),
            "time_separated": abs(m30 - m20) >= 1.0}
```

- [ ] **Step 6: 통과 확인 + 커밋**

Run: `uv run pytest tests/test_closing_demand.py -q`
Expected: PASS (전체)
```bash
git add src/bakery/analysis/closing_demand.py tests/test_closing_demand.py
git commit -m "feat: A2 depth elasticity 추정기 + depth 시간-confound 진단"
```

---

### Task 4: A3 surplus counterfactual 추정기 (실측 폐기)

진짜 잔량 S=closing+waste에 마감판매가 얼마나 반응하나. 기울기 높음=공급주도 떨이(α 낮음), 포화=수요주도 base(α 높음). **A3는 방향/상한 참조**(깔끔한 점 α 아님) — 정직하게 그 역할로. 정상판매를 수요수준 통제.

**Files:**
- Modify: `src/bakery/analysis/closing_demand.py`
- Test: `tests/test_closing_demand.py`

**Interfaces:**
- Consumes: `build_closing_panel` 출력 (`closing_qty, surplus, normal_qty, dow, trend`).
- Produces: `@dataclass SurplusResult(n:int, slope:float, se:float|None, clearance_high:float, note:str)`, `fit_surplus_counterfactual(panel) -> SurplusResult`. `slope=d(closing)/d(surplus)`, `clearance_high = 고surplus 4분위에서 closing/surplus 평균`.

- [ ] **Step 1: 실패 테스트 — known-answer(기울기 회복)**

```python
from bakery.analysis.closing_demand import fit_surplus_counterfactual

def _surplus_panel(slope, n=200):
    rng = np.arange(n)
    surplus = 10.0 + rng % 40          # varies 10..49
    closing = slope * surplus          # supply-driven if slope~1
    return pd.DataFrame({
        "category_id": ["bread"]*n,
        "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng, "D"),
        "closing_qty": closing, "surplus": surplus,
        "normal_qty": 100.0, "dow": rng % 7, "trend": rng,
    })

def test_surplus_slope_supply_driven():
    res = fit_surplus_counterfactual(_surplus_panel(slope=0.9))
    assert res.slope == pytest.approx(0.9, abs=0.05)   # tracks surplus → supply-driven
    assert res.clearance_high == pytest.approx(0.9, abs=0.05)

def test_surplus_slope_saturated():
    # closing fixed regardless of surplus → demand-driven base
    p = _surplus_panel(slope=0.9); p["closing_qty"] = 8.0
    res = fit_surplus_counterfactual(p)
    assert res.slope == pytest.approx(0.0, abs=0.05)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closing_demand.py -k surplus -q`
Expected: FAIL (`fit_surplus_counterfactual` 없음)

- [ ] **Step 3: 구현**

```python
@dataclass(frozen=True)
class SurplusResult:
    n: int
    slope: float
    se: float | None
    clearance_high: float
    note: str

def fit_surplus_counterfactual(panel):
    p = panel[panel["surplus"] > 0].copy()
    if len(p) < 20:
        return SurplusResult(len(p), float("nan"), None, float("nan"), "insufficient rows")
    y = p["closing_qty"].to_numpy(float)
    dow = pd.get_dummies(p["dow"], prefix="dow", drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(p)), p["surplus"], p["normal_qty"], p["trend"], dow])
    out = _ols_hc3(y, X, treat_idx=1)
    slope, se = (out if out is not None else (float("nan"), None))
    q75 = p["surplus"].quantile(0.75)
    high = p[p["surplus"] >= q75]
    clearance = float((high["closing_qty"] / high["surplus"]).mean()) if len(high) else float("nan")
    note = "supply-driven (low α)" if slope > 0.5 else "demand-limited (higher α)"
    return SurplusResult(len(p), float(slope), se, clearance, note)
```

- [ ] **Step 4: 통과 확인 + 커밋**

Run: `uv run pytest tests/test_closing_demand.py -k surplus -q`
Expected: PASS
```bash
git add src/bakery/analysis/closing_demand.py tests/test_closing_demand.py
git commit -m "feat: A3 surplus counterfactual 추정기 (실측 폐기 기반)"
```

---

### Task 5: A1 kink-in-time 추정기 (RD at closing onset)

하루 판매율 q(t). 마감 onset t0 직전 추세 외삽 = counterfactual 제값율. 마감창 관측−counterfactual = 유도 I. α_A1 = B/C. 저녁 commute 상승으로 **하한**.

**Files:**
- Modify: `src/bakery/analysis/closing_demand.py`
- Test: `tests/test_closing_demand.py`

**Interfaces:**
- Consumes: `DiscountSales.rows` (date,hour,minute,qty,label,item_id), `item_to_category`.
- Produces: `build_intraday_curve(rows, item_to_category, category, bin_min=15) -> pd.DataFrame`(category-date별 시간bin qty, onset 표시); `@dataclass KinkResult(n_days:int, base:float, closing_total:float, alpha:float, note:str)`; `fit_kink(curve) -> KinkResult`.

- [ ] **Step 1: 실패 테스트 — known-answer(유도분 분리)**

```python
from bakery.analysis.closing_demand import build_intraday_curve, fit_kink

def _intraday_rows(days=30):
    # pre-onset(17-19h) flat rate 2/bin; closing window(20-21h) observed 5/bin.
    # counterfactual base in closing window = 2/bin. induced=3/bin. α = base/closing = 2/5=0.4
    recs = []
    for d in range(days):
        date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=d)
        for h in [17, 18, 19]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 2,
                         "label": "none", "item_id": "A"})
        for h in [20, 21]:
            recs.append({"date": date, "hour": h, "minute": 0, "qty": 5,
                         "label": "closing", "item_id": "A"})
    return pd.DataFrame(recs)

def test_kink_recovers_alpha():
    rows = _intraday_rows()
    curve = build_intraday_curve(rows, pd.Series({"A": "bread"}), "bread", bin_min=60)
    res = fit_kink(curve)
    assert res.alpha == pytest.approx(0.4, abs=0.05)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closing_demand.py -k kink -q`
Expected: FAIL

- [ ] **Step 3: 구현 (onset=첫 closing bin, pre-onset 평균율 외삽)**

```python
PRE_ONSET_START_HOUR = 17

@dataclass(frozen=True)
class KinkResult:
    n_days: int
    base: float
    closing_total: float
    alpha: float
    note: str

def build_intraday_curve(rows, item_to_category, category, bin_min=15):
    df = rows.copy()
    df["category_id"] = df["item_id"].map(item_to_category)
    df = df[df["category_id"] == category].copy()
    df["tod"] = df["hour"] + df["minute"] / 60.0
    df["bin"] = (df["tod"] // (bin_min / 60.0)).astype(int)
    df["is_closing"] = df["label"] == "closing"
    g = df.groupby(["date", "bin"], observed=True).agg(
        qty=("qty", "sum"), closing=("is_closing", "max"),
        hour=("hour", "min")).reset_index()
    return g

def fit_kink(curve):
    days = curve["date"].nunique()
    if days == 0:
        return KinkResult(0, float("nan"), float("nan"), float("nan"), "no data")
    pre = curve[(~curve["closing"].astype(bool)) & (curve["hour"] >= PRE_ONSET_START_HOUR)]
    win = curve[curve["closing"].astype(bool)]
    if len(pre) == 0 or len(win) == 0:
        return KinkResult(days, float("nan"), float("nan"), float("nan"), "no pre/closing bins")
    pre_rate = pre["qty"].mean()                 # counterfactual per-bin rate
    bins_per_day = win.groupby("date").size().mean()
    base = pre_rate * bins_per_day * days        # counterfactual base over closing window
    closing_total = win["qty"].sum()
    alpha = float(np.clip(base / closing_total, 0.0, 1.0)) if closing_total > 0 else float("nan")
    return KinkResult(days, float(base), float(closing_total), alpha,
                      "lower-bound (evening commute uplift)")
```

- [ ] **Step 4: 통과 확인 + 커밋**

Run: `uv run pytest tests/test_closing_demand.py -k kink -q`
Expected: PASS
```bash
git add src/bakery/analysis/closing_demand.py tests/test_closing_demand.py
git commit -m "feat: A1 kink-in-time 추정기 (마감 onset RD)"
```

---

### Task 6: α_A 집계 (3각 삼각측량 → 구간)

세 추정을 하나의 α_A 구간으로. A1/A2=하한, A3=상한 참조. 정직한 구간 + 어느 방법이 어느 쪽 bound인지.

**Files:**
- Modify: `src/bakery/analysis/closing_demand.py`
- Test: `tests/test_closing_demand.py`

**Interfaces:**
- Consumes: `KinkResult`, `DepthResult`, `SurplusResult`.
- Produces: `@dataclass AlphaEstimate(alpha_low:float, alpha_high:float, a1:float, a2:float, a3_slope:float, note:str)`, `aggregate_alpha(kink, depth, surplus) -> AlphaEstimate`.

- [ ] **Step 1: 실패 테스트**

```python
from bakery.analysis.closing_demand import (
    aggregate_alpha, KinkResult, DepthResult, SurplusResult)

def test_aggregate_alpha_interval():
    kink = KinkResult(30, 2.0, 5.0, 0.40, "")
    depth = DepthResult(200, 50.0, 2.0, 10.0, 0.45, "")
    surplus = SurplusResult(200, 0.9, 0.05, 0.9, "supply-driven (low α)")
    est = aggregate_alpha(kink, depth, surplus)
    # lower bound = max of the two lower-bound methods (A1, A2)
    assert est.alpha_low == pytest.approx(0.45, abs=1e-6)
    assert est.a1 == pytest.approx(0.40, abs=1e-6)
    assert est.a2 == pytest.approx(0.45, abs=1e-6)
    assert 0.0 <= est.alpha_low <= est.alpha_high <= 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closing_demand.py -k aggregate -q`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
@dataclass(frozen=True)
class AlphaEstimate:
    alpha_low: float
    alpha_high: float
    a1: float
    a2: float
    a3_slope: float
    note: str

def aggregate_alpha(kink, depth, surplus):
    lowers = [v for v in (kink.alpha, depth.alpha) if v == v]   # drop NaN
    alpha_low = max(lowers) if lowers else float("nan")
    # A3 supply-driven(높은 slope)면 상한을 낮게 당김; demand-limited면 1.0까지 허용
    alpha_high = 1.0
    if surplus.slope == surplus.slope and surplus.slope > 0.5:
        alpha_high = float(np.clip(1.0 - (surplus.clearance_high or 0.0) + alpha_low, alpha_low, 1.0))
    note = f"lower=max(A1,A2) bounds; A3 {surplus.note}"
    return AlphaEstimate(alpha_low, alpha_high, kink.alpha, depth.alpha, surplus.slope, note)
```

- [ ] **Step 4: 통과 확인 + 커밋**

Run: `uv run pytest tests/test_closing_demand.py -q`
Expected: PASS (전체)
```bash
git add src/bakery/analysis/closing_demand.py tests/test_closing_demand.py
git commit -m "feat: α_A 3각 집계 (A1/A2 하한 · A3 상한 참조 구간)"
```

---

### Task 7: CLI `closing-demand` + 리포트 CSV

실데이터(광교)로 전 파이프라인 실행 → α_A 구간 + 방법별 결과 CSV.

**Files:**
- Modify: `src/bakery/analysis/closing_demand.py` (오케스트레이터 `run_closing_demand`)
- Modify: `src/bakery/cli.py` (`closing-demand` 명령)
- Test: `tests/test_closing_demand.py` (오케스트레이터 smoke — 합성 입력)

**Interfaces:**
- Consumes: 위 모든 함수 + `load_sales_with_discount`, 폐기 로더, bonavi category map.
- Produces: `run_closing_demand(rows, waste, item_to_category, category="bread") -> dict` (keys: `alpha`(AlphaEstimate), `depth`, `surplus`, `kink`, `panel`). CLI가 `reports/closing_alpha_estimates.csv` 등 기록.

- [ ] **Step 1: 실패 테스트 — 오케스트레이터 smoke (합성)**

```python
from bakery.analysis.closing_demand import run_closing_demand

def test_run_closing_demand_smoke():
    rows = _intraday_rows(days=40)             # Task5 헬퍼 재사용 (모듈 상단으로)
    waste = pd.DataFrame({"date": rows["date"].unique(),
                          "item_id": "A", "waste_qty": 3})
    out = run_closing_demand(rows, waste, pd.Series({"A": "bread"}), category="bread")
    assert set(out) == {"alpha", "depth", "surplus", "kink", "panel"}
    assert 0.0 <= out["alpha"].alpha_low <= 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closing_demand.py -k run_closing -q`
Expected: FAIL

- [ ] **Step 3: 오케스트레이터 구현**

```python
def run_closing_demand(rows, waste, item_to_category, category="bread"):
    panel = build_closing_panel(rows, waste, item_to_category)
    cat_panel = panel[panel["category_id"] == category]
    depth = fit_depth_elasticity(cat_panel)
    surplus = fit_surplus_counterfactual(cat_panel)
    curve = build_intraday_curve(rows, item_to_category, category)
    kink = fit_kink(curve)
    alpha = aggregate_alpha(kink, depth, surplus)
    return {"alpha": alpha, "depth": depth, "surplus": surplus,
            "kink": kink, "panel": cat_panel}
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_closing_demand.py -q`
Expected: PASS (전체)

- [ ] **Step 5: CLI 명령 추가**

`cli.py`에 기존 명령 패턴을 따라 `closing-demand` 추가: `load_sales_with_discount()` → 폐기 로더 → bonavi category map → `run_closing_demand` → `reports/closing_alpha_estimates.csv`(a1,a2,alpha_low,alpha_high,a3_slope,note), `closing_panel.csv` 기록 + α 구간 print. (실제 폐기 로더/카테고리 map 함수명은 discount.py/bonavi_loader.py 현행 확인 후 사용.)

- [ ] **Step 6: 실데이터 스모크 (수동)**

Run: `uv run bakery closing-demand`
Expected: α_A 구간 출력 + `reports/closing_alpha_estimates.csv` 생성. depth 시간-confound 플래그 로그 확인.

- [ ] **Step 7: 커밋**

```bash
git add src/bakery/analysis/closing_demand.py src/bakery/cli.py tests/test_closing_demand.py
git commit -m "feat: closing-demand CLI + α_A 리포트 (Phase A 완결)"
```

---

### Task 8: Phase B counterfactual formulation 스파이크 (결정 문서)

Phase B 본구현 전, B-1(salvage-newsvendor) vs B-2(시뮬레이션+censoring)를 **실데이터로 판정**한다. 코드 산출 아님 — 결정 문서 + 소규모 탐색.

**Files:**
- Create: `docs/superpowers/spikes/2026-07-04-phaseB-counterfactual-formulation.md`

**탐색 항목 (문서에 결과 기록):**
- [ ] **Step 1: 순환 위험 확인** — α가 타깃과 비용 yardstick 양쪽에 들어가 상쇄되는지 실데이터 소표본으로 재현. B-1이 이를 어떻게 끊는지(salvage 채널로 α를 "salvage vs 진짜수요"에만 진입) 서술.
- [ ] **Step 2: B-1 성립조건 점검** — newsvendor critical ratio에 salvage(마감회수) 반영 가능한지. 필요 입력(정가, 할인율, c 격자, 관측 폐기·마감·매진)이 다 있는지 데이터로 확인.
- [ ] **Step 3: B-2 가정 비용 평가** — censoring 위 수요분포 추정에 필요한 가정 목록화. 광교 무품절일 0 문제(W0 기록)와의 상호작용.
- [ ] **Step 4: 결정** — B-1 primary 확정 or B-2 fallback 조건. c 역추정(α_A ∩ α\*(c)) 구현 방식 스케치. 고객사 c 제공 시 슬롯.
- [ ] **Step 5: 커밋 + 후속 플랜 예약**

```bash
git add docs/superpowers/spikes/2026-07-04-phaseB-counterfactual-formulation.md
git commit -m "spike: Phase B counterfactual formulation 결정 (B-1 salvage-newsvendor)"
```
→ 결정 후 `docs/superpowers/plans/`에 Phase B 구현 플랜 별도 작성.

---

## Self-Review

**Spec coverage:**
- Phase A A1/A2/A3 → Task 5/3/4 ✓
- α_A 구간 삼각측량 → Task 6 ✓
- cost-free 성질(원가율 불필요) → Phase A 전체가 c 미참조 ✓
- Phase B α\*(c) + 원가율 역추정 → Task 8 스파이크에서 formulation 확정 후 후속 플랜(설계상 명시적 분리) ✓
- 교차검증(α_A ∩ α\*(c)) → Task 8 Step 4에 스케치, 후속 플랜에서 구현 ✓
- 데이터 항등식(normal+closing, closing20+30, closing+waste=surplus) → Task 2 테스트 ✓
- degenerate 처리(마감0·단일depth) → Task 3 insufficient-variation 가드 ✓
- 범위(광교 bread/pastry, sandwich/cake 제외) → Global Constraints + category 파라미터 ✓
- 절대규칙(leakage/censored/시간순/WAPE) → Global Constraints, Phase B에서 강제 ✓

**Placeholder scan:** 코드 스텝은 실제 코드 포함. CLI Task7 Step5는 "현행 함수명 확인 후"라는 조건부가 있으나 이는 실데이터 로더 시그니처가 코드에 존재(discount.py/bonavi_loader.py)하므로 실행 시 grep 1회로 해소 — 가짜 아님.

**Type consistency:** `_ols_hc3(y,X,treat_idx)->tuple|None` 전 태스크 일관. dataclass 필드명(KinkResult.alpha, DepthResult.alpha, SurplusResult.slope/clearance_high) Task6 집계에서 동일 참조 ✓.

**주의(정직):** A1/A2/A3의 α 매핑(base/mean, clip 등)은 known-answer 테스트로 mechanics는 검증되나 실데이터 해석은 Phase A 실행 후 민감도로 재점검 필요. A3는 점 α 아닌 방향/상한 참조로 명시.

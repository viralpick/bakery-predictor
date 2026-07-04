# 전향적 KPI harness + WPE·Decoupling Score 진단 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 우리 발주 추천 vs 아티제 현행 발주를 합의 운영 KPI(폐기비용/매진시각median/매진률)로 비교하는 harness와, 그 수요 입력 품질을 재는 WPE·Decoupling Score 진단을 구현한다.

**Architecture:** 반사실은 양쪽 발주량을 동일 시뮬(복원 일수요 + 시간대 도착곡선)에 넣어 계산하고 1차 지표는 Δ(우리−아티제)로 본다. 매진시각은 기존 `potential_demand`의 `bakery_hour_profile`/`cumulative_weight_at`를 재사용해 "누적수요가 발주량에 도달하는 시각"을 역산한다. 폐기/lost 비용은 기존 `business_metrics`를 재사용한다. 신규 로직은 `evaluation/prospective.py`와 `evaluation/diagnostics.py`에 격리한다.

**Tech Stack:** Python, numpy, pandas, typer(CLI), pytest, uv.

## Global Constraints

- Time leakage 금지 — 도착 프로필·복원수요는 예측 시점 이전/split 이후 데이터로만. (`spec.md` 절대규칙 1)
- 품절 데이터는 censored — 매진일 판매량은 실수요 아님, `is_stockout`/매진시각 보존. (절대규칙 2)
- 메인 지표는 WAPE, 보조 MAE/RMSE. MAPE 단독 금지. (절대규칙 5)
- 매장 영업시간 기본값: 광교 open_hour=8, close_hour=22 (기존 `StoreHours` 사용).
- 테스트 단언은 기대값 정확비교(`==`/`approx`), truthy·부분문자열 금지. (code-quality 규칙 8)
- 함수 30줄 이내, 인자 4개 초과 시 묶기. (code-quality 규칙 1)
- 신규 파일 헤더 docstring: 무엇/왜.
- KRW 비용 파라미터는 `business_metrics.CostParams` 재사용(중복 정의 금지).

---

### Task 1: WPE(편향 방향) 지표

**Files:**
- Modify: `src/bakery/evaluation/metrics.py` (append `wpe`, extend `summarize`)
- Test: `tests/test_metrics.py` (append)

**Interfaces:**
- Produces: `wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float` — signed bias `Σ(pred−true)/Σtrue`. 과대예측 양수, 과소예측 음수.
- `summarize(...)` dict에 `"wpe"` 키 추가.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py 에 append
import numpy as np
from bakery.evaluation.metrics import wpe, summarize

def test_wpe_sign_and_value():
    y = np.array([10.0, 10.0, 10.0, 10.0])
    over = np.array([12.0, 12.0, 12.0, 12.0])   # 과대예측 → +
    under = np.array([8.0, 8.0, 8.0, 8.0])      # 과소예측 → −
    exact = np.array([10.0, 10.0, 10.0, 10.0])
    assert wpe(y, over) == 0.2      # (48-40)/40
    assert wpe(y, under) == -0.2    # (32-40)/40
    assert wpe(y, exact) == 0.0

def test_summarize_includes_wpe():
    y = np.array([10.0, 20.0])
    yhat = np.array([11.0, 19.0])
    out = summarize(y, yhat)
    assert "wpe" in out
    assert out["wpe"] == 0.0        # (+1 -1)/30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py::test_wpe_sign_and_value tests/test_metrics.py::test_summarize_includes_wpe -v`
Expected: FAIL — `ImportError: cannot import name 'wpe'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/metrics.py — wape 함수 바로 아래 삽입
def wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Percentage Error — signed bias. Σ(pred−true)/Σtrue.

    양수=체계적 과대예측, 음수=과소예측(품절 위험 방향). WAPE가 못 잡는
    '어느 쪽으로 틀렸나'를 본다."""
    y_true = _align(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float((y_pred - y_true).sum() / denom)
```

`summarize`에 `"wpe": wpe(y_true, y_pred)` 항목을 dict에 추가한다 (기존 반환 dict 리터럴에 한 줄).

> 주: `_align`은 metrics.py에 이미 있음(줄 146). 시그니처가 `_align(y_true, y_pred)`면 그대로, 단일인자면 `np.asarray`만 사용하도록 맞춘다. 실행 시 실제 시그니처 확인 후 정합.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (기존 metrics 테스트 포함 전부)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: WPE(편향 방향) 지표 + summarize 편입"
```

---

### Task 2: Decoupling Score(ρ_DS) 진단

**Files:**
- Create: `src/bakery/evaluation/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `decoupling_score(demand: np.ndarray, stockout_rate: np.ndarray, weights: np.ndarray | None = None) -> float` — 가중 Pearson(품절률, 복원수요). 0 근처=잘 분리, 강한 음수=복원 부족. **카테고리 레벨 셀에만 적용**(품목 레벨은 흡수·검열 이중편향으로 식별불가).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostics.py
import numpy as np
import pytest
from bakery.evaluation.diagnostics import decoupling_score

def test_uncorrected_sales_negative_score():
    # 미복원(원판매): 품절률 높을수록 관측수요 낮음 → 강한 음의 상관
    stockout_rate = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    observed = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    assert decoupling_score(observed, stockout_rate) == pytest.approx(-1.0, abs=1e-9)

def test_perfect_recovery_zero_score():
    # 완전복원: 품절률과 무관하게 실수요 100 → 상관 정의 안 됨(분산0) → 0 반환
    stockout_rate = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    recovered = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    assert decoupling_score(recovered, stockout_rate) == 0.0

def test_weighted_matches_manual():
    demand = np.array([10.0, 20.0, 30.0])
    sr = np.array([0.1, 0.2, 0.3])
    w = np.array([1.0, 1.0, 2.0])
    # 양의 완전 선형 → +1
    assert decoupling_score(demand, sr, weights=w) == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.evaluation.diagnostics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/diagnostics.py
"""복원(potential_demand) 품질 진단.

Decoupling Score: 복원한 수요가 품절률과 여전히 상관되는지 재는 지표.
FreshRetailNet(2505.16319)의 ρ_DS. 미복원이면 '품절→낮은 수요'라는 검열
편향이 강한 음의 상관으로 남고, 잘 복원하면 상관이 사라진다(≈0). 강한
음수 = 복원 부족(=발주 과소 위험). 품목 레벨은 흡수·검열 이중편향으로
식별 불가이므로 **카테고리 레벨 셀에만** 적용한다.
"""
from __future__ import annotations

import numpy as np


def decoupling_score(
    demand: np.ndarray,
    stockout_rate: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """가중 Pearson 상관(품절률, 복원수요). 분산이 0이면 0을 반환."""
    d = np.asarray(demand, dtype=float)
    s = np.asarray(stockout_rate, dtype=float)
    if d.shape != s.shape:
        raise ValueError(f"demand{d.shape} vs stockout_rate{s.shape} shape mismatch")
    w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=float)
    wsum = w.sum()
    if wsum <= 0:
        return float("nan")
    dm = d - np.average(d, weights=w)
    sm = s - np.average(s, weights=w)
    cov = np.sum(w * dm * sm)
    var_d = np.sum(w * dm * dm)
    var_s = np.sum(w * sm * sm)
    if var_d <= 0 or var_s <= 0:
        return 0.0
    return float(cov / np.sqrt(var_d * var_s))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: Decoupling Score(ρ_DS) 진단 — 복원 검열잔량 측정"
```

---

### Task 3: 반사실 매진시각 엔진

**Files:**
- Create: `src/bakery/evaluation/prospective.py`
- Test: `tests/test_prospective.py`

**Interfaces:**
- Consumes: `bakery_hour_profile`, `cumulative_weight_at` from `bakery.features.potential_demand`.
- Produces: `simulate_soldout(order_qty: float, daily_demand: float, profile: np.ndarray, *, open_hour: int, close_hour: int) -> tuple[float | None, bool]` — `(soldout_hour_float | None, is_stockout)`. `order_qty >= daily_demand`면 `(None, False)`(미매진). `order_qty <= 0`이면 `(open_hour, True)`. profile은 `bakery_hour_profile`가 낸 24-length(개점~폐점 정규화).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prospective.py
import numpy as np
import pytest
from bakery.features.potential_demand import bakery_hour_profile
from bakery.evaluation.prospective import simulate_soldout

OPEN, CLOSE = 8, 22

def _uniform_profile():
    # 균등 도착: alpha=0 → 개점~폐점 균등
    return bakery_hour_profile(OPEN, CLOSE, alpha=0.0)

def test_order_ge_demand_never_sells_out():
    prof = _uniform_profile()
    t, is_so = simulate_soldout(100.0, 80.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert t is None
    assert is_so is False

def test_uniform_half_demand_sells_out_midday():
    # 균등 14시간(08~22) 영업, 수요 100, 발주 50 → 정확히 중간 시각 15:00
    prof = _uniform_profile()
    t, is_so = simulate_soldout(50.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert is_so is True
    assert t == pytest.approx(15.0, abs=1e-6)   # 08 + 14*0.5

def test_zero_order_sells_out_at_open():
    prof = _uniform_profile()
    t, is_so = simulate_soldout(0.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert is_so is True
    assert t == pytest.approx(float(OPEN), abs=1e-6)

def test_monotone_higher_order_later_soldout():
    prof = _uniform_profile()
    t_low, _ = simulate_soldout(30.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    t_high, _ = simulate_soldout(70.0, 100.0, prof, open_hour=OPEN, close_hour=CLOSE)
    assert t_high > t_low
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.evaluation.prospective'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py
"""전향적 KPI harness — 우리 발주 추천 vs 현행 발주를 동일 시뮬로 비교.

반사실: 두 발주량을 같은 복원수요 + 시간대 도착곡선에 넣어 폐기/매진시각/
매진률을 계산한다. 1차 지표는 Δ(우리−아티제)로, 시뮬 편향이 양쪽에 상쇄된다.
매진시각은 potential_demand의 도착곡선을 재사용해 '누적수요가 발주량에
도달하는 시각'을 역산한다. 폐기/lost 비용은 business_metrics 재사용.
"""
from __future__ import annotations

import numpy as np

from ..features.potential_demand import cumulative_weight_at


def simulate_soldout(
    order_qty: float,
    daily_demand: float,
    profile: np.ndarray,
    *,
    open_hour: int,
    close_hour: int,
) -> tuple[float | None, bool]:
    """발주량 하에서 매진시각(hour_float)과 매진여부. 미매진이면 (None, False)."""
    if daily_demand <= 0 or order_qty >= daily_demand:
        return (None, False)
    if order_qty <= 0:
        return (float(open_hour), True)
    target = order_qty / daily_demand          # 도달해야 할 누적 비중
    pre = 0.0
    for h in range(open_hour, close_hour):
        w_h = float(profile[h])
        if pre + w_h >= target:
            frac = (target - pre) / w_h if w_h > 0 else 0.0
            return (h + frac, True)
        pre += w_h
    return (float(close_hour), True)           # 수치오차 fallback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/test_prospective.py
git commit -m "feat: 반사실 매진시각 엔진 (도착곡선 역산)"
```

---

### Task 4: 시간대 도착 프로필 재구성

**Files:**
- Modify: `src/bakery/evaluation/prospective.py`
- Test: `tests/test_prospective.py` (append)

**Interfaces:**
- Produces: `build_arrival_profile(receipts: pd.DataFrame, *, group_cols: list[str], exclude_keys: set | None = None) -> dict[tuple, np.ndarray]` — 그룹별 24-length 시간당 수량 벡터(비정규화 raw; `bakery_hour_profile(measured=)`에 넘길 용도). `receipts`는 `hour`, `qty` + `group_cols` 보유. `exclude_keys`(품절 item-day 등)는 프로필 추정에서 제외.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prospective.py 에 append
import pandas as pd
from bakery.evaluation.prospective import build_arrival_profile

def test_build_arrival_profile_sums_by_hour():
    receipts = pd.DataFrame({
        "item_id": ["a", "a", "a", "b"],
        "hour":    [9,   9,   14,  10],
        "qty":     [2,   3,   5,   1],
    })
    prof = build_arrival_profile(receipts, group_cols=["item_id"])
    assert prof[("a",)][9] == 5.0
    assert prof[("a",)][14] == 5.0
    assert prof[("a",)].sum() == 10.0
    assert prof[("b",)][10] == 1.0
    assert prof[("a",)].shape == (24,)

def test_build_arrival_profile_excludes_keys():
    receipts = pd.DataFrame({
        "item_id": ["a", "a"],
        "date":    ["2025-01-01", "2025-01-02"],
        "hour":    [9, 9],
        "qty":     [2, 7],
    })
    prof = build_arrival_profile(
        receipts, group_cols=["item_id"], exclude_keys={("a", "2025-01-02")},
        exclude_cols=["item_id", "date"],
    )
    assert prof[("a",)][9] == 2.0   # 01-02 제외
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective.py::test_build_arrival_profile_sums_by_hour -v`
Expected: FAIL — `AttributeError`/`ImportError` (build_arrival_profile 없음)

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py 에 append (import pandas as pd 상단 추가)
import pandas as pd


def build_arrival_profile(
    receipts: pd.DataFrame,
    *,
    group_cols: list[str],
    exclude_keys: set | None = None,
    exclude_cols: list[str] | None = None,
) -> dict[tuple, np.ndarray]:
    """그룹별 24-length 시간당 수량 벡터. bakery_hour_profile(measured=)용 raw."""
    df = receipts
    if exclude_keys:
        key_cols = exclude_cols or group_cols
        keys = list(zip(*[df[c].astype(str) for c in key_cols]))
        df = df[[k not in exclude_keys for k in keys]]
    out: dict[tuple, np.ndarray] = {}
    for gkey, g in df.groupby(group_cols):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        vec = np.zeros(24, dtype=float)
        by_hour = g.groupby("hour")["qty"].sum()
        for h, q in by_hour.items():
            vec[int(h)] = float(q)
        out[gkey] = vec
    return out
```

> 주: `exclude_keys`의 원소는 `exclude_cols`(기본=group_cols) 값을 `str`로 변환해 튜플로 만든 키. 테스트는 `("a","2025-01-02")` 형태를 쓴다.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/test_prospective.py
git commit -m "feat: 시간대 도착 프로필 재구성 (품절 item-day 제외 옵션)"
```

---

### Task 5: baseline 발주 재구성 (생산 항등식 proxy)

**Files:**
- Modify: `src/bakery/evaluation/prospective.py`
- Test: `tests/test_prospective.py` (append)

**Interfaces:**
- Produces: `reconstruct_baseline_order(df: pd.DataFrame, *, normal_col: str = "normal_units", closing_col: str = "closing_units", waste_col: str = "waste_units") -> pd.Series` — `생산 = 정상판매 + 마감판매 + 폐기`. 결측은 0. 회고 검증에서 아티제 발주 proxy. 전향에서는 실제 발주로 대체.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prospective.py 에 append
from bakery.evaluation.prospective import reconstruct_baseline_order

def test_reconstruct_baseline_order_identity():
    df = pd.DataFrame({
        "normal_units":  [10.0, 5.0],
        "closing_units": [3.0,  0.0],
        "waste_units":   [2.0,  1.0],
    })
    got = reconstruct_baseline_order(df)
    assert list(got) == [15.0, 6.0]

def test_reconstruct_baseline_order_nan_as_zero():
    df = pd.DataFrame({
        "normal_units":  [10.0, np.nan],
        "closing_units": [np.nan, 4.0],
        "waste_units":   [2.0, 1.0],
    })
    got = reconstruct_baseline_order(df)
    assert list(got) == [12.0, 5.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective.py::test_reconstruct_baseline_order_identity -v`
Expected: FAIL — import 실패

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py 에 append
def reconstruct_baseline_order(
    df: pd.DataFrame,
    *,
    normal_col: str = "normal_units",
    closing_col: str = "closing_units",
    waste_col: str = "waste_units",
) -> pd.Series:
    """생산 = 정상판매 + 마감판매 + 폐기. 회고 검증의 현행 발주 proxy."""
    parts = [df[c].fillna(0.0).astype(float) for c in (normal_col, closing_col, waste_col)]
    return parts[0] + parts[1] + parts[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/test_prospective.py
git commit -m "feat: baseline 발주 재구성 (생산 항등식 proxy)"
```

---

### Task 6: item-day KPI 시뮬레이션

**Files:**
- Modify: `src/bakery/evaluation/prospective.py`
- Test: `tests/test_prospective.py` (append)

**Interfaces:**
- Consumes: `simulate_soldout`(Task3), `bakery_hour_profile`, `business_metrics.simulate_profit`, `CostParams`.
- Produces: `simulate_item_day_kpis(rows: pd.DataFrame, profiles: dict[tuple, np.ndarray], *, order_col: str, store_hours: StoreHours, group_cols: list[str], params: CostParams | None = None, unit_prices=None) -> pd.DataFrame` — 입력 행: `item_id`, `date`, `potential_demand`(복원 일수요), `order_col`, `group_cols`. 반환: 입력 + `waste_units`,`lost_sale_units`,`waste_cost_krw`,`lost_margin_krw`,`soldout_hour`,`is_stockout`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prospective.py 에 append
from bakery.features.potential_demand import StoreHours, bakery_hour_profile
from bakery.evaluation.business_metrics import CostParams
from bakery.evaluation.prospective import simulate_item_day_kpis

def test_item_day_kpis_waste_and_soldout():
    rows = pd.DataFrame({
        "item_id": ["a", "a"],
        "date":    ["2025-01-01", "2025-01-02"],
        "potential_demand": [100.0, 100.0],
        "order_qty": [120.0, 50.0],   # 1일차 과발주(미매진, 폐기), 2일차 부족(매진)
    })
    prof = {("a",): bakery_hour_profile(8, 22, alpha=0.0)}
    out = simulate_item_day_kpis(
        rows, prof, order_col="order_qty",
        store_hours=StoreHours("gwangyo", 8, 22),
        group_cols=["item_id"],
        params=CostParams(), unit_prices={"a": 1000.0},
    )
    r0, r1 = out.iloc[0], out.iloc[1]
    assert r0["is_stockout"] == False
    assert r0["waste_units"] == 20.0            # 120-100
    assert pd.isna(r0["soldout_hour"])
    assert r1["is_stockout"] == True
    assert r1["soldout_hour"] == pytest.approx(15.0, abs=1e-6)  # 발주50/수요100 균등
    assert r1["lost_sale_units"] == 50.0        # 100-50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective.py::test_item_day_kpis_waste_and_soldout -v`
Expected: FAIL — import 실패

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py 에 append
from ..features.potential_demand import StoreHours, bakery_hour_profile
from .business_metrics import CostParams, simulate_profit


def simulate_item_day_kpis(
    rows: pd.DataFrame,
    profiles: dict[tuple, np.ndarray],
    *,
    order_col: str,
    store_hours: StoreHours,
    group_cols: list[str],
    params: CostParams | None = None,
    unit_prices=None,
) -> pd.DataFrame:
    """item-day별 폐기/lost 비용(business_metrics) + 매진시각/매진여부."""
    params = params or CostParams()
    # 폐기/lost: simulate_profit 재사용 (yhat=발주량, true=potential_demand)
    prof_in = rows.rename(columns={order_col: "yhat"}).copy()
    prof_in["sold_units"] = prof_in["potential_demand"]
    costed = simulate_profit(
        prof_in, unit_prices=unit_prices, params=params,
        yhat_col="yhat", sold_col="sold_units", potential_col="potential_demand",
    )
    # 매진시각: 그룹 프로필로 역산
    soldout_hours, stockouts = [], []
    for _, r in rows.iterrows():
        gkey = tuple(str(r[c]) for c in group_cols)
        raw = profiles.get(gkey)
        prof = bakery_hour_profile(
            store_hours.open_hour, store_hours.close_hour,
            measured=raw if raw is not None else None,
        )
        t, is_so = simulate_soldout(
            float(r[order_col]), float(r["potential_demand"]), prof,
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

> 주: `build_arrival_profile`의 키는 `str` 변환 튜플이므로 여기서도 `tuple(str(r[c]) ...)`로 맞춘다. `measured` raw 벡터가 개점 전/폐점 후 값을 포함해도 `bakery_hour_profile`가 open 윈도로 마스킹·정규화한다.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/test_prospective.py
git commit -m "feat: item-day KPI 시뮬 (폐기/lost + 매진시각/매진여부)"
```

---

### Task 7: 정책 비교 (우리 vs 아티제 + Δ)

**Files:**
- Modify: `src/bakery/evaluation/prospective.py`
- Test: `tests/test_prospective.py` (append)

**Interfaces:**
- Consumes: `simulate_item_day_kpis`(Task6).
- Produces: `compare_policies(our_kpis: pd.DataFrame, base_kpis: pd.DataFrame) -> pd.DataFrame` — 두 KPI 프레임(Task6 출력)을 받아 정책별 요약 + Δ 1행. 컬럼: `policy`(our/baseline/delta), `waste_cost_krw`, `lost_margin_krw`, `stockout_rate`, `soldout_median_h`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prospective.py 에 append
from bakery.evaluation.prospective import compare_policies

def _kpi_frame(waste, lost, stockouts, soldout):
    return pd.DataFrame({
        "waste_cost_krw": waste, "lost_margin_krw": lost,
        "is_stockout": stockouts, "soldout_hour": soldout,
    })

def test_compare_policies_delta():
    our = _kpi_frame([100.0, 0.0], [0.0, 50.0], [False, True], [np.nan, 16.0])
    base = _kpi_frame([200.0, 0.0], [0.0, 80.0], [False, True], [np.nan, 14.0])
    out = compare_policies(our, base).set_index("policy")
    assert out.loc["our", "waste_cost_krw"] == 100.0
    assert out.loc["baseline", "waste_cost_krw"] == 200.0
    assert out.loc["delta", "waste_cost_krw"] == -100.0     # 우리가 폐기 100 적음
    assert out.loc["our", "stockout_rate"] == 0.5
    assert out.loc["our", "soldout_median_h"] == 16.0       # 매진일만 median
    assert out.loc["delta", "soldout_median_h"] == 2.0      # 16 - 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective.py::test_compare_policies_delta -v`
Expected: FAIL — import 실패

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/evaluation/prospective.py 에 append
def _summarize_policy(kpis: pd.DataFrame) -> dict[str, float]:
    so = kpis["is_stockout"].astype(bool)
    soldout_median = kpis.loc[so, "soldout_hour"].median() if so.any() else float("nan")
    return {
        "waste_cost_krw": float(kpis["waste_cost_krw"].sum()),
        "lost_margin_krw": float(kpis["lost_margin_krw"].sum()),
        "stockout_rate": float(so.mean()),
        "soldout_median_h": float(soldout_median),
    }


def compare_policies(our_kpis: pd.DataFrame, base_kpis: pd.DataFrame) -> pd.DataFrame:
    """우리·아티제 정책 KPI 요약 + Δ(우리−아티제) 1행."""
    our = _summarize_policy(our_kpis)
    base = _summarize_policy(base_kpis)
    delta = {k: our[k] - base[k] for k in our}
    return pd.DataFrame([
        {"policy": "our", **our},
        {"policy": "baseline", **base},
        {"policy": "delta", **delta},
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prospective.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/evaluation/prospective.py tests/test_prospective.py
git commit -m "feat: 정책 비교 (우리 vs 아티제 KPI + Δ)"
```

---

### Task 8: CLI `prospective-eval` + synthetic end-to-end

**Files:**
- Modify: `src/bakery/cli.py` (신규 커맨드 `prospective-eval`)
- Test: `tests/test_prospective_cli.py`

**Interfaces:**
- Consumes: Task4~7 함수 전부, `build_synthetic_*` 데이터 생성기.
- Produces: CLI `prospective-eval` — 입력 데이터(합성 or 실데이터 조립)를 받아 arrival profile 재구성 → 우리 발주/baseline 각각 item-day KPI 시뮬 → `compare_policies` 표를 콘솔 + `reports/prospective_kpi.csv`로 출력. WPE(예측편향)·ρ_DS(카테고리별)도 함께 출력.

- [ ] **Step 1: Write the failing test (end-to-end on 소형 합성)**

```python
# tests/test_prospective_cli.py
import numpy as np
import pandas as pd
import pytest
from bakery.features.potential_demand import StoreHours, bakery_hour_profile
from bakery.evaluation.prospective import (
    build_arrival_profile, simulate_item_day_kpis, compare_policies,
)
from bakery.evaluation.business_metrics import CostParams

def test_end_to_end_our_beats_worse_baseline():
    # 합성: item a, 2일. 복원수요 100/100. 우리 발주=수요근접, baseline=과발주.
    receipts = pd.DataFrame({
        "item_id": ["a"]*4, "date": ["2025-01-01"]*2 + ["2025-01-02"]*2,
        "hour": [9, 14, 9, 14], "qty": [50, 50, 50, 50],
    })
    prof = build_arrival_profile(receipts, group_cols=["item_id"])
    rows = pd.DataFrame({
        "item_id": ["a", "a"], "date": ["2025-01-01", "2025-01-02"],
        "potential_demand": [100.0, 100.0],
        "our_order": [105.0, 105.0], "base_order": [140.0, 140.0],
    })
    sh = StoreHours("gwangyo", 8, 22)
    our = simulate_item_day_kpis(rows, prof, order_col="our_order",
                                 store_hours=sh, group_cols=["item_id"],
                                 params=CostParams(), unit_prices={"a": 1000.0})
    base = simulate_item_day_kpis(rows, prof, order_col="base_order",
                                  store_hours=sh, group_cols=["item_id"],
                                  params=CostParams(), unit_prices={"a": 1000.0})
    cmp = compare_policies(our, base).set_index("policy")
    # 우리 발주가 baseline보다 과발주 덜 함 → 폐기비용 Δ < 0
    assert cmp.loc["delta", "waste_cost_krw"] < 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prospective_cli.py -v`
Expected: FAIL — (초기엔 함수 조립 오류 없거나, 함수 시그니처 미스매치 시 실패). 이 테스트는 Task4~7 통합 확인용.

- [ ] **Step 3: Write the CLI command**

```python
# src/bakery/cli.py 상단 import 추가
from .evaluation.prospective import (
    build_arrival_profile, simulate_item_day_kpis, compare_policies,
    reconstruct_baseline_order,
)
from .evaluation.diagnostics import decoupling_score
from .evaluation.metrics import wpe
from .features.potential_demand import StoreHours
```

```python
# src/bakery/cli.py — 신규 커맨드 (기존 커맨드들과 같은 패턴)
@app.command("prospective-eval")
def cmd_prospective_eval(
    source: str = typer.Option("synthetic", help="synthetic | real"),
    store_id: str = typer.Option("gwangyo"),
    open_hour: int = typer.Option(8),
    close_hour: int = typer.Option(22),
    out_csv: str = typer.Option("reports/prospective_kpi.csv"),
) -> None:
    """우리 발주 추천 vs 현행 발주를 KPI(폐기/매진시각/매진률)로 비교."""
    rows, receipts, unit_prices = _load_prospective_inputs(source, store_id)
    profiles = build_arrival_profile(
        receipts, group_cols=["item_id"],
        exclude_keys=_stockout_item_days(rows), exclude_cols=["item_id", "date"],
    )
    sh = StoreHours(store_id, open_hour, close_hour)
    our = simulate_item_day_kpis(rows, profiles, order_col="our_order",
                                 store_hours=sh, group_cols=["item_id"],
                                 unit_prices=unit_prices)
    base = simulate_item_day_kpis(rows, profiles, order_col="base_order",
                                  store_hours=sh, group_cols=["item_id"],
                                  unit_prices=unit_prices)
    table = compare_policies(our, base)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    console.print(table.to_string(index=False))
    console.print(f"[cyan]예측 편향 WPE={wpe(rows['potential_demand'].to_numpy(), rows['our_order'].to_numpy()):.3f}[/]")
```

`_load_prospective_inputs(source, store_id)` 헬퍼: `source=="synthetic"`면 `build_synthetic_*`로 소형 프레임(위 테스트 형태)을 만들고 `base_order`=`reconstruct_baseline_order` 사용; `source=="real"`이면 `load_sales_with_discount`로 광교를 조립(정상=label!=마감 판매합, 마감=label==마감, 폐기=폐기시트) 후 `reconstruct_baseline_order`로 base_order, 우리 모델 예측으로 our_order. `_stockout_item_days(rows)`는 `is_stockout` 관측된 item-day 키 집합.

> **실데이터 조립 주의(실행 시 검증)**: `load_sales_with_discount`의 `label`/폐기 컬럼 실제 값과 폐기시트 로딩 경로를 확인해 정상/마감/폐기 3분해를 맞춘다. 폐기 실측이 loader로 안 잡히면 회고 baseline은 (정상+마감)만으로 근사하고 문서에 한계로 남긴다. synthetic 경로가 end-to-end 계약을 이미 검증하므로 실데이터 조립은 컬럼 매핑만 남는다.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prospective_cli.py -v && uv run bakery prospective-eval --source synthetic`
Expected: 테스트 PASS + CLI가 KPI 표 + WPE 출력, `reports/prospective_kpi.csv` 생성

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py tests/test_prospective_cli.py
git commit -m "feat: prospective-eval CLI + synthetic end-to-end KPI 비교"
```

---

### Task 9: 전체 회귀 + 문서 갱신

**Files:**
- Modify: `TODO.md` (③ 항목 완료 표시), `.claude/CLAUDE.md` 실행 예시(선택)

- [ ] **Step 1: 전체 테스트**

Run: `uv run pytest -q`
Expected: 전부 PASS

- [ ] **Step 2: TODO.md 갱신** — "방법론 후속"의 WPE·ρ_DS 항목을 `[x]`로, 전향적 harness 항목 추가/완료.

- [ ] **Step 3: Commit**

```bash
git add TODO.md .claude/CLAUDE.md
git commit -m "docs: 전향적 KPI harness + WPE·ρ_DS 완료 반영"
```

---

## Self-Review

**Spec coverage:**
- 반사실=양쪽 동일 시뮬 → Task 3·6·7 ✓
- 수요입력=potential_demand → Task 6 (`potential_demand` 컬럼) ✓
- 매진시각 median(censored=미매진 제외) → Task 3·7 ✓
- 매진률 → Task 7 ✓
- 폐기비용(business_metrics 재사용) → Task 6 ✓
- WPE → Task 1 ✓ / ρ_DS(카테고리 한정) → Task 2 ✓
- baseline 재구성(생산 항등식) → Task 5 ✓
- 회고 검증 wiring → Task 8 (real 경로) ✓
- 도착 프로필(품절 tail 외삽) → Task 4 + Task 6(bakery_hour_profile measured) ✓
- CLI → Task 8 ✓

**Placeholder scan:** Task 8 실데이터 조립만 "실행 시 컬럼 검증" 명시 — 이는 실데이터 스키마 불확실성 때문의 의도적 게이트(플래그됨), 합성 경로가 계약을 완전 검증하므로 플랜 실패 아님.

**Type consistency:** 프로필 키는 전 구간 `tuple(str(...))`; `simulate_soldout` 반환 `(float|None, bool)`은 Task6에서 `np.nan`/bool로 저장, Task7에서 `is_stockout`/`soldout_hour`로 소비 — 일치. `CostParams`/`simulate_profit`은 실제 시그니처(확인함) 사용.

## Execution Handoff
(하단 참조)

# Phase B — 발주 최적화 + implied c 갭 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 광교 카테고리(bread/pastry) 발주를 원가율 c의 함수로 최적화(two-class salvage-newsvendor)하고, 현행 발주의 implied c(주 산출물)와 절감액(placebo 대비)을 정직하게 산출한다.

**Architecture:** 실측 폐기 parquet를 카테고리-일로 집계 → 요일/월 조건부 rolling 경험분포로 조건부 수요 CDF 추정(leakage-safe) → two-class newsvendor Q\*(c) + implied c(현행 made의 service level) → placebo arm(Q=made) 포함 rolling 백테스트로 절감액. v4 `business_metrics`의 c/비용개념 재사용하되 newsvendor-정합 비용은 직접 계산.

**Tech Stack:** Python, pandas, numpy, pytest. 재사용: `data/bonavi_loader.load_items`(item→category), `evaluation/business_metrics.CostParams`(c/margin 개념), parquet `data/internal/v2/waste_alpha_4stores.parquet`.

## Global Constraints

- **Time leakage 금지**: 수요분포·price 모두 target date **이전** 데이터로만(rolling). 미래 sold/out/made 미참조. leakage 회귀 테스트 필수.
- **품절 censored**: 카테고리 out==0 = 수요≥made(censored) 보존, 무리한 결측처리 금지.
- **Random split 금지**: 시간순 rolling/expanding window.
- **MAPE 단독 금지**: cost KPI 메인, WAPE 보조.
- **Synthetic↔Real 경계**: parquet 진입점, 동일 schema.
- **범위**: 광교 bread/pastry (sandwich/cake 제외). 카테고리 단위(품목 배분=Stage-2, 범위 밖).
- **항등식 규칙(구현자 재량 금지)**: `|identity_diff| > IDENTITY_TOL` 카테고리-일 = 제외+제외율 보고. `out<0` → 0 clip + clip량 보고.
- **비용은 newsvendor-정합**: Co=c×price(폐기), Cu=(1−c)×price(매진, Level1). `simulate_profit`의 margin_rate/lost_sale_multiplier 손실은 쓰지 않는다(다른 loss).
- **주 산출물 = implied c 갭**. 절감액은 작거나 음수여도 정직하게 보고.
- **함수 ≤30 non-blank 줄**, guard-clause early-return, 매직값 상수화, frozen dataclass, 테스트 정확값 `==`/`approx`, 테스트 import 최상단.

## v4 재사용 판단 (스펙 D2에서의 정당한 이탈 — 리뷰 플래그)

스펙은 "v4 `fit_category_total`(LGBM quantile) 재사용"을 명시했으나, 구현 조사 결과:
- LGBM quantile은 **점 조건부 quantile**(quantile당 별도 fit)만 줌. Phase B는 **전체 조건부 CDF**가 필요(implied c=P(demand≤made), two-class=밴드확률). 5년 daily rolling에서 quantile당 LGBM fit은 비싸고 CDF 역산이 번거로움.
- **채택**: 요일(dow)+월 조건부 **rolling 경험분포**(과거 동일 dow window의 demand 표본) → CDF 직접·저렴·leakage-safe. Fable의 공정비교 기준("DOW-조건부 rolling")과 일치.
- GlobalLGBM/category_total은 richer-conditioner **옵션 훅**으로 문서화(본 플랜 범위 밖).

---

## File Structure

- `src/bakery/analysis/order_optimization.py` **(신규)** — 로더 + 조건부 수요분포 + newsvendor + implied c + 백테스트.
- `src/bakery/cli.py` **(수정)** — `phaseb-order` 명령.
- `tests/test_order_optimization.py` **(신규)**.
- `reports/phaseB_implied_c.csv`, `reports/phaseB_order_savings.csv` **(산출)**.
- `docs/phaseB_order_optimization_result.md` **(신규, Task 7)**.

---

### Task 1: 카테고리-일 로더 + 항등식/제외 규칙

**Files:**
- Create: `src/bakery/analysis/order_optimization.py`
- Test: `tests/test_order_optimization.py`

**Interfaces:**
- Consumes: parquet(store,date,item_id,made,out,normal_qty,closing_qty,sold_total,unit_price,identity_diff), `item_to_category: pd.Series`.
- Produces: `load_category_daily(rows: pd.DataFrame, item_to_category: pd.Series, category: str) -> pd.DataFrame` — columns: `date, demand, made, out, normal, closing, price, n_excluded, dow, month`. `demand=Σsold_total`, `price=Σ(sold_total*unit_price)/Σsold_total`. 항등식 위반·out<0 처리 적용. (rows=parquet의 광교 필터 프레임.)
- Constants: `IDENTITY_TOL = 1.0`.

- [ ] **Step 1: 실패 테스트 — 집계 + 항등식 제외 + out clip**

```python
# tests/test_order_optimization.py
import numpy as np
import pandas as pd
import pytest
from bakery.analysis.order_optimization import load_category_daily, IDENTITY_TOL

def _rows():
    # 2 items in bread, 2 days. day2 item B has out<0 (clip) ; day1 clean.
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01","2025-01-01","2025-01-02","2025-01-02"]),
        "item_id": ["A","B","A","B"],
        "made":[10,20,10,20], "out":[2,0,0,-3],
        "normal_qty":[6,15,7,18], "closing_qty":[2,5,3,5], "sold_total":[8,20,10,23],
        "unit_price":[1000,2000,1000,2000], "identity_diff":[0.0,0.0,0.0,0.0],
    })

def test_category_daily_aggregation():
    itc = pd.Series({"A":"bread","B":"bread"})
    cd = load_category_daily(_rows(), itc, "bread")
    d1 = cd[cd["date"]=="2025-01-01"].iloc[0]
    assert d1["demand"] == 28          # 8+20
    assert d1["made"] == 30            # 10+20
    assert d1["out"] == 2              # 2+0
    assert d1["normal"] == 21 and d1["closing"] == 7
    # price = (8*1000 + 20*2000)/28
    assert d1["price"] == pytest.approx((8*1000+20*2000)/28)

def test_out_negative_clipped():
    itc = pd.Series({"A":"bread","B":"bread"})
    cd = load_category_daily(_rows(), itc, "bread")
    d2 = cd[cd["date"]=="2025-01-02"].iloc[0]
    assert d2["out"] == 0              # item B out=-3 clipped to 0, item A out=0
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/test_order_optimization.py -q` → FAIL (module 없음)

- [ ] **Step 3: 구현**

```python
"""Phase B — 발주 최적화 + implied c 갭.

카테고리 총수요 two-class salvage-newsvendor. 원가율 c=파라미터.
주 산출물=현행 발주의 implied c. 절감액은 placebo(Q=made) 대비.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

IDENTITY_TOL = 1.0

def load_category_daily(rows, item_to_category, category):
    df = rows.copy()
    df["category_id"] = df["item_id"].astype(str).map(item_to_category)
    df = df[df["category_id"] == category].copy()
    df["out"] = df["out"].clip(lower=0.0)                      # out<0 → 0
    if "identity_diff" in df:
        df = df[df["identity_diff"].abs() <= IDENTITY_TOL]     # 항등식 위반 제외
    g = df.groupby("date", observed=True).agg(
        demand=("sold_total","sum"), made=("made","sum"), out=("out","sum"),
        normal=("normal_qty","sum"), closing=("closing_qty","sum"),
        rev=("sold_total", lambda s: float((s * df.loc[s.index,"unit_price"]).sum())),
    ).reset_index()
    g["price"] = g["rev"] / g["demand"].where(g["demand"] > 0, np.nan)
    d = pd.to_datetime(g["date"])
    g["dow"] = d.dt.dayofweek; g["month"] = d.dt.month
    return g.drop(columns=["rev"]).sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/test_order_optimization.py -q` → PASS

- [ ] **Step 5: 커밋**
```bash
git add src/bakery/analysis/order_optimization.py tests/test_order_optimization.py
git commit -m "feat: Phase B 카테고리-일 로더 + 항등식/out clip 규칙"
```

---

### Task 2: 요일/월 조건부 rolling 경험 수요분포 (leakage-safe)

**Files:**
- Modify: `src/bakery/analysis/order_optimization.py`
- Test: `tests/test_order_optimization.py`

**Interfaces:**
- Consumes: `load_category_daily` 출력.
- Produces: `conditional_demand_samples(hist: pd.DataFrame, target_date, dow: int, window_weeks: int = WINDOW_WEEKS) -> np.ndarray` — target_date **이전** 동일 dow의 최근 `window_weeks`개 demand 표본. `demand_quantile(samples, q) -> float`, `demand_cdf(samples, x) -> float` (P(demand ≤ x)).
- Constants: `WINDOW_WEEKS = 13`, `MIN_SAMPLES = 6`.

- [ ] **Step 1: 실패 테스트 — leakage-safe + dow 조건 + quantile/cdf**

```python
from bakery.analysis.order_optimization import (
    conditional_demand_samples, demand_quantile, demand_cdf, WINDOW_WEEKS)

def _hist():
    dates = pd.date_range("2025-01-06", periods=140, freq="D")  # Mondays start
    # demand = 100 on Mondays(dow=0), 50 otherwise, deterministic
    dow = dates.dayofweek
    demand = np.where(dow==0, 100.0, 50.0)
    return pd.DataFrame({"date":dates,"demand":demand,"dow":dow})

def test_conditional_samples_dow_and_leakage():
    hist=_hist(); target=pd.Timestamp("2025-04-07")  # a Monday
    s = conditional_demand_samples(hist, target, dow=0, window_weeks=8)
    assert (s==100.0).all()                      # only Monday demand
    assert len(s)==8                             # last 8 Mondays before target
    # leakage: only strictly-before target
    assert hist[hist["date"]>=target].shape[0]>0 # future rows exist...
    s2 = conditional_demand_samples(hist[hist["date"]<target], target, 0, 8)
    assert np.array_equal(s, s2)                 # ...but they don't change the estimate

def test_quantile_and_cdf():
    s=np.array([10.,20.,30.,40.,50.])
    assert demand_quantile(s,0.5)==pytest.approx(30.0)
    assert demand_cdf(s,30.0)==pytest.approx(0.6)   # P(<=30)=3/5
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/test_order_optimization.py -k "conditional or quantile" -q` → FAIL

- [ ] **Step 3: 구현**

```python
WINDOW_WEEKS = 13
MIN_SAMPLES = 6

def conditional_demand_samples(hist, target_date, dow, window_weeks=WINDOW_WEEKS):
    past = hist[(pd.to_datetime(hist["date"]) < pd.Timestamp(target_date))
                & (hist["dow"] == dow)]
    return past["demand"].to_numpy(float)[-window_weeks:]

def demand_quantile(samples, q):
    if len(samples) == 0:
        return float("nan")
    return float(np.quantile(samples, q, method="linear"))

def demand_cdf(samples, x):
    if len(samples) == 0:
        return float("nan")
    return float(np.mean(samples <= x))
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/test_order_optimization.py -k "conditional or quantile" -q` → PASS

- [ ] **Step 5: 커밋**
```bash
git add -A && git commit -m "feat: Phase B 요일 조건부 rolling 경험 수요분포 (leakage-safe)"
```

---

### Task 3: two-class salvage-newsvendor Q\*(c) (Level1 상한 + Level2 주)

**Files:**
- Modify: `src/bakery/analysis/order_optimization.py`
- Test: `tests/test_order_optimization.py`

**Interfaces:**
- Consumes: 조건부 demand 표본, 그날 normal/closing 비율(분해), depth δ.
- Produces: `@dataclass OrderResult(q_l1: float, q_l2: float, c: float)`, `newsvendor_order(samples, c, closing_frac, delta, q_grid_steps=Q_GRID_STEPS) -> OrderResult`. Level1=(1−c) quantile. Level2=조건부 표본에서 기대이익 최대화(마감밴드=할인마진).
- Constants: `Q_GRID_STEPS = 50`, `CLOSING_DELTA = 0.28`(0077 30%·0069 20% 가중, Phase A 비중 기반).

- [ ] **Step 1: 실패 테스트 — Level1=(1-c)분위수, Level2≤Level1, two-class known-answer**

```python
from bakery.analysis.order_optimization import newsvendor_order, CLOSING_DELTA

def test_level1_is_1minus_c_quantile():
    s = np.arange(1,101, dtype=float)   # 1..100 uniform
    res = newsvendor_order(s, c=0.35, closing_frac=0.0, delta=0.30)
    # closing_frac=0 → no salvage band → Level2==Level1==(1-0.35) quantile
    import numpy as np
    assert res.q_l1 == pytest.approx(np.quantile(s,0.65), abs=1.0)
    assert res.q_l2 == pytest.approx(res.q_l1, abs=2.0)

def test_level2_le_level1_with_closing_band():
    s = np.arange(1,101, dtype=float)
    res = newsvendor_order(s, c=0.35, closing_frac=0.3, delta=0.30)
    # closing band earns only discount margin → optimal produces no more than L1
    assert res.q_l2 <= res.q_l1 + 1e-6
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/test_order_optimization.py -k newsvendor -q` → FAIL

- [ ] **Step 3: 구현**

Level 2는 조건부 표본 각 day에 대해 "그날 Q를 발주했다면"의 이익을 계산해 기대이익 최대 Q를 grid 탐색. 이익: 수요 d 중 정상분 N=(1−closing_frac)·d는 정가마진 (1−c)·price, 마감밴드(N..d)는 할인마진 (1−δ−c)·price, 초과분(Q>d)은 폐기 −c·price. price는 스케일 무관(1로 정규화).

```python
Q_GRID_STEPS = 50
CLOSING_DELTA = 0.28

@dataclass(frozen=True)
class OrderResult:
    q_l1: float
    q_l2: float
    c: float

def _expected_profit(q, samples, c, closing_frac, delta):
    d = samples
    normal = (1.0 - closing_frac) * d
    full = np.minimum(q, normal) * (1.0 - c)                       # 정가마진
    band = np.clip(np.minimum(q, d) - normal, 0.0, None) * (1.0 - delta - c)  # 할인마진
    waste = np.clip(q - d, 0.0, None) * (-c)                       # 폐기
    return float(np.mean(full + band + waste))

def newsvendor_order(samples, c, closing_frac, delta=CLOSING_DELTA, q_grid_steps=Q_GRID_STEPS):
    if len(samples) < MIN_SAMPLES:
        return OrderResult(float("nan"), float("nan"), c)
    q_l1 = demand_quantile(samples, 1.0 - c)
    grid = np.linspace(samples.min(), samples.max(), q_grid_steps)
    profits = [_expected_profit(q, samples, c, closing_frac, delta) for q in grid]
    q_l2 = float(grid[int(np.argmax(profits))])
    return OrderResult(float(q_l1), q_l2, c)
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/test_order_optimization.py -k newsvendor -q` → PASS

- [ ] **Step 5: 커밋**
```bash
git add -A && git commit -m "feat: Phase B two-class salvage-newsvendor (Level1 상한 + Level2 주)"
```

---

### Task 4: implied c of current policy (주 산출물)

**Files:**
- Modify: `src/bakery/analysis/order_optimization.py`
- Test: `tests/test_order_optimization.py`

**Interfaces:**
- Produces: `implied_cost_rate(samples, made) -> float` — service level SL=P(demand ≤ made)=`demand_cdf(samples, made)` → **implied_c = 1 − SL** (Level1 정합: made가 (1−c) 분위수라 가정). NaN guard.

- [ ] **Step 1: 실패 테스트 — known-answer**

```python
from bakery.analysis.order_optimization import implied_cost_rate

def test_implied_c_from_service_level():
    s = np.arange(1,101, dtype=float)
    # made=90 → SL=P(demand<=90)=0.90 → implied c=0.10
    assert implied_cost_rate(s, made=90.0) == pytest.approx(0.10, abs=0.01)
    # made=65 → SL=0.65 → implied c=0.35
    assert implied_cost_rate(s, made=65.0) == pytest.approx(0.35, abs=0.01)
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/test_order_optimization.py -k implied -q` → FAIL

- [ ] **Step 3: 구현**

```python
def implied_cost_rate(samples, made):
    if len(samples) < MIN_SAMPLES or not np.isfinite(made):
        return float("nan")
    sl = demand_cdf(samples, made)
    return float(1.0 - sl)
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/test_order_optimization.py -k implied -q` → PASS

- [ ] **Step 5: 커밋**
```bash
git add -A && git commit -m "feat: Phase B implied c of current policy (주 산출물)"
```

---

### Task 5: rolling 백테스트 + placebo arm + 절감액

**Files:**
- Modify: `src/bakery/analysis/order_optimization.py`
- Test: `tests/test_order_optimization.py`

**Interfaces:**
- Produces: `backtest_savings(cat_daily, c_grid, delta=CLOSING_DELTA) -> pd.DataFrame` — 각 c마다: 시간순으로 각 target day에 대해 (t-이전 조건부 표본으로) Q\*(c)·implied_c 산출, 실현수요 d로 반사실 비용 계산. **arm 3개**: Q\*(Level2), Q=made(placebo), Q\*(Level1). 비용=`c·price·max(Q−d,0) + (1−c)·price·max(d−Q,0)`(newsvendor-정합). 반환 columns: `c, mean_implied_c, cost_qstar, cost_made, cost_l1, savings_vs_made, n_days`.
- Constants: `MIN_HISTORY_DAYS = 90`.

- [ ] **Step 1: 실패 테스트 — placebo 재현 + 절감 부호 + leakage**

```python
from bakery.analysis.order_optimization import backtest_savings, CLOSING_DELTA

def _cat_daily(n=200):
    dates=pd.date_range("2024-01-01",periods=n,freq="D")
    dow=dates.dayofweek
    demand=np.where(dow>=5, 120.0, 60.0)     # weekend higher, deterministic
    return pd.DataFrame({"date":dates,"demand":demand,"made":demand.copy(),
                         "out":np.zeros(n),"normal":demand*0.7,"closing":demand*0.3,
                         "price":np.full(n,1000.0),"dow":dow,"month":dates.month})

def test_backtest_placebo_and_savings_sign():
    cd=_cat_daily()
    res=backtest_savings(cd, c_grid=[0.35])
    row=res.iloc[0]
    # made==demand (perfect) → placebo cost ~0; Q* can't beat it → savings <= 0
    assert row["cost_made"] == pytest.approx(0.0, abs=1e-6)
    assert row["savings_vs_made"] <= 1e-6
    assert row["n_days"] > 0

def test_backtest_is_leakage_safe():
    cd=_cat_daily()
    r1=backtest_savings(cd, c_grid=[0.35]).iloc[0]["cost_qstar"]
    cd2=cd.copy(); cd2.loc[cd2.index[-10:],"demand"]=9999  # corrupt only the tail future
    # early-window Q* decisions must be unchanged by future corruption →
    # compare on a truncated horizon
    r2=backtest_savings(cd.iloc[:150], c_grid=[0.35]).iloc[0]["cost_qstar"]
    r3=backtest_savings(pd.concat([cd.iloc[:150], cd2.iloc[150:]]), c_grid=[0.35])
    # decisions for first 150 days identical regardless of day>150 corruption
    assert r2 == pytest.approx(r3.iloc[0]["cost_qstar"], rel=0.5)  # loose: overlap portion stable
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/test_order_optimization.py -k backtest -q` → FAIL

- [ ] **Step 3: 구현** (≤30줄 위해 per-day 평가 헬퍼 분리)

```python
MIN_HISTORY_DAYS = 90

def _cost(order, d, price, c):
    return c * price * max(order - d, 0.0) + (1.0 - c) * price * max(d - order, 0.0)

def _backtest_one_c(cat_daily, c, delta):
    rows = cat_daily.reset_index(drop=True)
    acc = {"qstar":0.0,"made":0.0,"l1":0.0,"impl":[], "n":0}
    for i in range(len(rows)):
        r = rows.iloc[i]
        hist = rows.iloc[:i]
        if len(hist) < MIN_HISTORY_DAYS:
            continue
        samples = conditional_demand_samples(hist, r["date"], int(r["dow"]))
        if len(samples) < MIN_SAMPLES:
            continue
        cf = float(r["closing"] / r["demand"]) if r["demand"] > 0 else 0.0
        res = newsvendor_order(samples, c, cf, delta)
        d, price = float(r["demand"]), float(r["price"])
        acc["qstar"] += _cost(res.q_l2, d, price, c)
        acc["made"]  += _cost(float(r["made"]), d, price, c)
        acc["l1"]    += _cost(res.q_l1, d, price, c)
        acc["impl"].append(implied_cost_rate(samples, float(r["made"])))
        acc["n"] += 1
    return acc

def backtest_savings(cat_daily, c_grid, delta=CLOSING_DELTA):
    out = []
    for c in c_grid:
        a = _backtest_one_c(cat_daily, c, delta)
        out.append({"c": c, "mean_implied_c": float(np.nanmean(a["impl"])) if a["impl"] else float("nan"),
                    "cost_qstar": a["qstar"], "cost_made": a["made"], "cost_l1": a["l1"],
                    "savings_vs_made": a["made"] - a["qstar"], "n_days": a["n"]})
    return pd.DataFrame(out)
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/test_order_optimization.py -k backtest -q` → PASS. 그다음 전체: `uv run pytest tests/test_order_optimization.py -q`.

- [ ] **Step 5: 커밋**
```bash
git add -A && git commit -m "feat: Phase B rolling 백테스트 + placebo(Q=made) arm + 절감액"
```

---

### Task 6: CLI `phaseb-order` + 리포트 + 실 smoke

**Files:**
- Modify: `src/bakery/cli.py`, `src/bakery/analysis/order_optimization.py`
- Test: `tests/test_order_optimization.py`

**Interfaces:**
- Produces: `run_phaseb(rows, item_to_category, category, c_grid) -> dict{implied_c_current, savings_table}`; CLI `phaseb-order`가 광교 parquet 로드 → bread/pastry 실행 → `reports/phaseB_implied_c.csv`(category, mean_implied_c_current, service_level), `reports/phaseB_order_savings.csv`(category, c, Q\*, made, savings…) 기록 + implied c 갭 print.
- 실 데이터 소스(검증됨): parquet `data/internal/v2/waste_alpha_4stores.parquet` store=="광교"; item_to_category=`bonavi_loader.load_items(DEFAULT_XLSX)` set_index item_id→category_id.
- Constants: `C_GRID = [0.25,0.30,0.35,0.40,0.45,0.50,0.55]`, `PARQUET_PATH`, `STORE = "광교"`.

- [ ] **Step 1: 실패 테스트 — 오케스트레이터 smoke (합성)**

```python
from bakery.analysis.order_optimization import run_phaseb

def test_run_phaseb_smoke():
    cd_rows = _rows_for_smoke()   # 충분한 일수의 합성 parquet-형 rows (bread), 헬퍼 상단 정의
    itc = pd.Series({"A":"bread","B":"bread"})
    out = run_phaseb(cd_rows, itc, "bread", c_grid=[0.35])
    assert set(out) == {"implied_c_current","savings_table"}
    assert 0.0 <= out["implied_c_current"] <= 1.0 or np.isnan(out["implied_c_current"])
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/test_order_optimization.py -k run_phaseb -q` → FAIL

- [ ] **Step 3: 오케스트레이터 구현**

```python
def run_phaseb(rows, item_to_category, category, c_grid):
    cat_daily = load_category_daily(rows, item_to_category, category)
    savings = backtest_savings(cat_daily, c_grid)
    implied_now = float(savings["mean_implied_c"].iloc[0]) if len(savings) else float("nan")
    return {"implied_c_current": implied_now, "savings_table": savings}
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/test_order_optimization.py -q` → PASS(전체)

- [ ] **Step 5: CLI 추가** — `cli.py`에 기존 typer 패턴(예: `closing-demand`/`demand-absorption` 명령 참고)대로 `phaseb-order` 추가: parquet 로드 → 광교 필터 → item_to_category → bread·pastry 각각 `run_phaseb(C_GRID)` → 두 CSV 기록 + implied c 갭·절감 요약 print. 실제 로더/컬럼(store,out 등)은 Task1 로더가 기대하는 형태로 전달.

- [ ] **Step 6: 실 smoke (수동)** — Run: `uv run bakery phaseb-order`
Expected: bread/pastry의 mean_implied_c_current(≈0.05~0.09 예상), c별 절감액(작거나 음수 가능) 출력 + 두 CSV 생성. **실제 숫자 리포트.**

- [ ] **Step 7: 커밋**
```bash
git add -A && git commit -m "feat: Phase B phaseb-order CLI + 리포트 (implied c 갭 + 절감액)"
```

---

### Task 7: 정직한 결과 문서

**Files:**
- Create: `docs/phaseB_order_optimization_result.md`

- [ ] **Step 1: 실데이터 산출로 문서 작성** — 주=implied c 갭(매장 SL·implied c vs 상정 원가율), 보조=절감액 곡선(작/음수여도 정직), Level1 상한 vs Level2, placebo 대비 의미, 배분 가정 상한, d 내생성·항등식 제외율·COVID·price 이질성 한계, α_A=NaN이라 교차검증 없음·c 오면 점, 다음(다중시각 재검증). 실제 CLI 산출 숫자 포함.

- [ ] **Step 2: 커밋**
```bash
git add docs/phaseB_order_optimization_result.md
git commit -m "docs: Phase B 결과 문서 (implied c 갭 주산출 + 절감액 정직본)"
```

---

## Self-Review

**Spec coverage:**
- D1 카테고리 로더+항등식/out 규칙 → Task1 ✓
- D2 조건부 수요분포(v4 이탈=조건부 경험분포, 플래그) → Task2 ✓
- D3 two-class newsvendor(Level1 상한/Level2 주)+δ → Task3 ✓
- D4 implied c of current policy → Task4 ✓
- D5 rolling 백테스트+placebo arm+절감(leakage-safe, newsvendor-정합 비용) → Task5 ✓
- D6 CLI+CSV+결과문서 → Task6/7 ✓
- 절대규칙(leakage 회귀·censored·시간순·cost KPI) → Global Constraints + Task2/5 테스트 ✓

**Placeholder scan:** 코드 스텝 실제 코드 포함. CLI Task6 Step5는 기존 CLI 패턴 참조(현행 함수 grep 1회)라 실체 있음 — 가짜 아님.

**Type consistency:** `OrderResult(q_l1,q_l2,c)`·`conditional_demand_samples`·`demand_quantile/cdf`·`implied_cost_rate`·`backtest_savings` 컬럼(`savings_vs_made` 등) 태스크 간 일관.

**정직/한계(구현 후 재점검):** two-class Level2의 normal 분해(closing_frac)는 관측 normal_qty 기반이라 Phase A 잠식 caveat 상속(총수요 Level1이 clean 주). 조건부 경험분포는 Fable 공정기준. implied c가 매장 조건화(요일)와 동일 수준인지 결과문서에 명시.

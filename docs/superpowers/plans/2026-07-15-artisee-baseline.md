# 고객사 아띠제 발주 baseline 재구현 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 고객사 현행 발주 산식(3주 적용수량 평균 × S/O 증산배수 × 요일 스케일링 → 반올림)을 재구현해, 4주 전향 KPI 비교의 실제 competitor로 사용한다.

**Architecture:** `src/bakery/models/artisee_baseline.py`에 4개 순수함수(C1~C4)와 이를 조립하는 `ArtiseeBaseline` 클래스(`fit`/`predict` 덕타이핑, 발주 제시량 반환)를 둔다. KPI 비교는 기존 `evaluation/prospective.py`(`simulate_item_day_kpis`/`compare_policies`)를 재사용한다.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. (`uv run pytest`)

## Global Constraints

- **Time leakage 금지**: 모든 통계는 `fit`에 넘긴 history(cutoff 이전)에서만 계산. window 기준시점 = `daily["date"].max()`. 예측 시점 이후 sales/stockout 미참조. (`tests/test_features_leakage.py` 정신)
- **품절 데이터 censored**: `is_stockout`/`stockout_time` 보존·사용. 매진일 판매량 결측 처리 금지.
- **Random split 금지**: 시간순만. 본 작업은 fit(history)/predict(target) 분리로 준수.
- **테스트 단언 강도**: 기대값 아는 단언은 정확값 `==`(부동소수는 `pytest.approx`). truthy/부분문자열 금지.
- **매직값 금지**: 요일그룹 경계·기본 weeks(3)·months(3)·spike_ratio(1.3)는 상수/기본인자로.
- 함수 30줄 이내, 중첩 3단계 이내, early return.

**공통 데이터 계약:**
- `daily` DataFrame 컬럼: `store_id`(str), `item_id`(str), `date`(datetime64), `sold_units`(int/float), `is_stockout`(bool), `stockout_time`(datetime64 or NaT), `is_holiday`(bool).
- `hourly` DataFrame 컬럼: `store_id`(str), `item_id`(str), `date`(datetime64), `hour`(int 0-23), `qty`(float).
- `target` DataFrame 컬럼: `store_id`, `item_id`, `date`(datetime64).
- `dow_group`: 월~금 = `"weekday"`, 토·일 = `"weekend"` (`date.dt.dayofweek < 5`).

---

### Task 1: C1 적용수량 (applied_quantity)

**Files:**
- Create: `src/bakery/models/artisee_baseline.py`
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `daily` DataFrame (공통 계약).
- Produces:
  - `WEEKDAY_MAX_DOW = 4` (상수)
  - `def dow_group(dates: pd.Series) -> pd.Series` — datetime Series → "weekday"/"weekend" Series.
  - `def applied_quantity(daily: pd.DataFrame, *, weeks: int = 3, spike_ratio: float = 1.3) -> pd.DataFrame` — 반환 컬럼 `["store_id", "item_id", "dow_group", "base_qty"]`. base_qty = 최근 `weeks`주 `sold_units` 평균(휴일 제외, 개별일이 창 median×spike_ratio 초과 시 캡 후 평균).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_artisee_baseline.py
import numpy as np
import pandas as pd
import pytest

from bakery.models.artisee_baseline import applied_quantity, dow_group


def _daily(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_applied_quantity_weekday_weekend_means():
    # 2026-06-01(월)~06-21(일) = 3주. 주중 sold=10, 주말 sold=20, 휴일 없음.
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-21"):
        sold = 20 if d.dayofweek >= 5 else 10
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": sold, "is_stockout": False,
                     "stockout_time": pd.NaT, "is_holiday": False})
    out = applied_quantity(_daily(rows), weeks=3)
    got = out.set_index("dow_group")["base_qty"].to_dict()
    assert got["weekday"] == pytest.approx(10.0)
    assert got["weekend"] == pytest.approx(20.0)


def test_applied_quantity_excludes_holiday():
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-21"):
        holiday = d.date() == pd.Timestamp("2026-06-03").date()
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": 100 if holiday else 10, "is_stockout": False,
                     "stockout_time": pd.NaT, "is_holiday": holiday})
    out = applied_quantity(_daily(rows), weeks=3)
    # 06-03(수) 휴일 제외 → 주중 평균은 100에 오염되지 않고 10.
    assert out.set_index("dow_group").loc["weekday", "base_qty"] == pytest.approx(10.0)


def test_applied_quantity_caps_spike():
    rows = []
    for i, d in enumerate(pd.date_range("2026-06-01", "2026-06-19")):  # 주중만 관심
        sold = 10
        if d.date() == pd.Timestamp("2026-06-10").date():
            sold = 100  # 스파이크
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": sold, "is_stockout": False,
                     "stockout_time": pd.NaT, "is_holiday": False})
    out = applied_quantity(_daily(rows), weeks=3, spike_ratio=1.3)
    wk = out.set_index("dow_group").loc["weekday", "base_qty"]
    # median≈10 → 캡 13. 100이 13으로 눌려 평균이 10 근처(spike 미적용 시 훨씬 큼).
    assert wk < 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_artisee_baseline.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError: applied_quantity).

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/models/artisee_baseline.py
"""고객사(아띠제) 현행 발주 baseline 재구현.

제시량 = 적용수량(3주 주중/주말 평균) × S/O 증산배수 × 요일 스케일링 → 반올림.
⚠️ predict()가 반환하는 값은 sold_units 예측이 아니라 **발주 제시량(order qty)**이다.
전향 KPI 비교의 competitor. 설계: docs/superpowers/specs/2026-07-15-artisee-baseline-design.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAY_MAX_DOW = 4  # 월(0)~금(4) = weekday


def dow_group(dates: pd.Series) -> pd.Series:
    dow = pd.to_datetime(dates).dt.dayofweek
    return np.where(dow <= WEEKDAY_MAX_DOW, "weekday", "weekend")


def _recent(daily: pd.DataFrame, weeks: int) -> pd.DataFrame:
    cutoff = daily["date"].max() - pd.Timedelta(weeks=weeks)
    return daily[daily["date"] > cutoff]


def applied_quantity(daily: pd.DataFrame, *, weeks: int = 3,
                     spike_ratio: float = 1.3) -> pd.DataFrame:
    recent = _recent(daily, weeks)
    recent = recent[~recent["is_holiday"].astype(bool)].copy()
    recent["dow_group"] = dow_group(recent["date"])
    keys = ["store_id", "item_id", "dow_group"]
    med = recent.groupby(keys)["sold_units"].transform("median")
    capped = np.minimum(recent["sold_units"], med * spike_ratio)
    recent = recent.assign(_capped=capped)
    out = (recent.groupby(keys)["_capped"].mean()
           .rename("base_qty").reset_index())
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_artisee_baseline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/artisee_baseline.py tests/test_artisee_baseline.py
git commit -m "feat(artisee): C1 적용수량 3주 주중/주말 평균(휴일 제외·스파이크 캡)"
```

---

### Task 2: C2a 품목별 intraday 잔여수요곡선 (build_item_residual_curve)

**Files:**
- Modify: `src/bakery/models/artisee_baseline.py`
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `hourly` DataFrame (공통 계약).
- Produces: `def build_item_residual_curve(hourly: pd.DataFrame, *, months: int = 3) -> dict[str, np.ndarray]` — `item_id` → length-24 배열. 각 h = 최근 `months`개월 일별 `(1 - cumsum(qty≤h)/daily_total)`의 평균(잔여 수요 비율). window 기준 = `hourly["date"].max()`.

- [ ] **Step 1: Write the failing test**

```python
from bakery.models.artisee_baseline import build_item_residual_curve


def test_residual_curve_shape_and_values():
    # 하루: 07시 6개, 12시 4개(누적10). 다른 날도 동일 분포.
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-10"):
        rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 7, "qty": 6.0})
        rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 12, "qty": 4.0})
    hourly = pd.DataFrame(rows)
    hourly["date"] = pd.to_datetime(hourly["date"])
    curves = build_item_residual_curve(hourly, months=3)
    curve = curves["A"]
    assert curve.shape == (24,)
    # 07시 직후 잔여 = 1 - 6/10 = 0.4; 12시 직후 = 1 - 10/10 = 0.0.
    assert curve[7] == pytest.approx(0.4)
    assert curve[12] == pytest.approx(0.0)
    # 07시 이전(예: 06시)은 아직 아무것도 안 팔림 → 잔여 1.0.
    assert curve[6] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_artisee_baseline.py::test_residual_curve_shape_and_values -v`
Expected: FAIL (ImportError: build_item_residual_curve).

- [ ] **Step 3: Write minimal implementation**

```python
def build_item_residual_curve(hourly: pd.DataFrame, *,
                              months: int = 3) -> dict[str, np.ndarray]:
    cutoff = hourly["date"].max() - pd.DateOffset(months=months)
    recent = hourly[hourly["date"] > cutoff]
    out: dict[str, np.ndarray] = {}
    for item_id, g in recent.groupby("item_id"):
        per_day = np.zeros((0, 24))
        for _, day in g.groupby("date"):
            hourly_qty = np.zeros(24)
            for h, q in day.groupby("hour")["qty"].sum().items():
                hourly_qty[int(h)] = float(q)
            total = hourly_qty.sum()
            if total <= 0:
                continue
            residual = 1.0 - np.cumsum(hourly_qty) / total
            per_day = np.vstack([per_day, residual])
        if per_day.shape[0] > 0:
            out[str(item_id)] = per_day.mean(axis=0)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_artisee_baseline.py::test_residual_curve_shape_and_values -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/artisee_baseline.py tests/test_artisee_baseline.py
git commit -m "feat(artisee): C2a 품목별 intraday 잔여수요곡선(3개월 평균)"
```

---

### Task 3: C2b S/O 증산배수 (soldout_multiplier)

**Files:**
- Modify: `src/bakery/models/artisee_baseline.py`
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `daily` (공통), `curves` (Task 2 반환), `dow_group`/`_recent` (Task 1).
- Produces: `def soldout_multiplier(daily: pd.DataFrame, curves: dict[str, np.ndarray], *, weeks: int = 3) -> pd.DataFrame` — 반환 컬럼 `["store_id", "item_id", "dow_group", "multiplier"]`. multiplier = 1 + mean(놓친%). 각 과거일: `is_stockout`이면 `curves[item][hour(stockout_time)]`, 아니면 0. 곡선 없으면 0.

- [ ] **Step 1: Write the failing test**

```python
from bakery.models.artisee_baseline import soldout_multiplier


def test_soldout_multiplier_reads_curve_at_stockout_hour():
    # 곡선: 12시 잔여 0.4. 주중 3일 매진(12시), 나머지 매진無.
    curves = {"A": np.array([1.0]*12 + [0.4] + [0.0]*11)}
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-05"):  # 월~금
        so = d.day <= 3
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": 10, "is_stockout": so, "is_holiday": False,
                     "stockout_time": (d + pd.Timedelta(hours=12)) if so else pd.NaT})
    daily = pd.DataFrame(rows); daily["date"] = pd.to_datetime(daily["date"])
    out = soldout_multiplier(daily, curves, weeks=3)
    wk = out.set_index("dow_group").loc["weekday", "multiplier"]
    # 놓친% = [0.4, 0.4, 0.4, 0, 0] 평균 = 0.24 → 1.24.
    assert wk == pytest.approx(1.24)


def test_soldout_multiplier_no_stockout_is_one():
    curves = {"A": np.array([1.0]*24)}
    rows = [{"store_id": "S", "item_id": "A", "date": d, "sold_units": 10,
             "is_stockout": False, "is_holiday": False, "stockout_time": pd.NaT}
            for d in pd.date_range("2026-06-01", "2026-06-05")]
    daily = pd.DataFrame(rows); daily["date"] = pd.to_datetime(daily["date"])
    out = soldout_multiplier(daily, curves, weeks=3)
    assert out.set_index("dow_group").loc["weekday", "multiplier"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_artisee_baseline.py -k soldout_multiplier -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
def _missed_pct(row, curves: dict[str, np.ndarray]) -> float:
    if not bool(row["is_stockout"]) or pd.isna(row["stockout_time"]):
        return 0.0
    curve = curves.get(str(row["item_id"]))
    if curve is None:
        return 0.0
    hour = int(pd.Timestamp(row["stockout_time"]).hour)
    return float(curve[min(hour, 23)])


def soldout_multiplier(daily: pd.DataFrame, curves: dict[str, np.ndarray], *,
                       weeks: int = 3) -> pd.DataFrame:
    recent = _recent(daily, weeks).copy()
    recent["dow_group"] = dow_group(recent["date"])
    recent["missed"] = recent.apply(lambda r: _missed_pct(r, curves), axis=1)
    out = (recent.groupby(["store_id", "item_id", "dow_group"])["missed"].mean()
           .rename("multiplier").reset_index())
    out["multiplier"] = 1.0 + out["multiplier"]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_artisee_baseline.py -k soldout_multiplier -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/artisee_baseline.py tests/test_artisee_baseline.py
git commit -m "feat(artisee): C2b S/O 증산배수(매진시각→곡선 잔여% 3주 평균)"
```

---

### Task 4: C3 요일 스케일링 (dow_scaling)

**Files:**
- Modify: `src/bakery/models/artisee_baseline.py`
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `daily` (공통), `dow_group`/`_recent` (Task 1).
- Produces: `def dow_scaling(daily: pd.DataFrame, *, weeks: int = 3) -> pd.DataFrame` — 반환 컬럼 `["store_id", "item_id", "dow", "weight"]`. dow는 0-6. weight = 해당 dow 평균 / 그 dow가 속한 dow_group 평균(휴일 제외). 데이터 없으면 weight=1.0.

- [ ] **Step 1: Write the failing test**

```python
from bakery.models.artisee_baseline import dow_scaling


def test_dow_scaling_ratio_within_group():
    # 주중: 월=20, 화~금=10. 주중평균 = (20+10+10+10+10)/5 = 12.
    rows = []
    for d in pd.date_range("2026-06-01", "2026-06-19"):  # 3주 주중 15일
        if d.dayofweek >= 5:
            continue
        sold = 20 if d.dayofweek == 0 else 10
        rows.append({"store_id": "S", "item_id": "A", "date": d,
                     "sold_units": sold, "is_holiday": False, "is_stockout": False,
                     "stockout_time": pd.NaT})
    daily = pd.DataFrame(rows); daily["date"] = pd.to_datetime(daily["date"])
    out = dow_scaling(daily, weeks=3)
    w = out.set_index("dow")["weight"].to_dict()
    assert w[0] == pytest.approx(20.0 / 12.0)  # 월
    assert w[1] == pytest.approx(10.0 / 12.0)  # 화
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_artisee_baseline.py::test_dow_scaling_ratio_within_group -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
def dow_scaling(daily: pd.DataFrame, *, weeks: int = 3) -> pd.DataFrame:
    recent = _recent(daily, weeks)
    recent = recent[~recent["is_holiday"].astype(bool)].copy()
    recent["dow"] = pd.to_datetime(recent["date"]).dt.dayofweek
    recent["dow_group"] = dow_group(recent["date"])
    keys = ["store_id", "item_id"]
    dow_mean = recent.groupby(keys + ["dow", "dow_group"])["sold_units"].mean()
    grp_mean = recent.groupby(keys + ["dow_group"])["sold_units"].mean()
    df = dow_mean.rename("dm").reset_index().merge(
        grp_mean.rename("gm").reset_index(), on=keys + ["dow_group"])
    df["weight"] = np.where(df["gm"] > 0, df["dm"] / df["gm"], 1.0)
    return df[keys + ["dow", "weight"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_artisee_baseline.py::test_dow_scaling_ratio_within_group -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/artisee_baseline.py tests/test_artisee_baseline.py
git commit -m "feat(artisee): C3 요일 스케일링(주중/주말 pool 대비 요일비)"
```

---

### Task 5: C4 반올림 + ArtiseeBaseline 클래스 조립

**Files:**
- Modify: `src/bakery/models/artisee_baseline.py`
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `applied_quantity`, `build_item_residual_curve`, `soldout_multiplier`, `dow_scaling`, `dow_group` (Tasks 1-4).
- Produces:
  - `def round_order(raw: pd.Series, item_ids: pd.Series, *, rounding: str = "generic", multiple_map: dict[str, int] | None = None) -> pd.Series` — generic=round; multiple_map[item]=N이면 N배수로 floor.
  - `class ArtiseeBaseline` — `__init__(self, *, weeks=3, curve_months=3, rounding="generic", multiple_map=None)`, `fit(self, daily: pd.DataFrame, hourly: pd.DataFrame) -> ArtiseeBaseline`, `predict(self, target: pd.DataFrame) -> pd.Series` (제시량, target.index 정렬).

- [ ] **Step 1: Write the failing test**

```python
from bakery.models.artisee_baseline import ArtiseeBaseline, round_order


def test_round_order_generic_and_multiple():
    raw = pd.Series([12.4, 12.6, 13.0])
    items = pd.Series(["A", "A", "B"])
    generic = round_order(raw, items, rounding="generic")
    assert list(generic) == [12.0, 13.0, 13.0]
    mult = round_order(raw, items, rounding="multiple", multiple_map={"A": 3, "B": 6})
    # A=3배수 floor: 12.4→12, 12.6→12; B=6배수 floor: 13→12.
    assert list(mult) == [12.0, 12.0, 12.0]


def _make_history():
    daily_rows, hourly_rows = [], []
    for d in pd.date_range("2026-06-01", "2026-06-21"):
        sold = 20 if d.dayofweek >= 5 else 10
        so = (d.dayofweek < 5) and (d.day % 5 == 0)
        daily_rows.append({"store_id": "S", "item_id": "A", "date": d,
                           "sold_units": sold, "is_stockout": so, "is_holiday": False,
                           "stockout_time": (d + pd.Timedelta(hours=12)) if so else pd.NaT})
        hourly_rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 7, "qty": 6.0})
        hourly_rows.append({"store_id": "S", "item_id": "A", "date": d, "hour": 12, "qty": 4.0})
    daily = pd.DataFrame(daily_rows); daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.DataFrame(hourly_rows); hourly["date"] = pd.to_datetime(hourly["date"])
    return daily, hourly


def test_artisee_baseline_predict_positive_order():
    daily, hourly = _make_history()
    model = ArtiseeBaseline(weeks=3, curve_months=3).fit(daily, hourly)
    target = pd.DataFrame({"store_id": ["S", "S"], "item_id": ["A", "A"],
                           "date": pd.to_datetime(["2026-06-22", "2026-06-27"])})  # 월, 토
    pred = model.predict(target)
    assert (pred.to_numpy() > 0).all()
    assert pred.index.equals(target.index)
    # 주말(토) 제시량 > 주중(월) — weekend base 20 > weekday base 10.
    assert pred.iloc[1] > pred.iloc[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_artisee_baseline.py -k "round_order or predict_positive" -v`
Expected: FAIL (ImportError: ArtiseeBaseline / round_order).

- [ ] **Step 3: Write minimal implementation**

```python
def round_order(raw: pd.Series, item_ids: pd.Series, *, rounding: str = "generic",
                multiple_map: dict[str, int] | None = None) -> pd.Series:
    if rounding == "generic" or not multiple_map:
        return raw.round()
    ns = item_ids.map(lambda i: multiple_map.get(str(i), 1)).astype(float)
    floored = np.floor(raw / ns) * ns
    return pd.Series(floored.to_numpy(), index=raw.index)


class ArtiseeBaseline:
    name = "artisee_baseline"

    def __init__(self, *, weeks: int = 3, curve_months: int = 3,
                 rounding: str = "generic", multiple_map: dict[str, int] | None = None):
        self.weeks = weeks
        self.curve_months = curve_months
        self.rounding = rounding
        self.multiple_map = multiple_map
        self._base = self._mult = self._dow = None

    def fit(self, daily: pd.DataFrame, hourly: pd.DataFrame) -> "ArtiseeBaseline":
        curves = build_item_residual_curve(hourly, months=self.curve_months)
        self._base = applied_quantity(daily, weeks=self.weeks)
        self._mult = soldout_multiplier(daily, curves, weeks=self.weeks)
        self._dow = dow_scaling(daily, weeks=self.weeks)
        return self

    def predict(self, target: pd.DataFrame) -> pd.Series:
        if self._base is None:
            raise RuntimeError("call fit() before predict()")
        out = target.copy()
        out["dow"] = pd.to_datetime(out["date"]).dt.dayofweek
        out["dow_group"] = dow_group(out["date"])
        keys = ["store_id", "item_id", "dow_group"]
        merged = (out.merge(self._base, on=keys, how="left")
                     .merge(self._mult, on=keys, how="left")
                     .merge(self._dow, on=["store_id", "item_id", "dow"], how="left"))
        merged["base_qty"] = merged["base_qty"].fillna(0.0)
        merged["multiplier"] = merged["multiplier"].fillna(1.0)
        merged["weight"] = merged["weight"].fillna(1.0)
        raw = merged["base_qty"] * merged["multiplier"] * merged["weight"]
        order = round_order(raw, merged["item_id"], rounding=self.rounding,
                            multiple_map=self.multiple_map)
        return pd.Series(order.to_numpy(), index=target.index, name="artisee_order")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_artisee_baseline.py -v`
Expected: PASS (전체).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/artisee_baseline.py tests/test_artisee_baseline.py
git commit -m "feat(artisee): C4 반올림 + ArtiseeBaseline fit/predict 조립"
```

---

### Task 6: Leakage 가드 테스트

**Files:**
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `ArtiseeBaseline` (Task 5).
- Produces: (테스트만) — predict가 target date 이후 데이터에 영향받지 않음 검증.

- [ ] **Step 1: Write the failing test**

```python
def test_predict_ignores_future_data():
    daily, hourly = _make_history()
    target = pd.DataFrame({"store_id": ["S"], "item_id": ["A"],
                           "date": pd.to_datetime(["2026-06-22"])})
    base = ArtiseeBaseline().fit(daily, hourly).predict(target)
    # cutoff 이후(미래) 폭발적 수요를 history에 추가해도 fit은 max(date) 기준 3주만 봄.
    future = daily.copy()
    extra = daily.tail(1).copy()
    extra["date"] = pd.to_datetime(["2026-07-15"]); extra["sold_units"] = 9999
    future = pd.concat([future, extra], ignore_index=True)
    # 미래를 넣으면 window 기준이 옮겨가므로, 이 테스트는 "window 밖 과거"를 검증:
    old = daily.copy()
    old_extra = daily.head(1).copy()
    old_extra["date"] = pd.to_datetime(["2026-01-01"]); old_extra["sold_units"] = 9999
    old = pd.concat([old_extra, old], ignore_index=True)
    with_old = ArtiseeBaseline().fit(old, hourly).predict(target)
    # 2026-01-01은 3주 창(06-01 이전) 밖 → 예측 불변.
    assert with_old.iloc[0] == base.iloc[0]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_artisee_baseline.py::test_predict_ignores_future_data -v`
Expected: PASS (구현이 `_recent`로 window 제한하므로 통과해야 함). 만약 FAIL이면 window 로직 버그 — `_recent` 확인.

- [ ] **Step 3: (필요 시) 수정**

`_recent`가 `daily["date"].max()` 기준 `weeks`주만 남기는지 확인. 이미 통과면 skip.

- [ ] **Step 4: Commit**

```bash
git add tests/test_artisee_baseline.py
git commit -m "test(artisee): leakage 가드 — window 밖 과거 데이터 예측 불변"
```

---

### Task 7: prospective 통합 스모크 + CLI 배선

**Files:**
- Modify: `src/bakery/cli.py` (prospective-eval baseline 소스 옵션)
- Test: `tests/test_artisee_baseline.py`

**Interfaces:**
- Consumes: `ArtiseeBaseline` (Task 5), `evaluation.prospective.simulate_item_day_kpis`/`compare_policies`, `features.potential_demand.StoreHours`.
- Produces: `compare_policies`가 우리 발주 KPI vs ArtiseeBaseline 제시량 KPI Δ를 산출하는 통합 스모크. CLI `prospective-eval`에 `--baseline artisee` 옵션(기본 유지 reconstruct proxy).

- [ ] **Step 1: Write the failing integration test**

```python
from bakery.evaluation.prospective import simulate_item_day_kpis, compare_policies
from bakery.features.potential_demand import StoreHours


def test_artisee_order_feeds_prospective_compare():
    daily, hourly = _make_history()
    model = ArtiseeBaseline().fit(daily, hourly)
    rows = pd.DataFrame({
        "store_id": ["S", "S"], "item_id": ["A", "A"],
        "date": pd.to_datetime(["2026-06-22", "2026-06-23"]),
        "potential_demand": [10.0, 11.0], "our_order": [12.0, 12.0],
    })
    rows["artisee_order"] = model.predict(rows).to_numpy()
    hours = StoreHours(open_hour=7, close_hour=22)
    profiles: dict[tuple, np.ndarray] = {}
    ours = simulate_item_day_kpis(rows, profiles, order_col="our_order",
                                  store_hours=hours, group_cols=["item_id"],
                                  demand_col="potential_demand")
    theirs = simulate_item_day_kpis(rows, profiles, order_col="artisee_order",
                                    store_hours=hours, group_cols=["item_id"],
                                    demand_col="potential_demand")
    cmp = compare_policies(ours, theirs)
    assert set(cmp["policy"]) == {"our", "baseline", "delta"}
    assert "waste_cost_krw" in cmp.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_artisee_baseline.py::test_artisee_order_feeds_prospective_compare -v`
Expected: 먼저 `compare_policies` 반환 policy 라벨 확인 필요 — 실패 시 `_summarize_policy`/`compare_policies` 실제 라벨("our"/"baseline"/"delta")에 맞춰 단언 수정(코드 변경 아님, 테스트 정합).

- [ ] **Step 3: CLI 배선**

`src/bakery/cli.py`의 `prospective-eval` 커맨드에 `--baseline` 옵션 추가(`typer.Option("proxy")`). `artisee` 선택 시 `_real_prospective_inputs`(또는 synthetic) 산출 history로 `ArtiseeBaseline().fit(...).predict(...)`을 baseline order로 사용. 기본값 `proxy`는 기존 `reconstruct_baseline_order` 유지. import 추가:
```python
from .models.artisee_baseline import ArtiseeBaseline
```
(정확한 배선 지점 = `_load_prospective_inputs`/`prospective_eval` 함수 본문. history/hourly 컬럼 매핑은 `_real_prospective_inputs`가 내보내는 스키마에 맞춘다.)

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/test_artisee_baseline.py -v` 및 `uv run pytest`
Expected: 신규 테스트 PASS, 기존 테스트 무회귀 (전체 pass 카운트 확인, `--color=no`).

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/artisee_baseline.py src/bakery/cli.py tests/test_artisee_baseline.py
git commit -m "feat(artisee): prospective KPI 비교 통합 + CLI --baseline artisee 옵션"
```

---

## Self-Review

**1. Spec coverage:**
- §5 C1 적용수량 → Task 1 ✓ / C2a 곡선 → Task 2 ✓ / C2b 증산배수 → Task 3 ✓ / C3 요일 → Task 4 ✓ / C4 반올림 → Task 5 ✓
- §4.2 fit/predict 덕타이핑(ABC 미상속) → Task 5 클래스 ✓
- §6 prospective 통합 + CLI → Task 7 ✓
- §7 leakage 안전 → Task 6 ✓ + Global Constraints
- §8 범위(광교 브레드) → 데이터 로딩은 CLI `_real_prospective_inputs` 재사용(Task 7). 단위 테스트는 합성 데이터.
- §9 배수 N 미수령 → `multiple_map` 주입구(Task 5 round_order) ✓

**2. Placeholder scan:** Task 7 Step 3의 CLI 배선은 "정확한 지점"을 함수명으로 특정하되 실제 배선 코드는 `_real_prospective_inputs` 스키마 확인 후 작성 — 이 한 곳만 탐색 의존(합성 경로 스모크는 완결). 그 외 placeholder 없음.

**3. Type consistency:** `applied_quantity`→`base_qty`, `soldout_multiplier`→`multiplier`, `dow_scaling`→`weight`, predict merge 키(`store_id,item_id,dow_group` / `...,dow`) 전 태스크 일치. `round_order(raw, item_ids, ...)` 시그니처 Task 5 정의·사용 일치. `build_item_residual_curve` 반환 `dict[str,ndarray]` Task 2 정의·Task 3/5 소비 일치.

**주의(구현자):** Task 7 Step 2에서 `compare_policies` 반환 라벨을 실제 코드(`prospective.py:140` 부근)로 확인 후 단언 맞출 것. `_make_history`/`_daily` 헬퍼는 Task 5에서 최초 정의되므로 Task 1~4 테스트는 인라인 `_daily`만 사용(순서 무관하게 동작하도록 각 테스트가 자체 프레임 구성).

# 특수일 레벨-앵커 Prior 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 트리가 못 잡는 sharp rare 이벤트(xmas)의 반복 고유 레벨을, 예측 이후 leakage-safe 블렌드로 주입하는 `EventLevelPrior` 모듈을 만들고 4매장 리포트 백테스트에 배선한다.

**Architecture:** 신규 `src/bakery/models/event_prior.py`에 post-model 블렌드 컴포넌트를 만든다. 기존 `fit_category_total`/`CategoryTotalModel` 시그니처는 불변. 리포트 `windowed_backtest`가 fold마다 pre-test history로 prior를 fit하고 예측값을 blend한다. 곱셈 보정으로 production 버퍼 비율을 보존한다.

**Tech Stack:** Python, pandas, numpy, pytest, LightGBM (기존).

## Global Constraints

- Time leakage 금지: prior는 예측 date보다 **엄격히 이전**(`date < D`) 이벤트만 사용. 신규 `test_event_prior_leakage.py`가 통과해야 함.
- 테스트 단언: 기대값 아는 것은 정확값 비교(`==` 또는 `pytest.approx`). truthy/부분문자열 금지.
- 기존 시그니처/호출부 변경 금지 (opt-in 합성만).
- 타깃 컬럼명: 리포트는 `TARGET = "adjusted_demand_unit"`. prior는 `target_col` 인자로 받아 하드코딩 금지.
- pytest 실행: `uv run pytest` (repo addopts에 `-q` 있음 — 추가 `-q` 금지).

---

### Task 1: `EventLevelPrior` 코어 (fit / level_for)

**Files:**
- Create: `src/bakery/models/event_prior.py`
- Test: `tests/test_event_prior.py`

**Interfaces:**
- Produces:
  - `EventLevelPrior(events: dict[str, tuple[int, int]] = {"xmas": (12, 25)}, k: float = 1.5)`
  - `.fit(history: pd.DataFrame, date_col: str = "date", target_col: str = "adjusted_demand_unit") -> "EventLevelPrior"` (self 반환)
  - `.is_event_day(date: pd.Timestamp) -> bool`
  - `.level_for(date: pd.Timestamp) -> tuple[float | None, int]` — (prior_level, n_past). date보다 엄격히 이전 이벤트 actual 평균. n_past==0 → (None, 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_prior.py
import numpy as np
import pandas as pd
import pytest

from bakery.models.event_prior import EventLevelPrior

TARGET = "adjusted_demand_unit"


def _daily(start="2021-06-01", periods=1800, value=100.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    df = pd.DataFrame({"date": dates, TARGET: float(value)})
    # xmas 마다 뚜렷한 레벨 심기: 2021=300, 2022=310, 2023=320, 2024=330
    for yr, lvl in {2021: 300.0, 2022: 310.0, 2023: 320.0, 2024: 330.0}.items():
        df.loc[df["date"] == pd.Timestamp(yr, 12, 25), TARGET] = lvl
    return df


def test_is_event_day():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    assert p.is_event_day(pd.Timestamp(2023, 12, 25)) is True
    assert p.is_event_day(pd.Timestamp(2023, 12, 24)) is False


def test_level_for_averages_strictly_past_events():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    # 2024-12-25 예측 → 과거 3개(300,310,320) 평균 = 310, n_past=3
    level, n_past = p.level_for(pd.Timestamp(2024, 12, 25))
    assert n_past == 3
    assert level == pytest.approx(310.0)


def test_level_for_first_occurrence_returns_none():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    level, n_past = p.level_for(pd.Timestamp(2021, 12, 25))
    assert level is None
    assert n_past == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_prior.py -v`
Expected: FAIL — `ModuleNotFoundError: bakery.models.event_prior`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/models/event_prior.py
"""Post-model level-anchor prior for sharp rare calendar events (xmas 등).

트리가 학습창당 2~5샘플의 sharp 이벤트를 못 잡는 문제를, 예측 이후
leakage-safe 레벨-앵커 블렌드로 보정한다. 자세한 배경은
docs/superpowers/specs/2026-07-12-event-level-prior-design.md 참조.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EVENTS: dict[str, tuple[int, int]] = {"xmas": (12, 25)}
DEFAULT_K = 1.5


class EventLevelPrior:
    def __init__(self, events: dict[str, tuple[int, int]] | None = None, k: float = DEFAULT_K):
        self.events = dict(events) if events is not None else dict(DEFAULT_EVENTS)
        self.k = k
        self._event_actuals: list[tuple[pd.Timestamp, float]] = []  # (date, actual)

    def fit(self, history: pd.DataFrame, date_col: str = "date",
            target_col: str = "adjusted_demand_unit") -> "EventLevelPrior":
        d = history[[date_col, target_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col])
        mask = d[date_col].apply(self.is_event_day)
        ev = d[mask].dropna(subset=[target_col])
        self._event_actuals = sorted(
            (row[date_col], float(row[target_col])) for _, row in ev.iterrows()
        )
        return self

    def is_event_day(self, date: pd.Timestamp) -> bool:
        date = pd.Timestamp(date)
        return any((date.month, date.day) == (m, day) for m, day in self.events.values())

    def level_for(self, date: pd.Timestamp) -> tuple[float | None, int]:
        date = pd.Timestamp(date)
        past = [a for (ed, a) in self._event_actuals if ed < date]
        if not past:
            return None, 0
        return float(np.mean(past)), len(past)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_prior.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/event_prior.py tests/test_event_prior.py
git commit -m "feat(event_prior): EventLevelPrior 코어 fit/level_for (xmas 레벨-앵커)"
```

---

### Task 2: `blend` — expected/production 곱셈 보정

**Files:**
- Modify: `src/bakery/models/event_prior.py`
- Test: `tests/test_event_prior.py`

**Interfaces:**
- Consumes: Task 1의 `EventLevelPrior`, `.level_for`, `.is_event_day`.
- Produces:
  - `.blend(dates, base_expected, base_production) -> tuple[np.ndarray, np.ndarray]`
    - `dates`: array-like of Timestamp/datetime; `base_expected`, `base_production`: array-like float.
    - 이벤트일만 교정, 비이벤트일·n_past==0·base_exp<=0 → 원값 유지.
    - `shrink = n_past/(n_past+k)`, `blended_exp = shrink*prior + (1-shrink)*base_exp`,
      `correction = blended_exp/base_exp`, `blended_prod = base_prod*correction`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_prior.py 에 append
def test_blend_corrects_only_event_day():
    p = EventLevelPrior(k=1.5).fit(_daily(), target_col=TARGET)
    dates = [pd.Timestamp(2024, 12, 24), pd.Timestamp(2024, 12, 25)]  # non-event, event
    base_exp = np.array([200.0, 240.0])
    base_prod = np.array([230.0, 276.0])  # buffer 1.15x
    exp2, prod2 = p.blend(dates, base_exp, base_prod)
    # non-event day unchanged
    assert exp2[0] == pytest.approx(200.0)
    assert prod2[0] == pytest.approx(230.0)
    # event day: prior=310 (n_past=3), shrink=3/4.5=0.6667
    shrink = 3 / (3 + 1.5)
    expected_exp = shrink * 310.0 + (1 - shrink) * 240.0
    assert exp2[1] == pytest.approx(expected_exp)
    # production keeps buffer ratio: correction = expected_exp/240
    correction = expected_exp / 240.0
    assert prod2[1] == pytest.approx(276.0 * correction)


def test_blend_first_occurrence_unchanged():
    p = EventLevelPrior().fit(_daily(), target_col=TARGET)
    dates = [pd.Timestamp(2021, 12, 25)]  # n_past=0
    exp2, prod2 = p.blend(dates, np.array([250.0]), np.array([280.0]))
    assert exp2[0] == pytest.approx(250.0)
    assert prod2[0] == pytest.approx(280.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_prior.py::test_blend_corrects_only_event_day -v`
Expected: FAIL — `AttributeError: 'EventLevelPrior' object has no attribute 'blend'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/models/event_prior.py 에 EventLevelPrior 메서드 추가
    def blend(self, dates, base_expected, base_production):
        dates = [pd.Timestamp(d) for d in dates]
        exp = np.asarray(base_expected, dtype=float).copy()
        prod = np.asarray(base_production, dtype=float).copy()
        for i, d in enumerate(dates):
            if not self.is_event_day(d):
                continue
            prior, n_past = self.level_for(d)
            if prior is None or exp[i] <= 0:
                continue
            shrink = n_past / (n_past + self.k)
            blended_exp = shrink * prior + (1 - shrink) * exp[i]
            correction = blended_exp / exp[i]
            exp[i] = blended_exp
            prod[i] = prod[i] * correction
        return exp, prod
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_event_prior.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/event_prior.py tests/test_event_prior.py
git commit -m "feat(event_prior): blend — expected/production 곱셈 보정(버퍼 보존)"
```

---

### Task 3: 전용 leakage 테스트

**Files:**
- Create: `tests/test_event_prior_leakage.py`

**Interfaces:**
- Consumes: Task 1·2의 `EventLevelPrior`.

- [ ] **Step 1: Write the failing test (실제로는 PASS 기대 — 회귀 가드)**

```python
# tests/test_event_prior_leakage.py
import numpy as np
import pandas as pd
import pytest

from bakery.models.event_prior import EventLevelPrior

TARGET = "adjusted_demand_unit"


def _daily_with_future(future_xmas_value: float) -> pd.DataFrame:
    dates = pd.date_range("2021-06-01", periods=1400, freq="D")
    df = pd.DataFrame({"date": dates, TARGET: 100.0})
    for yr, lvl in {2021: 300.0, 2022: 310.0, 2023: 320.0}.items():
        df.loc[df["date"] == pd.Timestamp(yr, 12, 25), TARGET] = lvl
    # 미래 이벤트(2024-12-25 이후는 데이터 없음이지만, 만약 오염되면 값이 바뀌게)
    df.loc[df["date"] == pd.Timestamp(2023, 12, 25), TARGET] = future_xmas_value
    return df


def test_level_for_ignores_future_events():
    # 예측 date 2023-12-25: 과거는 2021(300),2022(310)만 써야 함 → 2023 자기값 무관
    p_a = EventLevelPrior().fit(_daily_with_future(320.0), target_col=TARGET)
    p_b = EventLevelPrior().fit(_daily_with_future(999.0), target_col=TARGET)
    lvl_a, n_a = p_a.level_for(pd.Timestamp(2023, 12, 25))
    lvl_b, n_b = p_b.level_for(pd.Timestamp(2023, 12, 25))
    assert n_a == 2 and n_b == 2
    assert lvl_a == pytest.approx(305.0)  # (300+310)/2
    assert lvl_b == pytest.approx(305.0)  # 미래 자기값 오염 없음


def test_blend_at_date_unaffected_by_same_or_future_data():
    p = EventLevelPrior().fit(_daily_with_future(320.0), target_col=TARGET)
    exp2, _ = p.blend([pd.Timestamp(2023, 12, 25)], np.array([200.0]), np.array([230.0]))
    # prior=305, n_past=2, shrink=2/3.5
    shrink = 2 / (2 + 1.5)
    assert exp2[0] == pytest.approx(shrink * 305.0 + (1 - shrink) * 200.0)
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_event_prior_leakage.py -v`
Expected: PASS (2 tests) — `level_for`가 `date < D` 필터로 미래 불변.

- [ ] **Step 3: Commit**

```bash
git add tests/test_event_prior_leakage.py
git commit -m "test(event_prior): leakage 가드 — prior가 미래 이벤트에 불변"
```

---

### Task 4: 리포트 `windowed_backtest` 배선

**Files:**
- Modify: `scripts/store_predictive_power.py` (import 추가 + `windowed_backtest` 내부 blend 삽입)
- Test: 수동 재측정 (백테스트 실행)

**Interfaces:**
- Consumes: Task 1·2의 `EventLevelPrior`.
- 삽입 지점: `scripts/store_predictive_power.py:124-125` (`exp_pred`/`prod_pred` 계산 직후).

- [ ] **Step 1: import 추가**

`scripts/store_predictive_power.py` 상단 import 블록(`from bakery.models.category_total import ...` 근처)에 추가:

```python
from bakery.models.event_prior import EventLevelPrior
```

- [ ] **Step 2: blend 삽입**

`windowed_backtest` 안, 현재:

```python
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        actual = test_df[target_col].values
```

를 아래로 교체 (train_df는 rolling 730이지만 prior는 pre-test 전체 history 사용 → 더 많은 과거 이벤트 확보, leakage-safe):

```python
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        # 특수일 레벨-앵커 prior: pre-test 전체 history로 fit (train window보다 길게, leakage-safe)
        hist = df[df["date"] < test_start_date]
        prior = EventLevelPrior().fit(hist, target_col=target_col)
        exp_pred, prod_pred = prior.blend(test_df["date"].values, exp_pred, prod_pred)
        actual = test_df[target_col].values
```

- [ ] **Step 3: 스모크 — import·실행 안 깨지는지**

Run: `cd /Users/taehoonkim/dev/bakery-predictor && PYTHONPATH=scripts uv run python -c "import store_predictive_power"`
Expected: 에러 없이 종료 (exit 0).

- [ ] **Step 4: 4매장 재측정 실행 (event-window 지표 확인)**

Run: `cd /Users/taehoonkim/dev/bakery-predictor && PYTHONPATH=scripts uv run python scripts/store_predictive_power.py`
(진입점 `main()` → `reports/store_predictive_power.html` 생성. 실행 로그의 fold WAPE로 확인.)
Expected: 에러 없이 완료. 광교 xmas event-day 오차가 시뮬(0.231→0.088) 방향으로 개선. 전체 WAPE는 거의 불변이어도 정상.

- [ ] **Step 5: Commit**

```bash
git add scripts/store_predictive_power.py
git commit -m "feat(report): windowed_backtest에 EventLevelPrior 배선 (xmas 레벨-앵커)"
```

---

### Task 5: 전체 회귀 + event-window 재측정 검증

**Files:**
- Create: `scripts/verify_event_prior.py` (event-window 지표 base vs blend 비교, scratchpad prior_sim 로직 정식화)

**Interfaces:**
- Consumes: Task 1·2 모듈, 리포트 빌드 함수(`build_store_daily`, `build_category_daily`, `build_features`).

- [ ] **Step 1: 전체 테스트 실행**

Run: `uv run pytest`
Expected: 신규 7 테스트 포함 전체 PASS, 기존 회귀 0.

- [ ] **Step 2: event-window 검증 스크립트 작성·실행**

4매장 × xmas에 대해 leave-past-out(expanding) base WAPE/bias vs blend WAPE/bias 출력. (scratchpad `prior_sim.py`를 `scripts/`로 정식화, k=1.5.)
Expected: 4매장 모두 xmas event-day WAPE 개선 (광교 0.231→~0.09, 삼성 over 완화, 메세나 0.174→~0.11, 광화문 0.151→~0.06).

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_event_prior.py
git commit -m "test(event_prior): 4매장 xmas event-window 재측정 검증 스크립트"
```

---

## Self-Review

**Spec coverage:**
- §3.1 컴포넌트 → Task 1·2 ✓
- §3.2 블렌드 공식 → Task 2 ✓
- §3.3 통합(windowed_backtest) → Task 4 ✓ (CLU 배선은 spec §7에서 후속으로 명시 = 스코프 밖)
- §3.4 leakage 안전성 + 전용 테스트 → Task 3 ✓
- §4 테스트 → Task 1·2·3 ✓
- §5 성공 기준 → Task 5 ✓

**Placeholder scan:** 없음. 모든 코드/명령/기대값 명시.

**Type consistency:** `level_for -> (float|None, int)`, `blend -> (ndarray, ndarray)`, `fit -> self`. Task 1·2·3·4 전부 동일 시그니처 사용. `target_col` 인자 일관.

**해결됨:** Task 4 Step 4 실행 명령 확정 — `PYTHONPATH=scripts uv run python scripts/store_predictive_power.py` (`main()` → `reports/store_predictive_power.html`).

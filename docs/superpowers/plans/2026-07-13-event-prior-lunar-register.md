# EventLevelPrior 음력 지원 + 설/추석 등록 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `EventLevelPrior`에 음력 이벤트(설/추석) 지원을 추가하고, 리포트에 **광교 추석 · 메세나 설**을 등록한다(배포 코드 median+min_events=2 기준 순개선 확인된 것만). xmas 동작 불변.

**Architecture:** (1) `EventLevelPrior`에 `lunar_events` 파라미터(음력 date-map) 추가, `is_event_day`가 양력+음력 둘 다 매칭. (2) 리포트 `STORE_EVENT_PRIORS`를 매장별 `{events, lunar_events}`로 재구조화 + `windowed_backtest`에 `lunar_events` threading. (3) verify를 config 기반으로 재작성해 등록 이벤트 순개선 확인 + HTML 재생성. 설계 = docs/superpowers/specs/2026-07-12-event-level-prior-design.md §7·§8.

**Tech Stack:** Python, pandas, numpy, pytest. 음력 날짜 = `bakery.data.calendar.LUNAR_EVENT_DATES` ({"days_to_chuseok": {year:date}, "days_to_seollal": {year:date}}, 2028까지).

## Global Constraints

- Time leakage 금지: 음력도 `level_for`의 `ed < date` 필터로 과거만. 음력 date-map은 사전 확정 캘린더(누수 없음).
- 하위호환: `EventLevelPrior(lunar_events=None)` → 음력 없음(양력만, 기존 동작). `windowed_backtest`의 새 `lunar_events` 파라미터 기본 None.
- 등록은 **배포 코드(median+min_events=2) 순개선 확인된 것만**: 광교 추석(0.214→0.145), 메세나 설(0.179→0.101). 광화문 설=악화·메세나 추석=악화라 **미등록**.
- 테스트 단언 정확값. `uv run pytest`(추가 `-q` 금지).

---

### Task 1: EventLevelPrior 음력 지원

**Files:**
- Modify: `src/bakery/models/event_prior.py` (`__init__`, `is_event_day`)
- Modify: `tests/test_event_prior.py`

**Interfaces:**
- Produces: `EventLevelPrior(events=None, k=1.5, min_events=2, lunar_events: dict[str, dict[int, str]] | None = None)`. `is_event_day`가 양력 `(month,day)` 또는 음력 date-set 매칭.

- [ ] **Step 1: Write failing test (RED)**

`tests/test_event_prior.py`에 추가(파일 상단 import에 이미 있는 것 재사용):

```python
def _lunar_daily():
    # 추석 date-map (실제 LUNAR_EVENT_DATES 형식)으로 합성 시리즈
    chuseok = {2021: "2021-09-21", 2022: "2022-09-10", 2023: "2023-09-29", 2024: "2024-09-17"}
    dates = pd.date_range("2021-06-01", periods=1400, freq="D")
    df = pd.DataFrame({"date": dates, TARGET: 100.0})
    for yr, lvl in {2021: 200.0, 2022: 210.0, 2023: 220.0}.items():
        df.loc[df["date"] == pd.Timestamp(chuseok[yr]), TARGET] = lvl
    return df, {"chuseok": chuseok}


def test_is_event_day_matches_lunar_date():
    _, lunar = _lunar_daily()
    p = EventLevelPrior(lunar_events=lunar)
    assert p.is_event_day(pd.Timestamp(2023, 9, 29)) is True   # 추석 당일
    assert p.is_event_day(pd.Timestamp(2023, 9, 28)) is False  # 전날


def test_level_for_lunar_uses_past_lunar_actuals():
    df, lunar = _lunar_daily()
    p = EventLevelPrior(lunar_events=lunar).fit(df, target_col=TARGET)
    # 2024 추석(2024-09-17) 예측 → 과거 3개(200,210,220) median=210
    level, n_past = p.level_for(pd.Timestamp(2024, 9, 17))
    assert n_past == 3
    assert level == pytest.approx(210.0)
```

- [ ] **Step 2: RED 확인**

Run: `uv run pytest tests/test_event_prior.py::test_is_event_day_matches_lunar_date -v`
Expected: FAIL — 현재 lunar_events 파라미터 없음(TypeError) 또는 매칭 안 됨.

- [ ] **Step 3: 음력 지원 구현**

`src/bakery/models/event_prior.py` `__init__`:
```python
    def __init__(self, events: dict[str, tuple[int, int]] | None = None,
                 k: float = DEFAULT_K, min_events: int = 2,
                 lunar_events: dict[str, dict[int, str]] | None = None):
        self.events = dict(events) if events is not None else dict(DEFAULT_EVENTS)
        self.k = k
        self.min_events = min_events
        self.lunar_events = dict(lunar_events) if lunar_events else {}
        self._lunar_dates: set[pd.Timestamp] = set()
        for datemap in self.lunar_events.values():
            for date_str in datemap.values():
                self._lunar_dates.add(pd.Timestamp(date_str).normalize())
        self._event_actuals: list[tuple[pd.Timestamp, float]] = []
```

`is_event_day` (양력 매칭 뒤 음력 date-set 확인):
```python
    def is_event_day(self, date: pd.Timestamp) -> bool:
        date = pd.Timestamp(date)
        if any((date.month, date.day) == (m, day) for m, day in self.events.values()):
            return True
        return date.normalize() in self._lunar_dates
```

(`fit`/`level_for`/`blend`은 `is_event_day` 기반이라 변경 불필요.)

- [ ] **Step 4: GREEN + 무회귀**

Run: `uv run pytest tests/test_event_prior.py tests/test_event_prior_leakage.py -v`
Expected: 전부 PASS(신규 2 포함). 기존 xmas 테스트는 lunar_events 기본 {}라 불변.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/event_prior.py tests/test_event_prior.py
git commit -m "feat(event_prior): 음력 이벤트(설/추석) 지원 — lunar_events date-map"
```

---

### Task 2: 리포트 설/추석 등록 + lunar_events 배선

**Files:**
- Modify: `scripts/store_predictive_power.py`

**Interfaces:**
- Consumes: Task 1의 `EventLevelPrior(lunar_events=...)`.
- Produces: `windowed_backtest(..., events=None, lunar_events=None)`; `STORE_EVENT_PRIORS[label] = {"events":..., "lunar_events":...}`.

- [ ] **Step 1: import + config 재구조화**

`scripts/store_predictive_power.py` import 블록에 추가:
```python
from bakery.data.calendar import LUNAR_EVENT_DATES
```

기존 `STORE_EVENT_PRIORS`(현재 `{label: dict(XMAS)}`)를 아래로 교체:
```python
XMAS = {"xmas": (12, 25)}
SEOLLAL = {"seollal": LUNAR_EVENT_DATES["days_to_seollal"]}
CHUSEOK = {"chuseok": LUNAR_EVENT_DATES["days_to_chuseok"]}
# 매장×이벤트 opt-in. 배포 코드(median+min_events=2) OOS 순개선 확인된 것만 등록:
#   광교 추석 0.214→0.145, 메세나 설 0.179→0.101. (광화문 설·메세나 추석=악화라 미등록)
STORE_EVENT_PRIORS: dict[str, dict[str, dict]] = {
    "광교":       {"events": dict(XMAS), "lunar_events": dict(CHUSEOK)},
    "삼성타운":   {"events": dict(XMAS), "lunar_events": {}},
    "메세나폴리스": {"events": dict(XMAS), "lunar_events": dict(SEOLLAL)},
    "광화문":     {"events": dict(XMAS), "lunar_events": {}},
}
```

- [ ] **Step 2: windowed_backtest에 lunar_events 파라미터**

`windowed_backtest` 시그니처의 `events` 파라미터 옆에 keyword-only `lunar_events: dict | None = None` 추가. 내부 prior 생성:
```python
        prior = EventLevelPrior(events=events, lunar_events=lunar_events).fit(hist, target_col=target_col)
```

- [ ] **Step 3: 4 call site threading (events + lunar_events 둘 다)**

각 call site에서 `cfg = STORE_EVENT_PRIORS.get(sd.label, {})`로 꺼내 둘 다 전달. run_store 3곳:
```python
    cfg = STORE_EVENT_PRIORS.get(sd.label, {})
    main = windowed_backtest(sd.feat, window_days=DEFAULT_WINDOW_DAYS, n_folds=n_folds,
                             events=cfg.get("events"), lunar_events=cfg.get("lunar_events"))
```
(sensitivity 루프·variant도 동일하게 `events=cfg.get("events"), lunar_events=cfg.get("lunar_events")`. `cfg`는 run_store 함수 시작부에서 한 번만 꺼내 재사용.)

`run_compute`의 anchor(광교):
```python
    gw_cfg = STORE_EVENT_PRIORS.get("광교", {})
    ares = windowed_backtest(gw.feat, window_days=1825, n_folds=26,
                             events=gw_cfg.get("events"), lunar_events=gw_cfg.get("lunar_events"))
```

- [ ] **Step 4: 스모크**

Run: `PYTHONPATH=scripts uv run --with matplotlib python -c "import store_predictive_power"`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/store_predictive_power.py
git commit -m "feat(report): 광교 추석·메세나 설 등록 + lunar_events 배선"
```

---

### Task 3: verify config화 + 재측정 + 전체 회귀

**Files:**
- Modify: `scripts/verify_event_prior.py` (config 기반 per-store 등록 이벤트 검증)

**Interfaces:**
- Consumes: Task 1·2.

- [ ] **Step 1: verify를 STORE_EVENT_PRIORS 기반으로 확장**

`scripts/verify_event_prior.py`가 각 매장의 `STORE_EVENT_PRIORS[label]`로 `EventLevelPrior(events=..., lunar_events=...)`를 구성하고, **등록된 각 이벤트**(xmas + 등록된 음력)에 대해 base-vs-blend WAPE(leave-past-out)를 출력하도록 수정. import: `from store_predictive_power import STORE_EVENT_PRIORS`. 각 이벤트일 = `EventLevelPrior.is_event_day`가 True인 feat의 날짜. blend는 실제 `EventLevelPrior.blend` 호출(재구현 금지).

- [ ] **Step 2: verify 실행 — 등록 이벤트 순개선 확인**

Run: `PYTHONPATH=scripts uv run python scripts/verify_event_prior.py`
Expected: 등록 이벤트 전부 blend ≤ base. 특히 광교 추석 ~0.145(base 0.214), 메세나 설 ~0.101(base 0.179), xmas 4매장 유지. 등록 이벤트 중 blend>base면 STOP·보고.

- [ ] **Step 3: 전체 테스트**

Run: `uv run pytest`
Expected: 전체 PASS(1 pre-existing 네트워크 무관), 회귀 0.

- [ ] **Step 4: HTML 재생성**

Run: `PYTHONPATH=scripts uv run --with matplotlib python scripts/store_predictive_power.py`
Expected: exit 0, `reports/store_predictive_power.html` 갱신(광교 추석·메세나 설 반영).

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_event_prior.py
git commit -m "test(event_prior): verify를 STORE_EVENT_PRIORS 기반 per-store 검증으로 확장"
```

---

## Self-Review

**Spec coverage:** §7 음력 date-map → Task 1 ✓. §8 opt-in(설/추석 등록) → Task 2 ✓. 검증 → Task 3 ✓.
**Placeholder scan:** 없음.
**Type consistency:** `EventLevelPrior(events, k, min_events, lunar_events)`, `windowed_backtest(..., events=None, lunar_events=None)`, `STORE_EVENT_PRIORS[label]={"events","lunar_events"}`. Task 1·2·3 일관.
**하위호환:** `lunar_events=None`→음력 없음(기존). `windowed_backtest` 기존 호출 없음(전부 Task2서 threading). xmas 테스트 불변.
**등록 규율:** 배포 코드 순개선만(광교 추석·메세나 설). 미등록=광화문 설·메세나 추석(악화), 약한 것(광교 설·광화문 추석)은 이번 미등록.

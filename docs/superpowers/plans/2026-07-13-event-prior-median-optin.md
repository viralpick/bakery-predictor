# EventLevelPrior v2 (median + opt-in + min_events) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `EventLevelPrior`를 median 요약통계 + `min_events` 단일샘플 가드로 강화하고, 리포트에 매장×이벤트 opt-in config(`STORE_EVENT_PRIORS`)를 배선한다. 기본 동작은 xmas 전매장 유지.

**Architecture:** 3개 국소 변경. (a) `level_for`의 `np.mean`→`np.median`. (c) `EventLevelPrior.__init__`에 `min_events`, `blend`에 가드. (b) `scripts/store_predictive_power.py`에 `STORE_EVENT_PRIORS` config + `windowed_backtest(events=...)` 파라미터 threading(4 call site, 전부 `run_store` 안 `sd.label` 접근 가능). 설계 = docs/superpowers/specs/2026-07-12-event-level-prior-design.md §8.

**Tech Stack:** Python, pandas, numpy, pytest, LightGBM.

## Global Constraints

- Time leakage 금지: `level_for`는 `date < D` 이벤트만(불변). 기존 `tests/test_event_prior_leakage.py` 통과 유지.
- 기존 시그니처 하위호환: `EventLevelPrior(min_events=2)`는 새 기본값이나 기존 호출 `EventLevelPrior()`가 계속 동작해야 함. `windowed_backtest`의 새 `events` 파라미터는 기본 None(→ xmas).
- 테스트 단언: 정확값 비교(`==`/`pytest.approx`). truthy/부분문자열 금지.
- 테스트 명령: `uv run pytest` — repo addopts에 `-q` 있음, 추가 `-q` 금지(`-qq`로 요약 소실). clean 출력 필요시 `--color=no`.
- median 교체로 기존 event_prior 테스트의 기대값이 바뀔 수 있음 — 바뀌는 테스트는 정확값을 median 기준으로 갱신(마이그레이션).

---

### Task 1: median 요약통계 + 기존 테스트 마이그레이션

**Files:**
- Modify: `src/bakery/models/event_prior.py:42` (`level_for` 반환)
- Modify: `tests/test_event_prior.py` (mean 기대값 → median)
- Modify: `tests/test_event_prior_leakage.py` (mean 기대값 → median)

**Interfaces:**
- Produces: `level_for(date) -> (float|None, int)` — 이제 과거 이벤트 actual의 **median**(기존 mean).

- [ ] **Step 1: 기존 테스트를 median 기대값으로 갱신 (RED)**

`tests/test_event_prior.py`의 `_daily` fixture는 xmas 값 {2021:300, 2022:310, 2023:320, 2024:330}을 심는다. `test_level_for_averages_strictly_past_events`는 2024 예측 시 과거 3개(300,310,320)를 본다. mean=median=310으로 동일(홀수 3개). 하지만 이름이 mean 전제이므로 median임을 드러내고, median≠mean이 되는 케이스를 **추가**해 median을 고정한다.

`tests/test_event_prior.py`에 아래 테스트를 추가(기존 테스트는 mean=median=310이라 그대로 통과):

```python
def test_level_for_uses_median_not_mean():
    # 과거 4개 이벤트에 outlier 심어 median≠mean 구분
    df = _daily()
    # 2025 xmas 예측 시 과거 2021..2024 = {300,310,320,330} + outlier 하나로 교체
    df.loc[df["date"] == pd.Timestamp(2024, 12, 25), TARGET] = 900.0  # outlier
    p = EventLevelPrior().fit(df, target_col=TARGET)
    level, n_past = p.level_for(pd.Timestamp(2025, 12, 25))
    assert n_past == 4
    # median of [300,310,320,900] = (310+320)/2 = 315  (mean would be 457.5)
    assert level == pytest.approx(315.0)
```

- [ ] **Step 2: RED 확인**

Run: `uv run pytest tests/test_event_prior.py::test_level_for_uses_median_not_mean -v`
Expected: FAIL — 현재 mean 반환 457.5 ≠ 315.0.

- [ ] **Step 3: level_for를 median으로 교체**

`src/bakery/models/event_prior.py:42`:
```python
        return float(np.median(past)), len(past)
```
(주석/docstring에 "median"임을 반영: `level_for` 상단에 한 줄 주석 `# median: anomaly-robust (v2, design §8)`.)

- [ ] **Step 4: GREEN + 기존 event_prior 테스트 무회귀**

Run: `uv run pytest tests/test_event_prior.py tests/test_event_prior_leakage.py -v`
Expected: 전부 PASS. (기존 blend 테스트의 prior=310은 과거 3개(300,310,320) median=mean=310이라 불변; leakage 테스트의 prior=305는 과거 2개 median=mean=305라 불변 — 둘 다 안 깨짐. 만약 깨지면 그 기대값을 median으로 정정.)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/event_prior.py tests/test_event_prior.py
git commit -m "feat(event_prior): level_for 요약통계 mean→median (anomaly-robust, v2)"
```

---

### Task 2: min_events 단일샘플 가드

**Files:**
- Modify: `src/bakery/models/event_prior.py` (`__init__` + `blend`)
- Modify: `tests/test_event_prior.py`

**Interfaces:**
- Consumes: Task 1의 `level_for`.
- Produces: `EventLevelPrior(events=None, k=1.5, min_events=2)` — `blend`는 `n_past < min_events`면 해당 날짜 base 유지.

- [ ] **Step 1: Write failing test (RED)**

`tests/test_event_prior.py`에 추가:

```python
def test_blend_skips_when_below_min_events():
    # 2022 xmas 예측: 과거 1개(2021=300)뿐 → n_past=1 < min_events=2 → base 유지
    p = EventLevelPrior(min_events=2).fit(_daily(), target_col=TARGET)
    dates = [pd.Timestamp(2022, 12, 25)]
    exp2, prod2 = p.blend(dates, np.array([250.0]), np.array([280.0]))
    assert exp2[0] == pytest.approx(250.0)   # 단일샘플이라 미보정
    assert prod2[0] == pytest.approx(280.0)


def test_blend_applies_when_at_min_events():
    # 2023 xmas 예측: 과거 2개(300,310) median=305, n_past=2 == min_events → 보정
    p = EventLevelPrior(min_events=2).fit(_daily(), target_col=TARGET)
    exp2, _ = p.blend([pd.Timestamp(2023, 12, 25)], np.array([200.0]), np.array([230.0]))
    shrink = 2 / (2 + 1.5)
    assert exp2[0] == pytest.approx(shrink * 305.0 + (1 - shrink) * 200.0)
```

- [ ] **Step 2: RED 확인**

Run: `uv run pytest tests/test_event_prior.py::test_blend_skips_when_below_min_events -v`
Expected: FAIL — 현재 n_past=1도 blend(shrink=1/2.5=0.4) 되어 250→base와 다른 값.

- [ ] **Step 3: min_events 구현**

`src/bakery/models/event_prior.py` `__init__` 시그니처:
```python
    def __init__(self, events: dict[str, tuple[int, int]] | None = None,
                 k: float = DEFAULT_K, min_events: int = 2):
        self.events = dict(events) if events is not None else dict(DEFAULT_EVENTS)
        self.k = k
        self.min_events = min_events
        self._event_actuals: list[tuple[pd.Timestamp, float]] = []
```

`blend`의 가드(현재 `if prior is None or exp[i] <= 0:`):
```python
            prior, n_past = self.level_for(d)
            if prior is None or n_past < self.min_events or exp[i] <= 0:
                continue
```

- [ ] **Step 4: GREEN + 무회귀**

Run: `uv run pytest tests/test_event_prior.py tests/test_event_prior_leakage.py -v`
Expected: 전부 PASS. (leakage 테스트 `test_blend_at_date...`는 2023 예측 n_past=2 ≥ 2라 계속 보정, 불변.)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/event_prior.py tests/test_event_prior.py
git commit -m "feat(event_prior): min_events 가드 — 단일샘플 prior 차단(기본 2)"
```

---

### Task 3: 리포트 매장×이벤트 opt-in 배선

**Files:**
- Modify: `scripts/store_predictive_power.py` (`STORE_EVENT_PRIORS` config + `windowed_backtest` `events` 파라미터 + 4 call site threading)

**Interfaces:**
- Consumes: Task 1·2의 `EventLevelPrior`.
- Produces: `windowed_backtest(df, *, window_days, ..., events=None)` — `events`가 `EventLevelPrior(events=events)`로 전달(None → 기본 xmas).

- [ ] **Step 1: STORE_EVENT_PRIORS config 추가**

`scripts/store_predictive_power.py`의 STORES 정의 근처에 추가(현행 보존 = 전매장 xmas만; 설/추석/어린이날은 별도 등록):
```python
# 매장×이벤트 opt-in (verify_event_prior OOS 순개선 매장만 등록).
# 현행: xmas만(4매장 전부 개선 검증됨). 설/추석/어린이날은 per-event 검증 후 등록.
XMAS = {"xmas": (12, 25)}
STORE_EVENT_PRIORS: dict[str, dict[str, tuple[int, int]]] = {
    "광교": dict(XMAS),
    "삼성타운": dict(XMAS),
    "메세나폴리스": dict(XMAS),
    "광화문": dict(XMAS),
}
```

- [ ] **Step 2: windowed_backtest에 events 파라미터 추가**

`windowed_backtest` 시그니처에 keyword-only `events: dict | None = None` 추가. 내부 prior 생성(현재 `EventLevelPrior().fit(...)`):
```python
        prior = EventLevelPrior(events=events).fit(hist, target_col=target_col)
```

- [ ] **Step 3: run_store의 3 call site에 threading**

`run_store(sd, ...)` 안 3곳(main 1103, sensitivity 1113, variant 1119)에 `events=STORE_EVENT_PRIORS.get(sd.label)` 추가. 예:
```python
    main = windowed_backtest(sd.feat, window_days=DEFAULT_WINDOW_DAYS, n_folds=n_folds,
                             events=STORE_EVENT_PRIORS.get(sd.label))
```
(sensitivity 루프·variant 호출도 동일하게 `events=STORE_EVENT_PRIORS.get(sd.label)` 추가.)

- [ ] **Step 4: main()의 anchor call site threading**

`main()` 안 anchor 호출(~1234) `windowed_backtest(gw.feat, window_days=1825, n_folds=26)`에 `events=STORE_EVENT_PRIORS.get("광교")` 추가. (gw=광교 StoreData이므로 라벨 고정.)

- [ ] **Step 5: 스모크 — import 안 깨짐**

Run: `PYTHONPATH=scripts uv run python -c "import store_predictive_power"`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/store_predictive_power.py
git commit -m "feat(report): STORE_EVENT_PRIORS 매장×이벤트 opt-in 배선(기본 xmas 전매장)"
```

---

### Task 4: 재측정 검증 + 전체 회귀

**Files:**
- (변경 없음 — verify_event_prior.py는 실제 EventLevelPrior를 호출하므로 median+min_events 자동 반영)

**Interfaces:**
- Consumes: Task 1·2·3.

- [ ] **Step 1: 전체 테스트**

Run: `uv run pytest`
Expected: event_prior 신규 3테스트 포함 전체 PASS(1 pre-existing 네트워크 `test_grounding_openai_adapter` 실패는 무관). 기존 회귀 0.

- [ ] **Step 2: verify 재실행 — xmas 무회귀 확인**

Run: `PYTHONPATH=scripts uv run python scripts/verify_event_prior.py`
Expected: 4매장 xmas 여전히 개선(median+min_events로 소폭 변동 가능 but 방향 유지: 광교 ~0.09, 삼성 over 완화, 메세나 ~0.11, 광화문 ~0.06). 만약 xmas가 base보다 나빠지면 STOP·보고(min_events가 너무 공격적일 수 있음).

- [ ] **Step 3: Commit (검증 로그 문서화 불필요, 코드 변경 없음)**

변경 파일 없으면 커밋 생략. verify 출력은 리뷰 리포트에 첨부.

---

## Self-Review

**Spec coverage (§8):**
- (a) median → Task 1 ✓
- (b) opt-in config + threading → Task 3 ✓
- (c-②) min_events 가드 → Task 2 ✓
- (c-①) closed-day → 보류(스코프 밖, spec §8 명시) ✓
- xmas 무회귀 검증 → Task 4 ✓

**Placeholder scan:** 없음. 모든 코드/명령/기대값 명시.

**Type consistency:** `level_for -> (float|None,int)`, `blend -> (ndarray,ndarray)`, `EventLevelPrior(events,k,min_events)`, `windowed_backtest(...,events=None)`. Task 1·2·3 일관.

**하위호환:** `EventLevelPrior()` 무인자 계속 동작(min_events 기본 2). `windowed_backtest` 기존 호출은 events=None→xmas. 기존 leakage 테스트 기대값(prior 305/310) median=mean이라 불변.

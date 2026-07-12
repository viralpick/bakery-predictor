# 특수일 레벨-앵커 Prior (event-level-prior) 설계

**날짜**: 2026-07-12
**브랜치**: feat/bulk-filter-alpha-margin-policy
**관련 메모리**: `project_special_day_feature_spec`, `project_distributional_forecasting_stack`, `project_margin_buffer_optimization`

## 1. 배경 / 문제

광교 크리스마스(12/25)는 5년 실측 349·350·339·303·311로 **절대 레벨이 ~330에 고정**돼 있으나, 모델은 이를 못 맞춘다:

| xmas | 요일 | actual | base pred | 오차 |
|---|---|---|---|---|
| 2021 | 토(주말) | 349 | 421 | **+20%** (over) |
| 2022 | 일(주말) | 350 | 401 | **+14%** (over) |
| 2023 | 월(평일) | 339 | 234 | **−31%** (under) |
| 2024 | 수(평일) | 303 | 224 | **−26%** (under) |
| 2025 | 목(평일) | 311 | 236 | **−24%** (under) |

**메커니즘**: 이벤트가 요일 계절성을 덮어써서 그 날을 "이벤트 고유 레벨"로 끌어당긴다. 모델은 주말 xmas를 평소 바쁜 주말로(→over), 평일 xmas를 평소 한산한 평일로(→under) 예측한다. 부호(광교↑/삼성↓)는 상권 요일 패턴에서 자동으로 나온다.

**도메인 인과 (광교 xmas)**: 활성 품목 수가 주말 xmas에 급감(2021 37·2022 35 vs 평소 주말 42~44) → 케이크 생산에 인력 쏠려 빵 생산이 **공급 상한**에 걸림. 즉 수요 uplift가 아니라 반복되는 공급 제약. 이 인과가 레벨 안정성을 설명하고, 레벨-앵커 접근의 신뢰도를 높인다(내년 xmas에도 같은 제약 반복 → 상한이 곧 운영 정답).

## 2. 왜 트리 피처로 안 되나 (기각된 대안)

- **당일 dummy** (spec 원안): category-total은 1행/일, 기본 학습창 730일 안에 xmas 당일은 **2행**. LightGBM `min_child_samples` 기본값 20이 2행 leaf 분리를 금지. 실증: `min_child_samples=2`까지 낮춰도 `xmas_gain=0.0` — 2샘플 dummy가 gain 경쟁에서 절대 안 뽑힘.
- **`is_weekend × is_xmas` interaction**: 평일-xmas는 창당 1행으로 셀이 더 쪼개져 더 확실히 죽는다.
- **곱셈 uplift(1.25)**: actual/baseline 비율이 0.94→1.39로 비정상(non-stationary, baseline 드리프트 때문). 균일 곱셈은 주말 xmas를 더 과대화 → 틀린 설계.

결론: 희소(창당 <~20샘플) + sharp(정상패턴 덮어쓰기) 신호는 트리가 창 안에서 학습 불가. **레벨-앵커 prior(post-model)만이 유효.**

## 3. 설계

### 3.1 컴포넌트 — `src/bakery/models/event_prior.py`

```
EventLevelPrior(events={"xmas": (12, 25)}, k=1.5)
```
- `events`: 이벤트 레지스트리 (name → (month, day) solar). 설/추석/어린이날은 나중에 등록만 하면 확장 (음력은 후속에서 date-map 지원 추가).
- `.fit(history_df, date_col="date", target_col=...)`: 과거 이벤트일의 (date, target) 저장.
- `.level_for(date) -> (prior_level | None, n_past)`: **date보다 엄격히 이전** 이벤트 actual만 평균. n_past==0이면 (None, 0).
- `.blend(dates, base_expected, base_production) -> (exp', prod')`: 이벤트일만 교정.

### 3.2 블렌드 공식

이벤트일 D, `n_past = len(past events before D)`:
```
shrink       = n_past / (n_past + k)                       # n_past=0 → 0 → 교정 없음
prior        = mean(past event actuals before D)
blended_exp  = shrink * prior + (1 - shrink) * base_exp
correction   = blended_exp / base_exp                       # base_exp>0 가정, 0이면 교정 skip
blended_prod = base_prod * correction                       # 분위수 버퍼 비율 보존, prod≥exp 유지
```
- 비이벤트일: base 그대로.
- 레벨(요일 무관)에 앵커 → 요일 덮어쓰기 자동 처리, 매장 부호 자동 처리.

### 3.3 통합 (opt-in, 기존 시그니처 불변)

- `scripts/store_predictive_power.py::windowed_backtest`: fold마다
  ```
  hist = df[df.date < test_start_date]      # pre-test only
  prior = EventLevelPrior(...).fit(hist, target_col=TARGET)
  exp_pred, prod_pred = prior.blend(test_df.date, exp_pred, prod_pred)
  ```
  헤드라인/폴드 지표가 교정 반영. **1차 목표 = 4매장 재측정.**
- CLI 예측 경로(`predict-next-week`/v6) 배선은 **후속** (리포트 검증 후 별도 작업).

### 3.4 leakage 안전성

- prior는 test date 이전만 사용. 이중 방어: (a) fit은 pre-test history로 호출, (b) `level_for`는 `date < D` 필터.
- **전용 leakage 테스트**: `tests/test_event_prior_leakage.py` — prior at date D가 D 이후 데이터 추가/변경에 불변임을 단언. (기존 feature-level leakage 테스트는 post-model 레이어 미커버.)

## 4. 테스트

- `tests/test_event_prior.py` (unit, 정확값 비교):
  - 이벤트일 식별: 12/25 → event, 12/24 → non-event.
  - `level_for`: 과거 actual 평균 정확값, n_past 정확값, n_past=0 → (None, 0).
  - shrink 공식 정확값 (k=1.5, n_past=2 → 0.5714… → 근사 허용은 float 비교로).
  - blend: 이벤트일만 교정, 비이벤트일 불변, production correction 비율 == expected correction 비율.
  - 첫 발생(n_past=0): base 그대로.
- `tests/test_event_prior_leakage.py`: 미래 데이터 불변.

## 5. 성공 기준

1. `uv run pytest` 전부 통과 (신규 테스트 포함, 기존 회귀 없음).
2. 4매장 재측정에서 **xmas event-day WAPE 개선** (광교 시뮬 0.231→0.088 재현 방향). 전체 WAPE는 희석돼 거의 안 움직여도 정상 — **event-window 한정 지표로 판단**.
3. 매장별 부호 자동 처리 확인 (삼성 over 완화, 광교/메세나/광화문 under 완화).

## 6. 한계 (전달 시 함께)

- 이벤트당 과거 3~5샘플 → OOS 증거 얇음. 강한 shrinkage로 방어.
- 첫 발생(n_past=0) 교정 불가 — 구조적.
- xmas 전용. 설/추석/어린이날은 메커니즘이 다를 수 있어(휴무/가족수요) 각각 별도 검증 후 등록.
- 검열: 일부 xmas는 조기품절(2023 early=24) → 관측은 수요 하한이나, 공급 상한이 반복되면 상한 앵커가 운영상 정답.
- **mean은 추세를 지연**: `level_for`가 과거 이벤트 실측의 단순 평균 → YoY 상승 추세가 있으면 과소(fixture 300·310·320 평균 310 vs 2024 실제 330). 지금은 base가 레벨을 아예 못 잡아 mean으로 shrink하는 게 순이득이지만, base가 상승 xmas를 정확히 잡는 상황이면 blend가 오히려 끌어내려 오차↑. (최종 리뷰 권고)
- **하향 보정의 운영 리스크**: 삼성타운은 보정이 **하향**(base 과대→매진위험 높은 날 생산 축소)이며 3~5샘플 앵커에 기반. 방향(피크일 공급 감소)을 PoC 운영자에게 명시 필요. (최종 리뷰 권고)

## 7. YAGNI / 스코프 밖

- 음력 이벤트 date-map 지원: 설/추석 등록 시 추가 (이번 아님).
- CLI/production 예측 배선: 후속.
- 이벤트별 개별 k 튜닝: 우선 공용 k=1.5, 필요 시 후속.

## 8. v2 후속 (median + 매장×이벤트 opt-in + min_events 가드)

xmas 배포(PR#34) 후, 다른 이벤트 판정용 4매장×4이벤트 OOS에서 나온 3개 패턴에 대응:
1. **prior는 base가 나쁠 때만 이득** — base 좋은 매장(메세나 어린이날 base 0.046)엔 순손실.
2. **mean은 anomaly에 취약** — median이 더 강함(메세나 2024 추석 82 anomaly, 설 메세나 .122→.097).
3. **오피스 설/추석 = 사실상 휴무**(n=1) — 단일 샘플 prior 위험.

**변경**:
- **(a) median**: `level_for`가 `np.mean`→`np.median`(요약통계 교체, 파라미터 없음). xmas는 mean≈median(무승부)이라 회귀 무시할 수준.
- **(b) 매장×이벤트 opt-in**: `EventLevelPrior(events=...)`가 이미 opt-in 지점. 리포트에 `STORE_EVENT_PRIORS` config(매장 label→events) 추가, `windowed_backtest`에 `events` 파라미터 threading. **기본값 = 전매장 xmas만(현행 보존)**. 설/추석/어린이날 실제 등록은 별도 per-event 결정(이번 스코프 아님).
- **(c-②) min_events 가드**: `EventLevelPrior(min_events=2)` — `n_past < min_events`면 blend skip. 단일 샘플 prior 차단. xmas 운영엔 무영향(next xmas n_past=5), 초기 backtest fold(2022 n_past=1)만 base 유지.

**보류(c-①)**: 오피스 휴무일 closed-day(예측 0) 처리 — 비즈니스 영업 캘린더 입력이고 광교 PoC critical path 밖. 실 영업정보 확보 시.

# 운영 예측 패널 (C1 3단계) — 설계 스펙

작성 2026-07-29. 1단계=`lead_days`/`anchor_dow`(PR#64), 2단계=`align_features`(PR#66).
3단계는 **예측 시점에 실제로 쓸 수 있는 값으로 feature를 다시 정의**한다.

## 1. 문제 — 2단계가 남긴 것

2단계는 원점 이후를 보는 자기회귀(AR) feature를 **NaN으로 지웠다**. 그 결과 운영 성능이
`[0.0783, 0.1207]` 이라는 **너무 넓은 구간**으로만 묶였다(근거 `docs/operational_horizon_ar_align_result.md`).
폭의 원인 두 개는 둘 다 "지우기"의 부작용이다:

- **(a) 원점 시점의 최신 실측을 버린다.** lag 집합 {1,7,14,28}에 6이 없어, 화요일 원점에서
  다음주 월요일(`target−6`)을 볼 때 화요일 값을 못 쓴다.
- **(b) train/test 가용성 불일치.** blinding이 test 행에만 걸려 **train AR 결측 0.0% vs
  test 34.3%** → 모델은 "lag 없을 때"를 배운 적이 없다(공변량 shift). wpe가 +0.010 →
  **−0.092** 로 부호가 뒤집힌 것이 그 증상이다.

→ 지우는 게 아니라 **원점 기준으로 다시 만들어야** 한다.

## 2. 왜 "요일별(=horizon별) 모델"을 기본안으로 두지 않는가

원점이 화요일로 고정되면 **horizon offset이 대상 요일을 결정**한다(h=+6 ⇒ 항상 월요일).
따라서 "horizon별 7모델" = "요일별 7모델"이고, 각 모델의 학습 행이 그 요일만 남는다:
**1,826행 → 요일당 약 260행(7배 감소).**

이 프로젝트 근거로 기본안에서 제외한다:

1. **정확도 헤드룸이 거의 없다고 이미 실측**됐다(잔차 무편향·무구조, 진짜 gap은 분산 과소추정 —
   `project_distributional_forecasting_stack`). 이 국면에서 데이터를 7배 버려 요일 특화를
   얻는 건 순손실 가능성이 크다.
2. **요일은 이미 feature다.** `dow_sin/cos`·`is_weekend` 가 있고 공휴일×요일 상호작용도
   트리가 학습한다(평일 공휴일 +35% / 주말 ≈0, `docs/holiday_premium_decomposition.md`).
   요일로 쪼개면 트리가 이미 하는 일을 데이터를 깎아가며 중복한다.
3. **전례.** 옛 `scripts/operational_backtest.py` 가 horizon별 모델을 썼고 결과가
   0.77%~62.47% 로 튀었다 — n=3~10 소표본 시그니처다.

요일별 모델은 **horizon 진단이 "특정 offset이 구조적으로 안 맞는다"를 보여줄 때의 fallback**이며,
그때도 1순위는 **offset별 후처리 보정**이다(이 프로젝트는 `event_prior` 처럼 post-model 보정이
잘 먹혔다).

## 3. 설계 — 예측 패널 (forecast panel)

행 하나 = **(원점, 대상일)** 쌍. 한 번 만들고, 백테스트는 **원점으로** 자른다.

| 열 그룹 | 내용 |
|---|---|
| 키 | `origin_date`, `target_date`, `horizon_offset`(= target − origin) |
| 원점 기준 AR | `y_origin`, `y_origin_lag1..3`, `rmean7/28(≤origin)`, `rstd7/28(≤origin)`, `ewma7/28(≤origin)` |
| ★ 같은 요일 | `y_same_dow_latest`(원점 이전 마지막 같은 요일), `y_same_dow_mean4`(최근 4회 평균) |
| 대상일 캘린더 | 기존 `add_cyclic_calendar` / `add_holiday_features` / `add_event_features` 를 **target_date** 에 적용 |
| 대상일 외부 | 기존 `add_weather_features` / `add_competitor_features` (§6 캐비앗 참조) |
| 타깃 | `actual` (= `adjusted_demand_unit`) |

### ★ `y_same_dow_latest` 가 핵심이다
지금 `lag7`이 하던 "지난주 같은 요일" 역할을 대체하는데 **항상 가용**하다. 화요일 원점에서
월요일 대상이면 직전 월요일(=`target−7`, 원점−1), 일요일 대상이면 `target−14`(원점−2)를
자동으로 집는다. 반면 현 lag 집합은 일요일 대상일 때 `lag7`이 원점 이후라 그냥 버려진다.

### 원점 cadence — 학습은 매일, 평가는 화요일만
- **학습**: 모든 날짜를 원점으로 생성(daily origins) → 행 ≈ 1,826 × 7 = **약 12,800행**.
  offset별 오차 구조를 전 범위에서 배운다(direct multi-horizon 표준 증강).
- **평가**: 운영 cadence인 **화요일 원점만** 사용 → fold당 7행(월~일), 52 fold = **`n_test`=364**
  로 기존 실험과 **직접 비교 가능**.
- ⚠️ daily origins는 같은 `actual` 이 7행에 중복 등장한다(상관 있는 행). direct multi-horizon
  에서 표준이지만, 유효 표본이 12,800이 아니라는 점은 리포트에 적는다.

### 두 문제가 동시에 해소된다
| 문제 | 해소 |
|---|---|
| train/test 가용성 불일치 | 모든 행이 **정의상** 원점-가용 feature만 가짐 → 공변량 shift 소멸 |
| 데이터 7배 감소(요일별 모델의 대가) | 단일 모델이 12,800행 전부 학습 → **오히려 증가** |
| 364 fit 비용(offset별 모델의 대가) | fold당 1 fit 유지 → **52×2 = 104 fit**(현재와 동일), 행수만 7배 |

## 4. 아키텍처 — `windowed_backtest` 를 건드리지 않는다

패널은 fold를 **원점**으로 자르므로 현 fold 로직(대상일 7일 블록)과 계약이 다르다.
하드 게이트가 걸린 `backtest_core.windowed_backtest` 를 더 파라미터화하지 않고 **형제 함수**를 둔다.

- `src/bakery/features/forecast_panel.py` — `build_forecast_panel(...)` 신규 프리미티브
- `src/bakery/harness/panel_backtest.py` — `panel_backtest(...) → BacktestResult`
  (**같은 반환 shape** 이라 `metrics_from_preds`·`report.py` 무변경 소비)
- `harness/config.py` — `ExperimentSpec.engine: "windowed" | "panel" = "windowed"`
- `harness/runner.py` — engine 분기
- `experiments/gwangyo_operational_panel.yaml`

### fold 정의 (leakage-safe)
평가 원점 `O`(화요일)마다:
- **test** = `origin_date == O` 인 행(7개, offsets 6~12 → 다음주 월~일)
- **train** = `target_date <= O` **AND** `target_date >= O − window_days` 인 행
  - `target_date <= O` 면 `origin = target − h <= O − 6 < O` 이므로 **원점도 자동으로 O 이전**이다
  - 즉 "O 시점에 실측이 확정된 행만 학습" — 이게 leakage 차단의 전부다
- `min_train_rows` 미달 fold는 스킵(기존과 동일 규약)
- `event_prior` 는 기존과 동일하게 post-model 블렌드로 적용하고, history는 `target_date <= O` 로 fit

## 5. 검증 계획

1. **패널 정합성**: 임의 (origin, target) 표본에서 `y_same_dow_latest` 의 출처 날짜가
   (a) 대상일과 같은 요일이고 (b) `<= origin` 인지 **정확값**으로 검증.
2. **leakage 회귀**: 모든 원점-기준 feature의 출처 날짜 최대값이 `<= origin` 임을 검증.
   train 행의 `target_date` 최대값이 `<= O` 임을 검증.
3. **기존 경로 불변**: `engine="windowed"` 기본값에서 **52-fold rtol=1e-9 동등성 게이트를
   수정 없이 통과**해야 한다(패널은 별도 함수라 구조적으로 보장되지만 게이트로 확인).
4. **horizon별 진단**: offset 6~12별 WAPE·WPE를 fold 결과에 남긴다. `horizon_offset` 을
   feature로 주는 것만으로 offset별 편향이 자동 해소되지는 않는다 —
   체계적으로 어긋나는 offset이 있으면 **offset별 후처리 보정**을 다음 단계로 얹는다.
5. **비교**: `gwangyo_train_aged5`(하한) / `gwangyo_lead5_ar_aligned`(느슨한 상한)과
   **같은 대상일 블록(월~일)·같은 `n_test`=364** 로 대조. 기대 = 구간 안쪽으로 수렴.

## 6. 캐비앗 — 이번에 정렬하지 **않는** 축

<!-- 이 절을 지우지 말 것. 정렬한 축과 안 한 축을 섞으면 "운영 검증 완료"로 오독된다. -->
**날씨는 여전히 관측값(`weather_observed`)을 쓴다.** 현 헤드라인도 그렇다
(`add_weather_features` 기본 경로). D+6~D+12 를 예측하면서 실제로는 그 날의 관측 기온·강수를
쓰는 것이므로 **이 축은 아직 낙관적**이다. 운영에서는 중기예보를 써야 하고
`paths.dataset("forecast_mid_term_daily")` 가 이미 등록돼 있다(CLI 주석에도 "horizon
D+4~D+10 커버"라고 적혀 있다).

이번 단계에서 바꾸지 않는 이유: **한 번에 한 축만 움직여야** 패널 효과를 식별할 수 있다.
날씨 축 정렬은 **4단계**로 분리한다. 그 전까지 이 실험 수치는 "AR 축은 정렬됐고 날씨 축은
미정렬"로 인용해야 한다.

또한 강수는 이 프로젝트에서 **약한 2차 변수**로 실측됐고(±5%, 매장별 부호 상이) 극한날씨
트랙은 전부 noise로 drop됐으므로, 날씨 축의 기대 영향은 AR 축보다 작을 것으로 본다(미검증).

## 7. 성공 기준

- `engine: panel` 실험이 52 fold·`n_test`=364로 돌고, WAPE가 `[0.0783, 0.1207]` **구간 안쪽**으로
  수렴한다(수렴 자체가 (a)(b) 해소의 증거).
- 기존 동등성 게이트 무수정 통과 · 전체 스위트 green.
- horizon별 진단이 리포트에 남는다.
- 헤드라인(`gwangyo_default.yaml`)은 **무변경** — 교체 판단은 architect.

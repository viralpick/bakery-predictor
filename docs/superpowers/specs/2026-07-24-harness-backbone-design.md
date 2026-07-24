# Harness Backbone 설계 (실험 단일 표면)

**날짜**: 2026-07-24
**상태**: 설계 승인 대기
**범위**: 실험 오케스트레이션 backbone — 전처리 일반화는 **별도 스펙**으로 분리 (본 문서 "차기 스펙" 참조)

---

## 1. 문제 정의

지금까지 빠른 실험을 위해 코드가 82개의 일회성 스크립트(`scripts/*.py`)에 흩어져 있다. 그 결과:

- **모델 혼란**: "lightgbm v1/v2 중 뭘 쓰지", "v6가 최신 아냐?" — v0~v6이 사실 **층위가 다른 것들**인데 한 줄로 발전한 버전처럼 취급됨.
- **엔진 혼란(★설계 중 실제 발생)**: 백테스트 엔진이 **둘**이다 — cli.py의 item-level `run_backtest` vs `scripts/store_predictive_power.windowed_backtest`(카테고리 총량+event_prior). **후자가 발행 헤드라인(고객 지표)인데 src에 없고 scripts에 숨어 있어**, 이 설계 초안조차 처음엔 item-level을 canonical로 잘못 잡았다. 이 혼란 제거가 backbone의 1차 목적.
- **타깃 혼란**: 측정헌장은 `adjusted_demand`가 기준인데 스크립트마다 `potential_demand`(폐기)를 다시 봄.
- **품질 편차**: 데이터 분석·그래프가 매번 새로 짜여 포맷/품질이 천차만별.
- **재현 불가**: "이 숫자가 어떤 설정에서 나왔는지" 추적이 안 됨.

핵심 목표: **"골라서 돌리는 단일 표면"** 을 만들어, 무엇을 써야 맞는지 헷갈리는 것을 구조적으로 없애고, 새 데이터/스펙이 와도 데이터만 갈아끼우면 재실행되도록 한다.

## 2. 핵심 설계 원칙

1. **표면은 하나** — `src/bakery`(프리미티브 라이브러리)는 유지하되, 오케스트레이션 스파인이 **두 곳에 숨어 있다**: (a) `cli.py`(2786줄)의 item-level `run_backtest` 경로, (b) `scripts/store_predictive_power.py`(1411줄)의 `windowed_backtest` — **카테고리 총량+event_prior = 발행 헤드라인 엔진**. 둘 다 `harness/`로 **추출**해 CLI·YAML runner가 공유한다. ★이 "숨은 두 번째 스파인"이 바로 backbone의 존재이유(어느 게 canonical인지 헷갈렸던 근본 원인).
2. **YAML 설정 = 1실험** — `experiments/*.yaml` 한 파일이 하나의 실험. 설정이 산출물과 함께 남아 재현·diff·버전관리가 된다.
3. **canonical 강제 + 경고** — 측정헌장(`adjusted_demand` 기본, WAPE 메인, random split 금지)을 config 검증에서 코드로 강제한다. 폐기된 선택(`potential_demand`)은 명시적 override 없으면 에러.
4. **재개 가능** — 실험을 단계별 캐시 아티팩트로 쪼개, 바뀐 지점 이후만 재실행한다 (E2E 반복 회피).
5. **재구현 금지** — harness는 `src/bakery`를 호출하는 얇은 오케스트레이션. 모델/피처/평가 로직을 다시 짜지 않는다.

## 3. 디렉토리 구조

```
src/bakery/                프리미티브 라이브러리 (그대로 유지)
src/bakery/harness/        ★신설 — 단일 스파인
    config.py                실험 스펙 스키마 + 로더 + canonical 강제/경고
    registry.py              data/target/forecaster/layer/window "골라 쓰기" (taxonomy)
    event_priors.py          STORE_EVENT_PRIORS named preset (scripts에서 승격, 단일 출처)
    backtest_core.py         windowed_backtest 코어 추출(category_total+event_prior)
    runner.py                ExperimentSpec → 단계 실행 → RunResult (캐시/재개)
    report.py                RunResult → 표준 지표/KPI 테이블
    eda.py                   재사용 데이터분석/가설검증
    viz/
        data_report.html        데이터분석 표준 리포트 템플릿
        result_report.html      결과분석 표준 리포트 템플릿
cli.py                     harness 호출 thin wrapper로 점진 전환 (기존 커맨드 인터페이스 유지)
experiments/*.yaml         1파일 = 1실험
scripts/_archive/          기존 82개 스크립트 보존 (삭제 아님)
```

## 4. 모델 Taxonomy (registry가 담는 것)

flat 리스트가 아니라 **kind별 분류**. `v0~v6`은 한 줄 버전이 아니라 층위가 다르다.
★**canonical(발행 헤드라인) = 카테고리 총량 `category_total` + `event_prior` 레이어**. 나머지는 비교용 보조.

| kind | 인터페이스 | 멤버 | 위상 |
|---|---|---|---|
| `category_total` | `fit_category_total`→expected/quantile (카테고리 총량 granularity) | category_total | ★**canonical 기본**. 발행 WAPE 엔진(`windowed_backtest`). |
| `distributional` | `predict_dist/quantile/sigma/median` | distributional_total | **기본 병행**. σ 출력으로 위험수치 산정. |
| `point_forecaster` | `fit`/`predict`→Series | seasonal_naive, moving_average, lightgbm(v0~v3), artisee_baseline | **비교용 보조** (item-level, cli 경로). |
| `composite_pipeline` | category_total → item_proportion → new_product_tracker | v4 full stack | **하위 배분 레이어** (총량→품목). canonical은 총량까지, item 배분은 이 위에. |
| `post_layer` | 예측 위 후처리 | event_prior(특수일 레벨 블렌드), decision(구 v6: 발주량+위험수치+lineage), conformal_order(발주 보정) | forecaster가 아님 |

- **granularity 주의**: canonical = category_total **총량 레벨**(`windowed_backtest`가 내는 expected를 groupby sum). full v4(→item_proportion→new_product)와 섞지 않는다. item 배분은 하위 레이어.
- **event_prior = 완전 후처리(post-model 블렌드)**: 예측기가 낸 값을 특수일(xmas·설·추석·어린이날 등)에만 레벨-앵커로 보정. `prior.blend(dates, expected, production)`이 **expected(예측)·production(발주) 둘 다** 보정. ★추출 시 보존 규칙 2개: (1) **leakage-safe** — prior는 train window가 아니라 **pre-test 전체 history**(`date < test_start`)로 fit(헌장 1번 직결). (2) 특수일 날짜 설정은 `event_priors.py` 프리셋 단일 출처. distributional_total에도 prior를 걸지는 Phase 2 결정(현 canonical 검증은 category_total 경로).
- `v5 conformal_interval` = **DEPRECATED** 로 등록. 선택 시 경고.
- 스택 흐름: `category_total(총량 예측) → event_prior(특수일 보정) → [decision(발주량+위험수치)]`. distributional_total은 총량 예측의 대안 forecaster(σ 제공).
- **"v2 vs v6"는 비교 대상이 아니다** — 하나는 예측기, 하나는 결정 레이어. **"cli lightgbm_v2 = 헤드라인"도 오해였다** — 헤드라인은 카테고리 총량 스택이고 item-level lightgbm은 보조.

## 5. Config 스키마 (YAML)

```yaml
# experiments/gwangyo_default.yaml
name: gwangyo_default
data:
  source: real                 # real | synthetic
  store: store_gw01            # 광교 단독 (헌장: 검증은 광교. real store_id = store_gw01)
target: adjusted_demand        # 기본값. potential_demand 지정 시 allow_deprecated 없으면 ERROR
forecaster: [category_total, distributional_total]  # ★기본 = 발행 헤드라인 + 분포 병행
layers: [event_prior]          # canonical 스택의 일부. decision 은 opt-in (기본 미포함)
event_priors: gwangyo          # ★STORE_EVENT_PRIORS 프리셋 키 (event/lunar 날짜 설정, 아래 참조)
window:
  scheme: expanding            # expanding | rolling. random split 스키마 거부
  n_folds: 52                  # 발행 헤드라인 = 52 folds (약 12개월 test span)
  window_days: 730             # rolling window 길이
metrics:                       # 생략 시 아래 6종 헌장 기본 세트
  - wape                       # 메인 (★카테고리 총량 WAPE = 빵 총량 = 발행 지표)
  - wpe                        # 편향 방향
  - waste_rate                 # 폐기율 (1차 KPI)
  - soldout_median             # 매진 시각 중앙값
  - stockout_item_rate         # 전체 제품 중 실제 품절 발생 비율 (헌장 관점②)
  - shortfall_day_rate         # 전체 날 중 adjusted_demand > 발주량 비율 (헌장 관점①)
```

**STORE_EVENT_PRIORS 정착 (advisor #3, 숨은 설정 제거):** 현재 매장별 event/lunar 날짜가 `scripts/store_predictive_power.py`에 하드코딩. harness 승격 시 **canonical 위치 = `src/bakery/harness/event_priors.py`의 named preset dict**(키 예: `gwangyo`). config는 `event_priors: gwangyo`로 프리셋을 참조만 한다(YAML에 날짜 나열 금지 — 단일 출처 유지). 프리셋 미지정 + forecaster에 event_prior layer 있으면 경고.

### canonical 강제 규칙 (config.py 검증)

| 규칙 | 동작 |
|---|---|
| `target` 미지정 | `adjusted_demand` 기본 |
| `target: potential_demand` | `allow_deprecated: true` 없으면 **ERROR** |
| `forecaster`/`layers` 미지정 | 기본 스택: `forecaster=[category_total, distributional_total]`, `layers=[event_prior]` (canonical 헤드라인) |
| `event_priors` 미지정 + layers에 event_prior 있음 | **경고** (프리셋 키 필요 — `harness/event_priors.py`) |
| `metrics` 미지정 | 헌장 6종 기본 세트 |
| `metrics: [mape]` 단독 | **경고** (희소 품목 폭발, WAPE 병기 권장) |
| random split 스키마 | **거부** (time leakage 헌장 1번) |
| `forecaster: conformal_interval` (v5) | **경고** (DEPRECATED) |

## 6. 단계별 실행 + 재개 (runner)

실험을 단계별 캐시 아티팩트로 쪼갠다. E2E는 config가 바뀐 지점부터만 다시 돈다.

```
[1 load] → [2 features] → [3 fit] → [4 predict] → [5 evaluate] → [6 report/viz]
   ↓캐시      ↓캐시          ↓model.pkl  ↓predictions.csv  ↓metrics.json  ↓html
```

- 각 단계의 캐시 키 = **(그 단계에 영향 주는 config 부분 + 상위 아티팩트 해시)**.
- config 무변경 + 아티팩트 존재 → 그 단계 **건너뛰고 재사용**.
- CLI:
  - `harness run <config>` — 필요한 것만 실행 (변경분부터)
  - `harness run <config> --from evaluate` — 예측 캐시 쓰고 평가부터
  - `harness run <config> --only report` — 그래프만 다시
- 원리는 워크플로우 `resumeFromRunId`와 동일: 바뀐 지점 이후만 재실행.

## 7. 표준 산출물 (RunResult) — 결과분석 통일

모든 실험이 **동일 구조**를 반환하여, 어떤 모델이든 같은 기준으로 tracking:

```
reports/<name>/
    predictions.csv        store_id, [category_id|item_id], date, y_true, yhat, [sigma], is_stockout, ...
    metrics.json           WAPE/WPE(category 총량 헤드라인 + item·total 보조) + 운영KPI 6종
    config_resolved.yaml   실제 적용된 설정 (재현 기록, 기본값 확정 포함)
```

- **granularity 컬럼**: canonical(category_total)은 `category_id` 레벨, item-level 보조 forecaster는 `item_id` 레벨. predictions.csv에 granularity 컬럼을 명시해 report.py가 헤드라인(카테고리 총량 WAPE)을 올바른 축에서 집계.
- `report.py`는 이 구조만 소비. forecaster 종류에 무관.

## 8. 그래프 — 2개 HTML 템플릿 (기존 자산 승격, 신규 작성 아님)

★기존에 재사용할 시각화 자산이 있다. "새로 만든다"가 아니라 **승격/일반화**한다:

| 기존 자산 | 정체 | 승격 대상 |
|---|---|---|
| `scripts/build_dashboard.py` (1209줄, plotly) | 4매장 종합 EDA HTML(탭·매장별 12차트) | **`data_report.html`의 prior art** — 차트 구성·plotly 패턴 재사용, 데이터소스만 harness eda.py 출력으로 교체 |
| `scripts/weekly_overlay_series.py` (신규, 3ee0991) | 주간 오버레이 **데이터 조립**(실측/예측/발주 동일축, canonical 3cat) | **`result_report.html`의 데이터 소스 패턴** — 이미 canonical 엔진(windowed_backtest) 위에 서 있어 RunResult와 수렴 |

- `data_report.html` ← `eda.py` 출력 소비: 분포/결측/커버리지/가설검증 표준 패널. **build_dashboard.py 차트 로직 승격.**
- `result_report.html` ← `RunResult` 소비: 카테고리 총량 WAPE 추이 / 오차분해 / KPI 카드 6종 / 실험간 비교. **weekly_overlay 오버레이 패턴 승격.**
- 데이터만 갈아끼우면 항상 같은 포맷. `dataviz` 스킬 기준 적용(라이트/다크, 접근성).

## 9. 구현 순서

| Phase | 내용 | 완료 기준 (acceptance) |
|---|---|---|
| **1. 스파인** | config / registry / runner + **`windowed_backtest` 코어 추출**(카테고리 총량+event_prior) | ★**엔진 동등성**: harness가 추출한 스파인의 예측값이 `scripts/store_predictive_power.windowed_backtest`와 **정확일치**(동일 folds=52·window=730·STORE_EVENT_PRIORS[gwangyo]). `fit_category_total`(random_state=42)·`EventLevelPrior`(비랜덤) 결정성 확인됨 → 정확일치 유효. **발행 WAPE(~8.03)는 sanity anchor로 병기.** |
| **2. 결과분석** | report.py + result_report.html (weekly_overlay 승격) | RunResult에서 6종 metric + KPI 카드가 표준 포맷으로 생성. 실험 2개 비교 뷰 동작. |
| **3. 데이터분석** | eda.py + data_report.html (build_dashboard 승격) | 데이터소스만 갈아끼워 표준 EDA/가설검증 리포트 재생성. |
| **4. (분리)** | 전처리 일반화 | **별도 스펙** (아래) |

**acceptance 원칙 (advisor #2):** "발행 WAPE 재현"은 목표 숫자가 durable하지 않으면 반증 불가한 게이트라, **엔진 동등성**(추출 스파인 == windowed_backtest, 예측값 정확일치)으로 검증한다. 발행 숫자를 찾을 필요 없이 배선 충실성이 증명된다. 추출 대상은 **`windowed_backtest` 코어만**(fit_category_total+event_prior+metrics); 1411줄 스크립트의 plot_*/find_optimal_*/buffer_analysis는 Phase 2(viz)·마진최적화 영역이라 끌어오지 않는다.

**`scripts/store_predictive_power.py` 처리 (사용자 결정 2026-07-24):** 코어만 harness로 추출하고 **원본은 당분간 유지**. 원본의 `windowed_backtest`는 harness 코어를 import하는 얇은 래퍼로 점진 전환(plot_*/find_optimal_* 등 나머지는 그대로) → 기존 리포트 파이프라인 안 깨짐. 두 구현 공존 금지(단일 출처).

기존 82개 스크립트는 `scripts/_archive/`로 보존. 신규 실험은 표면(YAML)으로만.

## 10. 차기 스펙 (본 문서 범위 밖 — TODO에 기록)

backbone(실험/측정 평면)과 성격이 다른 두 평면은 **별도 스펙**으로 분리한다. 둘 다 backbone이 만든 canonical 진입점을 **소비**하는 쪽이라, backbone Phase 1~3 완료 후 착수한다.

**(A) 전처리 일반화 스펙** — 데이터 수령 평면. 새 데이터/스펙이 와도 loader/adapter만 추가해 canonical schema로 진입시키는 일반화. 매번 새 로더를 짜지 않고 어댑터 패턴으로 대응. (참조: 진행 중인 신규 라인레벨 데이터 통합 `bonavi_loader_v2`, `docs/superpowers/specs/2026-07-23-new-data-integration-design.md`)

**(B) AOS/온톨로지 예측 배선 정합 스펙** — 제품/데모 평면. 현재 `src/bakery/ontology/scenario.py`가 `GlobalLGBM(feature_set="v2")`를 **직접 fit**(cli·scripts에 이어 네 번째 예측 배선처). 이를 harness canonical 스택 진입점 호출로 **재배선**해 온톨로지·시나리오·writeback·loop가 단일 예측 엔진을 공유하게 한다. `decision`(v6)은 이미 순수 post-prediction이라 본 스펙의 `post_layer`로 흡수됨 — 별도 처리 불필요. AOS 자체(scenario/writeback/loop 기계)는 backbone의 소비자라 범위 밖. (참조: `project_poc_v7_aos_demonstrator`)

## 11. 리스크

- **표면 3중화 위험**: cli.py 스파인 추출을 소홀히 하고 runner만 추가하면 표면이 늘어난다 → Phase 1에서 cli.py ↔ harness 관계를 명시적으로 정리한다.
- **taxonomy 누락**: 새 모델이 4개 kind 어디에도 안 맞으면 registry 확장 필요 → kind 추가는 설계 리뷰 대상.
- **캐시 무효화 버그**: config 변경이 캐시 키에 안 잡히면 낡은 결과 재사용 → 해시 키에 포함되는 config 필드를 명시적으로 열거하고 테스트.
- **재현 acceptance 실패**: Phase 1에서 windowed_backtest와 예측값이 안 맞으면, 추출 과정에서 로직이 미묘하게 바뀐 것 → 진행 중단하고 원인 규명.
- **엔진 승격 폭발반경**: canonical을 카테고리 총량으로 바꾸면 taxonomy·기본스택·acceptance가 다 연동. §4/§5/§9 정합을 self-review에서 재확인(기본 config가 canonical 엔진과 일치하는지 — item-level 기본이 남으면 모순).
- **granularity 혼동**: canonical = 카테고리 총량 레벨. full v4(item 배분)와 섞으면 "총량 예측"과 "품목 배분"이 뒤엉킴 → registry에서 category_total(총량)과 composite_pipeline(배분)을 명확히 분리.

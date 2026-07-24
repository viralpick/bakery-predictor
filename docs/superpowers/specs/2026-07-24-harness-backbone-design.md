# Harness Backbone 설계 (실험 단일 표면)

**날짜**: 2026-07-24
**상태**: 설계 승인 대기
**범위**: 실험 오케스트레이션 backbone — 전처리 일반화는 **별도 스펙**으로 분리 (본 문서 "차기 스펙" 참조)

---

## 1. 문제 정의

지금까지 빠른 실험을 위해 코드가 82개의 일회성 스크립트(`scripts/*.py`)에 흩어져 있다. 그 결과:

- **모델 혼란**: "lightgbm v1/v2 중 뭘 쓰지", "v6가 최신 아냐?" — v0~v6이 사실 **층위가 다른 것들**인데 한 줄로 발전한 버전처럼 취급됨.
- **타깃 혼란**: 측정헌장은 `adjusted_demand`가 기준인데 스크립트마다 `potential_demand`(폐기)를 다시 봄.
- **품질 편차**: 데이터 분석·그래프가 매번 새로 짜여 포맷/품질이 천차만별.
- **재현 불가**: "이 숫자가 어떤 설정에서 나왔는지" 추적이 안 됨.

핵심 목표: **"골라서 돌리는 단일 표면"** 을 만들어, 무엇을 써야 맞는지 헷갈리는 것을 구조적으로 없애고, 새 데이터/스펙이 와도 데이터만 갈아끼우면 재실행되도록 한다.

## 2. 핵심 설계 원칙

1. **표면은 하나** — `src/bakery`(프리미티브 라이브러리)는 유지하되, `cli.py`(2786줄) 안에 private helper로 갇힌 오케스트레이션 스파인을 `harness/`로 **추출**한다. CLI 커맨드와 YAML runner **둘 다 이 스파인을 호출**한다. 새 runner를 CLI 옆에 나란히 세워 표면을 3개로 늘리지 않는다.
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

flat 리스트가 아니라 **kind별 분류**. `v0~v6`은 한 줄 버전이 아니라 층위가 다르다:

| kind | 인터페이스 | 멤버 | 비고 |
|---|---|---|---|
| `point_forecaster` | `fit`/`predict`→Series | seasonal_naive, moving_average, lightgbm(v0~v3), artisee_baseline | v0~v3 = feature_set 스위치 |
| `distributional` | `predict_dist/quantile/sigma/median` | distributional_total | σ 출력으로 위험수치 산정 |
| `composite_pipeline` | 다단계 | v4 category stack (category_total→item_proportion→new_product_tracker) | forecaster 계열 |
| `post_layer` | 예측 위 후처리 | event_prior(특수일 레벨 블렌드), decision(구 v6: 발주량+위험수치+lineage), conformal_order(발주 보정) | forecaster가 아님 |

- `v5 conformal_interval` = **DEPRECATED** 로 등록. 선택 시 경고.
- 스택 흐름: `forecaster(예측/분포) → event_prior(특수일 보정) → decision(발주량+위험수치)`
- **"v2 vs v6"는 비교 대상이 아니다** — 하나는 예측기, 하나는 결정 레이어.

## 5. Config 스키마 (YAML)

```yaml
# experiments/gwangyo_default.yaml
name: gwangyo_default
data:
  source: real                 # real | synthetic
  store: gwangyo               # 광교 단독 (헌장: 검증은 광교)
target: adjusted_demand        # 기본값. potential_demand 지정 시 allow_deprecated 없으면 ERROR
forecaster: [lightgbm_v2, distributional_total]   # 단일값 또는 리스트(나란히 비교)
layers: [event_prior]          # decision 은 opt-in (기본 미포함)
window:
  scheme: expanding            # expanding | rolling. random split 스키마 거부
  test_weeks: 8
metrics:                       # 생략 시 아래 6종 헌장 기본 세트
  - wape                       # 메인 (item / category / total)
  - wpe                        # 편향 방향
  - waste_rate                 # 폐기율 (1차 KPI)
  - soldout_median             # 매진 시각 중앙값
  - stockout_item_rate         # 전체 제품 중 실제 품절 발생 비율 (헌장 관점②)
  - shortfall_day_rate         # 전체 날 중 adjusted_demand > 발주량 비율 (헌장 관점①)
```

### canonical 강제 규칙 (config.py 검증)

| 규칙 | 동작 |
|---|---|
| `target` 미지정 | `adjusted_demand` 기본 |
| `target: potential_demand` | `allow_deprecated: true` 없으면 **ERROR** |
| `forecaster`/`layers` 미지정 | 기본 스택: `forecaster=[lightgbm_v2, distributional_total]`, `layers=[event_prior]` |
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
    predictions.csv        store_id, item_id, date, y_true, yhat, [sigma], is_stockout, ...
    metrics.json           WAPE/WPE(item·category·total) + 운영KPI 6종
    config_resolved.yaml   실제 적용된 설정 (재현 기록, 기본값 확정 포함)
```

`report.py`는 이 구조만 소비. forecaster 종류에 무관.

## 8. 그래프 — 2개 HTML 템플릿

- `data_report.html` ← `eda.py` 출력 소비: 분포/결측/커버리지/가설검증 표준 패널.
- `result_report.html` ← `RunResult` 소비: WAPE 추이 / 오차분해 / KPI 카드 / 실험간 비교.
- 데이터만 갈아끼우면 항상 같은 포맷. `dataviz` 스킬 기준 적용(라이트/다크, 접근성).

## 9. 구현 순서

| Phase | 내용 | 완료 기준 (acceptance) |
|---|---|---|
| **1. 스파인** | config / registry / runner (캐시·재개 포함) | ★**기존 backtest 결과 1개(광교 lightgbm_v2 adjusted_demand)를 YAML config로 재현해 같은 숫자**가 나온다. 추상화가 faithful하다는 증거. |
| **2. 결과분석** | report.py + result_report.html | RunResult에서 6종 metric + KPI 카드가 표준 포맷으로 생성. 실험 2개 비교 뷰 동작. |
| **3. 데이터분석** | eda.py + data_report.html | 데이터소스만 갈아끼워 표준 EDA/가설검증 리포트 재생성. |
| **4. (분리)** | 전처리 일반화 | **별도 스펙** (아래) |

기존 82개 스크립트는 `scripts/_archive/`로 보존. 신규 실험은 표면(YAML)으로만.

## 10. 차기 스펙 (본 문서 범위 밖 — TODO에 기록)

**전처리 일반화 스펙**: 새 데이터/스펙이 와도 loader/adapter만 추가해 canonical schema로 진입시키는 일반화. 데이터 수령 사이드라 성격이 달라 분리. (참조: 진행 중인 신규 라인레벨 데이터 통합 `bonavi_loader_v2`, `docs/superpowers/specs/2026-07-23-new-data-integration-design.md`)

## 11. 리스크

- **표면 3중화 위험**: cli.py 스파인 추출을 소홀히 하고 runner만 추가하면 표면이 늘어난다 → Phase 1에서 cli.py ↔ harness 관계를 명시적으로 정리한다.
- **taxonomy 누락**: 새 모델이 4개 kind 어디에도 안 맞으면 registry 확장 필요 → kind 추가는 설계 리뷰 대상.
- **캐시 무효화 버그**: config 변경이 캐시 키에 안 잡히면 낡은 결과 재사용 → 해시 키에 포함되는 config 필드를 명시적으로 열거하고 테스트.
- **재현 acceptance 실패**: Phase 1에서 기존 숫자가 안 맞으면, 추출 과정에서 로직이 미묘하게 바뀐 것 → 진행 중단하고 원인 규명.

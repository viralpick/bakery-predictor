# 데이터분석 + 가설검증 레이어 (Phase 6) — 설계 스펙

작성일 2026-07-28. 마스터 로드맵 6단계(유일 미착수). harness backbone(예측 평면) 위에 **입력 데이터 분석 + 가설 검증 평면**을 YAML 제어로 추가.

## 1. 문제 정의

지금까지 입력 데이터 분석(EDA)과 가설 검증은 `scripts/`에 흩어진 20+개 스크립트로 ad-hoc 실행됐다("뭘 돌려야 맞지" 혼란 — harness backbone이 예측에서 푼 것과 동일 문제). `src/bakery/analysis/`에 프리미티브 17개는 이미 승격돼 있으나, **조합·실행·리포트 표면이 없다.**

목표: **`bakery analysis-run <yaml>` 단일 진입점**으로
- **입력 데이터**(우리 모델 산출물이 아닌 raw/observed: 판매·생산·폐기·할인·재고) 분석을 켜고/끄고,
- **여러 가설**을 각각 켜고/꺼서 검증하고,
- 결과(수치 + 그래프)를 **자기포함 HTML** 하나로 뽑는다.

**입력 데이터의 정의(중요)**: canonical `bonavi_daily`(관측 sold/stockout/waste/capacity), `multistore_daily`(4매장), `bonavi_receipts`(라인레벨), 할인(`discount` v2 클린 parquet), 재고정보(QT_MADE/QT_OUT). **모델 예측값(yhat/expected/production)은 입력이 아니다** — 이 레이어는 데이터 자체와 가설을 본다(예측 성능은 harness-run 담당).

## 2. 설계 원칙 (harness 일관성)

- **harness backbone과 같은 패턴 미러링**: pydantic Spec + registry + runner + 자기포함 HTML report. `harness-run`(예측)과 `analysis-run`(분석)은 **형제 표면**, 성격이 달라 진입점 분리(사용자 결정).
- **재구현 금지·호출만**: registry의 각 분석/가설은 이미 있는 `src/bakery/analysis/` 프리미티브와 `scripts/`의 검증 로직을 **호출**한다. 스크립트에서 검증된 로직을 함수로 추출해 registry에 등록(scripts는 얇은 wrapper로 남기거나 점진 폐기).
- **on/off 선언적**: YAML에서 각 분석·가설을 boolean으로. 끈 것은 실행·리포트 모두 스킵(단 리포트에 "off"로 표기해 은폐 방지).
- **leakage 무관**: 이 레이어는 회고(retrospective) 분석·가설 검증이라 예측 leakage 규칙 밖. 단 가설이 lag/baseline을 쓰면 그 내부의 leakage-safe 계산(예: absorption의 4주 proxy)은 프리미티브가 이미 보장.

## 3. 아키텍처

**새 서브패키지 `src/bakery/analysis/lab/`** (프리미티브 `src/bakery/analysis/*.py`를 소비하는 오케스트레이션 레이어):

- `spec.py` — `AnalysisSpec`(pydantic): `name`, `data`(source/store), `data_analyses`(dict[str,bool]), `hypotheses`(dict[str,bool]), 선택적 per-item params.
- `registry.py` — 이름→핸들러 매핑. 두 종류: `DATA_ANALYSES`(입력 데이터), `HYPOTHESES`(가설). 각 핸들러 = `(inputs) -> AnalysisResult`.
- `inputs.py` — 입력 데이터 로더(canonical/multistore/receipts/discount/inventory를 spec.data 기준으로 한 번 로드해 핸들러에 전달; 중복 IO 방지).
- `runner.py` — `run_analysis(spec) -> AnalysisReport`. 켜진 항목만 실행, 결과 수집.
- `report.py` — `AnalysisReport` → 자기포함 HTML(plotly `fig_to_div` stateless, harness `report.py` 패턴 재사용). 2대 섹션(데이터분석 / 가설검증), 각 항목 = 제목 + 그래프 + 결과표 + 판정(가설의 경우).
- `result.py` — `AnalysisResult`(name, kind, figures: list[plotly], tables: list[DataFrame], verdict: str|None, notes).

**CLI**: `bakery analysis-run <yaml> [--out reports/analysis/]` → `analysis_report.html`.

**YAML 위치**: `experiments/analysis_*.yaml` (harness의 `experiments/*.yaml`과 같은 디렉토리, 접두어로 구분).

## 4. YAML 스키마 (예시)

```yaml
# experiments/analysis_gwangyo.yaml
name: analysis_gwangyo
data:
  source: real
  store: store_gw01        # store_gw01 | multistore(4매장)
data_analyses:
  sales_distribution: true       # eda01: 매장별 일별 매출(수량/매출액) 분포
  category_mix: true             # eda03: 카테고리 매출 비중(+월별 안정성)
  waste_rate: true               # eda02: 매장별/품목별 폐기율·재고·매진
  waste_alpha_identity: false    # eda04: production=normal+closing+waste+carry 항등식
  overproduction_breakdown: true # eda05: 과잉생산 카테고리
hypotheses:
  demand_absorption: true        # W0 카테고리 총량 수요이전 흡수(β/TOST)
  substitution: false            # MNL/nested/DiD substitution
  stockout_revenue: true         # 매진→매장 시간당 매출 무영향 가정
  closing_discount: true         # 마감할인 코드(0069/0077) 시각분포·α
  other_discounts: false         # 마감 외 할인코드 매장×시각 분포
  seasonal_bias: false           # 주말·여름 계절 편향
  weather_bias: false            # 극한날씨 nonlinear 편향
  weekday_bias: false            # 평일 과대예측(iso-waste)
  holiday_premium: false         # 공휴일 프리미엄 분해
  month_dow_adjust: false        # 월×요일 조정 전후
  popularity_stockout: false     # 인기품 매진 재검증
  modeling_v4_assumptions: false # v4 framework 4가정
```

on/off는 명시된 것만; 미명시 항목은 default off. `store: multistore`면 4매장 대상 핸들러(매장간 비교 가설)가 활성.

## 5. Registry 이식 범위 (기존 가설 전부 — 사용자 결정)

**데이터 분석 (eda01~05):** sales_distribution, category_mix, waste_rate, waste_alpha_identity, overproduction_breakdown.

**가설 (verify/diag/track/substitution/absorption):**
| registry 키 | 출처 스크립트 | 프리미티브 |
|---|---|---|
| demand_absorption | absorption_4stores, diag_assumptions_multistore | `analysis/demand_absorption.py` |
| substitution | substitution_4stores | `analysis/{substitution,mnl_substitution,nested_logit,substitution_did}.py` |
| stockout_revenue | verify_stockout_revenue_4stores/fixed | `analysis/self_fulfillment.py` |
| closing_discount | verify_closing_codes | `analysis/{discount,closing_demand}.py` |
| other_discounts | verify_other_discounts | `analysis/discount.py` |
| discount_regime | (신규 노출) | `analysis/discount_regime.py` |
| seasonal_bias | track3_seasonal_diagnose | `analysis/seasonal.py` |
| weather_bias | track4_weather_diagnose | — |
| weekday_bias | weekday_bias_isowaste | — |
| holiday_premium | holiday_premium_decompose | — |
| month_dow_adjust | verify_month_dow_adjust | — |
| popularity_stockout | revalidate_popularity_stockout | `analysis/popularity.py` |
| modeling_v4_assumptions | verify_hypotheses | `analysis/basket_composition.py` |
| event_prior_validation | verify_event_prior | `models/event_prior.py` |

**제외(DEPRECATED, v5 conformal 구간예측 폐기):** `diag_anchor_gh`, `diag_chuseok_gh`, `diagnose_conformal_residual`(conformal 계열 — 점추정+위험수치 전환으로 폐기). 이식하지 않고 스펙에 제외 근거 기록.

전부 이식은 **점진적으로**: 프레임워크(spec/registry/runner/report/inputs) 먼저, 각 가설은 "스크립트 로직 추출→registry 등록→HTML 그래프 확인"을 한 항목씩 TDD로. 구현 플랜에서 태스크 분할.

## 6. HTML Report 구조

harness `report.py`(자기포함 plotly, stateless `fig_to_div`) 패턴 재사용:
- 헤더: spec 이름, 데이터 소스, 실행 항목 on/off 요약 표.
- **섹션 A — 입력 데이터 분석**: 켜진 각 분석 = 제목 + plotly 그래프(들) + 요약표.
- **섹션 B — 가설 검증**: 켜진 각 가설 = 제목 + 그래프 + 결과표 + **판정**(지지/기각/불확실 + 근거 수치). 가설별 caveat(예: 데이터 한계, censoring) 명시.
- off 항목은 "(off)"로 표기(은폐 방지).

## 7. 성공 기준

- `bakery analysis-run experiments/analysis_gwangyo.yaml` → `analysis_report.html` 생성, 켜진 분석/가설만 실행·표시.
- 각 registry 항목이 출처 스크립트의 핵심 수치를 재현(이식 정확성 — 항목별 회귀 대조).
- data_analyses·hypotheses 각각 독립 on/off 동작(끈 것은 실행 안 됨).
- 입력 데이터만 사용(모델 예측값 미참조) — 데이터 분석/가설 검증 평면의 경계 준수.
- 기존 테스트 + 신규 항목 테스트 통과.

## 8. 리스크

- **이식 폭 큼**: 14+ 가설 전부 이식은 범위 큼 → 구현 플랜에서 프레임워크 + 항목별 태스크로 분할, 항목마다 회귀 대조.
- **스크립트 로직 추출 시 동작 변화**: 스크립트가 print 기반이라 반환값 계약이 없음 → 추출 시 순수함수화 + 출처 수치와 대조(harness 엔진 동등성 게이트와 같은 정신).
- **데이터 소스 정합**: 입력 로더가 canonical/multistore/discount를 신규 소스로 읽는지(phase 7 Task 0 closing 재배선 반영). event_prior fix(별도 PR)와 무관(이 레이어는 예측 안 함).
- **그래프 자산 승격**: `build_dashboard.py`(1209줄 plotly)·`weekly_overlay_series.py` 등 기존 시각화 재사용 가능 — 신규 작성 최소화.
- **DEPRECATED 혼입**: conformal 계열 3종 제외를 registry에 명시(실수로 이식 방지).

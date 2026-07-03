# 카테고리 총량 수요이전 흡수 직접 검증 (W0 게이트) — Design

**Date**: 2026-07-03
**Depends on**: daily 스키마(category_id/sold_units/is_stockout/stockout_time/open_hours), 기존 substitution 인프라(analysis/), q_order_top의 lost-hours 개념
**Status**: Design approved (2026-07-03)

## 목적

W0 게이트 = **"카테고리 총량 예측 → 품목 비율 배분"(v4 Stage 1→2) 모델링이 정당한가**의 판정. 절대규칙: "품목비율은 수요이전 검증 통과 시에만"(CLAUDE.md, poc_scope_v7 §W0). 통과해야 Stage 2(품목 비율 배분) 진입.

정당 조건: **개별 품목이 조기 품절돼도 그 수요가 같은 카테고리 다른 품목으로 흡수되어 카테고리 총 판매량이 보존**되어야 한다. 그래야 카테고리 총량이 안정적 예측 대상이 되고 품목 배분은 규칙으로 처리 가능.

## 기존 검증과의 차이 (왜 새로 필요한가)

기존 6개 방법(RD/MNL/Nested/DiD/매출검정/카테고리경계)은 W0를 **간접** 지지할 뿐, 카테고리 총량 흡수를 직접 재지 않았다:
- RD/MNL/Nested/DiD → **개별 i→j** 전환 측정 (β≈0, λ≈0.99). "개별 substitution 약함"이지 "카테고리 총량 보존"이 아님.
- 매출검정(2-1-a) → **매장 전체** 매출 (카테고리 아님), 게다가 "매진일=고수요일" confound에 노출("매진일 매출 +5~14%"는 흡수가 아니라 reverse causality일 수 있음).

본 검증은 **카테고리 총량 단위 흡수를 직접 측정**하고 그 confound를 명시적으로 통제한다.

## 핵심 난제 (LOCKED 이해)

**품절강도와 카테고리 총량이 둘 다 그날 고수요의 결과** (reverse causality). 순진하게 "품절 많은 날 총량 높다"를 보면 흡수가 아니라 고수요 공통원인을 흡수로 오독한다. 이 confound를 끊는 것이 설계 전체의 목적.

추가 제약:
- **광교 무품절일 = 0일** → "품절 vs 완전 무품절" 전체 효과는 식별 불가. "품절강도 연속" 회귀 형태로 간다.
- **potential_demand 금지** — `attach_potential_demand`가 `outflow_ratio`(카테고리 내 substitution 비율)로 censoring 보정을 shrink한다(potential_demand.py:140,189). 흡수 가정을 내장하므로 타깃/변수로 쓰면 순환논증. **타깃은 raw `sold_units`.**

## 핵심 설계 결정 (LOCKED)

### D1. leave-one-out 총량보존 계수 검정
흡수를 "카테고리 총 sold가 품절강도에 얼마나 반응하는가"의 회귀 계수 β로 잰다. β≈0 = 절단분이 나머지 품목으로 흡수돼 총량 보존, β<0 = walk-away(총량 감소). i 손실분(counterfactual 잠재수요) 추정이 불필요 → potential_demand 순환 회피. 흡수율(%) 정량은 손실분 추정이 필요해 순환 위험이라 채택하지 않음.

### D2. 이중 confound 통제
"고수요일=품절많은날"을 두 겹으로 끊는다:
1. **다른 카테고리 총 sold** (c 제외) — 그날 매장 전반 수요의 *c-독립* proxy. 전반적 고수요 흡수.
2. **c baseline 기대수요** = c의 최근 4주 동일요일 평균 sold — *c만 유독 잘 팔린 날*의 편향 차단. **lag 기반, 그날 이전 정보만**(time leakage 절대금지 규칙 준수).
단독 통제로는 부족: 통제 1만 쓰면 c-specific 수요충격이 β를 +로 편향(흡수 과대평가), 통제 2만 쓰면 그날 전반 traffic을 놓친다.

### D3. 판정 = TOST equivalence (게이트)
"β가 음수로 유의하지 않다"(귀무 채택)는 검정력 부족을 통과로 오독하므로 지양. **TOST equivalence test**로 "β의 90% CI가 [−δ, +δ] 안"일 때 흡수 선언. δ = 품절강도 IQR 변화가 카테고리 총량의 **5% 미만**에 해당하는 β 값. (경제적 무시가능성 임계.)

### D4. raw sold_units 타깃 (D 제약 계승)
종속·통제 모두 raw sold_units 기반. potential_demand 파생 변수 일절 미사용.

### D5. 다매장·카테고리별 일관성
광교 + 삼성/메세나/광화문 각각 실행. 삼성은 매출검정에서 lost-sales 신호가 있었으므로, 여기서 β<0이 나오면 상호검증(신뢰↑), 안 나오면 재검토 트리거. cake/시즌 카테고리는 "진짜 substitute 어려움" 예상 → 별도 리포트, 게이트 판정은 일반 카테고리 기준.

## 모델

```
Y_cd = β·T_cd + γ·OtherCatSold_cd + δ_b·c_baseline_cd + FE_dow + FE_month + trend + ε_cd
```
- **Y_cd** = 카테고리 c, 날짜 d의 총 sold_units (raw)
- **T_cd** = c 품절강도 = Σ_{i∈c} max(close_hour − stockout_time_i, 0), 매진 없는 품목 0 (q_order_top lost-hours 동형)
- **OtherCatSold_cd** = 같은 매장·날짜의 c 외 전 카테고리 sold 합
- **c_baseline_cd** = c의 (d 기준) 최근 4주 동일요일 평균 sold (lag; 부족하면 가용 주 평균, 최소 관측 수 미달 행은 제외)
- **FE_dow, FE_month, trend** = 요일·월 고정효과 + 선형 트렌드
- 추정: OLS(HC3 robust SE) 또는 매장 FE 포함. per (store) × per (category) 또는 category-pooled with store FE.

판정: TOST(β, δ) → 흡수 여부. 부호·유의성·매장/카테고리 일관성 보조 보고.

## 강건성

- **placebo**: T를 미래 품절강도(d+7)로 치환 → β 유의하면 허위상관 경보.
- **처치 sensitivity**: lost-hours vs 조기품절 품목 수(count) 두 정의로 β 재현성.
- **통제 ablation**: 통제 1만 / 2만 / 둘 다 → β 이동 방향 보고(이중 통제 정당성 입증).
- **cake 제외 vs 포함** 비교.

## 모듈 / 산출

- `src/bakery/analysis/demand_absorption.py`:
  - `build_absorption_panel(daily, *, close_hour=22, baseline_weeks=4) -> DataFrame` — (store, category, date) 패널 + T/Y/OtherCatSold/baseline/FE 재료. leakage-safe baseline.
  - `fit_absorption(panel, *, store_id=None) -> AbsorptionResult` — 회귀 + TOST. β, CI, δ, 판정.
  - `run_absorption(daily, ...) -> dict[store→list[AbsorptionResult]]` — 매장×카테고리 전체.
- CLI: `bakery demand-absorption --source real` → `reports/demand_absorption/{panel.parquet, results.csv, verdict.md}`
- 테스트: 합성 fixture로 (a) 완전흡수 시나리오 β≈0 회복, (b) walk-away 시나리오 β<0 회복, (c) baseline leakage-safe(미래 미참조), (d) TOST 판정 경계, (e) 처치=0 카테고리 가드.

## 성공 기준 (게이트)

- 일반 카테고리(cake/시즌 제외) 다수가 **TOST 흡수** 판정 → **W0 통과** → Stage 2 진입 허가.
- 광범위 β<0(walk-away) → **게이트 실패** → 카테고리 총량 모델링 재검토(품목 단위 유지 또는 흡수 부분반영).
- 혼재 → 카테고리/매장별 차등 결론 + 사용자 판단.

## 범위 밖

- 흡수율(%) 정량 (손실분 추정 순환 위험).
- cross-category 흐름 정량 (bread→pastry) — 별건.
- 무품절일 반사실 (데이터상 식별 불가).
- 운영 데이터(실 폐기량) 의존 캘리브레이션.

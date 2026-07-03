# W0 게이트 — 카테고리 총량 수요이전 흡수 검증 결과

**Date**: 2026-07-03
**설계**: `docs/superpowers/specs/2026-07-03-demand-absorption-design.md`
**코드**: `src/bakery/analysis/demand_absorption.py`, CLI `bakery demand-absorption --source real`
**데이터**: 광교 단독(store_gw01), bonavi 5년, raw sold_units

## 검증 질문

"카테고리 내 품목이 조기 품절되면 그 수요가 같은 카테고리 다른 품목으로 흡수되어 **카테고리 총 판매량이 보존**되는가?" (통과 시 v4 Stage 1→2 = 카테고리 총량 예측 → 품목 비율 배분 모델링 정당)

## 방법 (요약)

leave-one-out 총량보존 계수 β 회귀:
```
카테고리 총 sold = β·품절강도 + γ·다른카테고리 sold + δ·c baseline(lag) + 요일·월 FE + trend
```
- β≈0 → 품절돼도 총량 보존 = **흡수** / β<0 → **walk-away**
- 판정 = TOST equivalence, δ = 품절강도 IQR 변화가 카테고리 총량 5%에 해당하는 β
- 타깃 raw sold_units (potential_demand는 흡수 가정 내장 → 순환, 배제)
- confound(고수요일=품절많은날) 이중 통제: 다른 카테고리 traffic + c baseline(동일요일 4주 lag, leakage-safe)

## 결과 (광교, close_hour=22)

| 카테고리 | 품목수 | β | 90% CI | δ | 판정 |
|---|---|---|---|---|---|
| **bread** | 29 | **+0.014** | [−0.000, +0.028] | 0.042 | **absorb** ✅ |
| **pastry** | 54 | **+0.043** | [+0.027, +0.060] | 0.070 | **absorb** ✅ |
| sandwich | **1** | +0.041 | [+0.016, +0.066] | 0.028 | inconclusive → **구조적 제외** |
| cake | 3 | +0.063 | [+0.024, +0.102] | 0.025 | inconclusive → **시즌 별도** |

### placebo (미래 d+7 품절강도로 회귀 — 허위상관 체크)

| 카테고리 | placebo β | 판정 |
|---|---|---|
| bread | −0.015 | absorb |
| pastry | +0.043 | absorb |
| cake | +0.005 | inconclusive |
| sandwich | +0.007 | inconclusive |

## 해석

1. **walk-away(β<0)가 어떤 카테고리에서도 나오지 않았다** (실제·placebo 모두). 품절강도가 올라도 카테고리 총량이 **감소하지 않음** — 흡수 반대 신호 전무. 이것이 게이트 판정의 1차 근거.
2. **다품목 일반 카테고리(bread, pastry)는 TOST absorb** — β의 90% CI가 δ 안. 품절이 카테고리 총량을 무너뜨리지 않고 나머지 품목이 흡수.
3. **sandwich는 단일 품목** → 카테고리 내 흡수할 "다른 품목"이 없어 leave-one-out이 성립하지 않는다. 흡수 검증 대상이 아니라 구조적 제외 (품목=카테고리이므로 비율 배분도 불필요).
4. **cake는 시즌 3품목** → inconclusive지만 β>0(walk-away 아님). 설계대로 별도 모델(예약·시즌) 대상.

### 정직한 한계

- β가 미세 양수이고 placebo에서도 pastry β가 실제와 같게 나온다. 이는 β 양수가 순수 인과 흡수만이 아니라 **잔차 고수요 confound**(이중 통제 후에도 남는 그날 수요 상관)를 일부 포함함을 시사한다. 이중 통제로 편향을 줄였으나 완전 무편향 식별은 아니다.
- 그러나 이 한계는 흡수 **과대평가** 위험일 뿐, **walk-away 부재 확증**에는 영향이 없다 — placebo β가 잔차 confound 크기를 **δ 이내로 하한**시킨다: pastry는 placebo +0.043 < δ 0.070, bread는 |실제 0.014 − placebo (−0.015)| = 0.029 < δ 0.042. 즉 관측 β 전부가 confound라고 최악으로 가정해도 실제 인과 효과는 여전히 동등성 구간 안에 있다 → walk-away로 뒤집힐 여지가 없다. β<0 부재가 핵심 결론.
- 광교 무품절일 = 0일이므로 "품절 vs 완전 무품절" 전체 효과가 아니라 "품절강도 한계 효과"를 측정한 것이다.
- 광교 단독. 다매장(삼성/메세나/광화문) 일반화는 store_daily 경로로 후속(삼성은 매출검정에서 lost-sales 신호가 있어 우선 확인 대상).

## W0 게이트 판정: **통과 (일반 다품목 카테고리)**

- **bread, pastry 흡수 확인** → 카테고리 총량 예측 → 품목 비율 배분(v4 Stage 1→2) **정당성 지지**. Stage 2 진입 허가.
- **sandwich**: 단일 품목 → 비율 배분 불필요(품목=카테고리). Stage 2 대상 아님.
- **cake**: 시즌 별도 모델(예약·특수). 일반 흡수 framework 밖.

## 모델링 확정

1. **v4 3-stage 유지·정당화**: Stage 1(카테고리 총량 quantile) → Stage 2(품목 비율 배분). bread/pastry에 적용.
2. **카테고리별 차등**: sandwich는 단일품목 직접 예측, cake는 시즌 모델 분리.
3. **다매장 도착 시**: 동일 검증을 store_daily 경로로 재실행, 특히 삼성타운 β 부호 확인(walk-away 신호 상호검증).

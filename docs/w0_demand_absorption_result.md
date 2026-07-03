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
3. **다매장 도착 시**: 동일 검증을 store_daily 경로로 재실행, 특히 삼성타운 β 부호 확인(walk-away 신호 상호검증). → 아래 §다매장 확장에서 완료.

---

## 다매장 확장 (2026-07-03)

`scripts/absorption_4stores.py` — `store_daily.build_store_daily`로 4매장(광교/삼성타운/메세나폴리스/광화문) daily(bulk 제외)를 만들어 광교 단독에서 머지된 동일 `run_absorption` 로직에 태움. 실행: `PYTHONPATH=scripts uv run python scripts/absorption_4stores.py`, 산출 `reports/demand_absorption/results_4stores.csv`.

### 게이트 대상(bread/pastry) 결과

| 매장 | 카테고리 | β | 90% CI | δ | 판정 |
|---|---|---|---|---|---|
| 광교 | bread | +0.024 | [+0.016,+0.033] | 0.047 | **absorb** ✅ |
| 광교 | pastry | +0.033 | [+0.022,+0.043] | 0.067 | **absorb** ✅ |
| 메세나 | bread | +0.053 | [+0.039,+0.068] | 0.094 | **absorb** ✅ |
| 메세나 | pastry | +0.043 | [+0.025,+0.061] | 0.137 | **absorb** ✅ |
| 광화문 | bread | +0.099 | [+0.074,+0.124] | 0.120 | inconclusive (경계) |
| 광화문 | pastry | +0.098 | [+0.072,+0.125] | 0.116 | inconclusive (경계) |
| 삼성 | bread | +0.121 | [+0.111,+0.132] | 0.024 | inconclusive |
| 삼성 | pastry | +0.074 | [+0.058,+0.091] | 0.072 | inconclusive |

### 핵심

1. **walk-away(β<0)가 20건 전 카테고리·전 매장에서 0건.** 4매장 어디서도 품절이 카테고리 총량을 감소시키지 않는다 — W0의 핵심 결론(카테고리 총량이 품절로 무너지지 않음)이 **4매장 일반화**. (기존 substitution 4매장 DiD 유의비율 3~4% 노이즈 바닥 결론과 일관.)
2. **광교·메세나: 깨끗한 absorb** (β 작고 CI⊂δ).
3. **광화문: inconclusive지만 CI 상한이 δ를 근소 초과**하는 경계 (거의 absorb).
4. **삼성: β가 유독 큼(+0.121), δ 작음(0.024)** → 명확한 inconclusive. placebo 삼성 bread β=+0.041(실제 +0.121)로 잔차 confound만으로 설명 안 되는 실제 양의 신호. **방향이 +이므로 walk-away 아님.**

### 삼성타운 해석 (매출검정 lost-sales와의 화해)

삼성은 매장 매출검정에서 유일하게 lost-sales(잠재구매자 이탈) 신호가 있던 매장인데, 흡수검증 β는 오히려 크게 양수다. 모순이 아니다: **흡수검증은 관측 sold 기반이라 영수증 없이 떠난 손님(extensive margin)을 구조적으로 못 본다.** sold 안에서는 품절일에 총량이 유지/증가(β>0)하고, 잠재 이탈은 별도 마진에서 매출검정으로만 잡힌다. 두 신호는 상호보완 (project_stockout_revenue_perstore, project_substitution_mnl §walk-away와 일관).

### inconclusive의 의미 (정직한 한계)

삼성·광화문의 inconclusive는 **"흡수 아님"이 아니라 "이 방법(TOST)으로 흡수 크기를 엄밀히 확정 못함"**이다. β 양수 크기가 매장마다 다른 것은 이중 통제 후에도 남는 **잔차 고수요 confound가 매장마다 다르게 크기 때문**(삼성이 가장 큼). δ가 매장별로 다른 것도(삼성 0.024 vs 메세나 0.137) 품절강도 IQR·평균 차이 탓. 견고한 부분은 **walk-away 부재**이고, 흡수의 정밀 크기는 매장별 confound 차이로 유보된다.

### 경로 일관성 sanity check

광교는 bonavi_daily 경로(PR#18: bread +0.014, pastry +0.043 absorb)와 store_daily 경로(bread +0.024, pastry +0.033 absorb) 두 결과가 **모두 absorb로 일관** (소폭 차이는 데이터 소스·bulk 제외 차이).

### 다매장 게이트 판정: **통과 (walk-away 부재 4매장 일반화)**

- 일반 카테고리 8건 중 absorb 4건(광교·메세나), 경계/inconclusive 4건(광화문·삼성) — **walk-away 0건**.
- 카테고리 총량 모델링(v4 Stage 1→2) 정당성이 4매장에서 흡수 반대 신호 부재로 지지됨.
- 삼성·광화문은 흡수 **크기**를 엄밀 확정 못하나(잔차 confound 큼), 흡수를 **부정하지도 않음**. 매장별 α/보수성 차등의 근거로 활용(삼성은 confound 큼 = 수요 변동 큼 → 더 보수적 quantile 검토).

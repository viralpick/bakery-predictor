# Phase B — 발주 최적화 결과 (정직본)

**Date**: 2026-07-04
**매장/범위**: 광교(보나비), bread·pastry 카테고리 총량, 2021~2025 (백테스트 1,736일)
**질문**: 원가율 c에서 비용최적 발주는? 현행 발주는 마치 원가율 얼마처럼 행동하나(implied c)? 재최적화로 얼마 절감되나?
**배경**: Phase A가 structural α 식별 실패(α_A=NaN, 저녁 상시 마감할인 confound) → operational 전환.

## 요약 (한 줄)

**광교 매장의 발주는 이미 대략 cost-rational이다.** 현행 발주의 **implied c ≈ 0.20(bread) / 0.28(pastry)** — 베이커리 원가율 상식 범위. 그래서 우리의 (요일 조건부) newsvendor는 **가정 원가율이 ~0.45(bread)/0.55(pastry)를 넘어야만** 매장을 이기고, 그 아래에선 매장이 더 낫다. "큰 폐기 절감 여지"라는 가설은 이 분석으로는 지지되지 않는다.

## 주 산출물 — implied c of current policy

| 카테고리 | 조건부 service level | **implied c** |
|---|---|---|
| bread | 0.796 | **0.204** |
| pastry | 0.717 | **0.283** |

해석: 매장 현행 발주(made)는 그날 요일 조건부 수요분포에서 bread 79.6% / pastry 71.7% 분위수에 위치. 이는 "원가율 c=0.20/0.28인 newsvendor처럼 발주"와 동치. **이 값이 베이커리 실제 원가율(대략 0.25~0.40 추정)에 가깝다** → 매장 발주가 이미 합리적. 갭(implied c ≪ 실제 c)이 크지 않아 과잉생산 여지가 제한적.

> ⚠️ 조건부 vs pooled: 단순 pooled service level(made≥demand 비율)은 ~95%(→implied c 0.05)로 더 높게 나오나, 이는 그날 요일·변동성을 무시한 값. **요일 조건부(매장이 실제 직면하는 변동성 반영)가 방법론적으로 옳고**, 0.20/0.28이 정직한 수치다.

## 보조 — 재최적화 절감액 (5년 누적, c별)

`savings = cost(현행 made) − cost(newsvendor Q*(c))`, 동일 실현수요·newsvendor-정합 비용:

| c | bread 절감(원) | pastry 절감(원) |
|---|---|---|
| 0.25 | **−10.0M** | **−26.8M** |
| 0.35 | −5.7M | −21.1M |
| 0.45 | **+1.4M** | −11.6M |
| 0.55 | +11.5M | +3.1M |

**절감이 음수 = 매장이 우리 모델을 이긴다.** 절감이 양수로 도는 c(bread≈0.45, pastry≈0.55)는 각 카테고리의 implied c보다 훨씬 높다 — 즉 실제 원가율이 그만큼 높아야만(폐기가 그만큼 비싸야만) 생산을 줄이는 게 이득. 플러그인 placebo arm(Q=made)로 유령 절감(품목배분 완벽 가정분)은 분리했다.

**왜 매장이 이기나**: 매장은 요일·트렌드·행사·현장 정보를 이미 반영해 발주한다. 우리의 순진한 요일 조건부 rolling 분위수는 그 조건화 능력에 못 미친다(Fable 검토 예측대로). 이건 모델 실패가 아니라 **"매장 발주가 이미 좋다"는 정직한 발견**이다.

## 정직한 한계

- **부분일 제외**: identity_diff 위반 품목-행 제외로 bread **15.7%**·pastry **35.9%** 일자가 품목 하나 이상 빠져 그날 카테고리 총량이 과소집계될 수 있음(행 제외율 1.6%/2.7%). implied c에 소폭 영향 가능 — identity_diff 근본원인(어느 품목/기간) 미규명, 후속 필요.
- **censored 수요**: 카테고리 out==0(수요≥made) 일자(~1.5~1.9%, 카테고리 단위라 드묾) 보수 처리.
- **d 내생성**: 반사실은 made 근방 한계 재최적화로 한정(대폭 감산 시 수요분포 자체가 이동).
- **Level2 vs 스코어링**: Q\* Level2는 two-class 이익 최대화, 절감 스코어는 single-class newsvendor 비용 → 지표 불일치 소폭. cost_l1(single-class 최적)도 함께 보고, 결론(매장이 저-중 c서 이김) 불변.
- **c 미지**: implied c·절감 모두 c 곡선. 고객사 c 제공 시 점으로 확정.
- **α_A=NaN**: Phase A structural α 부재라 α_A ∩ α\*(c) 교차검증 없음.
- 광교 한정(카테고리 총량=참수요는 W0 흡수 가정, 현행 정책·공급 하).

## 재현
```bash
uv run bakery phaseb-order   # reports/phaseB_implied_c.csv, phaseB_order_savings.csv
```
데이터: `data/internal/v2/waste_alpha_4stores.parquet`(광교, made/out/normal/closing/sold_total/unit_price), 카테고리 map=`bonavi_loader.load_items`.

## 결론 + 다음

**결론**: 광교 발주는 implied c ≈0.20~0.28로 이미 합리적. 순진 재최적화의 절감은 저-중 원가율에서 음수 → PoC의 "폐기 절감" 가치는 (이 매장·이 카테고리 총량 수준에선) 제한적. 진짜 여지는 (a) 품목 배분(Stage-2) (b) 원가율이 정말 높은 품목 (c) 조건화를 매장보다 잘하는 수요모델에 있음.

**다음**:
1. identity_diff mismatch 근본원인 규명(부분일 제외 편향 제거).
2. Phase A 다중시각 재검증(할인정책 자연실험 등) — α 재식별.
3. 고객사 c 수령 → implied c 갭·절감 점 확정.
4. 수요모델을 v4 LGBM(트렌드·날씨·행사 조건화)로 격상 시 매장 대비 우위 재평가.

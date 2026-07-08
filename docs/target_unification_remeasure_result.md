# Target 통일(adjusted_demand) 재측정 결과 — PR#26·#27 재판정

**날짜**: 2026-07-08
**브랜치**: feat/stockout-redefine
**설계**: [target-unification-remeasure-design](superpowers/specs/2026-07-08-target-unification-remeasure-design.md)
**실행 조건**: `--source real --store-id store_gw01 --n-folds 8 --production-quantile 0.85` (full-window rolling, 광교 단독)
**원본 로그**: `reports/log_remeasure_{item,cat}_a{0.3,0.5,0.7,1.0}.txt`, 요약 `reports/remeasure_numbers.txt`, CSV `reports/remeasure_*.csv`

---

## 1. 배경 — 왜 재측정하나

전향 retro harness(`prospective-eval`)로 낸 두 과거 결론이 **잣대 오염**으로 무효화됐다.

- **PR#26 (item-level de-risk, [retro_harness_result](retro_harness_result.md))**: q0.85 발주 vs 아티제 현행. 발주 모델과 평가 잣대가 **둘 다 `potential_demand`** — target은 정합했으나 `potential_demand` 필드 자체가 오염됐다(로더가 하루 다중 품절이벤트 중 첫 것만 취해 `stockout_time`이 이르게 찍힘 → `is_stockout` 92% false → 배수 복원으로 수요 부풀림). 초과율 0.636, WPE −0.261.
- **PR#27 (category-level, [retro_harness_result](retro_harness_result.md) §category)**: 발주는 `adjusted_demand_unit`(마감할인 실수요 조정)로 학습 ↔ 평가는 `potential_demand`(조정 없음) → **target 불일치 confound**. 초과율 0.738(item-day), 0.900(카테고리-총합), WPE −0.306.

`is_stockout`는 재정의(92%→60.4%, 하위1/PR#28)로 이미 고쳤고, `potential_demand`는 폐기 확정, 수요 target은 **adjusted_demand**(정상판매 + 마감할인×α)로 확정됐다. 이번 재측정은 harness의 **발주 target과 평가 잣대를 둘 다 adjusted_demand로 통일**(Task 1~5)해 apples-to-apples로 다시 잰 것이다.

- item 경로: v2 LGBM을 `y_col="adjusted_demand"`로 재학습(+`run_backtest._clone` y_col 보존 수정) — potential 잔여 오염 제거.
- category 경로: 이미 adjusted 학습이라 **평가 잣대만** potential→adjusted로 교체.

---

## 2. 표 1 — 원본(potential 잣대) vs 재측정(adjusted 잣대), α=0.5, n_folds=8

Δ 부호 규약: **우리 − 아티제(baseline)**. waste/lost_margin은 원화, soldout_median_h는 시간(h). 재측정 Δ는 fold-평균(8-fold)이며 CI95는 §4 참조.

### item-level

| 지표 | PR#26 원본 (potential) | 재측정 (adjusted) |
|---|---|---|
| **calibration 초과율** P(demand>order), nominal 0.15 | **0.636** | **0.679** ⬆ (악화) |
| WPE (발주 편향) | −0.261 | −0.301 |
| stockout_rate Δ | −0.017 | **+0.603** (우리 0.679 / base 0.077) |
| waste_cost Δ (fold-mean) | +3.82M | **−3.01M** (부호 반전) |
| lost_margin Δ (fold-mean) | +52.3M† | +18.12M |
| soldout_median_h Δ | −2.03 | −4.24 |

### category-level

| 지표 | PR#27 원본 (potential) | 재측정 (adjusted) |
|---|---|---|
| **calibration 초과율** (item-day), nominal 0.15 | **0.738** | **0.458** ⬇ (개선) |
| calibration 초과율 (카테고리-총합) | 0.900 | 0.346 |
| WPE (발주 편향) | −0.306 | **+0.039** (편향 거의 소멸) |
| stockout_rate Δ | +0.084 | +0.380 (우리 0.458 / base 0.076) |
| waste_cost Δ (fold-mean) | −0.90M | −1.62M |
| lost_margin Δ (fold-mean) | +6.7M | +4.88M |
| soldout_median_h Δ | −1.35 | −0.64 |

† PR#26 원본의 lost_margin Δ는 aggregate 프린트 값(문서 [retro_harness_result](retro_harness_result.md)에는 fold-mean +6.5M로 별도 집계). 여기서는 aggregate-프린트 비교 축을 그대로 옮겼다. 재측정 값은 모두 fold-mean(8) + CI(§4)로 재확인.

**⚠️ KPI 부호 반전의 원인 (반드시 읽을 것)**: 재측정에서 waste Δ가 양→음으로, stockout Δ가 0→+0.6으로 뒤집힌 것은 "모델이 좋아져서"가 아니다. 잣대를 potential(부풀린 값)에서 adjusted(더 낮은 실수요 조정값)로 바꾸자, **아티제 현행 발주(baseline = 실제 생산량, 판매량 기반)가 adjusted 수요 대비 과잉생산으로 재해석**됐다(baseline waste 35M, stockout 0.077 = 거의 매진 안 남). 우리 q0.85 발주는 더 낮은 adjusted target을 겨눠 폐기는 적지만 심하게 과소서빙(우리 stockout 0.679, lost_margin 큼). 즉 KPI Δ 자체는 "우리가 겨누는 target(adjusted) ≠ 아티제가 실제 만든 양(sold 기반)"이라는 **스케일 불일치에 오염**된다. 순수하게 내부 정합적인 신호는 **calibration 초과율**(같은 잣대에서 order vs demand)이며, 판정은 그것을 근거로 한다.

---

## 3. 표 2 — α 민감도 (0.3 / 0.5 / 0.7 / 1.0), n_folds=8, q=0.85

초과율의 α-불변 여부 확인 + KPI 원화 Δ의 α-민감도. (Δ = 우리 − 아티제, fold-mean)

### item-level

| α | 초과율 | WPE | waste_cost Δ | lost_margin Δ | stockout_rate Δ | soldout_median_h Δ |
|---|---|---|---|---|---|---|
| 0.3 | 0.648 | −0.271 | −3.24M | +16.81M | +0.572 | −4.09 |
| 0.5 | 0.679 | −0.301 | −3.01M | +18.12M | +0.603 | −4.24 |
| 0.7 | 0.706 | −0.331 | −2.72M | +19.67M | +0.618 | −4.66 |
| 1.0 | 0.731 | −0.362 | −2.22M | +21.52M | +0.639 | −4.87 |

### category-level

| α | 초과율(item-day) | 초과율(cat-총합) | WPE | waste_cost Δ | lost_margin Δ | stockout_rate Δ | soldout_median_h Δ |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.452 | 0.317 | +0.043 | −1.98M | +4.96M | +0.374 | −0.74 |
| 0.5 | 0.458 | 0.346 | +0.039 | −1.62M | +4.88M | +0.380 | −0.64 |
| 0.7 | 0.465 | 0.368 | +0.035 | −1.23M | +4.85M | +0.376 | −0.67 |
| 1.0 | 0.471 | 0.384 | +0.033 | −0.60M | +4.80M | +0.376 | −0.47 |

**α-불변성**: 설계는 "발주·평가가 같은 α를 공유하므로 초과율은 대체로 α-불변"을 예상했다. 실측은 그보다 약하다.
- **category**: 거의 불변(0.452→0.471, 폭 0.019). 예상대로.
- **item**: **약하게 단조 증가**(0.648→0.731). 완전 불변 아님. α↑ → adjusted target에 마감할인 물량이 더 편입 → target 스케일·분산 상승 → q0.85 LGBM이 더 두꺼워진 tail을 못 덮어 초과율이 오히려 상승. 즉 α는 초과율에 2차 효과를 준다.

**KPI 원화 Δ의 α-민감도** (예상대로): α↑ → adjusted target이 sold 쪽으로 커짐 → 우리 발주도 커져 **waste Δ 절대값 축소**(item −3.24M→−2.22M, cat −1.98M→−0.60M), **lost_margin Δ 증가**(item +16.8M→+21.5M). 방향·규모 모두 α에 민감하므로, KPI 원화 수치는 α가 실증 확정되기 전까지 절대값으로 신뢰하면 안 된다.

---

## 4. 헤드라인 α=0.5 fold-집계 + CI95

### item (n=8)

| metric | fold-mean | CI95 |
|---|---|---|
| waste_cost_krw | −3.01M | [−3.45M, −2.57M] |
| lost_margin_krw | +18.12M | [+16.70M, +19.53M] |
| stockout_rate | +0.603 | [+0.578, +0.628] |
| soldout_median_h | −4.24 | [−4.47, −4.01] |

초과율 0.679, WPE −0.301. ρ_DS: bread 0.185 / pastry 0.196 / sandwich 0.277.
waste sanity: actual 19,614 / simulated 24,907 (ratio 1.27). scored 13,554 / 62,960 item-day, fold [0..7] 전부 채점.

### category (n=8)

| metric | fold-mean | CI95 |
|---|---|---|
| waste_cost_krw | −1.62M | [−2.13M, −1.10M] |
| lost_margin_krw | +4.88M | [+3.64M, +6.11M] |
| stockout_rate | +0.380 | [+0.339, +0.421] |
| soldout_median_h | −0.64 | [−0.87, −0.41] |

초과율 0.458(item-day) / 0.346(카테고리-총합, WAPE 0.104, WPE +0.039, 448 dates). scored 13,529 / 62,960 item-day, fold [0..7] 전부 채점.

모든 CI가 0을 가로지르지 않는다(각 지표 8-fold 모두 같은 부호) → 방향은 robust.

---

## 5. 판정 (정직 보고)

브리핑의 두 갈래 중 **"confound가 일부만 설명, 잣대 정리 후에도 genuine under-calibration 잔존"** 쪽이다. 더 정밀하게 갈라 보면:

1. **item-level: confound가 원인이 아니다 — under-calibration은 진짜다.**
   item 경로는 PR#26에서 애초에 target 불일치가 없었다(발주·평가 둘 다 potential). 잣대를 오염 없는 adjusted로 바꾸고 데이터까지 고쳐도 초과율은 0.636→**0.679**로 회복은커녕 소폭 악화, WPE −0.261→−0.301. **PR#26의 calibration negative는 데이터 아티팩트가 아니라 v2 quantile 모델 자체의 실제 under-calibration**이다. q0.85가 nominal 0.15를 겨눠야 하는데 68%가 order<demand — tail을 못 덮는다(신호 부족). α를 0.3~1.0로 흔들어도 0.65~0.73에 머물러 robust.

2. **category-level: 총합 BIAS는 confound 아티팩트였다(해소됨) — 그러나 분위 under-calibration은 잔존.**
   PR#27의 핵심 오염이던 **aggregate 편향**(WPE −0.306)은 잣대를 adjusted로 통일하자 **+0.039로 사실상 소멸** — 카테고리 발주는 이제 총량에서 불편(unbiased)하다. 초과율도 0.738→**0.458**로 크게 개선. **여기까지는 PR#27 negative가 target confound 아티팩트였음을 확인**한다. 그러나 item-day 초과율 0.458은 여전히 nominal 0.15의 3배. WPE≈0(총량 불편)인데 초과율이 0.46이라는 건, 카테고리→품목 배분 후 발주가 **q0.85가 아니라 사실상 median(불편 추정)** 수준으로 내려앉아 절반 이상의 품목-일에서 수요를 못 덮는다는 뜻이다. "낮게 편향"이 아니라 "분위 공격성이 배분에서 소실"되는 형태의 genuine 잔존 under-calibration.

3. **adjusted_demand target 전환은 정당하다 — 다만 문제를 고친 게 아니라 드러냈다.**
   통일된 adjusted 잣대는 (a) PR#27의 confound 편향을 제거했고 (b) apples-to-apples calibration을 가능케 했다. 그 결과 두 경로 모두 nominal 0.15로 **회복하지 못하는 진짜 quantile under-calibration**이 노출됐다. 즉 잣대 정리는 옳았고, 남은 과제는 발주 모델의 calibration(production quantile 상향 또는 conformal 보정)이지 잣대가 아니다.

**한 줄 요약**: category의 총합 편향은 confound 아티팩트(해소), item의 under-calibration은 진짜(불변). 통일 잣대에서 두 경로 다 초과율이 nominal 0.15로 회복하지 못함 → adjusted target 전환은 정당하나 v2/배분 quantile의 genuine under-calibration이 잔존하며 별도 calibration 교정이 필요하다.

---

## 6. 캐비엣

- **광교 단독**: `_load_real_daily`가 n_stores≠1이면 raise. 타매장(삼성 lost-sales 신호 등) 프로파일 다를 수 있음 — 4매장 확장은 별도 후속.
- **α 미확정**: 초과율은 category에서 거의 불변이나 item에서 약하게 α-민감하고, **KPI 원화 Δ는 방향·규모 모두 α에 민감**. α 실증 확정([closing_discount_true_demand_result](closing_discount_true_demand_result.md), Phase B) 전까지 KPI 절대값을 신뢰 금지.
- **KPI Δ vs baseline은 스케일 불일치에 오염**(§2 경고): 우리 target(adjusted)과 아티제 실제 생산(sold 기반)의 스케일이 달라, waste/stockout Δ는 "누가 더 낫다"의 순 신호가 아니다. 내부 정합 신호는 calibration 초과율. 실데이터(전향 운영 피드) 수령 시 baseline을 실제 발주로 교체해야 KPI 우열이 의미를 갖는다.
- **potential_demand 잔존 참조**: 이번 경로(`prospective-eval`) 밖 CLI(`backtest` 등)는 아직 potential 사용. 전역 감사·제거는 별도.
- **smoke vs full-window 격차**: n_folds=1 예비 smoke(최근 8주)는 item 0.432 / cat 0.429였으나, full-window n_folds=8은 item 0.679 / cat 0.458. **최근 창이 전체 이력보다 후하다** — 헤드라인은 full-window 값이다.

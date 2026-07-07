# 카테고리-레벨 발주 재측정 (v4 스택 harness 배선) — 설계

작성 2026-07-07. [[project_prospective_derisk_retro]] 후속. de-risk 회고가 드러낸
"전향 harness가 v4 카테고리 결정이 아니라 item-level v2 LGBM 지름길을 테스트 중"이라는
갭을 닫는다.

## 목적

전향 retro harness의 `our_order`를 **item-level v2 LGBM → v4 카테고리 스택(Stage1 총합
예측 + Stage2 품목 배분)**으로 교체하고, 동일 full-window 회고로 **calibration과 KPI를
재측정**한다. item-level 결과(초과율 0.636, soldout −2.03h, lost_margin +4~9M)와 나란히
비교해 "카테고리 결정이 아티제를 이기나 + calibration이 맞나"를 답한다.

핵심 가설: item-level 잠재수요는 검열+흡수 이중편향으로 식별 불가([[project_external_benchmark_research]])
→ 그 레벨의 q0.85가 4× 과소발주였다. **식별 가능한 총합 레벨에서 결정하면 calibration이
nominal(0.15)에 근접**할 것이다(확증 아님, 측정 대상).

## 중요 — "카테고리"의 실제 단위 (정확히)

`features/category_aggregate.build_category_daily`는 3개 타깃 카테고리(bread/pastry/sandwich,
cake 제외)를 `groupby("date")`로 **하나의 combined 일별 총합**으로 집계한다. 즉 v4 Stage1은
**per-category가 아니라 "타깃 카테고리 통합 총합"** 단일 시계열이다. 이는 광교에서
개별 substitution 효과가 약하고 수요가 "한 묶음"이라는 결론([[project_substitution_mnl]])으로
정당화된다. 본 스펙의 "카테고리 레벨"은 이 **통합 총합 레벨**을 뜻한다.

## 데이터 흐름

```
build_category_daily (adjusted_demand_unit target, 3카테고리 통합 총합)
  → fit_category_total(production_q=0.85)   [expanding split, leakage-safe]
  → 일별 통합 총합 q0.85 발주  (date → total_order)
  → item_proportion.distribute_total(history, total_by_date)   [history<target_date 비율]
  → item별 order_qty  (Σ items == total_order)
  → simulate_item_day_kpis  [기존, 실현수요 = potential_demand]  vs  baseline=아티제 생산량
  → compare_policies_by_fold / aggregate_fold_kpis  [기존 재사용]
```

## 컴포넌트 (전부 기존 재사용, 신규는 배선뿐)

- **Stage1**: `models/category_total.fit_category_total`(expected L1 + q quantile) + `features/category_aggregate.build_category_daily`. `production_q=0.85`로 호출(기본 0.90 아님 — 이전 retro와 비교 위해).
- **Stage2**: `models/item_proportion.distribute_total(history, total_by_date)` → `ItemProportionResult.quantities` (date/item_id/qty). `compute_proportions`는 `history < target_date`만 사용.
- **신규 (cli.py)**: `_category_order_predictions(store_id, *, production_quantile, val_weeks, n_folds)` — 위 스택을 fold별 val 창에서 돌려 item-level `our_order`(item_id/date/fold/our_order)를 반환. 시그니처·반환을 기존 `_quantile_backtest_predictions`와 맞춰 다운스트림(`_fill_our_order` 등) 무변경 재사용.
- **분기**: `prospective-eval`에 `--order-level {item|category}` 옵션(기본 `item` = 기존 동작 100% 보존). `category`면 `_category_order_predictions` 사용.
- KPI 집계·진단(초과율/waste sanity/fold CI)은 [[project_prospective_derisk_retro]]에서 만든 것 그대로.

## 측정·비교 (산출물)

1. **통합 총합 calibration 초과율** — P(일별 Σ_item potential_demand > 일별 total_order). item-level 0.636과 직접 대조 (핵심 질문). nominal = 1−0.85 = 0.15.
2. **item-level KPI (배분 후)** — Δ waste/stockout_rate/soldout_median_h/lost_margin, mean±95%CI, 8주×8fold. 이전 item-level retro + 아티제 baseline과 side-by-side 표.
3. **배분 sanity** — 각 date에서 Σ(품목 order) == total_order(보존), 비율 출처가 pre-cutoff인지.

`docs/retro_harness_result.md`에 "카테고리-레벨 재측정" 절 추가(item vs category 나란히).

## 절대 규칙 / 리스크 (플랜에서 반드시 검증)

- **Time leakage 2겹**:
  (a) `fit_category_total`은 item 경로처럼 **expanding train + val 이전만** 학습. `generate_time_splits` 창 경계를 category_daily에 그대로 적용하고, val 이후 데이터로 학습하지 않음을 leakage 테스트로 고정.
  (b) `distribute_total`의 품목 비율은 **`compute_proportions`가 history<target_date만** 쓰므로 설계상 안전하나, **`_per_item_signals`가 실제로 target_date 이후 판매를 참조하지 않는지 테스트로 검증**(이번 스펙 최상위 새 누수면). val 창 실판매가 비율에 새면 누수.
- **품절 데이터 censored**: 도착곡선·KPI는 기존 계약(품절 item-day exclude) 유지.
- **Random split 금지**: 시간순 rolling만(기존 harness 경로).
- **Target confound (문서화, 버그 아님)**: 발주는 `adjusted_demand_unit`(α-blended) 학습, calibration·KPI의 실현수요는 `potential_demand`. 우리 모델 target ≠ 실현수요 proxy라는 운영 현실 — caveat로 명시. 실현수요는 이전 retro와 동일하게 potential_demand로 통일.
- **Synthetic↔Real**: 신규 배선 함수 테스트는 합성 fixture로만.
- **단일매장 가드 유지**.

## 검증

- leakage 회귀(`test_split_leakage`/`test_features_leakage` + `assert_no_leakage`) 그린 유지.
- 신규 단위 테스트(정확값 단언, 합성 fixture):
  - 배분 보존: Σ(품목 order) == total_order.
  - 비율 pre-cutoff: target_date 이후 행을 넣어도 비율 불변.
  - `--order-level item` 기본값이 기존 결과 보존(회귀).
- item-level KPI/calibration은 WAPE/WPE·초과율 짝으로 본다.

## 명시적 비범위 (YAGNI)

- Phase B salvage-newsvendor 엔진 / q-스윕 / conformal 구현 / 신제품 tracker 고도화 / 다매장.
- 배분 규칙 자체 개선(현 `item_proportion` 기본 재사용 — 깊은 배분 품질 검증은 별도 스펙).
- per-category(4분할) 모델(현 v4는 통합 총합; 분할은 별도 결정).

## 성공 기준

- `prospective-eval --order-level category` 가 full-window 8×8fold를 돌려, **통합 총합 초과율**(vs 0.636)과 **배분 후 item-level Δ KPI(CI 포함)**를 산출하고, 이전 item-level retro와 side-by-side로 문서화된다.
- 결론(카테고리 결정이 calibration을 개선하나 / 아티제를 이기나)이 수치로 답해지고, 못 이기면 정직하게 보고한다(방향성 신뢰, target confound·배분오차·production-proxy baseline caveat 병기).

# 발주 quantile conformal calibration — 설계

**날짜**: 2026-07-09
**브랜치**: feat/order-quantile-calibration
**선행**: [[project_stockout_time_bug_and_adjusted_demand]] (③ 재측정 = adjusted 잣대에서도 두 경로 다 nominal 0.15 미회복 → genuine under-calibration), [[project_convex_loss]] (실단가 Cu/Co→newsvendor 0.74, q0.85 과공격), [[project_interval_conformal]] (v5 conformal 잔차마진 near-nominal, DEPRECATED as deliverable)

## 배경 / 문제

③ target 통일 재측정 결과, adjusted_demand 잣대에서도 발주 초과율이 nominal(0.15)로 회복하지 못했다: item 0.679, category 0.458(카테고리-총합 0.346). WPE는 category에서 ≈0(총량 불편)인데 초과율은 여전히 높다 = **분위 공격성이 소실된 genuine under-calibration**. v5 실험([[project_prospective_derisk_retro]] α-sweep)에서 base LGBM은 production_quantile을 0.99까지 올려도 nominal에 못 간다(신호 부족·희소 품목). 즉 **base quantile 상향만으로는 서비스레벨을 못 맞춘다** → 데이터 기반 사후 보정(conformal 잔차 마진)이 필요.

## 목표 / 비목표

**목표**
- item 발주가 **목표 서비스레벨을 실현 초과율로 신뢰성 있게 달성**하도록 하는 conformal 보정 레이어.
- 목표 서비스레벨 = knob, 기본 **cost-optimal 0.74**(Cu/(Cu+Co), CostParams margin_rate 0.5×lost_sale 1.7=0.85 vs cost_rate 0.30).
- 프론티어 스윙(s 0.70~0.95)으로 waste↔stockout 트레이드오프 정량화.

**비목표**
- 구간(interval) 예측 (v5 폐기, 점추정+위험수치 유지).
- category 경로 즉시 적용 (레이어는 path-agnostic 설계, category는 후속).
- cost 절대값으로 아티제 우열 판정 (③: KPI Δ vs baseline 스케일 오염, 전향 실측 전까지 순 신호=초과율).
- 4매장 (광교 단독).
- base 모델 자체 개선/교체 (보정은 사후 레이어).

## 결정 (승인됨)

1. **접근**: coverage calibration — 목표 서비스레벨을 conformal 잔차-마진으로 달성. (cost-optimal 재정식화·단순 q상향 기각.)
2. **기본 목표 레벨**: 0.74 + 프론티어 스윙.
3. **granularity**: scale-정규화 pooled conformal (per-item·per-category 기각 — 희소 품목 강건성).
4. **적용 경로**: item 우선, 레이어 path-agnostic.
5. **base 예측 = q0.5 (median)**: target과 분리 → 프론티어 스윙이 base 재학습 없이 저렴(fold당 base 1회, Q_s만 post-hoc).

## 방법 — one-sided, scale-normalized split-conformal (CQR류)

기호: item i, day. `y_i` = adjusted_demand(실현수요 잣대, ③). `ŷ_i` = base median 예측. `scale_i` = leakage-safe item 규모. 서비스레벨 `s`(기본 0.74).

1. **base**: v2 LGBM, `objective="quantile", alpha=0.5` (median). target과 무관하게 고정.
2. **scale_i**: 해당 item의 adjusted_demand 평균 — 첫 val 창(backtest 시작) **이전 전체 이력**에서 산출, floor `max(mean, 1.0)`. 모든 fold val이 그 이후라 leakage-safe(단일 scale/item).
3. **calibration set — cross-fold half-split** (⚠️nested 아님, lag history 단절 회피): expanding backtest의 각 fold는 기존과 동일하게 "그 fold val 직전 전체"로 base median 학습(lag/rolling 정상). n_folds개 val 예측을 시간순으로 **앞쪽 `cal_fold_frac`(기본 0.5, `floor(n_folds×frac)`)개 folds = calibration, 나머지 뒤쪽 folds = test**로 분할. base 모델 재설계 0, split-conformal 정합(cal이 test보다 시간상 앞).
4. **conformity score**: cal folds에서 `E_i = (y_i − ŷ_i) / scale_i` (one-sided; 양수=under-order).
5. **margin quantile**: `Q_s` = pooled(전 품목·전 cal folds) `E_i`의 `s`-분위 (numpy quantile, one-sided 상방). 유한표본 보정 `ceil((n+1)s)/n` 옵션(작은 n).
6. **calibrated order** (test folds): `order_i = ŷ_i + Q_s × scale_i`.
7. **보장**: exchangeability 하 `P(y_i > order_i) ≈ 1 − s`. 평가는 **test folds에서만** 측정.

프론티어: 동일 base·동일 cal `E_i`에서 `Q_s`만 여러 s로 재산출 → 재학습 0.

## 컴포넌트 (파일 구조)

- `src/bakery/models/conformal_order.py` (신규) — `ConformalOrderCalibrator`:
  - `fit(scores: np.ndarray, service_level: float) -> None` : normalized score(=E, 호출자가 `(y−ŷ)/scale`로 산출)의 `s`-분위 `Q_s` 저장.
  - `apply(base_pred: np.ndarray, scales: np.ndarray) -> np.ndarray` : `base_pred + Q_s×scale`, 음수 클립 0.
  - 순수·상태최소·path-agnostic(입력이 배열이라 item/category 무관). 기존 `ConformalInterval`은 interval·양방향·per-dow라 계약 불일치 → 재사용 대신 잔차분위 아이디어만 차용.
- `src/bakery/features/` 또는 cli 헬퍼 — item scale (train-only 평균) 계산.
- `src/bakery/cli.py` — `prospective-eval`에 `--calibrate/--no-calibrate`(기본 off로 하위호환) + `--target-service-level`(기본 0.74) + `--cal-fold-frac`(기본 0.5). base median 예측(fold별) 산출 → 앞쪽 folds 잔차·scale로 Q_s fit → 뒤쪽(test) folds our_order 교체. leakage: cal folds가 test folds보다 시간상 앞.
- `docs/order_conformal_calibration_result.md` — 진단·프론티어·판정 (구현 Task 후반).

## 데이터 흐름

```
adjusted_demand rows (③ 잣대)
  └─ base median 예측 (v2 LGBM alpha=0.5, expanding fold별 '직전 전체' 학습)
       │  scale_i = 첫 val 이전 이력 평균 (단일/item, leakage-safe)
       ├─ [앞쪽 cal folds] E_i=(y−ŷ)/scale  ─┐
       └─ [뒤쪽 test folds] ŷ_i, scale_i     │
                                              ▼
                    ConformalOrderCalibrator.fit(E, s) → Q_s
                    apply(ŷ_test, scale_test) = ŷ + Q_s×scale → our_order
                                              ▼
                    평가(test folds만): 실현 초과율 P(y>order)≈1−s (adjusted 잣대)
```

## 검증

- **단위** (`tests/test_conformal_order.py`):
  - `Q_s` = normalized score의 정확 분위 (알려진 배열로 `==`/approx).
  - `apply`: `ŷ+Q_s×scale` 정확값, 음수 클립.
  - scale floor(0/희소 → ≥1).
  - **coverage 계약**: 합성 데이터에서 s=0.8로 fit 후 test 초과율 ≈ 0.2 (허용오차 명시, exchangeable 합성이므로 tight).
- **leakage**: calibrator가 test 잔차를 안 본다(cal 인덱스가 test 이전) — 테스트로 고정. 기존 `test_split_leakage.py`/`test_features_leakage.py` 통과.
- **진단 스텝**: production_quantile 스윙(무보정)으로 base 미스칼 크기 기록.
- **엔드투엔드**: `prospective-eval --calibrate --target-service-level 0.74` 실현 초과율이 ≈0.26로 수렴하는지(raw q0.85 0.68 대비) full-window.
- 회귀: full-suite green.

## 리스크 / 캐비엣

- **exchangeability**: 시간 분포 shift(명절·12월 spike)서 coverage 약화 — rolling cal 완화, 잔존 리스크 문서화.
- **scale 저빈도**: floor·pooled로 완화하나 극저빈도 품목 마진 과대/과소 가능.
- **cal 표본 크기**: cal folds 너무 적으면 `Q_s` 불안정(interval 메모리 Mondrian min_n 교훈) — pooled(전 품목×전 cal folds)라 per-dow보다 견고. n_folds=8·frac 0.5면 cal 4 folds. n_folds 작을 때(≤2) 주의.
- **test folds 축소**: half-split이라 평가가 뒤쪽 절반 folds에서만 이뤄짐(n_folds=8→test 4 folds). full-window 대비 표본↓ — n_folds를 넉넉히(8+).
- **coverage만이 순 신호**: KPI Δ vs baseline은 ③ 스케일 오염 상존. 판정은 초과율 수렴 기준.
- **base=median 선택**: conformal이 서비스레벨 전부를 담당 → 마진이 큼. base=q_target 대비 스윙 효율↑이나 마진 분산이 scale 추정에 더 의존 — 진단에서 확인.
- 광교 단독; category·다매장은 후속.

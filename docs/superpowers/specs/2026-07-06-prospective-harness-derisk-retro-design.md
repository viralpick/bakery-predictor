# 전향 KPI harness de-risk + full-window 회고 — 설계

작성 2026-07-06. 범위 = 지난 회고 harness(PR#25, `docs/retro_harness_result.md`)의 후속으로,
**신규 데이터 없이 광교 5년 실데이터만으로** (1) 파이프라인을 안정화(de-risk)하고
(2) full-window rolling 회고로 KPI를 신뢰구간과 함께 확정한다.

## 목표 성격 (확정)

- **2단계 순차**: Phase 1 de-risk(배선/버그 안정화) → Phase 2 full-window 회고(성능 확정).
- 4주 전향(prospective) 검증은 되돌릴 수 없는 단계이므로, 그 전에 파이프라인을 실데이터로
  full-window 완주시켜 배선·계약·누수·calibration 리스크를 걷어내는 것이 상위 목표.

## 배경 — 현재 상태 (PR#25)

`prospective-eval --source real --store-id store_gw01`이 동작한다. 우리 발주(LightGBM v2,
production-quantile q0.85, expanding-window backtest, `assert_no_leakage` 경유) vs 아티제 현행
발주(재고정보 실측 생산량)를 동일 복원수요+시간대 도착곡선 시뮬에 넣어 운영 KPI를 비교한다.

8주 단일 fold 결과 (1,787 item-day):

| policy | waste_cost | stockout_rate | soldout_median_h |
|---|---|---|---|
| our q0.85 | 3.6M | 0.687 | 15.84 |
| baseline(생산량) | 1.5M | 0.805 | 16.55 |
| **Δ (우리−아티제)** | **+2.1M** | **−0.118** | **−0.71** |

ρ_DS: bread 0.117 / pastry 0.046 / sandwich 0.000. WPE = −0.310.

### 이미 식별된 3대 리스크 (이 spec이 겨냥)

1. **8주 데모 슬라이스** — 62,960 item-day 중 1,787만 채점. 5년 회고 성능이 아니다. (Phase 2)
2. **baseline = 생산량 ≈ 발주 가정** — 실제 발주 대조는 전향 피드 대기. proxy 품질 미특성화. (T1)
3. **calibration 신호** — WPE −0.31(q0.85가 복원수요보다 31% 아래)인데 폐기 > baseline. 분포 편중
   (일부 품목 대량 과발주 + 다수 과소) 의심. 품목별 quantile calibration 미검증. (T3)

추가: 재고정보 폐기량에 음수 2,061/62,960행(~3.3%, min −31, 반품/보정 추정) — actual-waste
sanity 전 처리 필요. (T2)

## 핵심 코드 지점

- `src/bakery/cli.py`
  - `_quantile_backtest_predictions` (≈1811): `generate_time_splits(..., n_splits=1, ...)` — **Phase 2 multi-fold 확장 지점**
  - `_our_order_predictions` / `_fill_our_order` (≈1833/1852): 최근 val 구간으로 예측 채움 + 창 밖 drop 로그
  - `_real_prospective_inputs` / `_assemble_real_rows`: 재고정보(생산량/폐기량) join — **음수폐기 처리 지점**
  - `cmd_prospective_eval` (≈1920): CLI 진입점
- `src/bakery/evaluation/prospective.py`: `reconstruct_baseline_order` / `simulate_item_day_kpis` / `compare_policies`
- `src/bakery/evaluation/metrics.py`: `wpe`, `wape` — calibration 진단 짝
- `src/bakery/evaluation/diagnostics.py`: `decoupling_score`
- `src/bakery/evaluation/backtest.py`: `run_backtest` (이미 다중 window 처리 — fold 루프 재사용)

## 접근 방식 — multi-fold 구조 (결정: A)

| | 방식 | 트레이드오프 |
|---|---|---|
| **A (채택)** | CLI 레벨에서 단일창 파이프라인을 fold별 N회 루프, per-fold KPI 수집 후 집계 | 기존 함수 재사용·침습 최소. leakage 경로 그대로 유지 |
| B | prospective 파이프라인 내부를 fold-aware로 리팩터 | 장기적 깔끔하나 회귀 위험 큼 |
| C | fold 벡터화 | 과함 |

`_quantile_backtest_predictions`를 `n_splits=N`로 열고, KPI 계산·집계 루프를 얇게 추가한다.
기존 leakage 방어(`generate_time_splits` expanding + `assert_no_leakage`)는 손대지 않는다.

## Phase 1 — de-risk (기존 8주 단일 fold 경로 위에서)

### T1 (2번) baseline proxy 특성화 + swap 지점 정비
- **무엇**: "생산량 ≈ 발주" 가정을 깨는 요인을 광교 실데이터로 정량화한다.
  - 당일 재생산 여지, 당일폐기 N 품목의 이월, 반품/보정(음수 폐기)
- **제약**: 아티제 실제 발주는 전향 피드가 와야 받는다 → 지금은 **가정 품질 특성화 + 교체 지점 정비**까지.
  진짜 발주와의 대조는 명시적으로 전향 단계로 미룬다.
- **swap 지점**: `base_order` 소스를 pluggable하게 정비 → 실발주 수령 시 무마찰 교체.
- **산출**: retro 문서에 "baseline proxy 신뢰도 + Δ KPI 신뢰구간" 절.

### T2 (1번) 음수 폐기 처리 + actual-waste sanity
- **무엇**: 재고정보 폐기량 음수 2,061행(~3.3%) 처리 방침 결정(clip-at-0 vs exclude) + 문서화.
- 현재 waste_cost는 전부 simulated(order−demand). 실측 폐기량과 대조하는 컬럼/진단 추가.
- **산출**: 처리 방침 문서 + actual vs simulated waste 대조.

### T3 (4번) calibration 진단 (conformal 없음)
- **무엇**: q0.85 예측의 경험적 초과율 P(potential_demand > our_order) 계산.
  보정돼 있으면 ≈ 1 − 0.85 = 0.15에 근접해야 한다.
- 품목별 + aggregate로 → **WPE −0.31 ↔ 폐기 > baseline 동시성**(분포 편중)을 규명.
- **비범위**: Conformal Prediction 래퍼 구현은 별도 spec. 여기서는 진단만.
- **산출**: calibration 진단 테이블(초과율/품목편중).

## Phase 2 — full-window rolling 회고

### T4 rolling multi-fold 예측
- `_quantile_backtest_predictions`를 `n_splits=N`로 확장. 8주 step, 최근 1~1.5년 → 6~10 fold.
- 초기 저데이터·코드나 이전·메뉴 개편 구간은 대표성 낮아 최근 1~1.5년으로 제한(무언 축소 금지, 로그 명시).
- KPI를 fold별로 계산(기존 단일창 KPI 파이프라인 재사용).

### T5 fold 간 집계
- Δ KPI(waste_cost / stockout_rate / soldout_median_h)를 **mean ± CI**로 집계.
- ρ_DS, WPE도 fold별 분산 병기.

### T6 문서 갱신
- `docs/retro_harness_result.md` 갱신(또는 신규 절) — 신뢰구간 붙은 Δ + Phase 1 caveat 반영.

## 검증

- **절대 규칙 1 (leakage 금지)**: `test_split_leakage.py` / `test_features_leakage.py` +
  `assert_no_leakage` 그린 유지가 착수·완료 조건.
- **절대 규칙 5 (WAPE 메인)**: calibration 진단은 WAPE/WPE 짝으로 본다.
- 신규 단위 테스트(정확값 `==` 단언, `code-quality` §8):
  - 음수폐기 처리 (T2) — 알려진 입력에 대한 clip/exclude 결과
  - calibration 초과율 (T3) — 합성 입력에 대한 경험적 초과율 정확값
  - multi-fold 집계 (T5) — fold별 KPI → mean/CI 정확값
- 단일매장 가드(`_load_real_daily` loud-fail) 유지.

## 명시적 비범위 (YAGNI)

- Conformal Prediction 구현 → 별도 spec (T3는 진단만).
- 아티제 실제 발주 피드 대조 → 4주 전향 단계.
- 다매장 확장(store-qualified receipts/merge/training-filter) → 이 묶음 밖.
- FreshRetailNet 외부 벤치(3번), classification 병렬 트랙 → 이 묶음 밖.

## 성공 기준

- Phase 1: 8주 경로에서 baseline proxy 신뢰도·음수폐기·calibration 3개 리스크가 정량화되고
  문서화됨. 파이프라인이 실데이터로 무결하게 돈다.
- Phase 2: 최근 1~1.5년 rolling 6~10 fold의 Δ KPI가 신뢰구간과 함께 확정됨.
  8주 슬라이스가 아닌, fold 간 분산을 가진 회고 성능 리포트가 산출됨.

# 회고 harness 실행 결과 — 광교 (prospective-eval --source real)

작성 2026-07-05. 재현: `uv run bakery prospective-eval --source real --store-id store_gw01`.

## 무엇을 한 것인가

전향적 KPI harness의 `--source real` 경로를 광교 과거 실데이터로 가동했다. **우리 발주(LightGBM v2, production-quantile q0.85)** vs **아티제 현행 발주(= 재고정보 실측 생산량)**를 동일 시뮬(복원수요 + 시간대 도착곡선)에 넣어 운영 KPI로 비교한다.

- baseline_order = 재고정보 `생산량`(실측), actual waste = `폐기량`(실측)
- 수요 입력 = `potential_demand`(품절 복원, bonavi_daily에 precompute)
- our_order = expanding-window backtest(누수 없음, `assert_no_leakage` 경유) q0.85 예측

## 결과 (8주 창, 단일 fold, 1,787 item-day)

| policy | waste_cost_krw | lost_margin_krw | stockout_rate | soldout_median_h |
|---|---|---|---|---|
| our (q0.85) | 3,608,963 | 45,081,430 | 0.687 | 15.84 |
| baseline (생산량) | 1,500,469 | 43,737,100 | 0.805 | 16.55 |
| **Δ (우리−아티제)** | **+2,108,495** | **+1,344,332** | **−0.118** | **−0.71** |

카테고리별 ρ_DS (Decoupling Score): bread 0.117 / pastry 0.046 / sandwich 0.000. 예측 편향 WPE = **−0.310**.

## 해석

- **매진률 −11.8%p** (0.805→0.687): q0.85 발주가 아티제 생산량보다 매진을 줄인다 — KPI 방향(매진↓)에 부합.
- **폐기비용 +2.1M**: 그 대가로 폐기가 늘었다. q0.85는 **매진↓ ↔ 폐기↑ trade-off**를 매진 쪽으로 당긴 것. 원가율/lost-sale 비용 가정에 따라 순이익 우열이 갈리므로, 실제 원가율 `c` 수령 후 재평가 필요.
- **ρ_DS ≈ 0 (전 카테고리)**: 복원 수요가 품절률과 잔여 상관이 거의 없다 = potential_demand 복원이 검열 편향을 잘 걷어냈다는 건강 신호. (sandwich=0은 품절 분산이 사실상 없어서.)
- **WPE −0.31 vs 폐기>baseline 동시성**: aggregate로 q0.85 발주가 복원수요보다 31% 아래인데 폐기는 더 많다 = 분포가 편중(일부 품목 대량 과발주, 다수 과소). 단일 fold quantile 모델의 **품목별 calibration 재검토 신호** — full-window/CV에서 확인 필요.

## 한계 (반드시 함께 읽을 것)

1. **8주 데모 슬라이스** — 전체 62,960 item-day 중 1,787만 채점(97% 창 밖 drop, 로그 명시). **5년 회고 성능이 아니다.** 배선·계약 검증 + 데모 자산 목적.
2. **단일 fold, CV 없음** — fold 간 분산 미측정.
3. **actual 폐기량 sanity 미반영** — 위 waste_cost는 전부 simulated(order−demand). 재고정보 실측 `폐기량`과의 대조는 별도이며, **음수 폐기량 2,061/62,960행(~3.3%, min −31, 반품/보정 추정)** 처리(clip/exclude+문서화) 후에 해야 한다.
4. **baseline = 생산량 ≈ 발주 가정** — 아티제 실제 발주가 생산량과 동일하다는 가정. 전향적 실데이터의 실제 발주 수령 시 교체.
5. **store_id 단일(광교)** — 학습필터/merge가 store_id 미적용(현재 단일매장 무해). 다매장 전 일괄 정리 필요.

## 다음

실데이터(전향적 운영 피드) 수령 시: (a) baseline을 실제 발주로 교체, (b) 음수 폐기 처리 후 actual-waste sanity 추가, (c) full-window/rolling CV로 q0.85 calibration·KPI 확정, (d) 실제 원가율로 순이익 우열 판정.

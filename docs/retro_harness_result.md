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

## Full-window rolling 회고 (8주 × 8 fold, 13,554 item-day)

작성 2026-07-07. 재현:
```bash
uv run bakery prospective-eval --source real --store-id store_gw01 \
  --production-quantile 0.85 --our-order-val-weeks 8 --n-folds 8 \
  --out-csv reports/prospective_kpi.csv
```

실행 로그: `our_order 8 fold(s)` / `fold 컬럼 보존됨: [0,1,2,3,4,5,6,7]`(8 fold 전부 채점, 무언 축소 없음) / `scored item-days: 13,554 / 62,960 (dropped 49,406 outside backtest val window)`. n_folds=8이 데이터 길이(최근 ~64주 = 8주×8 non-overlapping val 창 + expanding train)로 온전히 채점됐다 — 빈 fold 없음.

### fold별 Δ (우리 q0.85 − 아티제 생산량)

| fold | waste_cost_krw | lost_margin_krw | stockout_rate | soldout_median_h |
|---|---|---|---|---|
| 0 | +2,108,495 | +1,344,332 | −0.1181 | −0.579 |
| 1 | +1,898,097 | +1,124,766 | −0.1480 | −0.390 |
| 2 | +267,205 | +6,009,357 | −0.0054 | −2.501 |
| 3 | −370,905 | +8,619,009 | +0.0720 | −2.235 |
| 4 | +33,531 | +7,385,792 | +0.0329 | −1.365 |
| 5 | +445,219 | +10,133,086 | −0.0129 | −3.065 |
| 6 | −721,471 | +9,511,752 | +0.0655 | −2.855 |
| 7 | +158,243 | +8,187,277 | −0.0052 | −3.237 |

(fold 0 = 최근 8주 창 = 위 단일-fold 절과 동일 슬라이스. waste/lost_margin/stockout Δ가 정확히 일치 — 연속성 확인.)

### 집계 (mean ± 95% CI, n=8 fold)

| metric (Δ 우리−아티제) | mean | 95% CI | n(fold) |
|---|---|---|---|
| waste_cost_krw | +477,302 | [−224,509, +1,179,113] | 8 |
| lost_margin_krw | +6,539,421 | [+4,109,184, +8,969,658] | 8 |
| stockout_rate | −0.0149 | [−0.0704, +0.0406] | 8 |
| soldout_median_h | −2.028 | [−2.801, −1.255] | 8 |

- CI는 정규근사(mean ± 1.96·sem), fold 8개로 넓다 — 방향성 판정용.
- **calibration 초과율 P(demand>order)=0.636 (nominal 1−α=0.15)**: q0.85 발주가 복원수요 대비 **심각한 과소발주** — 목표(15%)의 4배 이상이 order<demand. 예측 편향 WPE=−0.261(발주가 복원수요보다 26% 아래)과 정합. conformal 보정 별도 spec에서 다룰 신호.
- **waste sanity ratio(simulated/actual)=0.678** (actual_total=18,843 / simulated_total=12,784, n_rows=13,554): 시뮬 폐기가 실측의 68% 수준. baseline=생산량 proxy 하에서 시뮬 폐기(생산−복원수요)와 실측 폐기(생산−판매)는 복원분만큼 구조적으로 다르므로 1에서 벗어나는 것이 정상. 방향(시뮬<실측)은 복원수요>실판매(품절 복원)와 정합 — proxy 신뢰도는 "구조적 차이는 설명 가능, 정량 대조는 실발주 수령 후" 수준.
- **baseline proxy 특성화**: stockout_share=0.924(store_gw01 TARGET_CATEGORIES daily의 is_stockout 비율 — 생산=판매+폐기 항등식이 복원분만큼 깨지는 item-day가 다수), negative_waste_share=0.027(재고정보 폐기량 음수 2,702 / 전체 99,357행).

### 해석 — 8주 데모 슬라이스가 full-window에서 유지되지 않는다

- **매진률 개선이 사라진다**: 단일 fold(fold 0)의 −11.8%p가 8 fold 평균 −1.5%p로 축소되고 CI가 0을 가로지른다([−7.0%p, +4.1%p]). fold 0·1(최근)만 강하게 음수, fold 3·4·6은 오히려 양수 — 매진 우위는 특정 최근 구간 아티팩트.
- **폐기 penalty도 방향 불확실**: 단일 fold +2.1M → 8 fold 평균 +477K, CI가 0 포함. pooled Δ(+3.8M)는 row 가중이라 fold-mean과 다르며, 방향성 판단은 fold-mean 기준.
- **일관되게 나쁜 두 축**: (a) soldout_median_h Δ가 8 fold 전부 음수(mean −2.0h, CI 전 구간 <0) — 우리 발주가 매진을 **더 이르게** 만든다(KPI "매진시각↑" 목표에 역행). (b) lost_margin_krw Δ가 CI 전 구간 양수(+4.1M~+9.0M) — 우리 발주의 lost-sale 마진이 일관되게 크다.
- **근본 원인은 과소발주**: calibration 0.636 + WPE −0.261이 (a)(b)를 설명한다. q0.85 모델이 복원수요를 체계적으로 하향 예측 → 만성 과소발주 → 매진↑·lost margin↑·조기 매진. fold 0에서 매진이 baseline보다 나았던 건 아티제 생산량이 **더** 과소였기 때문이며, full-window에서 그 격차가 사라진다.
- **함의**: 데모 슬라이스의 "매진↓" 서사는 full-window에서 무너진다. q0.85는 이 데이터에서 과소발주 쪽으로 미스캘리브레이션 — production quantile 상향 또는 calibration 보정 없이는 운영 우위 주장 불가.

### Phase 1 de-risk 반영 caveat

- **음수 폐기 clip은 이번 실행에 미적용**: `handle_negative_waste`(clip-at-0) primitive는 존재하나 prospective 경로(`_assemble_real_rows`)에 아직 배선되지 않았다. 따라서 waste sanity의 actual_total=18,843은 **raw waste_qty**(음수 미clip) 합이다. 음수 특성만 별도 측정: n_negative=2,702, min=−31.0(store_gw01 재고정보 99,357행 기준). actual-waste 정량 대조 전에 clip 배선 필요.
- **baseline=생산량 proxy**, 실발주 대조는 전향 단계 — swap 지점 준비됨(`select_base_order(source=...)`, 현재 'production'만 지원).
- **calibration은 진단만** — P(demand>order) 초과율 리포트일 뿐 conformal 보정 구현은 별도 spec.
- **단일 매장(광교)** — 학습필터/merge가 store_id 미적용(단일매장 무해), 다매장 전 정리 필요.

## 다음

실데이터(전향적 운영 피드) 수령 시: (a) baseline을 실제 발주로 교체, (b) 음수 폐기 clip을 `_assemble_real_rows`에 배선 후 actual-waste sanity 확정, (c) 과소발주 교정 — production quantile 상향 또는 conformal calibration으로 P(demand>order)를 nominal 0.15로 수렴, (d) 실제 원가율로 순이익 우열 판정.

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

작성 2026-07-07 (음수 폐기 clip 배선 후 재실행 — `03d3146`). 재현:
```bash
uv run bakery prospective-eval --source real --store-id store_gw01 \
  --production-quantile 0.85 --our-order-val-weeks 8 --n-folds 8 \
  --out-csv reports/prospective_kpi.csv
```

> **Headline (honest)**: 8주 슬라이스의 유망한 결과는 full-window에서 **유지되지 않는다**. stockout·waste Δ는 CI가 0을 가로지르고(방향 불확실), soldout_median Δ는 robustly 음수(~−2h, KPI 역행), lost_margin Δ는 robustly 양수(불리). 근본 원인은 q0.85의 **진짜 과소발주**(α-sweep로 확정). 방향성은 신뢰 가능하나 크기는 caveat(복원 타깃·생산량 proxy baseline·낮은 상관)로 제한적.

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
- **calibration 초과율 P(demand>order)=0.636 (nominal 1−α=0.15)**: q0.85 발주가 복원수요 대비 **심각한 과소발주** — 목표(15%)의 4배 이상이 order<demand. 예측 편향 WPE=−0.261(발주가 복원수요보다 26% 아래)과 정합. conformal 보정 별도 spec에서 다룰 신호. (진짜 under-calibration인지 harness 버그인지는 아래 "calibration 진단" α-sweep로 검증.)
- **waste sanity ratio(simulated/actual)=0.652** (actual_total=19,614 / simulated_total=12,784, n_rows=13,554): 시뮬 폐기가 실측의 65% 수준. **음수 폐기 clip 적용 후** 값이다 — clip 전 raw(actual_total=18,843, ratio=0.678) 대비 음수 2,702행이 0으로 올라가 actual_total이 상승(18,843→19,614)하고 ratio가 하락(0.678→0.652). baseline=생산량 proxy 하에서 시뮬 폐기(생산−복원수요)와 실측 폐기(생산−판매)는 복원분만큼 구조적으로 다르므로 1에서 벗어나는 것이 정상. 방향(시뮬<실측)은 복원수요>실판매(품절 복원)와 정합 — proxy 신뢰도는 "구조적 차이는 설명 가능, 정량 대조는 실발주 수령 후" 수준.
- **baseline proxy 특성화**: stockout_share=0.924(store_gw01 TARGET_CATEGORIES daily의 is_stockout 비율 — 생산=판매+폐기 항등식이 복원분만큼 깨지는 item-day가 다수), negative_waste_share=0.027(재고정보 폐기량 음수 2,702 / 전체 99,357행 — 이번 실행에서 clip-at-0 적용됨).

### calibration 진단 — 진짜 under-calibration (α-sweep로 확정)

초과율 0.636이 harness 배선 버그가 아니라 **모델의 실제 under-calibration**임을 최근 fold 단일-fold α-sweep로 검증했다:

| α (production_quantile) | exceedance P(demand>order) | pred mean |
|---|---|---|
| 0.50 | 0.861 | 6.47 |
| 0.85 | 0.687 | 10.13 |
| 0.95 | 0.538 | 12.60 |
| 0.99 | 0.354 | 15.92 |

- **exceedance가 α에 대해 매끄럽고 단조**로 감소하고 예측값이 교차하지 않는다 → quantile objective가 올바르게 배선됨(harness/파이프라인 버그 아님). α를 올리면 발주가 커지고 초과율이 규칙적으로 내려간다.
- **그럼에도 α=0.99조차 초과율 0.354에 머문다** → 모델이 타깃의 tail을 덮지 못한다. 극단 quantile까지 밀어도 nominal 0.01 근처에 못 간다 = 신호 부족.
- **근본 원인**: 타깃 `potential_demand`가 최대 3× 배율 기반 복원(reconstruction)이고, 대상 매장의 일 품절률이 ~91–99%라 복원분 분산이 크다. day-ahead feature와 타깃의 상관은 ~0.25–0.43에 불과해 복원된 수요 분산을 예측할 신호가 없다.
- **국소성**: 과거 conformal 파이프라인이 (덜 noisy한) 다른 타깃에서 near-nominal coverage에 도달한 바 있다 → 미스캘리브레이션은 quantile 방법 자체가 아니라 **이 타깃의 noise 구조에 국한**된 문제.

### 해석 — 8주 데모 슬라이스가 full-window에서 유지되지 않는다

- **매진률 개선이 사라진다**: 단일 fold(fold 0)의 −11.8%p가 8 fold 평균 −1.5%p로 축소되고 CI가 0을 가로지른다([−7.0%p, +4.1%p]). fold 0·1(최근)만 강하게 음수, fold 3·4·6은 오히려 양수 — 매진 우위는 특정 최근 구간 아티팩트.
- **폐기 penalty도 방향 불확실**: 단일 fold +2.1M → 8 fold 평균 +477K, CI가 0 포함. pooled Δ(+3.8M)는 row 가중이라 fold-mean과 다르며, 방향성 판단은 fold-mean 기준.
- **일관되게 나쁜 두 축**: (a) soldout_median_h Δ가 8 fold 전부 음수(mean −2.0h, CI 전 구간 <0) — 우리 발주가 매진을 **더 이르게** 만든다(KPI "매진시각↑" 목표에 역행). (b) lost_margin_krw Δ가 CI 전 구간 양수(+4.1M~+9.0M) — 우리 발주의 lost-sale 마진이 일관되게 크다.
- **근본 원인은 과소발주**: calibration 0.636 + WPE −0.261이 (a)(b)를 설명한다. q0.85 모델이 복원수요를 체계적으로 하향 예측 → 만성 과소발주 → 매진↑·lost margin↑·조기 매진. fold 0에서 매진이 baseline보다 나았던 건 아티제 생산량이 **더** 과소였기 때문이며, full-window에서 그 격차가 사라진다.
- **함의**: 데모 슬라이스의 "매진↓" 서사는 full-window에서 무너진다. q0.85는 이 데이터에서 과소발주 쪽으로 미스캘리브레이션 — production quantile 상향 또는 calibration 보정 없이는 운영 우위 주장 불가.

### Phase 1 de-risk 반영 caveat

- **음수 폐기 clip 이제 적용됨**(`03d3146`, `_real_prospective_inputs`에서 `handle_negative_waste(policy="clip")` 배선): 이번 실행 로그 `negative waste clipped: {'policy':'clip', 'n_negative':2702, 'min_value':-31.0}`. clip-at-0 후 음수 2,702행이 0으로 올라가 waste sanity의 actual_total=19,614(clip 전 18,843), ratio=0.652(clip 전 0.678). actual-waste 정량 대조는 이제 clip된 waste_qty 기준.
- **baseline=생산량 proxy**, 실발주 대조는 전향 단계 — swap 지점 준비됨(`select_base_order(source=...)`, 현재 'production'만 지원).
- **calibration은 진단만** — P(demand>order) 초과율 리포트 + 위 α-sweep 원인규명까지가 이 문서 범위. conformal 보정 **구현은 별도 spec으로 이연**.
- **단일 매장(광교)** — 학습필터/merge가 store_id 미적용(단일매장 무해), 다매장 전 정리 필요.

## 카테고리-레벨 재측정 (v4 통합 총합 q0.85 → 품목 배분)

작성 2026-07-07. 재현:
```bash
uv run bakery prospective-eval --source real --store-id store_gw01 \
  --production-quantile 0.85 --our-order-val-weeks 8 --n-folds 8 \
  --order-level category --out-csv reports/prospective_kpi_category.csv
```

n_folds=8 그대로 온전히 채점됨 (min_train 365일 + 8×8주 창이 실데이터 길이에 맞음, drop 없이 진행). 실행 로그: `category our_order 8 fold(s) × 8주, q=0.85, 448 dates × 56 items` / `fold 컬럼 보존됨: [0,1,2,3,4,5,6,7]` / `scored item-days: 13,529 / 62,960 (dropped 49,431 outside backtest val window)` — item-level 절(13,554/62,960)과 거의 동일한 창(±25 item-day, 카테고리→품목 배분 후 유효 행수 차이).

### fold별 Δ (우리 카테고리 발주 → 품목 배분 − 아티제 생산량)

| fold | waste_cost_krw | lost_margin_krw | stockout_rate | soldout_median_h |
|---|---|---|---|---|
| 0 | −1,000,865 | +14,605,010 | +0.0993 | −1.545 |
| 1 | −691,390 | +8,826,346 | +0.0755 | −1.049 |
| 2 | −875,838 | +3,828,709 | +0.0593 | −1.326 |
| 3 | −1,067,160 | +5,306,932 | +0.0863 | −0.984 |
| 4 | −932,719 | +5,036,037 | +0.0988 | −0.440 |
| 5 | −974,199 | +6,033,804 | +0.0989 | −1.603 |
| 6 | −1,209,204 | +5,832,561 | +0.1065 | −1.821 |
| 7 | −435,638 | +4,529,871 | +0.0459 | −2.040 |

### 집계 side-by-side (mean Δ ± 95% CI, n=8 fold)

| Δ (우리−아티제) | item-level (기존) | category-level (신규) |
|---|---|---|
| stockout_rate | −0.0149 [−0.0704, +0.0406] | **+0.0838 [+0.0687, +0.0989]** |
| waste_cost_krw | +477,302 [−224,509, +1,179,113] | **−898,377 [−1,064,029, −732,724]** |
| soldout_median_h | −2.028 [−2.801, −1.255] | −1.351 [−1.707, −0.995] |
| lost_margin_krw | +6,539,421 [+4,109,184, +8,969,658] | +6,749,909 [+4,323,118, +9,176,701] |
| **calibration 초과율**(item-day, P(demand>order)) | **0.636** | **0.738** (nominal 1−q=0.15) |
| 예측 편향 WPE(item-day) | −0.261 | −0.306 |

추가로 category-level에서만 나오는 카테고리-일 단위 진단(품목 배분 이전, Σdemand vs Σorder): **초과율 P(Σdemand>Σorder)=0.900, WAPE=0.316, WPE=−0.306, 448 dates** (nominal 1−q=0.15). WPE가 item-day 절과 정확히 −0.306으로 일치 — WPE=Σ(order−demand)/Σdemand는 합계 비율이라 품목 배분 전/후 값이 불변(배선 정합성 확인, 버그 아님).

waste sanity(category run): `{'actual_total': 19520.0, 'simulated_total': 12762.15, 'ratio': 0.654, 'n_rows': 13529}` — item-level 절의 0.652(actual 19,614/simulated 12,784)와 거의 동일. ρ_DS(이번 실행, full-window): bread 0.1751 / pastry 0.1252 / sandwich 0.1185 (8주 단일-fold 절의 0.117/0.046/0.000보다 전 카테고리에서 상승 — full-window에서 품절-복원수요 잔여 상관이 더 뚜렷해짐, 별도 조사 후보).

### 해석 — 카테고리-레벨이 item-level을 이기지 못한다 (target confound가 지배적)

- **폐기는 줄지만(−0.90M, CI 전 구간 <0, robust) 매진은 늘어난다(+0.084, CI 전 구간 >0, robust)** — item-level에서는 둘 다 CI가 0을 가로질러 방향 불확실이었는데, category-level은 정반대 방향(적게 만들고 더 자주 매진)으로 **명확하게** 갈린다. newsvendor trade-off가 "덜 발주" 쪽으로 뚜렷하게 밀린 것.
- **calibration이 개선되지 않고 악화된다**: item-day 기준 초과율 0.636→0.738, 카테고리-총합 기준으로는 0.900까지 올라간다(nominal 0.15 대비 6배). WPE도 −0.261→−0.306으로 편향이 더 깊어진다.
- **근본 원인 = target confound (버그 아님, 문서화된 예상 결과)**: 카테고리 발주는 `adjusted_demand ≈ normal + closing×0.5`(마감할인 실수요 조정 타깃)로 학습된 v4 category_total 모델의 출력이고, 여기서 평가 yardstick은 `potential_demand`(품절 복원, 마감할인 조정 없음)다. adjusted_demand는 구조적으로 potential_demand보다 낮은 스케일이므로 — 카테고리 총합 자체가 이미 "덜" 발주하게 설계된 타깃이고, 여기에 q0.85 quantile까지 겹쳐 item-level보다 더 심하게 과소발주한다. 이는 브리핑에서 예고한 그대로다.
- **lost_margin_krw과 soldout_median_h는 두 경로 모두 나쁘다**: lost_margin은 두 경로 모두 CI 전 구간 양수로 비슷한 규모(+6.5M vs +6.7M) — 카테고리 배분이 이 KPI를 고치지 못한다. soldout_median_h는 두 경로 모두 robust 음수(더 이르게 매진)지만 category가 약간 덜 나쁘다(−1.35h vs −2.03h) — 유일하게 category가 부분적으로 나은 지표지만 여전히 KPI 목표(매진시각↑)에 역행.
- **결론(honest)**: 카테고리-레벨 재측정은 item-level 대비 **더 나은 결과를 보여주지 않는다**. waste_cost 한 축만 방향이 뚜렷하게 좋아지고(감소), 그 대가로 stockout_rate가 뚜렷하게 나빠지며, calibration/WPE는 악화되고, lost_margin·soldout_median_h는 그대로 나쁘다. 아티제를 이기는 근거로 쓸 수 없다.

### caveat

- **target confound가 지배적 원인**(위 해석 참조) — 발주 target(adjusted_demand)과 평가 yardstick(potential_demand)의 스케일 불일치. 이 불일치를 통제한 apples-to-apples 비교(예: adjusted_demand를 yardstick으로 재평가)는 이 실행 범위 밖.
- **v4 카테고리 정의 = bread·pastry·sandwich 통합 총합** — "한 묶음" 수요로 모델링, 카테고리 총합을 품목 비율(`item_proportion`, pre-cutoff 계산)로 배분. 배분오차(총합은 맞아도 품목 단위에서 틀릴 가능성)는 별도 미검증.
- **baseline = 생산량 proxy** — item-level 절과 동일한 한계(실제 아티제 발주 아님, 전향적 실데이터 수령 시 교체 지점 준비됨).
- **n=8 fold, 정규근사 CI** — item-level과 동일한 표본 크기·근사 한계.

## 다음

실데이터(전향적 운영 피드) 수령 시: (a) baseline을 실제 발주로 교체, (b) 음수 폐기 clip을 `_assemble_real_rows`에 배선 후 actual-waste sanity 확정, (c) 과소발주 교정 — production quantile 상향 또는 conformal calibration으로 P(demand>order)를 nominal 0.15로 수렴, (d) 실제 원가율로 순이익 우열 판정, (e) 카테고리-레벨 target confound 해소(adjusted_demand 기준 재평가 또는 category_total 타깃을 potential_demand로 재학습) 후 item vs category 재대조.

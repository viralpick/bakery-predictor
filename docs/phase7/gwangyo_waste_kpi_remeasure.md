# 광교 폐기(1차 KPI) 재측정 — 빵 품목(타깃 베이커리)만, 새 166 canonical + 신규 closing (Phase 7 Task 3.5)

## 배경

[Task 3 리포트](gwangyo_headline_remeasure.md)에서 `harness-run`은 폐기 1차 KPI(vs 아띠제
실생산 QT_MADE)를 산출하지 못한다는 것이 확인됐다 — `harness-run`이 내는 `surplus_rate`는
자기참조적(모델 발주 vs 모델 자신의 실측치)이라 아띠제 실생산과 무관하다. 옛 "폐기
−33~40%"는 별도 경로 `scripts/unified_policy_kpi.py`(아띠제 실제 생산량 QT_MADE를
ground truth로 두고 여러 발주정책의 폐기를 비교)에서 나온 수치다. 이 태스크는 그 경로를
**새 166 canonical + 신규 closing 소스**에서, **빵 품목(타깃 베이커리)만**으로 재실행한다.

## Step 1 — 데이터 소스 감사 (게이트)

`unified_policy_kpi.py` → `bakery.cli._real_prospective_inputs` → `bakery.forecast.loaders.load_real_daily`
경로를 추적:

- **`load_real_daily(store_id)`** (`src/bakery/forecast/loaders.py:48`): `REAL_DAILY_PARQUET_PATH =
  paths.dataset("bonavi_daily")` = `data/processed/internal/bonavi_daily.parquet` (새 166
  canonical, 단일매장 store_gw01)를 읽고, `category_id.isin(TARGET_CATEGORIES)`로 필터.
  **옛 0520 xlsx 참조 없음.**
- **`TARGET_CATEGORIES = ("bread", "pastry", "sandwich")`** (`src/bakery/features/category_aggregate.py:27`,
  주석: "cake 제외 (사전 예약 + 시즌 특수)"). 실제 코드 스코프는 bread/pastry/sandwich이며
  **cake·sweets도 제외**된다 — task brief의 "bread/pastry/cake/sandwich, salad 제외" 서술과
  글자상 다르다. **검증**: `git show bdc3f19:src/bakery/features/category_aggregate.py`
  (옛 "폐기 −33~40%" 헤드라인을 낸 2026-07-15 커밋)에도 동일하게
  `TARGET_CATEGORIES = ("bread", "pastry", "sandwich")` — **옛 헤드라인도 cake·sweets를
  제외한 스코프였다.** 즉 이번 재측정이 옛 헤드라인과 스코프를 다르게 잡은 게 아니라,
  옛 스코프를 그대로 유지한 것 — like-for-like 비교의 전제 조건이 성립한다. (brief의
  "cake 포함" 서술은 166 canonical 전체를 가리킨 근사 표현으로 보이나, 그 경우 옛
  헤드라인과 스코프가 달라져 비교가 깨지므로 이번 재측정에서는 코드 기본값
  `TARGET_CATEGORIES`을 그대로 따랐다 — **cake·sweets 포함 재실행이 필요하면 별도 결정
  요청**.) 아래 표에서 실측 분포로 확인.
- **closing/adjusted_demand**: `_real_prospective_inputs`가 `build_item_adjusted_demand(daily,
  alpha=alpha)`를 호출(`src/bakery/cli.py:2564`). 그 함수(`src/bakery/features/category_aggregate.py:174`)는
  guarded fallback: `CLEAN_PARQUET_DEFAULT.exists()`면 `load_sales_with_discount_v2()` +
  `load_closing_returns_v2()`(신규), 아니면 `load_sales_with_discount()`(옛 0520) 폴백. 실행 전
  확인:
  ```
  uv run python -c "from bakery.analysis.discount import CLEAN_PARQUET_DEFAULT as p; print(p, p.exists())"
  → /Users/taehoonkim/dev/bakery-predictor/data/interim/sales_lines_clean.parquet True
  ```
  → **v2(신규) 분기 사용 확인.** `_category_order_predictions`가 쓰는 `build_category_daily`
  (같은 파일, category-level)도 동일한 guarded-fallback 패턴이며 같은 조건으로 v2 분기를 탄다.
- **재고정보(QT_MADE/QT_OUT)**: `REAL_INVENTORY_XLSX_PATH = paths.dataset("master_xlsx")` =
  `보나비 데이터_20260526.xlsx` — 이것은 `legacy_xlsx_0520`(20260520)이 **아니다**. 별도
  최신 스냅샷.
- **부수 발견(검증 완료, non-blocking)**: `_attach_unit_price`(category_aggregate.py:62, 카테고리-레벨
  `revenue`/`adjusted_demand_revenue` 보조 컬럼 계산용)만 `legacy_xlsx_0520`의 품목정보 시트
  (판매단가)를 그대로 읽는다. **검증**: `_category_order_predictions`(cli.py:2288)는
  `build_features(build_category_daily(...), target_col="adjusted_demand_unit", ...)` —
  발주량 모델의 학습/예측 target은 **unit 컬럼**이며 `revenue`가 아니다. `unified_policy_kpi.py`의
  KRW 변환도 `_load_unit_prices(REAL_INVENTORY_XLSX_PATH)` = 0526 master_xlsx를 명시적으로
  사용한다. → 0520 참조는 order 수량·폐기 KRW 산출 경로 어디에도 도달하지 않는 cosmetic
  참조(코드베이스 내 다른 diagnostic 커맨드용)로 확인됨.

### 실측 rows의 category_id 분포

`load_real_daily("store_gw01")` 직접 호출 결과(`_real_prospective_inputs`가 그대로 사용):

```
TARGET_CATEGORIES = ('bread', 'pastry', 'sandwich')
rows: 73208
category_id
pastry      43004
bread       21988
sandwich     8216
n items: 157
```

참고로 필터 이전 `bonavi_daily.parquet`(store_gw01) 전체는:

```
rows: 75615, n items: 166
category_id
pastry      43004
bread       21988
sandwich     8216
cake         2178
sweets        229
```

→ **166개 canonical 중 157개(cake 2178행·sweets 229행 제외)가 이 KPI 경로의 母집단.** 음료·완제품·
장기유통 품목은 canonical 자체에 없으므로 섞일 수 없음(Task 0/1 파운데이션 재설계에서 이미 확정).

**Step 1 결론(게이트 통과)**: 로드 경로는 새 166 canonical + 신규 closing(v2)을 사용하며,
스코프는 타깃 베이커리(bread/pastry/sandwich, 157/166품목)만. 옛 0520 xlsx/stale 캐시 참조 없음.

## Step 2 — 배선 수정

**불필요.** 감사 결과 이미 신규 소스(guarded fallback이 정확히 v2로 분기)이므로
`scripts/unified_policy_kpi.py` 및 그 하위 경로에 코드 변경 없음.

## Step 3 — 재실행

```
uv run python scripts/unified_policy_kpi.py
```

FOREGROUND 실행, 소요 3분 43초. 전체 콘솔 출력(그대로):

```
negative waste clipped: {'policy': 'clip', 'n_negative': 2702, 'n_total': 99357, 'min_value': -31.0}
생산 bulk 제외 57 item-day에서 938개 차감 (음수→0 clip 8개, 보수적)
our_order 8 fold(s), each 8주 (store=store_gw01, quantile α=0.85)
our_order scored item-days: 14,913 / 73,145 (dropped 58,232 outside backtest val window)
our_order fold 컬럼 보존됨: [0, 1, 2, 3, 4, 5, 6, 7]
category our_order 8 fold(s) × 8주, model=lightgbm, margin=q=0.85, event_prior=on, 448 dates × 65 items
category our_order 8 fold(s) × 8주, model=lightgbm, margin=nk(×1.0+15.0, base q=0.85), event_prior=on, 448 dates × 65 items
category our_order 8 fold(s) × 8주, model=lightgbm, margin=nk(×1.0+30.0, base q=0.85), event_prior=on, 448 dates × 65 items
category our_order 8 fold(s) × 8주, model=lightgbm, margin=conformal(s=0.85, cal_frac=0.5), event_prior=on, 224 dates × 50 items
공통 population: 7615 / 14913 item-days (conformal test창, 전 정책 발주 존재)

=== 폐기 (1차 KPI) — Δ는 현행 실측A / 현행시뮬B 둘 다 ===
policy                     basis           폐기KRW  ΔvsA(하한)  ΔvsB(공정)
actual_production          A_actual   16,782,180        ref
actual_production_sim      B_sim      18,490,266     +10.2%        ref
artisee_reimpl             B_sim      15,659,412      -6.7%     -15.3%
our_cat_quantile           B_sim      10,218,450     -39.1%     -44.7%
our_cat_nk15               B_sim       9,873,781     -41.2%     -46.6%
our_cat_nk30               B_sim      12,499,416     -25.5%     -32.4%
our_cat_conformal          B_sim      11,594,723     -30.9%     -37.3%

=== 매진 (2차 KPI) — 동일 B basis(발주<adjusted)로만 비교. actual A는 아래 censoring 패널 ===
policy                          매진①(전체)    매진②(SKU)
actual_production_sim             0.004       0.048
artisee_reimpl                    0.223       0.310
our_cat_quantile                  0.424       0.510
our_cat_nk15                      0.451       0.518
our_cat_nk30                      0.237       0.433
our_cat_conformal                 0.335       0.465

=== censoring 패널 — 현행 actual A(실측 폐기0) vs B(시뮬 발주<adjusted) ===
  매진①  A=0.009  B=0.004
  매진②  A=0.487  B=0.048   ← 실제 매진이 시뮬보다 훨씬 큼 = censoring

wrote reports/unified_policy_kpi.csv
※ 폐기 음수%=현행보다 덜 버림(좋음). adjusted=실수요 하한 → 모델폐기=상한 → 절감=하한(보수적).
※ 매진 모델 basis=발주<adjusted(헌장 §5B, 단 aggregation은 날별평균 — pooled 아님).
```

원자료: `reports/unified_policy_kpi.csv` (gitignored, 재현 가능 — 커밋 대상 아님).

## Step 4 — Delta 리포트

### 옛 헤드라인의 정확한 출처

옛 "폐기 −33~40%"는 `docs/three_way_baseline_comparison_20260715.md` §(b-통합)
(2026-07-15, 146 canonical, 옛 closing 소스)의 아래 수치다 — ground truth는 그때도
`actual_production`(실제 생산량 재구성):

| 정책 (옛, 146품목) | 폐기(KRW) | vs actual_production |
|---|---|---|
| actual_production (기준) | 10.44M | — |
| our_cat_quantile (q0.85) | 6.21M | **−40.5%** |
| our_cat_conformal (s0.85) | 7.00M | **−32.9%** |
| our_cat_nk (q0.85+40) | 11.46M | +9.8% |
| artisee_reimpl | 9.23M | −11.6% |

"−33~40%"는 정확히 conformal(−32.9%)~quantile(−40.5%) 두 정책의 범위였다 — nk/artisee_reimpl은
그 범위 밖(양수/−11.6%)이라 헤드라인에 포함되지 않았던 것으로 보인다.

**중요 — 옛 `actual_production` 기준값의 basis 확인**: 이 표를 낸 커밋(`adf4dc3`, 2026-07-15
19:14)의 `unified_policy_kpi.py`를 직접 확인(`git show adf4dc3:scripts/unified_policy_kpi.py`).
그때는 A/B 이중 basis 구분이 아직 없었고, `actual_production`의 waste(10.44M)도
`simulate_item_day_kpis(rows, profiles, order_col="order_actual_production", ...)`로
**시뮬레이션**된 값이다(`order_actual_production = base_order` = QT_MADE를 "발주"로 두고
adjusted_demand와 비교한 모델 basis) — 재고정보의 실측 QT_OUT을 그대로 합산한 값이
아니다. 즉 옛 `actual_production` 기준은 **이번 재측정의 `actual_production_sim`(B_sim,
ref_b=18,490,266)에 대응**하며, `actual_production`(A_actual, 실측 16,782,180)에 대응하지
않는다. 추가로 그 커밋은 `simulate_item_day_kpis`에 `unit_prices`를 전달하지 않아
`business_metrics.simulate_profit`이 **flat 3000원/개 fallback**(`avg_price = ... else 3000.0`,
`src/bakery/evaluation/business_metrics.py:88`)로 KRW를 계산했다 — 이번 재측정은 실제
품목단가(0526 master_xlsx)를 사용하므로 **절대 KRW 규모는 가격 basis 자체가 달라
직접 비교 불가**하나, 모든 정책이 동일 flat가/동일 실제가를 쓰므로 **정책 간 % delta는
가격 basis와 무관**(분자·분모에서 상쇄)하여 비교 가능하다.

### 새 166 + 신규 closing (이번 재측정, 두 basis 모두 표기)

| 정책 (신규, 157품목=bread/pastry/sandwich) | 폐기(KRW) | ΔvsA(실측) | ΔvsB(시뮬, 옛 basis와 동일) |
|---|---|---|---|
| actual_production (A, 실측) | 16.78M | — | (해당없음) |
| actual_production_sim (B, 시뮬) | 18.49M | +10.2% | — (ref) |
| our_cat_quantile (q0.85) | 10.22M | −39.1% | **−44.7%** |
| our_cat_conformal (s0.85) | 11.59M | −30.9% | **−37.3%** |
| our_cat_nk15 (q0.85+15) | 9.87M | −41.2% | −46.6% |
| our_cat_nk30 (q0.85+30) | 12.50M | −25.5% | −32.4% |
| artisee_reimpl | 15.66M | −6.7% | −15.3% |

*(신규 스크립트는 nk 버퍼를 +15/+30 두 지점으로 스캔한다 — 옛 +40과 직접 비교 불가.
가장 근접한 대응은 quantile↔quantile, conformal↔conformal.)*

### 옛 vs 새 — like-for-like 비교 (basis 정정: 옛 `actual_production` = 시뮬 basis → ΔvsB로 비교)

위 확인대로 옛 헤드라인의 `actual_production` 기준값은 이번 재측정의 **B_sim**
(`actual_production_sim`)에 대응한다. 따라서 like-for-like는 **ΔvsB**로 비교해야 한다
(ΔvsA는 basis가 달라 직접 대응하지 않음 — 참고용으로만 위 표에 병기):

| 정책 | 옛(146, 옛closing, 시뮬basis) | 새(157, 신규closing, ΔvsB) | delta |
|---|---|---|---|
| our_cat_quantile | −40.5% | **−44.7%** | **−4.2pp** (절감폭 확대) |
| our_cat_conformal | −32.9% | **−37.3%** | **−4.4pp** (절감폭 확대) |
| artisee_reimpl | −11.6% | −15.3% | −3.7pp (절감폭 확대) |

**결론**: "폐기 −33~40%" 헤드라인은 새 166(scoped 157, cake·sweets 제외 — 옛 헤드라인과
동일 스코프 확인됨) canonical + 신규 closing에서도 **같은 방향으로 재확인됐고, 절감폭은
오히려 소폭 커졌다** — quantile −44.7%(옛 −40.5%), conformal −37.3%(옛 −32.9%), 둘 다
4.2~4.4pp 더 절감. 옛 범위 "−33~40%"를 새 범위로 다시 쓰면 대략 **−37~45%**. 방향성
결론(우리 카테고리 총량 발주가 아띠제 실생산 대비 폐기를 크게 절감하되 카테고리 매진이
증가하는 트레이드오프)은 그대로 유지된다.

절대 KRW 규모는 basis가 이중으로 다르다 — (1) 품목 재정의+신규 closing이 母수요 자체를
바꿨고(actual_production_sim 기준 참고: 옛 커밋은 이 basis를 명시적으로 분리하지
않았음), (2) **옛 실행은 unit_prices를 넘기지 않아 flat 3000원/개**로 KRW를 계산했고
이번 실행은 **실제 품목단가**(0526 master_xlsx)를 사용했다 — 두 basis가 다르므로 절대
KRW(10.44M vs 18.49M 등) 자체는 비교 대상이 아니다. **% delta만 비교 가능**(가격 basis가
flat이든 실제든 정책 간 비율 계산에서 상쇄되므로).

### 매진(2차 KPI) — 참고

새 측정 매진②(SKU soldout rate, B basis): quantile 0.510, conformal 0.465,
artisee_reimpl 0.310, actual_production_sim 0.048. 옛(146) 매진② 대응값: quantile 0.497,
conformal 0.460, artisee_reimpl 0.311. **거의 변화 없음** — 매진 트레이드오프 방향·크기도
재확인됨(폐기 절감 정책일수록 매진 증가, 옛/새 동일).

## 경계 명시 (필수)

- 이 delta는 **146→157(166 canonical 중 bread/pastry/sandwich)** 품목/타깃정의 변화
  **+ 신규 closing 소스 교체분**을 함께 포함한 재측정이며, 두 요인을 분리하지 않았다.
- 날짜 범위는 여전히 **2021~2025-12**(canonical 데이터 범위) — **2026 전향(prospective) 검증이
  아니다.** 아띠제와의 실전 비교(4주 구축+4주 전향)는 별도 후속.
- `is_stockout`/`waste_qty`는 재고정보(master_xlsx 0526) 기반 재구성치이며, censoring 패널이
  보이듯 실제 매진은 시뮬(B)보다 훨씬 큼(매진② A=0.487 vs B=0.048) — 헌장 §6대로 모델 폐기
  절감은 **하한**(보수적)으로만 해석.
- `our_cat_nk15/nk30`은 옛 `nk(+40)`과 버퍼 크기가 달라 직접 비교 대상에서 제외했다(옛
  +40에 가장 가까운 건 없음 — 스캔 범위가 바뀌었다는 사실만 기록).

## 상호참조

- 총량 WAPE 재측정: [`docs/phase7/gwangyo_headline_remeasure.md`](gwangyo_headline_remeasure.md)
  (Task 3, 8.03%→7.72%, 동일 harness 경로·n_test 정확 일치로 확인된 유일한 methodologically
  클린 delta). 이 문서(Task 3.5)의 폐기 KPI는 harness-run 범위 밖의 **별도 파이프라인**
  (`unified_policy_kpi.py`, 아띠제 실생산 ground truth) 재실행 결과다 — 두 문서를 종합해야
  Phase 7 헤드라인 전체 그림이 된다.

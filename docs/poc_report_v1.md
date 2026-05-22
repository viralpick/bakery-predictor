# 베이커리 수요예측 PoC 리포트 v1

> 광교 매장(아티제 아브뉴프랑광교점, 점포코드 1000000047) 5년치 데이터로
> v0 → v3 모델 4단계 + production model까지 도입한 PoC 종합 정리.
> 노션 공유용. PM은 1~5장만 봐도 의사결정 가능. 엔지니어는 6장 이후 상세 참고.
>
> **v1 내 substitution 보강 (2026-05-21)**: 사용자 통찰을 반영해 **품목 간 substitution 효과**를
> 영수증 단위 cross-sell 분석(RD)으로 실측. potential_demand 보정에 substitution outflow
> 적용 — 사업 임팩트 추정의 over-estimate를 60% 축소했습니다.
>
> **다음 버전**: 사업 분석 6단계(A/B/G/F/F2/F3) 추가본 → `docs/poc_report_v2.md`

---

## 1. Executive Summary — PM 1분 읽기용

### 핵심 사업 임팩트 (광교 매장 24개월 환산)

| 항목 | 금액 |
|---|---|
| 실제 매출 | **8.34억원** |
| 실제 마진 (50% 가정) | 4.17억원 |
| 잃은 매출 (independent 보정 — over-estimate) | 5.91억원 |
| Substitution 평균 outflow rate | **55.3%** |
| **잃은 매출 (substitution-adjusted)** | **2.50억원** |
| **잃은 마진 (substitution-adjusted, cross-sell·평판 1.7× 가중)** | **2.12억원** |

→ **광교 매장은 v0 운영 시 매장 마진의 약 50%를 영구 손실** (substitution 반영).

### v0 vs v2/v3 비교 (4 fold backtest 누계, 7일 horizon, substitution-adjusted target)

| 모델 | wape | pct_under | **net_profit** |
|---|---|---|---|
| v0 (현재 운영 가정) | 0.215 ✓ | 12.5% | **−10.73M원** ← 마이너스 |
| v1 | 0.216 | 25.0% | −11.08M원 |
| **v2 (median)** | 0.771 | 12.5% | **+21.60M원** ← 1위 |
| **v3 (median)** | 0.828 | 0.0% | +21.52M원 |
| v2_q85 (production) | 1.210 | 0.0% | +16.52M원 |
| v3_q85 (production) | 1.214 | 0.0% | +15.11M원 |

→ **v0 → v2 도입 net_profit 차이: +32.33M원** (28일 backtest 누계).
24개월 환산 시 **약 3억원 추가 마진** (보수 추정, substitution 반영).

WAPE만 보면 v0가 좋아 보이지만 — **WAPE는 관찰값 fit, net_profit이 진짜 사업 가치**.

### 한 줄 결론

> **WAPE는 좋아 보이는 v0 모델이 사실은 매장을 매출 천장에 갇히게 만드는 함정. v2/v3의 quantile production model(α=0.85)이 진짜 사업 가치를 만든다. 다만 v3 외부 데이터의 효과는 다매장 비교 시점에 발현.**

---

## 2. 매장 정보 + 분석 기간

| 항목 | 값 |
|---|---|
| 매장 | 아티제 아브뉴프랑광교점 |
| 점포코드 | 1000000047 (보나비 시스템) |
| 주소 | 경기도 수원시 영통구 (광교2동) |
| 좌표 | lat 37.2853 / lon 127.0593 |
| 데이터 기간 | 2021-01-01 ~ 2025-12-31 (5년) |
| 데이터 row | 영수증 line item 458,366 → 일별 65,452 |
| 품목 수 | 146 active items (전체 마스터 417개) |
| 카테고리 | bread / pastry / cake / sandwich / sweets / beverage |
| 학습 윈도우 | 2024-01-01 ~ 2025-12-31 (외부 데이터 align) |

---

## 3. 핵심 인사이트 모음

### 인사이트 1 — Self-fulfilling stockout (자기실현적 품절)

광교 매장 인기 품목 top 10 분석:

| 지표 | 값 |
|---|---|
| 매주 7요일 모두에서 품절률 | **90~95%** |
| 매출 변동 계수 (CV) | **0.23~0.34** (매주 거의 동일량) |
| 평균 품절 시각 | 오후 14~18시 (이른 시간) |

→ **매장이 매주 같은 수요에 대해 같은 양만 만들고 같은 시각에 품절되는 사이클**에 갇혀 있음.
이것은 v0 모델 운영의 자기실현적 결과:

```
주차 1: 실 수요 20개 → v0가 10개 예측 → 16시 품절 → sold=10
        ↓
        "월요일 수요=10" 학습
        ↓
주차 2: 실 수요 20개 → 다시 10개 예측 → 또 16시 품절 → sold=10
        ↓
        영원히 천장 고착
```

### 인사이트 2 — v0 WAPE가 의미 없는 이유

```
WAPE = Σ|yhat - sold_units| / Σ sold_units
                       ↑
       관찰된 매출 (품절일에는 실수요보다 작음)
```

**v0 wape 0.2151은 "맞춤 능력"이지 "수요 예측 정확도"가 아님**:

- 주차 1: yhat=10, sold=10 → wape 0% (perfect)
- 주차 2: yhat=10, sold=10 → wape 0% (perfect)
- 매장의 진짜 잠재 매출 20개는 영원히 놓침
- **WAPE 0%여도 매주 50% 잠재 매출 손실** ← 측정 안 됨

→ WAPE는 v0의 자기실현적 손실을 못 잡는다.

### 인사이트 3 — Cross-sell 손실 (베이커리 관행)

베이커리 고객은 영수증 1건당 평균 2~4개 품목을 함께 구매.
**핵심 인기 품목 품절 시 → 고객이 다른 품목도 안 사고 그냥 나감.**

품절 1개의 사업 비용 환산:
- 직접 손실 마진: 500원 (단가 1000원 × 마진 50% 가정)
- + Cross-sell 손실: 잠재 구매 다른 품목 매출
- + 평판/재방문 손실: "그 매장 인기 품목 항상 품절" 평가 ↓
- **총합 ≈ 직접 마진의 1.7배 (PoC 디폴트 가정)**

→ 우리 `lost_sale_multiplier = 1.7` 파라미터의 근거.
실측치는 영수증 단위 cross-sell 분석으로 보강 가능 (보나비 데이터에 영수증번호 있음).

### 인사이트 4 — 폐기 < 품절 (베이커리 관행)

| 시나리오 | 직접 단위 비용 | 회수 가능성 |
|---|---|---|
| 폐기 1개 | 원재료 약 30% (300원) | 마감세일·직원식사로 일부 회수 |
| **품절 1개** | **잃은 마진 50% + cross-sell + 평판** | **0% 회수** |

→ **베이커리는 약간의 폐기를 감수하고 품절을 회피하는 것이 절대 유리.**
이게 quantile α=0.85 production model의 정당성.

### 인사이트 5 — v3 외부 데이터가 광교에서 무용한 이유

광교는 매장이 **1개**. 외부 features 15개는 다음 시나리오에서 가치 발현:

| Feature 그룹 | 광교 단독에서 동작? | 다매장에서 동작? |
|---|---|---|
| Cannibalization (매장×카테고리 lag) | ❌ 매장 1개라 cross-store 비교 불가 | ✅ |
| Competitor (반경 카운트) | ⚠️ store_id와 collinear (값 1개) | ✅ |
| Living population (행정동 인구) | ⚠️ 광교는 서울 데이터 없어 default | ✅ |
| Population (행정동 연령) | ⚠️ store_id와 collinear | ✅ |
| Consumption (행정동 소비) | ⚠️ 서울만이라 default | ✅ |

→ **v3의 net_profit이 v2와 동일(13.08M)인 이유**: 외부 features 모두 광교 1매장에선 상수 → LightGBM이 split 못 함 → 학습 정보 0.

### 인사이트 6 — 다매장 시 v3 가치 발현

서울 매장 1개+ 도착하면:

| Feature | 다매장에서 학습 가능한 패턴 |
|---|---|
| `competitor_bakery_500m` | 광교: 18개 / 서울 홍대: 62개 → "주변 경쟁 많을수록 매출 ↓" |
| `living_pop_lunch_share` | 광교: 0.20 / 여의도: 0.75 → "점심 인구 많은 매장은 sandwich 잘 팔림" |
| `pop_share_60_plus` | 광교: 0.16 / 청담: 0.26 → "고령층 많은 동네 bread 선호" |
| `consumption_food_retail` | 매장 행정동의 식료품 지출 → baseline 차이 |

광교 1개에선 모든 값이 상수 → 학습 정보 없음. 매장 2개+면 변동 있음 → LightGBM이 패턴 학습.

### 인사이트 7 — Substitution effect (대체재 효과)

**사용자 통찰**: A 품절일 때 손님 일부는 같은 카테고리 B 구매. A의 진짜 잃은 매출은 "A 잠재 수요 × (1 − substitution rate)".

**보나비 영수증 단위 cross-sell 분석으로 실측**:
- 영수증 1건당 평균 1.86 품목 — **48% 가 cross-sell (2품목+)**
- 광교 매장 평균 substitution outflow: **55.3%** (cap 0.7 적용)
- Top 14 인기 품목은 outflow가 cap에 saturate (강한 substitution 신호)

**방법론** — Regression Discontinuity on daily aggregate:
```
β_RD(i, j) = mean(j 매출 | i 일찍 품절) - mean(j 매출 | i 정상)
            ─────────────────────────────────────────────────
                          mean(j 매출)
                          
substitution(i→j) = max(0, β_RD) × (1 - co_occurrence(i, j))
outflow(i)        = Σ_j substitution(i→j), capped at 0.7
```

**영향**:
- 단순 보정 잃은 매출 5.91억 → substitution-adjusted **2.50억원** (over-estimate 58% 감소)
- v2 학습 target도 substitution-aware로 보정 → 더 신뢰성 있는 모델

### 인사이트 8 — 단순 모델이 데이터 부족 시 정답

| 시나리오 | 권장 모델 |
|---|---|
| **광교 단독** | **v0 (단순 매출 lag)** — 외부 features 무용 |
| 서울 + 경기 2~3매장 | v1~v2 효과 발현 시작 |
| 매장 10개+ 운영 | v3 외부 데이터 본격 가치 |
| 신규 매장 cold start | v3 (외부 매장 정적 메타가 사실상 유일한 baseline) |

→ **모델 복잡도는 데이터 양과 함께 점진적으로 올려야**. PoC는 v0~v3 인프라를 미리 갖춰 다매장 확장에 대비.

---

## 4. 메트릭 정의 (PM 필독)

### 4.1 WAPE (Weighted Absolute Percentage Error)

```
WAPE = Σ|예측 - 실제| / Σ 실제
```

**의미**: 실제 매출 단위로 평균 몇 % 어긋났는가.

**예시 (광교 v0 WAPE 0.2151)**:
- 평균 일별 품목당 매출 8개
- 평균 절대 오차 1.7개
- 즉 "8개 팔리는 날 평균 1.7개 정도 어긋남"

**한계**: 관찰값(sold_units)에만 fit하는 능력. 자기실현적 품절을 못 잡음.

### 4.2 pct_under / pct_over

```
pct_under = (예측 < 실제) 비율  ← 과소예측, 품절 위험
pct_over  = (예측 > 실제) 비율  ← 과대예측, 폐기 위험
```

**광교 v0**: pct_under 12.5% / pct_over 62.5%
→ "대부분 일자에 과대예측 (안전 사이드)"

베이커리 관행: **pct_under 낮은 게 운영상 더 중요** (품절 비용 > 폐기 비용).

### 4.3 net_profit (사업 KPI)

```
revenue_krw     = min(yhat, true_demand) × 단가 × 마진율       (매출 마진)
waste_cost_krw  = max(yhat - true_demand, 0) × 단가 × 원가율    (폐기 비용)
lost_margin_krw = max(true_demand - yhat, 0) × 단가 × 마진율 × 1.7  (품절 손실, cross-sell·평판)

net_profit = revenue - waste_cost - lost_margin
```

**파라미터** (PoC 디폴트, CLI 인자로 조정 가능):
- 마진율: 50%
- 원가율: 30%
- 품절 손실 multiplier: 1.7× (cross-sell + 평판)

**true_demand**:
- v2/v3는 `potential_demand` (품절 보정값) 사용
- v0/v1은 `sold_units` 그대로

### 4.4 asymmetric_loss

```
asymmetric_loss = (α × Σunder + β × Σover) / Σsold
  α = 1.7 × 0.50 = 0.85  (품절 단위 비용)
  β = 0.30                (폐기 단위 비용)
```

→ WAPE의 비대칭 비용 변형. 낮을수록 좋음.

---

## 5. 단계별 모델 발전 (v0 → v3)

### v0 — 내부 매출 시계열 baseline

| 항목 | 내용 |
|---|---|
| **Target** | `sold_units` (관찰값 그대로) |
| **Input** | categorical 5 + numeric 18 (date + sales lag/rolling) |
| **품절 인지** | ❌ 안 함 |
| **가치** | 기준점. 매출 시계열만으로 학습 가능한 한계 |

### v1 — + 캘린더 + 날씨

| 항목 | 내용 |
|---|---|
| **Target** | `sold_units` |
| **Input 추가** | calendar 8 + weather 10 = 18 (총 36 numeric) |
| **품절 인지** | ❌ 안 함 |
| **가치** | 외부 환경 영향 반영. 다만 매출 lag에 이미 흡수된 정보가 많아 marginal value 작음 |

### v2 — + Cannibalization + **품절 보정 target**

| 항목 | 내용 |
|---|---|
| **Target** | `potential_demand` ← 품절 보정 (핵심 변화) |
| **Input 추가** | cannibalization 6 = 42 numeric |
| **품절 인지** | ✅ 보정 target으로 자기실현적 사이클 깨기 |
| **가치** | 사업적 의미 큰 차이. WAPE는 비슷해도 net_profit 우월 |

`potential_demand` 계산:
- 무품절일: `potential = sold_units` (그대로)
- 품절일: `potential = sold / cumulative_combined_profile(stockout_time)`
  - `combined = 0.5 × hour_weight + 0.5 × uniform`
  - 안전장치: max 3× clip, min 0.15 floor

**예시**: 매출 10개, 16시 품절 → 16시까지 누적 가중치 0.5 → potential = 10/0.5 = 20

### v3 — + 외부 데이터 4축

| 항목 | 내용 |
|---|---|
| **Target** | `potential_demand` |
| **Input 추가** | competitor 6 + living_pop 3 + population 4 + consumption 2 = 15 (총 57 numeric) |
| **품절 인지** | ✅ |
| **가치** | **다매장 시나리오에서 발현**. 단일 매장에선 store_id categorical과 collinear |

### Production model (v2_q85, v3_q85)

| 항목 | 내용 |
|---|---|
| **알고리즘** | LightGBM quantile objective, α=0.85 |
| **예측** | 85% quantile (수요가 이만큼 이하일 확률 85%) |
| **운영** | 추천 생산량 — 품절 확률 15% 이하 목표 |
| **가치** | **net_profit 25.3% 향상** (광교 backtest 누계) |

---

## 6. Features 전체 정리 (엔지니어 상세)

### Categorical (5, 모든 모델 공통)

LightGBM native categorical encoding (one-hot 아닌 partitioning):

| Feature | 의미 |
|---|---|
| `store_id` | 매장 식별자 |
| `item_id` | 품목 식별자 |
| `category_id` | bread/pastry/cake/sandwich/sweets/beverage/etc |
| `dow` | 요일 (0=월 ~ 6=일) |
| `month` | 월 (1~12) |

### Numeric — v0 (18개)

#### Date 파생 (5)

| Feature | 의미 | Forecast-safe? |
|---|---|---|
| `is_weekend` | 주말 boolean | ✅ |
| `quarter` | 분기 1~4 | ✅ |
| `week_of_year` | 1~53 | ✅ |
| `day_of_month_sin` | 월 내 위치 (sin) — cyclic | ✅ |
| `day_of_month_cos` | 월 내 위치 (cos) — cyclic | ✅ |

> **why cyclic?** 30일/31일/2월28일 모두 "월 마지막" 의미 동일.
> raw 1~31 int로 두면 모델이 30 vs 31을 다르게 학습. sin/cos로 원형 인코딩하면 day 1과 last day가 인접 위치로 학습됨.

#### LAG 매출 (6)

| Feature | 의미 |
|---|---|
| `sales_lag_1` | 1일 전 매출 |
| `sales_lag_7` | 7일 전 매출 (= 1주 전 같은 요일) |
| `sales_lag_14` | 14일 전 매출 |
| `sales_lag_28` | 28일 전 매출 |
| `sales_same_dow_2w` | 2주 평균 (shift 7,14 평균) |
| `sales_same_dow_4w` | 4주 평균 (shift 7,14,21,28 평균) |

> `shift(N)`은 매장×품목 단위 그룹 내에서 미래 매출을 절대 못 보도록 강제. leakage 회귀 테스트로 보장.

#### ROLLING 매출 (7)

| Feature | 의미 |
|---|---|
| `roll_mean_7/14/28` | 직전 1~4주 평균 |
| `roll_std_7/28` | 직전 1주/4주 표준편차 |
| `roll_median_28` | 직전 4주 중앙값 |
| `same_dow_roll_mean_4w` | 4주 동일요일 rolling |

### Numeric — v1 추가 (+18 = 총 36)

#### Calendar (8)

천문연 특일정보 + 24절기 데이터 기반.

| Feature | 의미 |
|---|---|
| `is_public_holiday` | 공휴일 |
| `is_white_day` | 화이트데이 (3/14) boolean |
| `off_streak_length` | 연휴 연속 길이 |
| `off_position_in_streak` | 연휴 내 위치 |
| `days_to_xmas` | 크리스마스까지 signed int (-14~+14, 케이크 매출 사전 ↑) |
| `days_to_valentine` | 발렌타인까지 (sweets) |
| `days_to_children_day` | 어린이날까지 |
| `days_to_pepero` | 빼빼로데이까지 |

> **days_to_X 도입 이유**: 베이커리 케이크/sweets 매출은 이벤트 1주 전부터 점진적 증가.
> boolean 단일일 표시는 부족. -14 ~ +14 numeric으로 lead/lag 효과 학습.

#### Weather (10)

기상청 ASOS 일자료. 매장 매핑된 station_id 기반 (서울 108, 수원 119 등).

| Feature | 의미 |
|---|---|
| `avg_temp` / `max_temp` / `min_temp` | 일별 기온 (°C) |
| `diurnal_range` | 일교차 |
| `humidity` | 평균 습도 |
| `precipitation_mm` | 강수량 |
| `is_rain` | 강수 여부 boolean |
| `snow_depth_cm` | 적설량 |
| `is_snow` | 강설 여부 |
| `sunshine_hours` | 일조시간 |

> Threshold flags (is_heavy_rain, is_heatwave 등)는 raw 값에서 LightGBM이 자동 split 학습 가능하므로 제거 (PoC review).

### Numeric — v2 추가 (+6 = 총 42)

#### Cannibalization (매장×카테고리 store-level lag)

| Feature | 의미 |
|---|---|
| `store_stockout_rate_lag1` | 어제 매장 전체 품절률 |
| `store_stockout_rate_7d` | 직전 7일 매장 품절률 |
| `store_total_sold_lag1` | 어제 매장 전체 매출 |
| `cat_stockout_rate_lag1` | 어제 동일 카테고리 품절률 |
| `cat_stockout_rate_7d` | 직전 7일 카테고리 품절률 |
| `cat_total_sold_lag1` | 어제 동일 카테고리 전체 매출 |

> **다매장 비교 신호**. 광교 단독에서는 의미 작음 — 매장간 비교가 핵심.

### Numeric — v3 추가 (+15 = 총 57)

#### Competitor (6)

소상공인진흥공단(카페) + 행안부 LOCALDATA(베이커리) 기반. 시점 정합:
`license_date ≤ d AND (close_date is null OR close_date > d)`.

| Feature | 의미 |
|---|---|
| `competitor_bakery_500m` / `1km` | 반경 영업 중 제과점 수 (동적) |
| `competitor_cafe_500m` / `1km` | 반경 영업 중 카페 수 (정적, SBIZ snapshot) |
| `competitor_new_bakery_90d_1km` | 최근 90일 신규 제과점 |
| `competitor_closed_bakery_90d_1km` | 최근 90일 폐업 제과점 |

#### Living population (3) — 매장 정적 baseline

서울 SPOP_LOCAL_RESD_DONG (KT 통신 기반). 광교는 서울 데이터 없어 default fallback.

| Feature | 의미 |
|---|---|
| `living_pop_daily_avg` | 매장 행정동의 시간평균 생활인구 |
| `living_pop_lunch_share` | 점심대(11~13) 인구 / 일평균 |
| `living_pop_weekend_ratio` | 주말 / 평일 인구 비율 |

#### Population (4) — 행안부 행정동 주민등록

| Feature | 의미 |
|---|---|
| `pop_share_0_9` | 영유아 비중 (cake/sweets 친화) |
| `pop_share_20_39` | 청년층 비중 (sandwich/pastry 친화) |
| `pop_share_30_49_female` | 가족 구매층 비중 |
| `pop_share_60_plus` | 고령층 비중 (bread 친화) |

#### Consumption (2) — 서울 상권분석 행정동 소비

| Feature | 의미 |
|---|---|
| `consumption_total_log` | 분기 총 소비 (log scale) |
| `consumption_food_retail_log` | 음식+식료품 소비 (log) |

---

## 7. 모델링 알고리즘

### LightGBM (Gradient Boosted Decision Trees)

**Global model** — 모든 (store, item) pair를 한 모델로 학습:
- 매장간 / 품목간 정보 공유 (sparse 품목·신규 매장 cold start에 유리)
- LightGBM의 categorical encoding이 store_id, item_id 등을 partitioning으로 처리

**Hyperparameters** (`LGBMParams`):

| Param | Value | 의미 |
|---|---|---|
| `n_estimators` | 600 | 부스팅 round |
| `learning_rate` | 0.05 | |
| `num_leaves` | 63 | tree 복잡도 |
| `min_data_in_leaf` | 20 | leaf 최소 샘플 |
| `feature_fraction` | 0.9 | tree별 feature 90% random |
| `bagging_fraction` | 0.9 / `bagging_freq` 5 | 매 5 iter 데이터 90% bagging |
| `objective` | `regression` 또는 `quantile` | demand vs production |
| `alpha` | 0.5 또는 0.85 (quantile일 때) | production model용 |
| `metric` | `mae` | early stopping ref |

### Train/Predict 흐름

```
Fit:
  1. _check_keys + _check_feature_set_columns
  2. self._train_history = train 보존 (predict 시점에 lag/rolling 계산용)
  3. _build_features: date + lag + rolling + (v2+) cannibalization
  4. dropna lag 부족한 초반 row
  5. LightGBM categorical encoding (store_id/item_id/category_id/dow/month)
  6. lgb.train(params, Dataset, num_boost_round=600)

Predict:
  1. _join_history: target frame을 train history 위에 stack
  2. lag/rolling을 history+target 통합 frame에서 다시 계산
  3. target_mask로 horizon rows만 추출
  4. clip(yhat, min=0)
```

### Backtest — 시간순 expanding window

```
n_splits=4, horizon_days=7, step_days=7

Fold 1: train [2024-01 ~ 2025-12-03] | val [2025-12-04 ~ 12-10]
Fold 2: train [2024-01 ~ 2025-12-10] | val [2025-12-11 ~ 12-17]
Fold 3: train [2024-01 ~ 2025-12-17] | val [2025-12-18 ~ 12-24]
Fold 4: train [2024-01 ~ 2025-12-24] | val [2025-12-25 ~ 12-31]
```

**Random split 금지** (시계열 leakage). 각 fold별 WAPE/MAE/pct_under/pct_over 계산 → 평균.

### Production model layer (v2/v3)

```
demand_model (α=0.5) → yhat_potential_demand  (median 예측)
prod_model   (α=0.85) → recommended_production (85% quantile)
                                                     ↓
                          품절 확률 ≤ 15% 목표 안전 생산량
```

CLI:
```bash
uv run bakery predict-next-week --model lightgbm_v2 \
    --use-forecast --production-quantile 0.85
```

α 값으로 보수성 조정:
- 0.50 = median (균형)
- 0.65 = 단순 newsvendor 최적 (Cu=500, Co=300 가정)
- **0.85 = 디폴트** (cross-sell·평판 가중)
- 0.95 = 폐기 폭증, 품절 거의 0

---

## 8. 단계별 backtest 결과 (광교 매장, 4 fold × 7일)

### WAPE 기준 (관찰값 fit 능력)

| 모델 | wape_all | mae | pct_under | pct_over |
|---|---|---|---|---|
| **v0 (lightgbm)** | **0.2151** ✓ | 1.72 | 12.5% | 62.5% |
| v1 | 0.2162 | 1.73 | 25.0% | 62.5% |
| v3 | 0.2206 | 1.76 | 25.0% | 62.5% |
| v2 | 0.2220 | 1.77 | 12.5% | 62.5% |
| seasonal_naive | 0.2322 | 1.86 | 25.0% | 25.0% |
| v2_q85 (production) | 0.2386 | 1.91 | 12.5% | 75.0% |
| v3_q85 (production) | 0.2418 | 1.93 | 12.5% | 75.0% |
| moving_average | 0.2546 | 2.03 | 50.0% | 37.5% |

### 사업 KPI 기준 (net_profit)

| 모델 | asymmetric_loss | revenue | waste_cost | lost_margin | **net_profit** |
|---|---|---|---|---|---|
| **v3_q85** | **0.0953** ✓ | 16.43M | 2.10M | 1.25M | **13.08M** ✓ |
| **v2_q85** | 0.0958 | 16.44M | 2.14M | 1.23M | **13.08M** ✓ |
| seasonal_naive | 0.1296 | 15.29M | 1.35M | 3.18M | 10.76M |
| v0 (lightgbm) | 0.1305 | 15.05M | 1.02M | 3.60M | 10.44M |
| v1 | 0.1335 | 14.97M | 0.97M | 3.73M | 10.27M |
| moving_average | 0.1438 | 15.07M | 1.41M | 3.56M | 10.10M |
| v3 (median) | 0.1382 | 14.87M | 0.95M | 3.90M | 10.01M |
| v2 (median) | 0.1429 | 14.73M | 0.88M | 4.15M | 9.70M |

→ **WAPE 1위(v0)가 net_profit 4위. Production model이 사업 KPI 1·2위**.

### v3 = v2 동일 net_profit인 이유

광교 1매장에서 v3의 외부 features 15개가 모두:
- 상수값 (1매장 → store_id categorical과 collinear)
- 또는 default fallback (서울 데이터에 광교 dong 없음)

→ LightGBM이 외부 features로 split 못 함 → v2와 사실상 같은 모델.
**다매장 도착 시 v3 > v2 차이 발현 예상**.

---

## 9. 인프라 정리 — 무엇이 준비됐고 무엇이 남았나

### 준비 완료 (서울 매장 도착 시 즉시 가동)

| 영역 | 상태 |
|---|---|
| 데이터 schema (HOURLY/DAILY/WEATHER/CALENDAR/COMPETITOR/...) | ✅ |
| 7개 외부 데이터 ingest API + CLI | ✅ |
| 매장별 좌표·행정동·시군구·ASOS station·KMA 격자 매핑 | ✅ (yaml override 가능) |
| Feature engineering (76개 features) | ✅ |
| v0~v3 + production model 4종 LightGBM | ✅ |
| 시간순 backtest (n_splits×horizon×step) | ✅ |
| 사업 KPI 메트릭 (asymmetric loss, profit simulation) | ✅ |
| 보나비 xlsx → DAILY_COLUMNS 어댑터 | ✅ |
| 99/99 회귀 테스트 (leakage·schema·features) | ✅ |

### 남은 작업 (우선순위 순)

1. **Hourly granularity 도입** — 보나비 판매시간 활용. `hour_weights` 매장 실측 calibration → potential_demand 보정 정확도 ↑. (1~2일)
2. **서울 매장 데이터** — 다매장 비교로 v3 가치 검증. (데이터 도착 후 자동)
3. **카페 LOCALDATA history** — 휴게음식점 데이터셋 활용신청 후 cafe trend features unlock. (1일)
4. **영수증 cross-sell 분석** — 1.7× multiplier 실측 보정. (1~2일)
5. **경기 대체 데이터 source** — 생활인구·소비 대체 (옵션). (1~2일)
6. **상권 프로파일 산출물** — spec § 매장별 정성 요약 리포트. (1일)

---

## 10. 베이커리 도메인 관행 (PoC에 반영된 도메인 지식)

| 관행 | 우리 모델/파라미터 반영 |
|---|---|
| 매출 마진 ≈ 50% | `CostParams.margin_rate = 0.50` |
| 원재료 비율 ≈ 30% | `CostParams.cost_rate = 0.30` |
| 품절 cross-sell·평판 비용 ≈ 직접 마진의 1.7× | `lost_sale_multiplier = 1.7` |
| 폐기 < 품절 (베이커리는 약간 over 안전) | quantile production α = 0.85 |
| 인기 품목 오후 일찍 품절이 흔함 | hourly stockout_time 보존 + potential_demand 보정 |
| 베이커리 영수증 1건당 2~4개 품목 | cross-sell 가중치 1.7× 근거 |
| 크리스마스 케이크 매출 12/20~25 집중 | `days_to_xmas` numeric -14~+14 lead/lag |
| 발렌타인·화이트데이 sweets 매출 ↑ | `days_to_valentine` 등 |
| 매일 같은 인기 품목 같은 시각 품절 = 천장 시멘트화 | self-fulfilling 분석 |
| 베이커리 4-peak 시간대 (9시·13시·16시·19시) | synthetic DGP `_hour_weights` 디폴트 |
| 매장 신선식품이라 당일폐기 ≥ 95% | `당일폐기여부 Y` 397/417 품목 |
| A 품절 시 손님 일부 카테고리 내 B 구매 (substitution) | `substitution.py` outflow_ratio 영수증 RD 실측 (광교 평균 55%) |
| 영수증 1건당 평균 1.86개 품목, 52% 단품 구매 | cross-sell baseline (보나비 영수증 24만건) |

---

## 11. 다음 의사결정 — PM 보드용

### 즉시 운영 가능 (광교)

- **v2_q85 또는 v3_q85 production model 운영** → 24개월 환산 +2.3억원 추가 마진
- 인프라 그대로 사용 가능

### 의사결정 필요 사항

| 항목 | 결정 |
|---|---|
| Production quantile α 값 | 디폴트 0.85 / 매장별 카테고리별 차별 가능 |
| 마진율·원가율·multiplier | PoC 디폴트 사용 vs 실제 마진 데이터 적용 |
| 폐기 처리 운영 | 마감 할인 / 직원 식사 / 기부 (회수율 차이) |
| 인기 품목 추가 생산량 | 카테고리별 α 차별화 가능 |

### 검증 단계

1. **A/B test** (특정 매장에 v2_q85 운영 / 다른 매장에 v0 운영) — 1~2개월 후 실 net_profit 비교
2. 또는 매주 같은 인기 품목에서 v2_q85 추천량 비교 → 품절 시각 후퇴 확인

---

## 12. 산출물 위치

| 파일 | 내용 |
|---|---|
| `reports/business_report_kpi.csv` | 모델별 사업 KPI 누계 |
| `reports/business_report_folds.csv` | fold별 WAPE/MAE/pct_under |
| `reports/business_report_self_fulfilling.csv` | top 10 self-fulfilling 품목 |
| `reports/business_report_top_lost.csv` | top 10 잃은 매출 품목 |
| `reports/fold_results.csv` / `predictions.csv` | 일반 backtest 산출 |
| `reports/next_week_predictions.csv` | predict-next-week 출력 |
| `reports/feature_importance_*.csv` | 모델별 feature_importance |

CLI 명령 모음:
```bash
uv run bakery business-report                 # 종합 임팩트 리포트
uv run bakery backtest --source real --variants v0,v1,v2,v3 --include-production
uv run bakery predict-next-week --model lightgbm_v2 --use-forecast
uv run bakery format-bonavi                    # 보나비 xlsx → daily parquet
```

API 데이터 출처와 ingest 가이드는 **`docs/external_data_sources.md`** 참조.

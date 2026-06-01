# v4 베이커리 수요 예측 모델 — 종합 리포트

광교 매장 PoC. 5년 데이터 (2021-01-01 ~ 2025-12-31).

**최종 설정**:
- α=0.6, quantile=0.90, 시즌 16개 제외
- 카테고리 합산 (bread + pastry + sandwich)
- **Stage 1 모델: Ensemble (baseline + LGBM residual + dow safety)**
- 운영 시나리오: D=목요일 → D+4~D+10 (다음주 월~일) 발주 예측

---

## §1. Input 데이터

### 1-1. Internal (광교 매장 데이터)

보나비 xlsx 5 시트 → `bonavi_daily.parquet` + `bonavi_receipts.parquet`.

#### 1-1-A. daily 단위

| 컬럼 | 타입 | 의미 | 왜 필요한가 |
|---|---|---|---|
| `store_id` | string | 매장 ID (광교 = `store_gw01`) | 다매장 확장 시 group |
| `item_id` | string | POS 품목코드 | 품목별 lag/rolling |
| `category_id` | string | bread / pastry / sandwich / cake | 카테고리 합산 Stage 1 |
| `date` | date | 판매 일자 | 시계열 split + 외부 join |
| `sold_units` | int | 그날 품목 판매 수량 | target base |
| `is_stockout` | bool | 그날 품절 발생 여부 | censored demand 식별 |
| `stockout_time` | datetime | 품절 발생 시각 (HHMM) | Stage 2 stockout_signal |
| `open_hours` | int | 영업 시간 길이 | potential_demand 보정 |
| `capacity` | int | 매장 capacity 추정 | 폐기 추정 |

#### 1-1-B. receipts 단위

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `receipt_id` | string | 영수증 ID |
| `date` / `hour` / `minute` | datetime | 영수증 발생 시각 |
| `item_id` | string | 포함 품목 |

#### 1-1-C. 판매정보 raw (분석 모듈 전용)

| 컬럼 | 의미 |
|---|---|
| `할인코드` | 30개 코드 (마감/PAYCO/직원/B2B) → label 분류 |
| `할인금액` | 라인별 할인 금액 → 마감 할인 손실 |
| `단가` | 정가 → revenue target |
| `판매수량` | unit |

### 1-2. External

#### 1-2-A. 기상 (`weather_observed.parquet`)

수원 기상관측소 (station_id = 119).

| 컬럼 | 의미 |
|---|---|
| `avgTa` / `maxTa` / `minTa` | 일평균/최고/최저 기온 |
| `sumRn` | 일 강수량 (mm) |
| `avgRhm` | 일평균 상대습도 (%) |
| `avgTca` | 일평균 운량 (0-10) |
| `avgWs` | 일평균 풍속 (m/s) |
| `hr1MaxRn` / `hr1MaxRnHrmt` | 시간당 최대 강수 + 시각 → 영업시간 폭우 |

⚠ **운영 시 일기예보 사용** (forecast_short_term + forecast_mid_term). 예측 부정확이 정확도에 영향.

#### 1-2-B. 캘린더 (`calendar_raw.parquet`)

천문연 특일정보 API. `date` / `is_holiday` (대체공휴일 포함) / `name`.

#### 1-2-C. 경쟁점 (`competitor_raw.parquet`)

`business_id` / `category` / `license_date` / `close_date` / `lat` / `lon`. 광교 1km Haversine + 일자별 active 계산.

#### 1-2-D. 미사용 (다매장 도착 시 활성화)

`living_population`, `population`, `consumption` (광교 단독에선 효과 식별 어려움).

### 1-3. Stage 1 모델 입력 — 57 features

#### 카테고리 합 raw (5)

| Feature | 정의 |
|---|---|
| `sold_total_unit` | 그날 bread+pastry+sandwich 합 sold |
| `sold_total_revenue` | sold_total_unit × unit_price |
| `sold_closing` | 마감 할인 적용 unit 합 |
| `sold_closing_revenue` | 마감 unit × 정가 합 |
| `n_items_active` | 그날 판매 품목 수 |

#### 자기상관 (10)

| Feature | 정의 |
|---|---|
| `lag1`, `lag7`, `lag14`, `lag28` | 어제 / 1주 전 / 2주 전 / 4주 전 매출 |
| `rmean7`, `rmean28` | 7일 / 28일 rolling mean (shift(1)) |
| `rstd7`, `rstd28` | rolling std |
| `ewma7`, `ewma28` | EWMA halflife — 브랜드 인기 trend |

#### 캘린더 cyclic (10)

| Feature | 정의 |
|---|---|
| `dow`, `month`, `dom` | 원본 int |
| `dow_sin`, `dow_cos` | 주간 주기성 |
| `month_sin`, `month_cos` | 연간 주기성 |
| `dom_sin`, `dom_cos` | 월간 주기성 |
| `is_weekend` | 토/일 여부 |
| `is_public_holiday` | 대체공휴일 포함 공휴일 |
| `is_before_holiday` | 좁은 정의 — **오늘 영업일 + 내일 휴일** (연휴 직전 마지막 영업일) |

#### 특수일 — ±14일 cap (12)

| Feature | 효과 (광교 ±3 평일) |
|---|---|
| `days_to_xmas` | +8.7% (p=0.045) |
| `days_to_valentine` | +7.0% (p=0.040) |
| `days_to_white_day` | **+11.1%** (p=0.005) |
| `days_to_children_day` | +2.9% (p=0.52) |
| `days_to_chuseok` | **+11.4%** (p=0.029) |
| `days_to_seollal` | **+11.0%** (p=0.005) |
| `is_within7_*` (각 이벤트) | binary 보조 |

빼빼로 제외 (효과 미식별 + 데이터 한계).

#### 날씨 (10)

| Feature | 정의 |
|---|---|
| `avgTa`, `maxTa`, `minTa` | 기온 (직접) |
| `sumRn`, `avgRhm`, `avgTca`, `avgWs` | 강수/습도/운량/풍속 |
| `rain_level` | 강수 binning (0=없음 / 1=약함 / 2=보통 / 3=폭우) |
| `heavy_rain_in_biz_hours` | 영업시간 내 폭우 발생 |
| `apparent_temp` | 체감온도 (avgTa - 0.7 × avgWs) |

#### 경쟁점 (1)

`n_competitors_active` — 광교 1km 내 active bakery/cafe 일자별

---

## §2. Output 데이터

### 2-1. Stage 1 출력 (일 단위)

| 출력 | 의미 | 활용 |
|---|---|---|
| `baseline` | target_dow 최근 3~4주 평균 | 기준값 |
| `residual_q90` | LGBM 잔차 (q=0.90) | safety margin |
| `dow_extra_safety` | dow별 평균 shortfall | dow별 추가 보정 |
| **`production`** | **baseline + residual + dow_safety** | **발주 시스템 입력** |
| `production_revenue` | revenue target 모델 | 매출 추적 |

### 2-2. Stage 2 출력 (품목 × 일)

| 출력 | 의미 |
|---|---|
| `proportion[i]` | 품목 i의 발주 비율 (Σ=1) |
| `qty[i]` | 품목 i의 권장 발주 unit |
| `adj_trend` / `adj_stockout` / `adj_closing` / `adj_new` | 분위수 가중 boost factor |
| `recommendation` | STRONG_UP / UP / HOLD / DOWN / STRONG_DOWN |

### 2-3. Stage 3 출력 (신제품 4주 후)

| 출력 | 의미 |
|---|---|
| `decision` | promote / hold / fade_out |
| `avg_daily_sold` | 4주 평균 일 매출 |
| `closing_rate` | 4주 마감 할인 비율 |

### 2-4. 비지니스 KPI

| KPI | 정의 |
|---|---|
| 일 폐기 추정 | production − sold_total (positive) |
| 일 마감 할인 unit | label=closing receipt 합 |
| 카테고리 합 WAPE | |actual − predicted| / actual |
| 매진율 (production < actual) | 일별 발주 부족 비율 |
| 신제품 의사결정 적중률 | decision vs actual outcome |

---

## §3. 모델링 가정

### 3-1. 도메인 가정

| # | 가정 |
|---|---|
| 1-1-a | 광교 손님 ≥95%는 "빵" 자체 사러 옴 |
| 1-1-b | bread/pastry 수요 경계 흐릿 (자유 선택) |
| 1-1-c | sandwich도 양방향 substitute |
| 1-2-a | 개별 품목 매진 = 매출 손실 0 (다른 빵 대체) |
| 1-2-b | 인기품 잦은 매진 = 장기 만족도 ↓ |

### 3-2. 데이터 처리

| 가정 | 적용 |
|---|---|
| 시즌/프리미엄 16개 제외 | `seasonal.py` filter |
| 광교 95.2% 당일폐기 | 단기 폐기 매장 |
| 카테고리 4개 (한국 표준) | bonavi_loader.py |
| cake Stage 1 제외 | 사전 예약 + 시즌 |
| **마감 unit 60% 실수요** | **α=0.6** |
| 추석/설날 +11% 효과 | days_to_event 추가 |
| 빼빼로 제외 | 효과 미식별 + 데이터 한계 |

### 3-3. 모델 설계

| 가정 | 적용 |
|---|---|
| 카테고리 합이 item-level보다 안정 | 3-stage hierarchy |
| Stage 1 baseline = target_dow 평균 | naive baseline (광교 강한 dow 신호) |
| LGBM은 residual만 학습 | overfitting 회피 + worst case 보장 |
| dow별 extra safety | 학습 시 LGBM 미달분 dow별 보정 |
| Stage 2 분위수 가중 boost | 인기/과잉 강도 비례 |
| 신제품 = 진입 < 90일, 4주 후 의사결정 | Stage 3 룰 |

---

## §4. 가정 뒷받침 데이터 기반 근거

### 4-1. 가정 1-1-b — 강한 지지

| 측정 | 결과 |
|---|---|
| Nested Logit λ | 0.99~0.995 (IIA 거의 안 어김) |
| KS bread vs pastry 시간 분포 | statistic 0.031 (효과 미미) |
| pastry 시간대별 CV | 0.036 (시간 무관 일정 66%) |
| 평균 구매 시각 차이 | 0.25시 |

### 4-2. 가정 1-2-a — 강한 지지 ★★★

매진일 매장 매출 +5~14% **더 높음** (p<0.001):

| 인기품 | 매진일 unit/hr | 비매진일 | diff |
|---|---|---|---|
| C통팥빵 | 21.7 | 19.6 | **+11.0%** |
| 바게트 치즈식빵 | 21.3 | 18.9 | **+13.0%** |
| 치즈롤 | 21.5 | 18.9 | **+14.0%** |
| C바닐라빈 카스타드빵 | 21.0 | 19.6 | +7.4% |
| C크럼블소보로빵 | 20.7 | 19.9 | +4.4% |

→ 매진은 결과지 원인 X. 강한 substitution 직접 증거.

### 4-3. α=0.6 — 시간당 unit + 휴리스틱

| 측정 | 결과 |
|---|---|
| 마감 시간(20-21시) bread unit/hr | 7.6 |
| 정상 시간(11-15시) bread unit/hr | 11.0 |
| 비율 | 69% (마감/정상) |
| 정상 hour profile상 저녁 자연 비율 | 30-50% |
| 추정 induced 비율 | 30-50% → **실수요 50-70%** |
| 마감 시간대 매출 중 마감 할인 비중 | **92.4%** |

→ α=0.6 = 데이터 근거 중간값.

### 4-4. 특수일 효과 검증 (광교 5년, ±3일 평일)

| 이벤트 | unit 효과 | p-value | 결정 |
|---|---|---|---|
| **white_day** | **+11.1%** | **0.005** | 포함 |
| **xmas** | **+8.7%** | **0.045** | 포함 |
| **valentine** | **+7.0%** | **0.040** | 포함 |
| **chuseok (추석)** | **+11.4%** | **0.029** | 포함 |
| **seollal (설날)** | **+11.0%** | **0.005** | 포함 |
| children_day | +2.9% | 0.52 | 보조 |
| pepero | -3.2% | 0.50 | 제외 |

### 4-5. 시즌 16개 제외 근거

매출 비중 0.8%:
- 밤 시즌 5개 (가을~겨울, 마감 0건)
- 프리미엄 3개 (AOP / 밀레앙)
- 크리스마스 시즌 3개 (위싱 리스, 파네토네)
- 단종 N 머핀 4개
- 한정판 1개

### 4-6. 가정 1-2-b — 데이터 검증 불가

매장 매출 trend vs 매진 빈도 trend Pearson r=+0.58 (endogeneity, 매장 인기 변동에 잡힘). 가설로 박고 진행, proxy로만 추적.

---

## §5. 모델링 알고리즘

### 5-1. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│ Stage 1 — Ensemble (운영 시 D=목 → D+4~D+10 예측)            │
│  Baseline   : target_dow 최근 3~4주 평균                    │
│  LGBM       : residual quantile (q=0.90)                   │
│  dow safety : dow별 평균 shortfall                          │
│  Production : baseline + residual + dow_safety              │
└──────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 2 — 분위수 가중 분배                                   │
│  proportion[i] = base × adj_trend × adj_stockout × ...      │
└──────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 3 — 신제품 4주 진단                                    │
│  promote / hold / fade_out                                  │
└──────────────────────────────────────────────────────────┘
```

### 5-2. Stage 1 — Ensemble (상세)

#### Target

```python
sold_normal     = sold_units - closing_qty
adjusted_demand = sold_normal + sold_closing × 0.6
```

#### Baseline 계산

```python
def compute_baseline(df, h, target_col):
    """target_date - 7k 의 매출 (같은 dow, k주 전) 평균."""
    shifts = [7*k - h for k in [1,2,3,4] if 7*k - h > 0]
    lag_cols = pd.concat([df[target_col].shift(s) for s in shifts], axis=1)
    return lag_cols.mean(axis=1)
```

#### LGBM Residual + dow Safety

```python
# 학습 (D 이전 데이터)
residual = future_target - baseline
expected = LGBMRegressor(objective="regression_l1", n_estimators=200, max_depth=4)
quantile = LGBMRegressor(objective="quantile", alpha=0.90, ...)
expected.fit(features, residual)
quantile.fit(features, residual)

# dow별 extra safety
prod_pred = baseline + quantile.predict(features)
shortfall = max(0, future_target - prod_pred)
dow_extra_safety = shortfall.groupby(target_dow).mean()
# 광교 예: 월 +0.8, 화 +0.7, 수 +2.0, 목 +0.9, 금 +0.6, 토 +1.3, 일 +0.1

# 운영 예측
production = baseline_at_D + quantile.predict(D_features) + dow_extra_safety[target_dow]
```

### 5-3. Stage 2 — 분위수 가중 분배

```python
# per-category percentile (광교 baseline 부재 매장)
stockout_rank_pct = avg_stockout_h의 percentile (작을수록 인기)
closing_rank_pct  = closing_rate의 percentile (클수록 과잉)

# 연속 boost (강도에 비례)
adj_trend    = 1 + 0.20 × |trend_strength| × sign(trend_pct)
adj_stockout = 1 + 0.20 × (1 - stockout_rank_pct)
adj_closing  = 1 - 0.20 × closing_rank_pct
adj_new      = 1.20 if days_since_first < 90 else 1.00

raw_weight   = base_sold × adj_trend × adj_stockout × adj_closing × adj_new
proportion   = raw_weight / Σ raw_weight
final_qty[i] = Stage_1_production × proportion[i]
```

### 5-4. Stage 3 — 신제품 진단

```python
new_products = [item if (today - item.first_sold) < 90]

metrics = {
    "avg_daily_sold":  최근 28일 평균 일 sold,
    "closing_rate":    28일 closing_qty / 28일 sold,
}
cat_median = 그 카테고리 평균 sold 중앙값

if avg_daily_sold >= cat_median × 0.5 AND closing_rate < 0.20:
    decision = "promote"
elif avg_daily_sold < cat_median × 0.2 AND closing_rate >= 0.40:
    decision = "fade_out"
else:
    decision = "hold"
```

---

## §6. 성능 결과

### 6-1. Stage 1 운영 호환 backtest

| 지표 | 값 |
|---|---|
| **WAPE** | **29.37%** |
| **매진율** (production < actual) | **13.6%** |
| 평균 production | 304 unit/일 |
| 평균 실제 sold | 242 unit/일 |
| 발주 over | +62 unit/일 |

### 6-2. Horizon별

| horizon | dow | n | WAPE | 매진율 |
|---|---|---|---|---|
| D+4 | 월 | 7 | 29.89% | 14.3% |
| D+5 | 화 | 8 | 42.14% | 0.0% |
| D+6 | 수 | 4 | 40.73% | 0.0% |
| D+7 | 목 | 6 | 62.47% | 0.0% |
| D+8 | 금 | 6 | 36.30% | 0.0% |
| D+9 | 토 | 10 | 10.30% | **50.0%** |
| D+10 | 일 | 3 | 0.77% | 33.3% |

→ 토요일 매진율 50% 잔여 — 일부 토요일 특별 이벤트 매출 급증.

### 6-3. Feature importance (Stage 1)

| 그룹 | 비중 |
|---|---|
| 자기상관 (lag/rolling/ewma) | 48% |
| 날씨 | 30% |
| 캘린더 (cyclic) | 18% |
| 경쟁점 | 2% |
| 휴일 / 특수일 | 각 1% |

### 6-4. Stage 2 분포 (마지막 fold)

| 카테고리 | STRONG_UP | UP | HOLD | DOWN | STRONG_DOWN |
|---|---|---|---|---|---|
| bread | 0 | 1 | 5 | 2 | 2 |
| pastry | 1 | 2 | 7 | 6 | 2 |
| sandwich | 0 | 0 | 0 | 1 | 0 |
| **합** | **1** | **3** | **12** | **9** | **4** |

- STRONG_UP: 스트로베리 밀크 크림 쇼콜라 (combined 1.37 = 인기품 + 신제품 boost)
- STRONG_DOWN: 식빵의정성, 뺑오쇼콜라(Grand), 아티제 잡곡 식빵 (combined 0.68~0.79)

### 6-5. Stage 3 — 신제품 81개 후행 검증

| 의사결정 \ 실제 | faded | marginal | survived | 합 |
|---|---|---|---|---|
| promote | 7 | 8 | **45** | 60 |
| hold | 3 | 4 | 9 | 16 |

**적중률 76.3%** (58/76), promote 성공률 71.4%.

---

## §7. 비지니스 임팩트

### 7-1. 정량 측정

| 임팩트 | 값 |
|---|---|
| Stage 1 운영 WAPE | **29.37%** |
| 운영 매진율 | **13.6%** (60일에 ~8일) |
| 신제품 의사결정 적중률 | **76.3%** |
| 마감 할인 손실 ceiling (광교 5년) | 7,846만원 (연 1,569만원) |
| 연 폐기 손실 절감 추정 (보수) | ~2.3M원 |

### 7-2. 자동 식별 운영 정보

| 식별 | 액션 |
|---|---|
| STRONG_UP 품목 | 발주 강화 (인기 강도 비례, max +20%) |
| STRONG_DOWN 품목 | 발주 감소 (과잉 강도 비례, max -20%) |
| 신제품 (< 90일) | 1.2× boost + 4주 후 promote/hold/fade 자동 |
| 토요일/일요일 | dow extra safety 추가 |

### 7-3. ERP / 발주 시스템 연동 JSON

```json
{
    "date": "2026-05-24",
    "store_id": "store_gw01",
    "stage1": {
        "production_unit":     304,
        "baseline":            246,
        "lgbm_residual":       57,
        "dow_safety":          1.3,
        "production_revenue":  1199406,
        "expected_stockout_pct": 13.6
    },
    "stage2": [
        {"item_id":"151100000241", "name":"올리브치아바타", "qty":12, "recommendation":"STRONG_UP"},
        {"item_id":"151300002107", "name":"클래식 버터롤(10개입)", "qty":4, "recommendation":"STRONG_DOWN"}
    ],
    "stage3": [
        {"item_id":"151100003338", "name":"스트로베리 밀크 크림 쇼콜라", "decision":"promote"}
    ]
}
```

---

## §8. 위험 및 한계

### 8-1. 데이터

| 한계 | 영향 |
|---|---|
| 광교 단일 매장 | 외부 features 효과 분리 어려움 |
| PoC 데이터 셀렉션 의심 | 잼 / 굿즈 / 예약 0개 |
| 폐기 실측 부재 | ROI = 마감 할인 절감 lower bound |
| 사전 예약 데이터 | cake 분리 불가 |
| 빼빼로 굿즈 데이터 부재 | 빼빼로 효과 검증 불가 |

### 8-2. 모델

| 한계 | 영향 |
|---|---|
| 토요일 D+9 매진율 50% | 특별 이벤트 매출 급증 캡처 불가 |
| LGBM 정확도 부가가치 작음 | 광교 baseline 강한 매장 |
| 운영 WAPE ~30% | 광교 5년 ceiling |
| 운영 시 일기예보 사용 | 실측 vs 예보 차이 (~+1pp) |

### 8-3. 가정

| 한계 | 영향 |
|---|---|
| 1-2-b 장기 만족도 데이터 검증 X | proxy 추적만 |
| α=0.6 휴리스틱 | 폐기 실측 도착 시 재캘리브레이션 |
| fade_out 룰 광교 미발현 | 다매장 도착 시 완화 |

---

## §9. 향후 진행

### 즉시 가능

| 작업 | 기대 효과 |
|---|---|
| Stage 2 boost 강도 sensitivity 튜닝 | 광교 적합 boost 정밀화 |
| cake 시즌 분리 + 홀케이크 별도 모델 | cake 시즌 패턴 |

### 데이터 도착 후

| 데이터 | 작업 | 기대 효과 |
|---|---|---|
| 폐기 실측 | α 캘리브레이션 + 진짜 ROI | 보수 추정 → 실측 |
| 다매장 | 외부 features 본격 + fade_out 룰 | framework 일반화 |
| 사전 예약 | cake 사전 예약 분리 | cake 카테고리 합산 가능 |
| 영업시간 / 휴무일 | open/close_hour dynamic | potential_demand 정확도 |
| 채널 (배달 vs 매장) | 채널별 hour profile | 채널별 demand 분리 |
| 무게 / 사이즈 | 진짜 정규화 unit (kcal) | unit 비교 정확도 |
| 광교 이벤트 / promo | 토요일 outlier 캡처 | 운영 매진율 개선 |

---

## §10. 모듈 및 산출물

### 10-1. 코드

| 모듈 | 역할 |
|---|---|
| `src/bakery/analysis/discount.py` | 30개 할인코드 분류 + 마감 할인 식별 |
| `src/bakery/analysis/popularity.py` | 인기도 신호 + 권장 |
| `src/bakery/analysis/waste.py` | 폐기 추정 + WasteEstimator Protocol |
| `src/bakery/analysis/seasonal.py` | 시즌 16개 제외 |
| `src/bakery/features/category_aggregate.py` | 카테고리 합 daily + 57 features |
| `src/bakery/models/category_total.py` | Stage 1 LGBM (training reference) |
| `src/bakery/models/item_proportion.py` | Stage 2 분위수 가중 분배 |
| `src/bakery/models/new_product_tracker.py` | Stage 3 4주 진단 |
| `scripts/operational_5model_comparison.py` | 운영 호환 backtest |

### 10-2. 산출 (`reports/`)

| 파일 | 내용 |
|---|---|
| `discount_codes_gwangyo.csv` | 30개 할인코드 분류 |
| `closing_discount_by_category.csv` | 마감 할인 손실 |
| `product_demand_signals.csv` | 130개 품목 인기도 |
| `seasonal_excluded_items.csv` | 시즌 16개 명세 |
| **`v4_method_c_revised.csv`** | **Stage 1 최종 backtest** |
| `v4_feature_importance_*.csv` | feature importance |
| `v4_phase5_stage2_proportions_quantile.csv` | Stage 2 분배 |
| `v4_phase5_recommendations_quantile.csv` | 5단계 권장 |
| `v4_phase5_new_product_validation.csv` | Stage 3 81개 검증 |

### 10-3. 테스트

- **119/119 통과**

### 10-4. 문서

- `docs/modeling_v4.md` — framework 상세
- `docs/discount_business_impact.md` — 할인 분석 비지니스 임팩트
- `docs/poc_framework_pivot.md` — "카테고리=한묶음" 가설
- 본 리포트: `docs/v4_consolidated_report.md` — 종합

---

## §11. 결론

광교 매장 PoC를 통해 v4 framework:

1. **운영 시 카테고리 합 WAPE 29.37%** (실제 비지니스 발주 정확도)
2. **운영 매진율 13.6%** (가정 1-2-b 회피 목표 거의 충족)
3. **신제품 의사결정 자동화 76.3% 적중률**
4. **명시적 비지니스 분배 logic** — Strong Up/Down/New 자동 식별 (분위수 가중)
5. **연 폐기 손실 절감 추정 ~2.3M원** (보수, 폐기 실측 도착 시 재추정)

운영 도입 시:
- 매주 목요일에 다음주 월~일 발주 예측
- D+9 토요일 매진율 50% 잔여 → 광교 이벤트 / promo 데이터 도착 시 개선 가능
- 일기예보 사용 시 정확도 ~+1pp 영향 (weather 효과 작음)

데이터 도착 시 진짜 개선:
- 폐기 실측 → α 캘리브레이션 + 진짜 ROI
- 다매장 → framework 일반화 + LGBM 진짜 가치 검증

---

## §13. WAPE 분해 + Permutation Importance — 정직 평가

### 13-1. WAPE 30%의 진짜 분해

광교 매장 매출의 본질적 변동성 분석:

| dow | 평균 매출 | std | **CV (변동성)** |
|---|---|---|---|
| 월 | 244 | 41 | **16.8%** |
| 화 | 236 | 38 | **16.1%** |
| 수 | 224 | 31 | **13.7%** |
| 목 | 228 | 36 | **15.8%** |
| 금 | 242 | 33 | **13.8%** |
| 토 | 320 | 48 | **15.0%** |
| 일 | 308 | 46 | **14.8%** |

**같은 dow 4주 평균 baseline 잔차**: MAE/mean = **10.1%**

### WAPE 분해 결론

```
WAPE 18% (Naive baseline)   = 광교 매장의 본질적 변동성 ceiling
                              (같은 dow끼리도 매출 ±10-15% 자연 변동)
WAPE 18.54% (LGBM expected) = 우리 features 추가해도 거의 같음
                              (외부 이벤트 정보 없이는 features 추가 신호 X)
WAPE 29.37% (Ensemble q90)  = baseline + 의도적 over-prediction (매진 safety)
```

→ **WAPE 30% 운영 = 매진 safety buffer 비용**. **우리 features의 정확도 ceiling = 18-19%** (Naive baseline 수준).

→ 그 아래로 가려면 **외부 이벤트 정보 (마케팅 / 신상 / promo) 필수**.

### 13-2. Permutation Importance — split count의 환상 확인

`feature_importances_` (split count)는 변수가 split에 자주 사용되면 ↑. 실제 예측력과 다를 수 있음. **Permutation importance**로 진짜 측정:

#### 진짜 TOP 5 (Permutation)

| 순위 | feature | perm importance | split rank |
|---|---|---|---|
| 1 | **dow** | **+4.968 ★** | 15위 |
| 2 | **lag14** | **+4.768 ★** | 3위 |
| 3 | **lag7** | **+4.418 ★** | 1위 |
| 4 | lag1 | +0.759 | 2위 |
| 5 | lag28 | +0.478 | 4위 |

#### Split count TOP에 있지만 실제로는 noise

| feature | split rank | **perm rank** | **perm 영향** |
|---|---|---|---|
| **avgRhm (습도)** | **5위** | **41위** | **-0.063 (해로움!)** |
| avgWs (풍속) | 9위 | 43위 | **-0.103 (해로움)** |
| rstd28 | 8위 | 42위 | -0.095 |
| ewma28 | 5위 | 40위 | -0.049 |

→ **avgRhm은 진짜 noise**. split count 5위는 환상이었음.

### 13-3. 그룹별 진짜 importance

| 그룹 | split count 합 | **Permutation 합** | 진짜 가치 |
|---|---|---|---|
| **자기상관 (lag/rolling/ewma)** | 923 | **+10.75** | **★ 진짜 핵심** |
| **캘린더 (cyclic + dow)** | 295 | **+5.17** | **★ 진짜 효과** |
| 휴일 | 7 | +0.02 | 작음 (rare event) |
| 특수일 | 34 | +0.02 | 작음 (rare event) |
| **날씨** | 398 | **+0.005** | **사실상 0** |
| **경쟁점** | 41 | **-0.016** | **오히려 해로움** |

### 13-4. 날씨 영향력 0인 이유

**Importance 30%였던 날씨 features가 진짜 가치 ≈ 0인 이유** (3가지 조합):

#### A. 다른 features에 이미 흡수 ★ 가장 큰 원인

- **자기상관 lag**에 날씨 효과 이미 포함 (lag7 = 지난주 비/맑음 상태)
- **month/cyclic 계절성**에 weather 평균 효과 포함
- → 날씨 features 추가해도 새 신호 X (중복)

검증:
- month vs avgTa 강한 상관
- month vs avgRhm: r=+0.154
- month vs avgTca, sumRn: 강한 상관

#### B. 베이커리 본질적 특성

- 빵 = 식사/간식 → 음료/아이스크림 대비 날씨 둔감
- 단골 고정 비율 큼
- 극단 날씨 (폭우/폭염)만 영향 → rare event라 학습 sample 부족

#### C. 광교 매장 = 실내 입주 (아브뉴프랑)

- 옥외 매장 대비 비/눈 영향 작음

### 13-5. 모델 단순화 효과 검증

3 feature set 비교 (운영 호환 backtest, 16 Thursdays × 7 horizons):

| Feature set | n features | WAPE | 매진율 | 발주 over |
|---|---|---|---|---|
| **full** | 45 | 29.44% | **11.4%** | +63.3 |
| slim (날씨/경쟁점 제거) | 34 | 31.03% | 13.6% | +66.7 |
| **minimal (lag + dow + 특수일)** | **22** | **30.31%** | 13.6% | +65.4 |

→ **단순화해도 거의 같은 성능**. permutation 결과 검증됨.

운영 결정:
- **full (45) 유지**: 약간 더 정확 + safety 약간 ↑ (LGBM 보수성)
- **minimal (22) 가능**: 운영 부담 줄이고 싶을 때

### 13-6. 핵심 메시지 정정

| 이전 메시지 | 정정 |
|---|---|
| "weather importance 30%라 큰 가치" | **거짓. 진짜 가치 ≈ 0** |
| "avgRhm 결측 큰 영향" | **거짓. avgRhm 진짜 noise** |
| "WAPE 30%는 모델이 부족" | **부분 사실. 18%는 광교 본질적 noise. 30%는 매진 safety 비용** |
| "운영 시 일기예보 부정확 영향 +1~3pp" | **영향 거의 0 (날씨 진짜 가치 없음)** |

### 13-7. 진짜 의미

**광교 단독 5년 데이터의 진짜 ceiling = WAPE 18%** (naive baseline 수준). 다음을 의미:
- 우리 features (날씨/특수일/경쟁점) 추가 정확도 거의 X
- 진짜 강한 신호: **dow + lag (7/14)**
- 더 정확하려면 **외부 이벤트 정보 (마케팅, 신상, promo) 필수**
- ML 본질적 한계 — features에 없는 신호는 학습 불가

### 13-8. 의사결정 권장

| 결정 | 근거 |
|---|---|
| **모델: full 또는 minimal 둘 다 운영 가능** | 차이 미미 |
| **중기예보 weather 결측 걱정 X** | 진짜 가치 ≈ 0 |
| **다음 우선순위 = 외부 이벤트 데이터** | 본질적 ceiling 돌파 유일 길 |
| **다매장 도착 시 features 진짜 가치 재평가** | outdoor 매장은 날씨 영향 더 클 가능성 |

---

## §14. 최종 정직 결론 (PM 미팅용)

### 광교 PoC 정직 평가

1. **WAPE 30%는 광교 5년 데이터의 ceiling** — 알고리즘 튜닝으로 안 됨
2. **단순 dow 평균 vs 우리 모델 차이는 매진 safety + 자동화** (정확도는 비슷)
3. **날씨 / 경쟁점 features 진짜 가치 ≈ 0** (lag + cyclic에 흡수됨)
4. **본질적 limit 돌파는 외부 이벤트 정보 보충 필수**

### 진짜 PoC 가치

- 매진 회피 자동화 (45% → 14%)
- 품목별 발주 자동 분배 (Stage 2)
- 신제품 4주 자동 진단 76.3% 적중률 (Stage 3)
- 데이터 추가 인프라 (마케팅/다매장/폐기 plug-in 가능)

### 운영 production-ready 가는 길

- 알고리즘 ✗ (이미 ceiling)
- **데이터 ✓** (마케팅 캘린더 → 다매장 → 폐기 실측 → promo 일지)
- 합치면 WAPE 12-18%, 매진 1-3% 달성 가능

---

## §15. Target-date features 후속 (운영 dow 고정 문제 해결)

### 15-1. 문제

운영 시 D=목요일 고정 → cutoff 시점 features (dow, dom_sin, month_sin, is_weekend ...) 모두 **목요일 값**으로 잠김. LGBM이 target_date의 진짜 dow를 모름. baseline + dow_safety가 dow 효과를 흡수하지만, residual 모델이 dow를 활용 불가.

### 15-2. 변경 (1+2+3)

| 변경 | 내용 |
|---|---|
| 1. target_date 캘린더 (cyclic) | `tgt_dow_sin/cos`, `tgt_month_sin/cos`, `tgt_dom_sin/cos`, `tgt_is_weekend` |
| 2. target_date 휴일 + 특수일 | `tgt_is_holiday`, `tgt_is_before_holiday`, `tgt_days_to_{xmas,valentine,white_day,children_day,chuseok,seollal}`, `tgt_is_within7_*` |
| 3. 단기 features 제거 | `rmean7`, `rstd7`, `ewma7` (cutoff dow 의존 — 운영 시 D=목 한정 noise) |

총 features: 45 → **62** (target_date 17개 추가, 단기 3개 제거).

### 15-3. 결과 (운영 호환 backtest, 16 Thursdays × 7 horizons)

| Model | n_feat | WAPE | 매진율 | 발주 over |
|---|---|---|---|---|
| 이전 full (D dow 기반) | 45 | 29.44% | 11.4% | +63.3 |
| minimal (lag+dow+특수일) | 22 | 30.31% | 13.6% | +65.4 |
| **target_date + 단기 제거** | **62** | **29.28%** | **11.4%** | **+60.0** |

→ WAPE -0.16pp 미세 개선, 발주 over -3.3 (매진율 동일).

### 15-4. dow별 매진율 — 토/일 개선 확인

| target_dow | 이전 (full 45) | **target_date (62)** |
|---|---|---|
| 토 (D+9) | 50.0% | **40.0%** (-10pp) |
| 일 (D+10) | 33.3% | **0.0%** (-33pp) |

→ target_date features가 **주말 매진율** 개선 효과 ★. 평일 sample 적어 dow별 점수 변화는 noise 수준.

### 15-5. 결론

target_date features는 운영 ergonomics 면에서 옳은 방향 + 주말 매진율 개선. 다만 **전체 WAPE는 여전히 ~29%** — §13에서 밝힌 광교 단독 ceiling (18% 본질적 + 11% safety 비용) 그대로.

---

## §16. D+7 (목요일) WAPE 60% 진단 — 외부 사건 캡처 실패

### 16-1. 사실 확인

D+7 = 다음주 목요일 발주. target_date features 적용 후에도 **WAPE 60.8%** (6 sample). 다른 horizon (D+4~D+10) 대비 압도적.

### 16-2. D+7 sample raw

| D (목요일) | production | actual | over |
|---|---|---|---|
| 2025-04-10 | 281 | 202 | +79 |
| 2025-05-15 | 283 | 198 | +85 |
| 2025-06-19 | 320 | 204 | +116 |
| 2025-10-09 | 288 | 236 | +52 |
| **2025-11-27** | **372** | **178** | **+194** ★ |
| **2025-12-04** | **376** | **176** | **+200** ★ |

→ **마지막 2개 sample이 WAPE 60% 견인**. 11-12월 광교 매출 절벽.

### 16-3. 원인 — 광교 매장 2025-11-10 ~ 12-15 평일 매출 drop

| 구간 | 평균 매출 | 평소 대비 |
|---|---|---|
| 2025-01 ~ 2025-10 평일 평균 | ~248 | baseline |
| **2025-11-10 ~ 12-15 평일** | **~222** | **-10.4%** |

dow별 변화:
- 목 -20.9%, 수 -17.4%, 월 -14.5%, 금 -12% (평일 다 -10% 이상)
- 토 -2.3%, 일 -3.8% (주말은 거의 영향 없음)

12월 25일 (크리스마스) 매출 315로 회복 — drop은 **5주 일시적 외부 사건**.

### 16-4. 추정 원인 (features 부재)

- 광교 매장 운영 변화 (시간/품목 조정)
- 근처 경쟁점 진입 또는 마케팅
- 본사 캠페인 일시 종료
- 행정/외부 요인 (아브뉴프랑 행사 종료 등)

→ **features에 없으면 ML은 학습 불가**. 평소 dow 평균 + lag = 372로 평소 수준 예측 vs 실제 176. 알고리즘 한계가 아니라 **데이터 한계**.

### 16-5. 검증 — §13 "본질적 ceiling 18%" 가설 직접 지지

광교 5년 ceiling 주장의 대표 사례. **외부 이벤트 정보 없이는 dow 평균이 답**. 운영 시 매니저가 매장 변화 즉시 입력하는 채널 필요.

### 16-6. 의사결정 권장

| 액션 | 비고 |
|---|---|
| **광교 promo / 행사 일지 (1순위 데이터)** | 평일 매출 drop 원인 캡처 |
| 운영 시 매니저 입력 채널 | 매장 변화 즉시 반영 |
| baseline window 단축 (4주 → 2주) 시도 | drop 발생 후 빠른 적응 |
| 모델이 D+7 예측 시 confidence band 노출 | 매니저 판단 보조

---

## §17. PM 미팅용 압축 결론

### 17-1. 핵심 메시지

1. **광교 단독 5년 데이터 운영 ceiling = WAPE 29%, 매진 11%** (target_date features 적용).
2. WAPE 18%가 광교 본질적 변동성 ceiling. 30%는 매진 safety buffer 비용.
3. **알고리즘 튜닝으로 더 못 내려감** (full / minimal / target_date 거의 같음).
4. **다음 ceiling 돌파 = 외부 데이터** (마케팅, 다매장, promo 일지).
5. D+7 WAPE 60%는 11-12월 매출 drop outlier — 모델 한계 아니라 광교 외부 사건 정보 부재.

### 17-2. v4 PoC 진짜 가치 (Naive 대비)

- **매진율 45% → 11%** (safety 4배 ↑)
- **품목별 자동 분배** (Stage 2)
- **신제품 4주 자동 진단 76.3% 적중률** (Stage 3)
- **데이터 plug-in 인프라** (마케팅/다매장/폐기)

### 17-3. PM 결정 요청

1. 추가 데이터 우선순위 합의 (1순위 마케팅 캘린더 / 2순위 다매장 / 3순위 폐기 실측 / 4순위 광교 promo)
2. PoC → production 시점 결정 (광교 단독 vs 다매장 도착 대기)
3. 사전 예약 / 채널 분리 정보 확보 가능성

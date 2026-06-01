# v4 Modeling Framework — Category-Total + Item-Proportion + New Product Tracker

베이커리 도메인 가정 기반 3-stage hierarchical 발주 예측 모델. 광교 매장 PoC (5년 데이터).

**최종 설정**:
- α=0.6 (마감 unit의 60% 실수요)
- quantile=0.90 (production safety)
- Stage 1 모델: **Ensemble (baseline + LGBM quantile residual + dow safety)**
- 시즌 16개 제외, 카테고리 합산 (bread + pastry + sandwich)
- 운영 시나리오: D=목요일 → D+4~D+10 (다음주 월~일) 발주 예측

---

## §1. 도메인 가정

### 1-1. 고객 패턴

- **(a)** 광교 매장 고객 **≥95%** 는 특정 품목 타겟 아닌 **"빵"** 자체 사러 옴
- **(b)** `bread` (식사 대용) 과 `pastry` (단과자+페이스트리) **수요 경계 흐릿** — 손님 자유 선택
- **(c)** `sandwich` (광교 = 크로크무슈) 도 약간 경계 있으나 다른 카테고리와 양방향 substitute

### 1-2. 가정의 직접 함의

- **(a)** 빵 + sandwich 내부 **개별 품목 매진 = 매출 손실 아님**
- **(b)** 단 인기 품목 잦은 조기 매진 = 고객 만족도 ↓ → **장기 매출 위험**

### 1-3. 데이터 기반 검증

| 가정 | 검증 결과 | 결론 |
|---|---|---|
| 1-1-a 95% "빵" 자체 | 영수증 평균 1.98 품목, bread+pastry 혼합 영수증 45% | 유지 |
| 1-1-b bread/pastry 경계 흐릿 | KS statistic 0.031, pastry 시간대 CV 0.036, Nested λ≈0.99 | **강한 지지** |
| 1-1-c sandwich 경계 약함 | 광교 sandwich=1 품목 | 보류 |
| 1-2-a 개별 stockout = 매출 손실 0 | 매진일 매장 매출 +5~14% 더 높음 (p<0.001) | **강한 지지 ★** |
| 1-2-b 장기 만족도 → 매출 | endogeneity 통제 불가 | 가설 박고 진행 |

### 1-4. 데이터 처리 가정

| 가정 | 적용 |
|---|---|
| 시즌/프리미엄 16개 분석 제외 | `seasonal.py` filter |
| 광교 95.2% 당일폐기 → 단기 폐기 매장 | 회계식 폐기 추정 가능 |
| 카테고리 4개 — 한국 제과제빵 표준 | bonavi_loader.py |
| cake 카테고리 Stage 1 제외 (사전 예약 + 시즌) | TARGET_CATEGORIES |
| **마감 할인 unit의 60% 실수요** | **α=0.6** |
| 추석/설날 +11% 효과 (p<0.05) | days_to_chuseok / days_to_seollal 추가 |

---

## §2. 모델 아키텍처

### 2-1. 전체 흐름

```
┌──────────────────────────────────────────────────────────┐
│ Stage 1 — 카테고리 합 일 총수요 (Ensemble)                   │
│  Baseline   : target_dow 최근 3~4주 평균                    │
│  LGBM       : residual quantile (q=0.90) 학습              │
│  dow safety : dow별 평균 shortfall 추가                     │
│  Production : baseline + LGBM residual + dow safety        │
└──────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 2 — 품목 비율 분배 (분위수 가중)                       │
│  Input  : history + Stage 1 production                     │
│  Output : per-item qty[i] (Σ = Stage 1 total)              │
└──────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 3 — 신제품 4주 진단                                    │
│  Input  : 진입 < 90일 품목의 4주 누적 메트릭                   │
│  Output : promote / hold / fade_out                         │
└──────────────────────────────────────────────────────────┘
```

### 2-2. Stage 1 — Ensemble

#### Target 정의 (학습용)

```python
sold_normal     = sold_units − closing_qty
adjusted_demand = sold_normal + sold_closing × 0.6    # α=0.6
```

#### Baseline (target_dow 최근 평균)

```python
# D=cutoff, h=horizon (D+h가 예측 target)
# target_date - 7k = row d + h - 7k = row d 기준 shift(7k - h)
def compute_baseline(df, h, target_col):
    shifts = [7*k - h for k in [1,2,3,4] if 7*k - h > 0]
    lag_cols = pd.concat([df[target_col].shift(s) for s in shifts], axis=1)
    return lag_cols.mean(axis=1)
```

- h=4 (월): shifts = [3, 10, 17, 24]
- h=7 (목): shifts = [7, 14, 21] (3주 평균)
- h=9 (토): shifts = [5, 12, 19] (3주 평균)
- h=10 (일): shifts = [4, 11, 18]

→ target date의 같은 dow 매출 평균.

#### LGBM Residual + Quantile

```python
# residual = future_target − baseline
residual = sold(D+h) − baseline

# 학습 (D 이전 모든 데이터)
expected_model = LGBMRegressor(objective="regression_l1",  n_estimators=200, max_depth=4, num_leaves=15)
quantile_model = LGBMRegressor(objective="quantile", alpha=0.90, ...)
expected_model.fit(features, residual)
quantile_model.fit(features, residual)
```

#### dow별 Extra Safety

```python
# 학습 데이터에서 LGBM q90 production의 dow별 shortfall 평균
prod_pred = baseline + quantile_model.predict(features)
shortfall = max(0, future_target − prod_pred)
dow_extra_safety = shortfall.groupby(target_dow).mean()
```

#### 최종 Production

```python
# 운영 시 D 시점
baseline_D = compute_baseline at D
residual_pred = quantile_model.predict(D_features)
production = baseline_D + residual_pred + dow_extra_safety[target_dow]
```

### 2-3. Stage 1 Features (57개)

#### 카테고리 합 raw (5)
`sold_total_unit`, `sold_total_revenue`, `sold_closing`, `sold_closing_revenue`, `n_items_active`

#### 자기상관 (8)
`lag1`, `lag7`, `lag14`, `lag28`, `rmean7/28`, `rstd7/28`, `ewma7/28`

#### 캘린더 cyclic (10)
`dow`, `month`, `dom`, `dow_sin/cos`, `month_sin/cos`, `dom_sin/cos`, `is_weekend`, `is_public_holiday`, `is_before_holiday` (좁은 정의: 오늘 영업일 + 내일 휴일)

#### 특수일 (12)
양력 고정 4개: `days_to_xmas/valentine/white_day/children_day` (±14 cap, white_day +11.1% p=0.005, xmas +8.7% p=0.045, valentine +7.0% p=0.040)
음력 변동 2개: `days_to_chuseok` (+11.4% p=0.029), `days_to_seollal` (+11.0% p=0.005)
각각 binary indicator `is_within7_*`

#### 날씨 (10)
`avgTa`, `maxTa`, `minTa`, `sumRn`, `avgRhm`, `avgTca`, `avgWs`, `rain_level` (0~3), `heavy_rain_in_biz_hours`, `apparent_temp`

#### 경쟁점 (1)
`n_competitors_active` (광교 1km Haversine 내 active bakery/cafe, 일자별)

### 2-4. Stage 2 — 분위수 가중 분배

```python
# per-category 분위수 (광교 baseline 부재 매장 특성)
stockout_rank_pct = avg_stockout_h의 percentile (ascending, 작을수록 인기)
closing_rank_pct  = closing_rate의 percentile (ascending, 클수록 과잉)

# 연속 boost (강도에 비례)
adj_trend    = 1 + 0.20 × |trend_strength| × sign(trend_pct)   # ±15% threshold
adj_stockout = 1 + 0.20 × (1 - stockout_rank_pct)              # 인기 강도 비례
adj_closing  = 1 - 0.20 × closing_rank_pct                      # 과잉 강도 비례
adj_new      = 1.20 if days_since_first < 90 else 1.00         # binary

raw_weight   = base_sold × adj_trend × adj_stockout × adj_closing × adj_new
proportion   = raw_weight / Σ raw_weight
final_qty[i] = Stage_1_production × proportion[i]
```

### 2-5. Stage 3 — 신제품 4주 진단

```python
# 진입 < 90일 신제품 식별
new_products = [item if (today - item.first_sold) < 90]

# 4주 누적 메트릭
metrics = {
    "avg_daily_sold":  28일 평균 일 sold,
    "avg_stockout_h":  28일 매진 시각 평균,
    "closing_rate":    28일 closing_qty / 28일 sold,
}

# 의사결정
cat_median = 그 카테고리 모든 품목 평균 sold의 중앙값
if avg_daily_sold >= cat_median × 0.5 AND closing_rate < 0.20:
    decision = "promote"      # 정규 라인 편입
elif avg_daily_sold < cat_median × 0.2 AND closing_rate >= 0.40:
    decision = "fade_out"     # 단종
else:
    decision = "hold"         # 추가 관찰
```

### 2-6. 카테고리 매핑 (한국 제과제빵 표준)

```
"식빵" → bread (가드)
"케이크/타르트/몽블랑/바브카/파이" → cake
"크로크무슈/토스트/파니니/버거" → sandwich
"크루아상/데니쉬/스콘/머핀/롤/팥/앙금/크림빵/카스타드/슈크림/소보로/모카번/모카빵" → pastry
"바게트/치아바타/베이글/포카치아/호밀/통밀/잡곡" → bread
```

---

## §3. 성능 결과

### 3-1. Stage 1 — 운영 호환 backtest

운영 시나리오: D=목요일 cutoff → D+4~D+10 multi-horizon 예측. 16 Thursdays × 7 horizons = 평가.

| 지표 | 값 |
|---|---|
| **WAPE** | **29.37%** |
| **매진율** | **13.6%** (60일에 8일 미만) |
| 평균 production | 304 unit/일 |
| 평균 실제 sold | 242 unit/일 |
| 발주 over | +62 unit/일 |

### 3-2. Horizon별

| horizon | dow | n | WAPE | 매진율 | dow safety |
|---|---|---|---|---|---|
| D+4 | 월 | 7 | 29.89% | 14.3% | +0.8 |
| D+5 | 화 | 8 | 42.14% | 0.0% | +0.7 |
| D+6 | 수 | 4 | 40.73% | 0.0% | +2.0 |
| D+7 | 목 | 6 | 62.47% | 0.0% | +0.9 |
| D+8 | 금 | 6 | 36.30% | 0.0% | +0.6 |
| **D+9** | **토** | 10 | 10.30% | **50.0%** | +1.3 |
| D+10 | 일 | 3 | 0.77% | 33.3% | +0.1 |

→ 토요일 매진율 50% 잔여 (광교 5년 데이터의 한계).

### 3-3. Stage 1 Feature Importance

| 그룹 | 비중 |
|---|---|
| 자기상관 (lag/rolling/ewma) | 48% |
| 날씨 | 30% |
| 캘린더 (요일/월/주말) | 18% |
| 경쟁점 | 2% |
| 휴일 / 특수일 | 각 1% |

**TOP 5**: `lag28`, `avgRhm` (습도), `lag7`, `lag1`, `dom_sin`

### 3-4. Stage 2 분포 (마지막 fold)

| 카테고리 | STRONG_UP | UP | HOLD | DOWN | STRONG_DOWN |
|---|---|---|---|---|---|
| bread | 0 | 1 | 5 | 2 | 2 |
| pastry | 1 | 2 | 7 | 6 | 2 |
| sandwich | 0 | 0 | 0 | 1 | 0 |
| **합** | **1** | **3** | **12** | **9** | **4** |

대표:
- STRONG_UP: 스트로베리 밀크 크림 쇼콜라 (combined 1.37 = 인기품 + 신제품)
- STRONG_DOWN: 식빵의정성, 뺑오쇼콜라(Grand) (combined 0.68 = 추세↓ + 마감 높음)

### 3-5. Stage 3 — 신제품 81개 후행 검증

| 의사결정 \ 실제 | faded | marginal | survived | 합 |
|---|---|---|---|---|
| promote | 7 | 8 | **45** | 60 |
| hold | 3 | 4 | 9 | 16 |
| fade_out | 0 | 0 | 0 | 0 |

**적중률 76.3%** (58/76), promote 성공률 71.4%.

---

## §4. 평가 Framework

| 지표 | 분류 | 의미 |
|---|---|---|
| **카테고리 합 WAPE** (운영 horizon) | 정확도 | Stage 1 핵심 |
| **매진율** (production < actual) | 안전 | 가정 1-2-b 회피 |
| 인기품 평균 품절 시각 | 만족도 | proxy |
| 일 폐기 추정량 (실측 도착 시) | 비용 | 진짜 ROI |
| 신제품 4주 진단 적중률 | Stage 3 | 자동 의사결정 |

---

## §5. 한계

### 5-1. 데이터

| 한계 | 영향 |
|---|---|
| 광교 단일 매장 | 외부 features 효과 분리 어려움 |
| PoC 데이터 셀렉션 의심 | 잼/굿즈/예약 0개 (당일폐기 빵류 위주) |
| 폐기 실측 부재 | ROI = 마감 할인 절감 lower bound |
| 사전 예약 데이터 | cake 분리 불가 |
| 빼빼로 굿즈 데이터 부재 | 빼빼로 효과 검증 불가 |

### 5-2. 모델

| 한계 | 영향 |
|---|---|
| 토요일 D+9 매진율 50% | 특별 이벤트 매출 급증(-46, -64 unit) 캡처 불가 |
| LGBM의 정확도 부가가치 작음 (광교 단독) | baseline 자체가 강한 수요 매장이라 추가 신호 적음 |
| dow safety 작은 값 (1~2 unit) | 학습 데이터에서 LGBM q90이 거의 over-prediction |
| 운영 WAPE ~30% | 광교 5년 데이터 ceiling |

### 5-3. 가정

| 한계 | 영향 |
|---|---|
| 1-2-b 장기 만족도 데이터 검증 X | 가설 박고 진행, proxy 추적만 |
| α=0.6 휴리스틱 | 폐기 실측 도착 시 재캘리브레이션 |
| fade_out 룰 광교 미발현 | 다매장 도착 시 완화 |

---

## §6. 향후 진행

### 즉시 가능

| 작업 | 기대 효과 |
|---|---|
| Stage 2 boost 강도 sensitivity 튜닝 | 광교 적합 boost 정밀화 |
| cake 시즌 분리 + 홀케이크 별도 모델 | cake 시즌 특수 패턴 |

### 데이터 도착 후

| 데이터 | 작업 | 기대 효과 |
|---|---|---|
| 폐기 실측 (또는 입고량) | α 캘리브레이션 + 진짜 ROI | 보수 추정 → 실측 |
| 다매장 데이터 | 외부 features 본격 검증 + fade_out 룰 캘리브레이션 | framework 일반화 |
| 사전 예약 정보 | cake 사전 예약 분리 + 별도 모델 | cake 카테고리 합산 가능 |
| 영업시간 / 휴무일 | open/close_hour dynamic | potential_demand 정확도 |
| 채널 정보 (배달 vs 매장) | 채널별 hour profile + 별도 추정 | 채널별 demand 분리 |
| 무게/사이즈 정보 | 진짜 정규화 unit (kcal 또는 1회분량) | unit 비교 정확도 |
| 광교 이벤트 / promo 데이터 | 토요일 outlier 캡처 → 매진율 ↓ | 운영 WAPE / 매진율 개선 |

---

## §7. 모듈 및 산출물

### 7-1. 코드

- `src/bakery/features/category_aggregate.py` — 카테고리 합 daily + 57 features
- `src/bakery/models/category_total.py` — Stage 1 LGBM (training용 reference)
- `src/bakery/models/item_proportion.py` — Stage 2 분위수 가중 분배
- `src/bakery/models/new_product_tracker.py` — Stage 3 4주 진단
- `src/bakery/analysis/seasonal.py` — 시즌 16개 제외
- `src/bakery/analysis/discount.py` — 30개 할인코드 분류
- `scripts/operational_5model_comparison.py` — 운영 호환 backtest

### 7-2. 산출 (reports/)

- `v4_method_c_revised.csv` — Stage 1 최종 backtest (5 모델 비교)
- `v4_feature_importance_*.csv` — feature importance
- `v4_phase5_stage2_proportions_quantile.csv` — Stage 2 분위수 가중 분배
- `v4_phase5_recommendations_quantile.csv` — 5단계 권장
- `v4_phase5_new_product_validation.csv` — Stage 3 81개 후행 검증

### 7-3. 테스트

- **119/119 통과**

---

**Current state**: v4 framework Ensemble (baseline + LGBM residual + dow safety) 운영 호환 backtest 완료. 폐기 실측 / 다매장 데이터 도착 대기.

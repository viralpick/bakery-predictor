# 다매장 stockout 재정의 (store_daily → aggregate_daily 코어 공유) 설계

**날짜**: 2026-07-13
**관련**: `project_stockout_time_bug_and_adjusted_demand`, `project_stockout_remediation_roadmap`, 단매장 재정의(PR#28)

## 1. 배경 / 문제

단매장(`bonavi_loader`)은 `is_stockout`을 **폐기0 & 완판**(`QT_MADE>0 & QT_OUT<=0`)으로 재정의했으나(60.4%), 다매장 로더 `scripts/store_daily.build_store_daily`는 **옛 버그 로직**을 그대로 쓴다:
- `stockout.parquet`(SOLD_TIME만 있음)에서 `groupby(date,item)['stockout_time'].first()` — 하루 다중 품절이벤트(리필→재품절) 중 **첫 순간**만 취함 → stockout_time이 이르게(07~09시) 오염.
- `is_stockout = stockout_time.notna()` → **92.7%**(거의 전부 품절로 오판).

**데이터는 있다**: `data/internal/v2/inventory.parquet`에 4매장 전부 `QT_MADE`·`QT_OUT`가 있다(광교 1826일·삼성 1826·메세나 1826·광화문 1270, null 0). 재정의 `(QT_MADE>0 & QT_OUT<=0)` = 4매장 합 47.9%(매장별 상이, 광교 60.4%).

## 2. 목표 / 스코프

**포함**:
1. `store_daily`가 단매장 코어 로직(`aggregate_daily` + `assign_stockout_fields`)을 **재사용**하도록 교체 — 다매장 로더가 단매장 로더를 내부 포함하는 형태.
2. 회귀 테스트(신규 + 기존 `test_ontology_functions` 통과).
3. HTML 4매장 리포트·발주 무영향 재확인(예측·WAPE·버퍼는 stockout feature를 LEAK_COLS로 제외하므로 불변 — 이 사실을 테스트/문서로 고정).
4. **소비처 영향맵 문서화** — store_daily의 `is_stockout`/`stockout_time`을 쓰는 소비처를 열거하고 "재검증 필요/불필요" 분류. **실제 재검증은 사용자가 체크리스트 보고 선택**(이번 스코프 밖).

**제외**: 소비처(absorption/substitution/popularity/stockout_classifier 등) 실제 재실행·재판정은 후속. 데이터 fix를 먼저 머지.

## 3. 설계

### 3.1 코어 재사용 (핵심)

`bonavi_loader.aggregate_daily(sales, items, inventory, last_sale, ...)`은 **이미 store-agnostic 순수 함수** — xlsx가 아니라 DataFrame 4개를 받아 병합하고 `assign_stockout_fields`(폐기0&완판)를 적용한다. 다매장은 **V2 parquet → 이 함수의 입력 스키마로 어댑터**만 만들면 된다.

**입력 스키마 (aggregate_daily가 기대하는 것)**:
- `sales`: columns `store_id, item_id, date, qty` (bulk 제외된 정상 단품).
- `items`: `item_id, category_id`.
- `inventory`: `date, item_id, production_qty, waste_qty`.
- `last_sale`: `date, item_id, last_sale_ts`.

### 3.2 store_daily 어댑터 (V2 → 위 스키마)

```
build_store_daily(store_cd, store_id, exclude_bulk=True):
    sales_raw = read V2/sales.parquet, filter (CD_PARTNER, SALES_FG=0, CD_USERDEF2=SS)
    if exclude_bulk: line-level flag_bulk_lines 로 예약 라인 제거   # 기존 로직 유지
    sales = rename → {store_id, item_id(CD_ITEM), date(DT_SALE), qty(QT_SALE)}
    items = item_category_map() → {item_id, category_id}
    inventory = read V2/inventory.parquet, filter CD_PARTNER
              → {date, item_id, production_qty(QT_MADE), waste_qty(QT_OUT)}
    last_sale = sales(위 bulk 제외본)에서 SALES_TIME max per (date,item)
              → {date, item_id, last_sale_ts}
    daily = aggregate_daily(sales, items, inventory, last_sale)  # 단매장과 동일 코어
    return daily[기존 반환 컬럼: date,item_id,sold_units,store_id,category_id,stockout_time,is_stockout]
```

### 3.3 bulk 일관성

- **bulk 제외는 sales 계열에만** 적용(sold_units·last_sale_ts 둘 다 동일 line-level `flag_bulk_lines` 필터를 탐 → 정합). bulk=예약이라 실수요 아님, 제외가 맞음.
- **inventory(QT_MADE/QT_OUT)엔 bulk 개념 없음** — 생산/폐기는 물리 수량이라 예약과 무관, 그대로 사용. is_stockout 판정은 inventory 기반이라 bulk 필터와 독립.

### 3.4 last_sale_ts (구 헬퍼 안 씀)

기존 `absorption_4stores._last_sale_ts`는 **구 bulk 소스**(`sales_with_bulk_flag.parquet`, whole-receipt)를 써서 store_daily의 신규 line-level bulk와 불일치. **재사용하지 않고** store_daily 안에서 line-level bulk 제외본 sales의 SALES_TIME max로 직접 계산 → sold_units와 동일 필터 보장.

### 3.5 store_mapping (다매장 QT_MADE null 방지)

inventory는 CD_PARTNER(매장코드)로 필터하므로 cross-store blend 없음. store_cd로 정확히 거른 뒤 넘긴다(단매장의 `inv_store` 가드와 동일 취지).

## 4. HTML/발주 무영향 (검증 대상)

`is_stockout`/`stockout_time`/`n_stockout_items`/`n_early_stockout`은 전부 `category_total.LEAK_COLS`에 있어 **학습 feature에서 제외**(확인: `select_feature_cols`가 stockout 컬럼 0개 반환). 타겟은 `adjusted_demand`(closing/α 기반, 품절 플래그 안 씀). → 4매장 리포트 예측·WAPE·발주·버퍼 **불변**. 이 사실을 테스트로 고정하고 HTML 재생성 결과가 동일함을 확인.

## 5. 소비처 영향맵 (문서화 산출물)

store_daily의 `is_stockout`/`stockout_time`을 소비하는 스크립트 21개 + `test_ontology_functions`. 재정의(92.7%→~48%)가 **결과를 바꾸는** 소비처 분류:
- **바뀜(재검증 후보)**: `absorption_4stores`(W0), `substitution_4stores`, `revalidate_popularity_stockout`, stockout_classifier 학습.
- **불변**: `store_predictive_power`/`verify_event_prior`/발주(LEAK_COLS 제외), 예측 파이프라인.
체크리스트로 정리 → 사용자가 재검증 대상 선택.

## 6. 테스트

- `test_store_daily_redefine.py` (신규):
  - inventory 합성 fixture로 `aggregate_daily` 경유 is_stockout이 `(made>0 & waste<=0)` 정확 일치.
  - bulk 라인 제외가 sold_units·last_sale_ts 둘 다에 반영(동일 필터).
  - 재정의 후 is_stockout 비율이 옛 notna(~92%) 아님(정성: 크게 낮음) + 특정 합성행 정확값.
- HTML 무영향: `select_feature_cols`가 stockout 컬럼 0개 반환 단언(회귀 가드).
- 기존 `test_ontology_functions` 통과.

## 7. 한계

- 광화문은 inventory가 2022-07부터(1270일) — 그 이전 없음(단매장과 달리 짧음). merge how=left라 없는 날은 is_stockout=False(NaN→False), 정상.
- 소비처 재검증 미포함 — 데이터 fix 머지 후 사용자 선택.
- QT_OUT 음수 3.26% 존재(반품/조정?) — `waste<=0`이 음수도 완판으로 판정(단매장 공식과 동일, 의도적).

**Current goal**: 다매장 로더 stockout 재정의(단매장 코어 공유) + 무영향 확인 + 소비처 영향맵.
**Last decisions**: aggregate_daily 재사용(어댑터만 신규), bulk는 sales 계열만 제외, last_sale은 line-level bulk 일관 재구현.
**Open risks**: 소비처 21개 결과 변화(재검증 후속). 광화문 짧은 inventory.
**Next first step**: writing-plans로 태스크 분해 → store_daily 어댑터 교체부터 TDD.

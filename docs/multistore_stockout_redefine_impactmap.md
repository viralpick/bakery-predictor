# 다매장 stockout 재정의 — 소비처 영향맵

**배경**: Task 1 (commit `8d3037f`)은 `scripts/store_daily.py`의 `build_store_daily`가
`bakery.data.bonavi_loader.assign_stockout_fields`(단매장에서 이미 확정된 "폐기0 & 완판"
재정의, commit `121de35`)를 공유하도록 고쳤다. 즉 **이번 변경의 실제 범위는
`scripts/store_daily.py` 한 파일**이고, 그 이전까지 그 파일은 "첫 순간품절 이벤트"라는
옛 정의(is_stockout≈92%)를 썼다. 단매장 파이프라인(`bakery.data.bonavi_loader.py`)은
이 브랜치 이전부터 이미 같은 재정의를 쓰고 있었으므로 **이번 변경으로 전혀 건드리지 않았다**.

**핵심 판정 기준**: 소비처가 `scripts/store_daily.py`(다매장, 이번에 변경됨)에서 데이터를
받는가, 아니면 `bakery.data.bonavi_loader.py`(단매장, 이번에 안 건드림)에서 받는가.
후자는 grep에 `is_stockout`/`stockout_time`이 잡히더라도 이번 커밋과 무관 — **B(불변)**.

**재검증 여부는 전적으로 사용자(architect) 선택**이다. 이 문서는 "무엇이 바뀔 수 있는지"를
분류할 뿐, 실제로 재실행/재검증할지는 결정하지 않는다.

## 1. 확증: HTML/발주 경로 무영향

`tests/test_store_daily_redefine.py::test_stockout_cols_excluded_from_training_features`
— `category_total.select_feature_cols`가 반환하는 학습 feature 목록에 `stockout` 포함
컬럼이 0개임을 고정(PASS). `n_stockout_items`/`n_early_stockout`/`n_items_active`는
`LEAK_COLS`로 이미 제외되고, `is_stockout`/`stockout_time` 원본 컬럼 자체도 feature가
아니다. target은 `adjusted_demand`(마감/α 기반)로 stockout flag와 독립. 따라서
**HTML 리포트/발주(예측) 경로는 이번 재정의로 값이 달라지지 않는다.**

## 2. 소비처 분류

`grep -rln "is_stockout\|stockout_time" scripts/ src/ | grep -v test` 결과 **29개 파일**.
이 중 `src/bakery/data/bonavi_loader.py`는 **정의처(upstream)** — `assign_stockout_fields`(재정의
공식 자체)를 두는 곳이지 `scripts/store_daily.py`의 소비처가 아니다. 단매장 파이프라인의 원천이며
Task1이 안 건드림(`121de35`에서 이미 재정의 확정) → **B (upstream/불변)**. 아래 표는 나머지 28개
소비처를 분류한다(29 = 정의처 1 + 소비처 28).

| 소비처 | 어떻게 쓰는가 | 분류 | 비고 |
|---|---|---|---|
| `scripts/substitution_4stores.py` | `store_daily.build_store_daily`로 받은 daily를 `mnl_substitution.fit_mnl_per_category` / `nested_logit.fit_nested_logit`의 availability(`~is_stockout`)와 `substitution.py`/`substitution_did.py`의 대체 타이밍(`stockout_time` 시각)에 직접 투입 | **A (재검증 후보)** | MNL/Nested/DiD 대체효과 추정치가 옛 92%→새 60%대 정의로 바뀌면 availability set 자체가 달라져 결과가 변할 수 있음. `project_substitution_mnl` 메모리의 λ≈0.99 등 헤드라인이 이 경로 |
| `scripts/revalidate_popularity_stockout.py` | `old_daily = build_store_daily(...)`(원본, "옛 정의" 가정)와 `new_daily = apply_fixed_stockout(old_daily,...)`("새 정의")를 비교해 `item_proportion.compute_proportions`의 인기 신호 델타를 측정 | **A — 단, 스크립트 로직 자체가 이번 커밋으로 깨짐** | 이 스크립트의 전제("`build_store_daily`는 옛 정의를 반환한다")가 Task 1로 무효화됨. 이제 `old_daily`도 이미 새 정의라서 `old_daily`와 `new_daily`가 사실상 동일해지고 "옛 vs 새" 델타가 항상 ≈0으로 나오는 퇴화(degenerate) 상태. 재실행 전에 old-side를 옛 정의로 별도 고정하는 코드 수정이 필요 — 단순 재실행이 아니라 **코드 수정 후 재검증** 대상 |
| `scripts/absorption_4stores.py` | `store_daily.build_store_daily`로 daily를 받지만, 자체 함수 `apply_fixed_stockout()`으로 `is_stockout`/`stockout_time`을 inventory(QT_MADE/QT_OUT)에서 **이미 이 브랜치 이전부터** 로컬 override 해왔음(주석: "옛 '첫 순간품절 이벤트' 정의... 대체. W0 재검증 전용") | **사실상 무영향 (idempotent)** | Task 1 이전에도 이 스크립트는 새 정의를 직접 재계산해서 덮어썼으므로, `build_store_daily`가 이제 같은 정의를 반환해도 override 결과는 동일(같은 소스 데이터에서 같은 공식 재계산). `project_demand_absorption_w0` 헤드라인(β/TOST)에 영향 없음. **클린업 후보**: 이제 override가 중복이므로 `apply_fixed_stockout` 호출 제거 고려 가능(별도 스코프) |
| `scripts/outflow_compare_quick.py` | `bakery.data.bonavi_loader`(단매장) 경로의 `daily_mode["is_stockout"]`로 필터링 | **B (불변)** | `store_daily.py` import 없음 — 단매장 파이프라인 소비, 이번 커밋과 무관 |
| `scripts/verify_hypotheses.py` | `bakery.data.bonavi_loader`(단매장) 경로 daily의 `is_stockout`/`stockout_time`으로 조기/후기 품절 그룹 분리 | **B (불변)** | 동일 사유 — 단매장 경로 |
| `src/bakery/features/category_aggregate.py` | `n_stockout_items`, `n_early_stockout` 등 집계 → **모두 `LEAK_COLS`로 학습 feature에서 제외**(Step1 테스트가 고정) | **B (LEAK_COLS 제외, 불변)** | 다매장 경로에서도 호출되지만(Task2 테스트가 광교 다매장 경로로 직접 확인) feature로 안 새므로 예측 무영향 |
| `src/bakery/features/cannibalization.py` | `store_stockout_rate`/`cat_stockout_rate` 집계(단매장 lightgbm_regressor의 `_extra_history_cols`용) | **B (단매장 + LEAK 경로)** | `is_stockout` shift(1) 이력 feature로만 사용, 원본 flag는 target 아님 |
| `src/bakery/features/stockout_history.py` | `is_stockout`을 `shift(1)`한 이력 feature(`STOCKOUT_HISTORY_COLUMNS`) 생성 — `stockout_classifier` 학습 입력 | **B (단매장 파이프라인)** | `stockout_classifier`는 현재 어떤 scripts에서도 다매장 경로로 호출되지 않음(재측정은 PR#32, 단매장 기준) |
| `src/bakery/features/potential_demand.py` | `stockout_time`으로 시간비례 보정해 `potential_demand` 계산 | **B (단매장, 이미 폐기 확정)** | `potential_demand`는 real 경로에서 별도 사유(로더 버그, PR#31 감사)로 이미 deprecated → target은 `adjusted_demand`. 다매장 경로는 이 함수를 안 씀 |
| `src/bakery/models/new_product_tracker.py` | `stockout_freq`, `stockout_h` — 신제품 tracker 신호 | **B (단매장)** | 다매장 스크립트에서 호출 안 됨 |
| `src/bakery/models/item_proportion.py` | `is_stockout`/`stockout_time`으로 `avg_stockout_h`→`adj_stockout`→`proportion`(Stage2 품목비율 boost) | **B for 단매장 실사용 / A는 위 `revalidate_popularity_stockout.py`를 통해서만** | 함수 자체는 공유 코드라 "소비처"는 호출자(스크립트) 기준으로 분류. 다매장에서 이걸 부르는 건 `revalidate_popularity_stockout.py` 하나뿐(위에서 이미 A로 분류) |
| `src/bakery/models/lightgbm_regressor.py` | `is_stockout`을 cannibalization aggregate의 입력으로 `_extra_history_cols`에 포함(target 아님, shift 이력) | **B (단매장 + LEAK 경로)** | `feature_set="v2"` 경로 — 다매장 스크립트가 이 클래스를 직접 학습에 쓰지 않음(다매장 backtest들은 `category_total`/자체 LGBM 사용) |
| `src/bakery/models/stockout_classifier.py` | `is_stockout`을 라벨(y)로 학습(운영 watchlist) | **B (단매장, PR#32 재측정 완료)** | 다매장 스크립트 어디서도 이 클래스를 import하지 않음(grep 확인) |
| `src/bakery/ontology/functions.py`, `grounding/tools.py`, `schema.py` | `stockout_time`을 "관측치(예보 아님)"로 노출, 스키마 필드 정의 | **B (단매장, v6 decision layer)** | 다매장 파이프라인과 분리된 온톨로지 레이어 |
| `src/bakery/evaluation/metrics.py` | `is_stockout` 마스크로 censored-aware WAPE 등 계산 | **B (단매장)** | 다매장 backtest 스크립트들은 자체 metric 함수 사용 |
| `src/bakery/evaluation/prospective.py` | 전향 검증에서 `is_stockout` 비율/공유 계산 | **B (단매장, PR#26 전향 harness)** | 다매장 경로 미사용 |
| `src/bakery/data/schema.py`, `src/bakery/data/synthetic.py` | 스키마 상수/synthetic 데이터 생성 시 `is_stockout(_hour)` 필드 정의 | **B (스키마/synthetic, 다매장 무관)** | synthetic은 PoC 초기 데이터, real 경로와 별개 |
| `src/bakery/cli.py` | 리포트 출력(`stockout days: N/M`), `stockout_classifier` 검증 경로(`val["is_stockout"]`) | **B (단매장)** | `store_daily` import 없음 |
| `scripts/{rfecv_per_store,window_rfecv_pipeline,all4_unified_features,train_window_sensitivity,rfecv_composite,interval_backtest_4stores,window_composite_pipeline,grid_backtest_cache,alpha_sensitivity,permutation_importance_all4,auto_feature_selection,store_predictive_power,verify_event_prior}.py` (13개) | `store_daily.build_store_daily`를 **실제로 import·호출**해 다매장 daily를 받지만, grep에 `is_stockout`/`stockout_time` 자체가 안 잡힘 — 반환 daily를 LGBM feature/backtest에만 쓰고 stockout 컬럼은 소비 안 함(`store_predictive_power.py`의 "stockout_risk"는 몬테카를로 draw 기반 별도 산출물로 `is_stockout` 컬럼과 무관) | **B (커플링은 있으나 stockout pass-through)** | Step1 테스트가 확인한 LEAK_COLS 제외와 정합 — 다매장 backtest 헤드라인(RFECV, α sensitivity, window composite 등)은 이번 재정의로 안 바뀜 |
| `scripts/{diag_anchor_gh,diag_chuseok_gh,interval_anchor_compare}.py` (3개) | `store_daily`에서 **`STORE_MAP`만 import**(`build_store_daily` 미호출) — 매장 코드 매핑만 씀, daily 빌드는 다른 경로/모듈에 위임 | **B (다매장 stockout 경로와 무커플링)** | `grep -n "store_daily\|build_store_daily"` 확인: `from store_daily import STORE_MAP` 한 줄뿐. is_stockout 정의 변경과 무관 |
| `scripts/all4_stores_backtest.py` (1개) | `scripts/store_daily`를 **import하지 않음**. 자체 동명 함수 `build_store_daily`(line 49)를 정의하며 `bakery.data.bonavi_loader.map_category`만 가져와 sales/inventory에서 `adjusted_demand`를 직접 계산 — stockout 컬럼 자체를 만들지도 소비하지도 않음 | **B (독립 로더)** | Task1이 바꾼 `scripts/store_daily.py`와 코드 경로가 완전히 분리. 동명 함수라 grep에 잡혔을 뿐 커플링 없음 |
| `src/bakery/analysis/{demand_absorption,self_fulfillment,mnl_substitution,nested_logit,substitution,substitution_did,popularity}.py` (공유 분석 라이브러리, 그 자체로는 호출자 아님) | `is_stockout`(availability/`~is_stockout`)·`stockout_time`(대체·자기충족 타이밍)을 실제 분석 입력으로 사용 — 함수 자체는 데이터 소스를 안 가림 | **호출자 기준으로 분류** (라이브러리 자체는 A/B 아님) | `grep -rl "self_fulfillment\|analysis\.popularity\|demand_absorption" scripts/` → 다매장 호출자는 `absorption_4stores.py` 하나뿐(이미 위에서 분류). `mnl_substitution`/`nested_logit`/`substitution`/`substitution_did`의 다매장 호출자는 `substitution_4stores.py`(이미 A). `self_fulfillment`는 `src/bakery/cli.py`(단매장)에서만, `popularity.py`는 어떤 scripts/에서도 호출되지 않음(미사용) — 따라서 이 라이브러리들을 통한 다매장 영향은 이미 위 3개 행(`substitution_4stores`/`absorption_4stores`/`revalidate_popularity_stockout`)에 전부 반영됨, 추가 소비처 없음 |

**A 요약 (재검증 후보, 사용자 선택)**:
- `substitution_4stores.py` (MNL/Nested/DiD 대체효과) — 진짜 재검증 후보
- `revalidate_popularity_stockout.py` — 재검증 이전에 **코드 수정 필요**(old-side가 더 이상 "옛 정의"가 아님)
- `absorption_4stores.py` — 형식상 A지만 실질적으로 이미 override라 무영향(클린업만 고려)

**B 요약**: 나머지 25개 — 단매장 파이프라인(이미 재정의 적용된 지 오래) 또는 LEAK_COLS로
feature에서 제외됨. 다매장 backtest 17개 스크립트 포함, 모두 stockout 컬럼을 학습/HTML/발주
경로에 안 씀.

## 3. 후속 팔로업 후보 (이번 스코프 아님)

**last_sale 산출 방식의 단매장↔다매장 비대칭**: 단매장 `bonavi_loader.load_receipts_with_time`
(→ `last_sale_ts` → `stockout_time`)은 `flag_bulk_lines`로 예약(bulk) 라인을 제외하지 **않고**
전체 영수증에서 `timestamp.max()`를 취한다(`load_sales`는 bulk 제외하지만 `load_receipts_with_time`은
안 함 — 코드 확인: `src/bakery/data/bonavi_loader.py:169-207`, `flag_bulk_lines` 호출 없음).
즉 예약 주문의 결제 타임스탬프가 그날 마지막 실판매 시각을 오염시킬 수 있다.
반면 다매장 `scripts/store_daily.py`는 `ls = sales.copy()`(이미 bulk 제외된 `sales`)에서
`last_sale_ts`를 뽑아 더 일관적이다(코드: `store_daily.py:69-76`, 주석 "③ 마지막 실판매 시각 —
bulk 제외본 sales에서"). 단매장 쪽도 동일하게 bulk 제외본에서 `last_sale_ts`를 뽑도록 맞추는
것을 팔로업으로 고려할 만하다 — 이번 태스크 스코프 밖.

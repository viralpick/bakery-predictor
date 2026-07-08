# Target 통일(adjusted_demand) + PR#26·#27 재측정 — 설계

**날짜**: 2026-07-08
**브랜치**: feat/stockout-redefine (하위2 재검증 ③)
**선행**: [[project_stockout_time_bug_and_adjusted_demand]], [[project_category_order_remeasure]], [[project_prospective_derisk_retro]], [[project_stockout_remediation_roadmap]]

## 배경 / 문제

전향 retro harness(`prospective-eval`)는 발주 추천 vs 현행 발주를 KPI(폐기/매진시각/매진률)와 calibration으로 비교한다. 두 과거 결론이 오염됐음이 규명됐다:

- **PR#26 (item-level de-risk)**: q0.85 초과율 0.636(nominal 0.15 대비 4×). 발주 v2 LGBM과 평가 잣대가 **둘 다 `potential_demand`** — target은 정합하나, `potential_demand` 필드 자체가 오염(로더가 하루 첫 품절이벤트만 취해 `stockout_time`이 이르게 찍힘 → is_stockout 92% false → 배수 복원으로 수요 부풀림).
- **PR#27 (category-level)**: 초과율 0.738. 발주는 `adjusted_demand_unit` 학습 ↔ 평가는 `potential_demand` → **target 불일치 confound**.

`is_stockout` 재정의(92%→60.4%)로 데이터는 이미 고쳤다(하위1, PR#28). `potential_demand`는 폐기 확정, 수요 target은 **adjusted_demand**(정상판매 + 마감할인×α)로 확정됐다. 이제 harness의 발주·평가 target을 adjusted_demand로 **통일**해 apples-to-apples로 재측정한다.

## 목표 / 비목표

**목표**
- item-level `adjusted_demand` 필드 신설.
- 발주 target + 평가 잣대를 **둘 다** adjusted_demand로 통일 (item·category 경로).
- PR#26·#27 결론을 재측정: confound/오염 제거 후 초과율이 nominal(0.15) 근처로 회복하는지 판별.
- α=0.5 헤드라인 + {0.3,0.7,1.0} 민감도.

**비목표**
- 4매장 발주 검증 (별도 후속. harness는 광교 단독 전제 — `_load_real_daily`가 n_stores≠1이면 raise).
- α 자체의 실증 확정 (별도 [[project_closing_discount_alpha]]).
- `potential_demand` 필드/코드의 전역 제거 (다른 CLI 경로가 아직 참조 — 이번 경로만 오버라이드).

## 결정 (승인됨)

1. **잣대 granularity**: item-level adjusted_demand 신설. potential_demand가 쓰이던 모든 잣대 자리에 스왑. KPI 시뮬 구조 유지.
2. **범위**: 발주+평가 target 모두 adjusted_demand로 통일. item 경로 v2 LGBM 재학습(y_col 교체), category 경로는 이미 adjusted 학습이라 잣대만 교체.
3. **α**: 0.5 헤드라인 + {0.3,0.7,1.0} 민감도 스윙.

## 설계

### 컴포넌트 1 — item-level adjusted_demand 신설

- 신규 함수 (`features/category_aggregate.py`에 추가; item 레벨이라 형제 헬퍼로):
  ```
  adjusted_demand_item = sold_units − closing_qty_item × (1 − α)
  ```
  (대수: normal + closing×α = (sold − closing) + closing×α = sold − closing×(1−α))
- `closing_qty_item`: `load_sales_with_discount().closing_discount()`를 `(item_id, date)` 합산 후 daily에 left-join. 매칭 없으면 closing=0 → adjusted=sold.
- **leakage-safe**: 당일 관측 label. feature 아님(단, 아래 lag 전파 주의).
- 인터페이스: `build_item_adjusted_demand(daily, discount_rows=None, alpha=DEFAULT_ALPHA) -> daily(+adjusted_demand)`. 의존: closing discount 소스. 단독 테스트 가능.

### 컴포넌트 2 — 발주 target 통일

**item 경로** (`_quantile_backtest_predictions`, cli.py):
- daily에 item `adjusted_demand` 주입(컴포넌트 1).
- forecaster를 `GlobalLGBM(feature_set="v2", y_col="adjusted_demand", params=...)`로 구성.
- `run_backtest(daily, [forecaster], windows, y_col="adjusted_demand")`.
- **lag/rolling 자동 전파**: `GlobalLGBM._build_features`가 `add_lag_features(y_col=self.y_col)` / `add_rolling_features(y_col=self.y_col)`로 호출 → y_col을 adjusted로 바꾸면 lag/rolling history도 adjusted 기반으로 일관 전환됨(잔여 potential 오염 없음). ✅ 확인됨(lightgbm_regressor.py:209-210).

**⚠️ 필수 수정 — `run_backtest._clone` y_col 보존** (backtest.py:111-118):
- 현재 `_clone`이 LGBM 재생성 시 `params`/`feature_set`만 넘기고 **`y_col`을 버림** → 클론이 `_default_target("v2")="potential_demand"`로 되돌아감.
- 결과: y_col="adjusted_demand"를 줘도 모델은 여전히 potential 학습, 평가만 adjusted → 새 불일치(사일런트 오염).
- 수정: `_clone`에서 `if hasattr(forecaster, "y_col"): kwargs["y_col"] = forecaster.y_col`. 기존 호출자는 default와 동일하므로 backward-compatible.
- 이 수정은 leakage/기존 backtest 테스트로 회귀 확인.

**category 경로** (`_category_order_predictions`): 이미 `build_features(build_category_daily(), target_col="adjusted_demand_unit")` 학습. **변경 없음**.

### 컴포넌트 3 — 평가 잣대 통일 (cli.py)

`rows`의 실현수요 잣대를 `potential_demand` → item `adjusted_demand`로 교체:
- `_assemble_real_rows` / `REAL_ROWS_COLUMNS`: `adjusted_demand` 포함(주입).
- 초과율 `quantile_exceedance_rate` / `wpe` / `_decoupling_by_category`: adjusted 사용.
- category calibration(2074~): item adjusted를 date별 합산.
- `simulate_item_day_kpis`(prospective.py): `potential_col`/`prof_in["sold_units"]=prof_in["potential_demand"]` 참조를 adjusted_demand로. 시그니처에 잣대 컬럼명을 파라미터화(기본 하위호환)해 synthetic 경로 불변.

### 컴포넌트 4 — 재측정 실행 + 산출물

- `n_folds=8`(full-window — PR#26·#27 헤드라인과 비교 가능), `our_order_val_weeks` 헤드라인 설정 유지.
- `order_level` = item, category **둘 다**.
- α=0.5 헤드라인 + {0.3,0.7,1.0}.
- 비교 축(PR#26·#27 표와 동일): 초과율(nominal 0.15) / WPE / stockout_rate Δ+CI / waste_cost Δ+CI / lost_margin Δ+CI / soldout_median_h Δ.
- 문서 `docs/target_unification_remeasure_result.md`. PR#26·#27 결과 문서·메모리 헤드라인 정정.

## 데이터 흐름

```
bonavi_daily (fixed, is_stockout 60.4%)
  + closing_discount() [item,date 합산]
  → build_item_adjusted_demand (α)  ──┬─→ [발주] _quantile_backtest_predictions
                                       │      GlobalLGBM(v2, y_col=adjusted) → our_order(item)
                                       │      _category_order_predictions → our_order(category, 기존)
                                       └─→ [평가] rows.adjusted_demand
                                              초과율/WPE/ρ_DS/KPI 시뮬 잣대
```

## 검증

- **단위**: `build_item_adjusted_demand` — α=1 → adjusted==sold, α=0 → adjusted==normal(=sold−closing), closing 없는 item → adjusted==sold. 정확값 비교(`==`).
- **회귀**: `test_split_leakage.py` / `test_features_leakage.py` / 기존 backtest 테스트 통과(_clone 수정 후).
- **wiring**: `_clone`이 y_col 보존하는지 단위 테스트(forecaster.y_col="adjusted_demand" → 클론도 동일).
- **smoke**: `prospective-eval --source real --order-level item --n-folds 1`에서 잣대가 adjusted로 바뀐 것 확인(초과율 로그).
- **엔드투엔드**: 헤드라인 재측정 실행, 결과 문서화.

## 리스크 / 캐비엣

- **α 미확정**: 잣대·발주 절대 스케일이 α에 의존. calibration 초과율은 대체로 α-불변(발주·평가 동일 α)이나 KPI 원화값은 민감 → 스윙으로 보고.
- **potential_demand 잔존 참조**: 이번 경로 밖 CLI(`backtest` 등)는 여전히 potential 사용 — 전역 감사·제거는 별도.
- **광교 단독**: 타매장 품절 프로파일 다를 수 있음(삼성 lost-sales 신호). 4매장 확장은 후속.
- **결과 방향 미정**: 초과율이 회복 안 하면(adjusted에서도 under), 그건 "잣대 문제 아닌 v2 quantile 모델 자체의 under-calibration"이라는 새 결론 — 정직 보고.

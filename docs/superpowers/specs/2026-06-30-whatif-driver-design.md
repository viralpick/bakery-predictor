# Upstream Lever `what_if_driver` (v7 S6) — Design

**Date**: 2026-06-30
**Depends on**: S1 ontology schema (OntologyLink), S2 OntologyFunction registry, models/lightgbm_regressor (GlobalLGBM), decision/risk (simulate_item_risk), grounding/tools (TOOL_SPECS dispatch)
**Status**: Design approved (2026-06-30)

## 목적

closed-loop의 **상류 레버**를 구현한다 (하류 = S5 발주량 변경). 드라이버(날씨/캘린더)를 가상으로 바꾸면 **바인딩된 forecast 모델이 demand를 재계산**하고, 그 변화가 OntologyLink를 따라 위험/비용으로 전파되는 것을 시연한다. 팔란티어 Scenario(model-on-object)의 동형 구현으로, v7 thesis(AOS가 모델을 객체에 바인딩해 가정 변경을 전파)를 직격한다.

## 핵심 설계 결정 (LOCKED)

### D1. 실제 forecast 재실행 (옵션 A)
드라이버 변경 시 demand를 **실 LightGBM 모델 재예측**으로 산출한다(elasticity 손튜닝 배수 아님). 이유: 상류 Scenario의 demonstrable 가치 = "드라이버 property 변경 → 바인딩된 모델이 객체 demand 재계산 → link 전파"이며, 이것이 회사 thesis(model-on-object)의 정통 구현. elasticity mock은 "그건 네 모델이 아니다" 반론으로 thesis를 약화시킨다. weather lift가 낮은 건(메모리: ±5%, 매장별 부호 다름) 정직한 한계로 두고 calendar/holiday를 헤드라인 드라이버로 쓴다.

### D2. forecast 국소 도입 (전면 교체 아님)
`what_if_driver` 안에서만 before/after를 `model.predict`로 계산한다. 기존 5개 read 함수(rank_stockout_risk 등)의 `demand_point`은 **proxy(potential_demand) 그대로 유지**. 이유: S1~S5 회귀 리스크 0, before/after가 같은 모델이라 델타 공정, forecast wiring이 이 함수의 컴포넌트로 흡수돼 별도 sub-project 불필요.

### D2-1. feature_set = "v2" (target = potential_demand) — 절대규칙 #2 준수
모델은 `GlobalLGBM(feature_set="v2")`를 쓴다. 이유: **v2/v3만 target=potential_demand**(v0/v1=sold_units). 품절일 `sold_units`는 검열된 값이라 이를 demand로 stockout 위험 시뮬레이터에 넣으면 고수요일 위험을 체계적으로 과소평가한다(절대규칙 #2 위반). v2는 calendar+weather 드라이버를 포함하고, **cannibalization feature는 `GlobalLGBM._build_features`가 내부 계산**(carried sold_units/is_stockout)하므로 입력 프레임은 v1과 동일한 calendar+weather enriched 프레임이면 된다(외부데이터 불필요).

### D2-2. 입력 feature 프레임 조립 (forecast wiring 핵심)
ontology `DailyDataset`은 daily/calendar/weather를 분리 보관하나 `GlobalLGBM`은 단일 병합 프레임을 요구한다. 따라서 fit/predict 전에 cli의 검증된 조립을 재사용한다:
```python
enriched = add_weather_features(add_calendar_features(daily, calendar), weather)
```
daily는 이미 `potential_demand`/`sold_units`/`is_stockout` 컬럼 보유(loader). 드라이버 컬럼(is_rain/is_snow/is_weekend/is_off_day/is_public_holiday)은 add_calendar/weather가 추가. predict는 calendar/weather feature를 프레임에서 그대로 사용하고 lag/rolling은 history에서 재계산하므로, **드라이버 컬럼 override가 예측에 반영**되고 lag/rolling은 불변(leakage 안전).

### D3. ceteris paribus 재예측 + 복수 드라이버
before/after는 **같은 fitted 모델 + 드라이버 컬럼만 차이**. 나머지 feature(lag/rolling 등)는 기존 period row 값을 재사용(새 계산 없음 → leakage 안전). 델타 = 순수 드라이버 민감도.
- **복수 드라이버 동시 변경 지원** (`driver_overrides: dict[str, float]`). LightGBM 트리가 상호작용을 학습하므로 복합 시나리오("비 오는 휴일")의 효과가 자동 반영된다 — 팔란티어 Scenario의 정통(여러 가정 동시). 단일 드라이버는 dict 1원소 특수 케이스.
- **비논리 조합 차단 안 함** (검증 로직 복잡도 회피). 대신 perturb 조합이 train에 존재했는지 카운트 → 0이면 `out_of_support=True` 플래그로 외삽을 정직하게 경고(메모리 LightGBM 외삽 약점 노출).
- **기여 분해 안 함** (YAGNI — before/after 총 델타만).

### D4. 읽기 전용 (commit 분리)
`what_if_driver`는 state를 mutate하지 않는 분석 함수(`side="read"`). Scenario→commit closed-loop(결과를 S5 commit_order로 발주 확정)은 Stretch로 분리.

### D5. caller 주입 (결정론·leakage 명시)
- `train_cutoff`를 caller가 주입한다 — fit은 cutoff 이전 데이터로만(leakage 절대규칙). 시그니처에 드러나 숨은 split이 없다.
- `base_order`를 caller가 인자로 준다(before_demand 기반 정책 자동발주가 아니라 명시 발주량). 단순·명시.
- LGBM seed 고정 → predict 결정론. 같은 cutoff면 모델 캐시 재사용.

## 아키텍처

```
what_if_driver(daily, store_id, item_id, period, driver_overrides, *, base_order, train_cutoff, ...)
  1. model = _fit_demand_model(daily, train_cutoff, feature_set="v2")   # cutoff 이전만, 1회 캐시, seed 고정
  2. base_row  = period의 (store_id, item_id) feature row(들)            # 실측 드라이버
  3. before_demand = _predict_demand(model, base_row)
  4. pert_row  = base_row 복사 후 driver_overrides 적용                 # 드라이버 컬럼만 변경
  5. after_demand  = _predict_demand(model, pert_row)
  6. 하류 전파: simulate_item_risk(before_demand, base_order) vs (after_demand, base_order)
  7. out_of_support = _count_support(daily, store_id, driver_overrides) == 0
  8. propagation_path = _propagation_path(driver_overrides)             # 탄 OntologyLink
  → WhatIfDriverResult
```

## 컴포넌트

새 모듈 `src/bakery/ontology/scenario.py` (functions.py 비대화 방지 + forecast wiring 독립 책임):

| 컴포넌트 | 책임 |
|---|---|
| `WhatIfDriverResult` (frozen) | store_id, item_id, driver_overrides, before/after_demand, demand_delta, before/after_p_stockout, before/after_expected_cost, out_of_support, propagation_path |
| `what_if_driver(daily, store_id, item_id, period, driver_overrides, *, base_order, feature_set="v2", train_cutoff, risk=RiskParams())` | 오케스트레이션(위 1~8). 반환 WhatIfDriverResult |
| `_fit_demand_model(daily, train_cutoff, feature_set)` | cutoff 이전 데이터로 GlobalLGBM fit, seed 고정, (cutoff, feature_set) 키로 캐시 |
| `_predict_demand(model, row)` | feature row predict → demand float (period 다중 row면 합 또는 평균; 기존 `_item_demand_points` 집계와 일치) |
| `_count_support(daily, store_id, driver_overrides)` | 해당 store에서 perturb 조합과 일치하는 과거 row 수 |
| `_propagation_path(driver_overrides)` | weather 드라이버(is_rain/is_snow)→`dailysales_observed_on_weather`→`item_sold_as_dailysales`; calendar(is_weekend/is_off_day/is_public_holiday)→`dailysales_observed_on_calendar`→`item_sold_as_dailysales` |

등록:
- `FUNCTION_REGISTRY["what_if_driver"]` — `OntologyFunctionSpec(..., side="read")` (impl=scenario.what_if_driver)
- `grounding/tools.py` TOOL_SPECS + dispatch — LLM 노출. `driver_overrides`는 object, 키 enum = `["is_weekend","is_off_day","is_public_holiday","is_rain","is_snow"]`, 값 number, `additionalProperties: false`. (demand_diff_by_condition의 enum 패턴 답습)

## 데이터 흐름 (예시)

```
질문: "광교 다음 주에 비가 오면 크루아상(P012) 발주(48)의 매진 위험이 어떻게 되나?"
→ what_if_driver(daily, "광교", "P012", [7/6,7/12], {"is_rain": 1}, base_order=48, train_cutoff="2026-07-05")
→ before_demand=45, after_demand=42 (비 오면 -3), demand_delta=-3
   before_p_stockout=0.18 → after_p_stockout=0.09 (수요↓ → 매진위험↓)
   out_of_support=False, propagation_path=("dailysales_observed_on_weather","item_sold_as_dailysales")
→ 에이전트(grounded): 진짜 수치로 답 + NL 해석
```

## 에러 / 엣지

- **빈 base_row** (period에 해당 (store,item) 없음) → ValueError (명확 실패; 조용한 빈 예측 금지).
- **알 수 없는 driver 키** (5개 enum 외) → ValueError. (LLM 노출은 enum으로 강제되나 직접 호출 방어)
- **train_cutoff 이후 데이터 없음 / fit 불가** → ValueError.
- **out_of_support=True**: 차단하지 않고 결과 반환 + 플래그 (외삽 정직 경고).

## 테스트

- **결정론**: 같은 입력 → 같은 before/after (seed 고정 확인).
- **델타 방향**: 소형 합성 데이터로 드라이버 perturb 시 demand 변화 → risk/cost 전파 검증.
- **복수 드라이버**: dict 2+ 원소 동시 변경 동작. 단일(1원소)도 동일 경로.
- **out_of_support**: train에 없는 조합 → True; 있는 조합 → False.
- **propagation_path**: weather 키 → weather 링크, calendar 키 → calendar 링크, 혼합 → 둘 다.
- **leakage**: `_fit_demand_model`이 train_cutoff 이후 row를 fit에 쓰지 않음 (scenario 전용 leakage 테스트 + 기존 `test_features_leakage`/`test_split_leakage` 통과).
- **엣지**: 빈 base_row/알 수 없는 키/cutoff 이후 데이터 없음 → ValueError.
- **dispatch**: grounding tools에서 what_if_driver 호출 → JSON 직렬화.

## 범위 밖 (TODO)

- **Scenario→commit closed-loop** (what_if_driver 결과 → S5 commit_order 발주 확정) — Stretch, 분리.
- 드라이버 기여 분해 (개별 효과) — YAGNI.
- 비논리 조합 차단 로직 — out_of_support 플래그로 대체.
- forecast 전면 교체 (proxy 폐기) — 국소 도입 유지 (D2).

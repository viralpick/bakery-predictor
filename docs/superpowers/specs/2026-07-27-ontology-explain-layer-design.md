# Spec 5b — 온톨로지 2층 설명 레이어 (explain_category_total / explain_item_order)

- 날짜: 2026-07-27
- 로드맵: Harness Backbone 5단계(온톨로지 + 액션레이어)의 **하위 스펙 5b**
- 선행: **Spec 5a — Forecast Substrate**(✅ PR#58 머지 adcfdde). 본 스펙은 5a의 `ForwardForecast` 중간값을 소비.
- 상위 스펙: `docs/superpowers/specs/2026-07-24-harness-backbone-design.md` §10-(B)
- 관련 메모리: `project_harness_backbone`, `project_poc_v7_aos_demonstrator`, `project_modeling_v4`, `project_measurement_charter`

---

## 1. 배경 / 목적

architect의 5단계 핵심 요구 = **"왜 이 수량?"을 총량·품목 두 층위로 온톨로지가 추적/설명**. 두 예시:
1. "크리스마스 총량 왜 이렇게?" → 카테고리 총량 = base 예측 + event_prior 보정
2. "다음주 목요일 팥빵 왜 K개?" → 총량 × 품목 비중(품절/추세 반영) = qty, **수식까지**

5a가 forward 2층 예측을 `forecast_forward()` seam으로 추출하고 **중간값**(base/prior 총량 + `adj_*` 비중 factor)을 `ForwardForecast`로 노출했다. 5b는 이 중간값을 **분해 서술**하는 grounded 설명 함수 2개 + 도구 등록 + Q셋 delta 측정을 얹는다. **새 모델링 없음** — 5a 엔진이 낸 실제 값을 충실히 분해할 뿐.

### v7 thesis 연결
Flagship = **grounding 정확도 delta**([[project_poc_v7_aos_demonstrator]]): 온톨로지 붙인 에이전트(진짜 분해값) vs 안 붙인 LLM(추측)의 정답률 차이. 5b의 두 설명 질문이 이 delta에 직결된다.

---

## 2. 목표 / 비목표

### 목표
1. `explain_category_total`·`explain_item_order` — `ForwardForecast` 소비, 실제 중간값 분해(신규 모델링 0).
2. 두 함수를 `ToolSpec`로 grounded 에이전트에 노출.
3. Q셋 2개(`q_explain_total`, `q_explain_item`) + 결정론 gold + forward period 컨텍스트 + live delta 측정.

### 비목표 (명시적 제외)
- **새 예측/모델링** — 5a seam이 유일 엔진. 5b는 서술 레이어.
- **기존 `explain_order`(v6 apply_policy 경로) 변경** — 다른 발주 철학(demand+퍼센트 safety+floor+rounding), 별개 도구로 유지. 5b 체인은 seam `our_order` 경로(분위수 버퍼)를 분해.
- **scenario/what_if_driver 재배선** — 5a에서 제외한 것 계속 제외.
- **synthetic 소스 explain** — forecast_forward가 real 지향(광교). synthetic 데모는 범위 밖.
- **다중 카테고리** — 단일 빵 총량 전제(5a와 동일).

---

## 3. 설계 — 통합 단일 체인 (faithfulness 확정)

### 3.1 핵심: seam our_order 경로를 분해 (apply_policy 이중계상 금지)
seam의 `our_order` = `distribute_total(prior_prod)` — **분위수 버퍼(q0.85)가 이미 카테고리 총량에 포함**. 그 위에 `apply_policy`의 퍼센트 safety를 얹으면 **안전마진 이중계상**이 된다. 따라서 `explain_item_order`는 apply_policy가 아니라 seam 경로를 그대로 분해한다.

```
explain_item_order(store, item, horizon):
  prior_median  (카테고리 q0.5 수요)              예: 250   ← ForwardForecast.category_totals
  + 분위수 버퍼 (prior_prod − prior_median)         +50     ← q0.85 production safety (총량 레벨)
  = prior_prod  (카테고리 생산총량)                 300
  × proportion_i (품목 비중)                        0.20    ← base_sold×adj_trend×adj_stockout×adj_closing×adj_new / Σ
  = our_order_i (품목 생산량)                        60      ← ForwardForecast.item_quantities.our_order
  × round_unit=3 ceil (배수 라운딩)                  →60     ← 3/6/9 생산제약 (아띠제 배수생산)
  = final_item_order                                60
```
- 보존식: 체인 각 단계 재구성 == 최종. (총량 레벨은 `prior_median + 버퍼 = prior_prod`, 품목 레벨은 `prior_prod × proportion = our_order`, `ceil_3(our_order) = final`.)
- 배수 라운딩: 기존 `decision/policy.py::_round_up_to_unit(qty, unit)` 재사용, `round_unit=3`(3/6/9). 라운딩은 마지막 단계.

### 3.2 explain_category_total
```
explain_category_total(store, horizon):
  base_median   (Stage1 예측, event_prior 이전)      ← ForwardForecast.category_totals.base_median
  + event_prior 보정 (prior_median − base_median)     ← 특수일 레벨-앵커 실제 기여 (blend 후−전)
  = prior_median  (보정된 카테고리 수요)
  + 분위수 버퍼 (prior_prod − prior_median)
  = prior_prod  (카테고리 생산총량)
```

### 3.3 ★faithfulness — 실제 블렌드값, "룰" 금지 (advisor 원칙 계승)
`event_prior` 기여는 `prior_median − base_median`(엔진이 실제 낸 blend 차이)이지 **"크리스마스=고정 N개 룰"이 아니다**. event_prior는 pre-test 히스토리로 fit한 레벨-앵커 블렌드다. 설명 함수는 이 차이를 **실제 수치**로 서술하며, "룰은 N"이라는 이상화 라벨을 만들지 않는다. (5a §3.4·§5.3 계승 — grounding이 없애려는 fabrication을 설명 레이어가 만들면 안 됨.)

### 3.4 위치 / 출력
- 신규 `src/bakery/ontology/explain.py` (functions.py ~200줄이라 설명 책임 분리).
- 반환 = lineage 스타일 DataFrame(단계별 행: `step`, `value`/`contribution`, `detail`) — 기존 `DecisionLineage.to_records()`와 유사 형태로 CSV/드릴다운·LLM 서술 친화.
  - `explain_category_total` → [store_id, date, step, value, detail] (step: base_median / event_prior / prior_median / quantile_buffer / prior_prod)
  - `explain_item_order` → [store_id, item_id, date, step, value, detail] (step: category_total / proportion / item_order / batch_rounding / final)
- horizon: forward 대상(다가오는 K일). 함수가 `forecast_forward(store, horizon_days=...)`를 호출하고 요청 date로 슬라이스.

---

## 4. 설계 — 도구 등록 (grounded surface)

`grounding/tools.py`의 `TOOLS` 리스트에 `ToolSpec` 2개 추가:
- `explain_category_total` — "카테고리 총량 예측 분해: base 예측 → event_prior 보정 → 분위수 버퍼."
- `explain_item_order` — "품목 생산량 분해: 카테고리 총량 × 품목 비중(factor) → 배수 라운딩."

`dispatch`/`_call`에 두 함수 배선. 인자 스키마: store_id·item_id·date(또는 horizon). ToolSpec 인자가 데이터 컬럼/enum이면 명시([[feedback_llm_tool_schema]] — enum·strict nested required).

---

## 5. 설계 — Q셋 delta 측정

### 5.1 신규 질문 2개
`grounding/questions.py::QUESTIONS`에 추가:
- `q_explain_total` (NUMERIC) — "다음주 이 매장의 카테고리 생산총량은? 어떻게 나왔나?" → gold = `explain_category_total` 최종 prior_prod(결정론).
- `q_explain_item` (DECOMPOSITION) — "다음주 [특정 품목] 생산량은? 총량·비중으로 분해하면?" → gold = `explain_item_order`의 item_id + final_qty. **item 선택 결정론**: 기존 `q_order_top` 패턴 차용 — forward 대상 품목 중 결정론 규칙으로 1개 고정(예: our_order 최대 품목, 또는 rank_stockout_earliness top-1). Q 정의에 그 규칙을 fn_kwargs로 명시(수동 라벨 0).

grader 재사용(YAGNI): NUMERIC(rel tol), DECOMPOSITION(item_id + qty). 신규 grader 없음.

### 5.2 forward period 컨텍스트 (★설계 관건)
기존 `build_gold`의 `_ctx`는 **과거(historical) period**(eval-gold, min~max). 그러나 두 신규 질문은 **forward**(다음주 생산 설명)라 forward horizon이 필요하다.
- gold = **함수의 결정론 출력**(관측 진실 아님). `forecast_forward(use_forecast=False, 고정seed)`가 결정론이므로 forward horizon에서도 gold 재현 가능.
- `build_gold`에 두 신규 source_fn 분기 추가 시, `_ctx` 대신 **forward horizon 컨텍스트**를 쓴다(마지막 관측일 다음 K일). 이 forward 컨텍스트 헬퍼를 questions.py에 추가(예: `_forward_ctx(dataset, horizon_days)`).
- ⚠️ 기존 11개 질문의 `_ctx`(historical) 경로는 무변경 — guarded fallback(5a §5.3)과 정합. 신규 2개만 forward 컨텍스트.

### 5.3 arms (with/without AOS)
기존 `arms.py` 구조 재사용: grounded(도구 호출→진짜 분해값) vs rag_only(OntologyKnowledge만, 추측). 신규 질문도 동일 2-arm. OUTPUT_SCHEMAS에 신규 질문 응답 스키마 추가(numeric/decomposition).

---

## 6. Acceptance 기준

1. **faithfulness (reconcile)**
   - `explain_item_order` 최종(라운딩 전) == seam `our_order`(동일 store·item·horizon).
   - factor 곱 정규화 == `proportion`, `explain_category_total` prior_prod == seam `category_totals.prior_prod`.
   - event_prior 기여 == `prior_median − base_median`(실제 blend 차이, "룰" 라벨 없음).
2. **보존식**: 체인 각 단계 재구성 == 최종(총량·품목 레벨 각각).
3. **grounding 무회귀 + delta**: 기존 11 Q 통과, 신규 2 Q 추가. gold 결정론(수동 라벨 0). CI는 키 없이 통과(LLM mock/skip, Q셋·gold·채점 단위테스트). live delta: grounded > rag_only.
4. **기존 explain_order(v6) 무변경**: apply_policy 경로 테스트 그대로 green.
5. 전체 pytest 통과.

---

## 7. 리스크

- **이중계상 재발**: 누군가 explain_item_order에 apply_policy safety를 얹으면 분위수 버퍼와 중복 → reconcile 테스트(최종==our_order)가 잡는다. 체인은 seam 경로만.
- **forward 컨텍스트 gold 비결정성**: forecast_forward가 결정론이어야 gold 재현. use_forecast=False + 고정 seed 확인(5a에서 확립). LightGBM fit 결정성 회귀 시 gold 흔들림 → 게이트로 고정.
- **functions.py↔explain.py 순환/중복**: explain.py는 forecast_forward(forecast 패키지)·item_proportion 상수를 직접 소비, functions.py를 import하지 않도록. `_forward_demand_points`와 로직 겹치면 공유 헬퍼로.
- **per-call LightGBM refit**: explain 함수도 매 호출 forecast_forward fit(5a 백로그). Q셋 eval에서 반복 호출 시 느림 → 5b에서 결과 캐싱 검토(또는 후속).
- **ToolSpec 스키마 오류는 CI 못 잡음**: live smoke 필수([[feedback_llm_tool_schema]]).
- **round_unit=3 가정**: 아띠제 실제 배수(3/6/9)가 품목별로 다를 수 있음 → 라벨된 가정으로 두고 실데이터 수령 시 조정(품목별 배수는 후속).

---

## 8. 파일 요약

- Create: `src/bakery/ontology/explain.py` (설명 함수 2개)
- Modify: `grounding/tools.py`(ToolSpec 2 + dispatch), `grounding/questions.py`(Q 2 + forward_ctx + build_gold 분기), `grounding/arms.py`(OUTPUT_SCHEMAS), `grounding/constants.py`(필요 시)
- Create: `tests/test_ontology_explain.py`(reconcile·보존식), `tests/test_grounding_*.py` 확장(신규 Q gold·채점)

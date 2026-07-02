# q_order_top Q셋 재설계 (rank→explain 체인, 관측 매진시각 기반) — Design

**Date**: 2026-07-02
**Depends on**: S3 grounding eval (questions/tools/arms/scorer), v6 decision layer (apply_policy), data schema `stockout_time`
**Status**: Design approved (2026-07-02)

## 목적

grounding eval의 `q_order_top`("상위 품목의 권장 발주량은?")은 gold가 `_ctx`의 **매출(판매량 합) 1위** 품목 기준인데, grounded arm에는 매출 top을 식별할 도구가 없어(노출된 ranking 도구 = 매진위험) 구조적으로 풀 수 없는 질문이다. 이를 **"관측 매진 위험 1위 품목의 발주 분해"**로 재설계해 `rank → explain` 2-step 도구 체인 질문으로 만든다.

핵심 원칙(사용자 결정): **매진 위험 ≠ 판매량**. 다른 품목이 다 팔려서 결과적으로 많이 팔린 품목(예: 하루종일 팔리는 식빵)은 위험이 높은 게 아니다. **매진이 얼마나 일찍, 얼마나 자주 일어나는지**(관측 `stockout_time`)가 위험의 실체다. 이는 PoC 운영 KPI(매진 time median↑ / 매진률↓)와도 정렬된다.

## 핵심 설계 결정 (LOCKED)

### D1. 위험 점수 = 일평균 손실 영업시간 (관측 기반)
품목별 점수 = 기간 내 **전체 일**에 대한 `mean(max(close_hour − stockout_time의 시각(시간 float), 0))`. 매진 없는 날은 0 기여. 조기 매진일수록·자주 매진될수록 점수↑ — 빈도(매진률)와 조기성(매진 시각)을 단일 수치로 통합. 매진일-only median(조기성만), 매진률 우선 사전식(두 축 분리) 대안은 기각.

### D2. 새 함수 추가 (기존 rank_stockout_risk 무변경)
`rank_stockout_earliness`를 신규 OntologyFunction으로 추가한다. 기존 `rank_stockout_risk`(MC, P(수요>발주), 전방향)는 그대로 둔다 — **관측(이력) 도구와 예측(MC) 도구의 의미론 분리**. 기존 q_rank_top1/3/5·테스트·S5 closed-loop 전부 무접촉. 한 함수에 `basis` 모드 인자를 넣는 안은 LLM 인자 오용 위험(q_diff_offday 교훈)으로 기각.

### D3. close_hour = 라벨된 가정 (기본 22)
close 시각은 DailyDataset에 없다(daily에 `open_hours` 수만 존재). `close_hour: int = 22` 파라미터로 받는다 — bonavi 실데이터 loader 하드코딩(22)과 동일, synthetic `_ctx` 선정 매장 store_A도 22. `waste_cost`의 `unit_cost=1.0`과 같은 "라벨된 가정" 패턴. `max(..., 0)` clamp로 close 이후 stockout_time(실데이터 이상치)도 안전.

### D4. 결정론 gold + 비퇴화 가드
- 정렬 키 `(lost_hours_per_day desc, item_id asc)` — 동률 시 안정적(결정론 gold 필수조건).
- 기간 내 매진이 전무하면(전 품목 점수 0) "위험 1위"가 무의미 → `ValueError` (기존 non-finite gold 가드와 같은 정신).

### D5. 채점 = item_id + order_qty 둘 다
decomposition 답 스키마를 `{"item_id": string, "order_qty": number}`로 확장하고 **둘 다 맞아야 정답**(item exact match AND qty 1e-6). 수요가 비슷한 다른 품목을 집고도 qty 우연 일치로 통과하는 것을 차단하고, 실패 시 rank 단계/explain 단계 진단이 가능하다.

### D6. Q셋 범위 = q_order_top 재설계 + 신규 rank Q 1개 (10→11문항)
- **q_order_top(재설계)**: "관측 매진 위험(일평균 손실 영업시간 기준) 1위 품목은 무엇이고, 그 품목의 권장 발주량은?" — grounded arm은 `rank_stockout_earliness(k=1)` → `explain_order(item)` 2-step 체인 필요. **유일한 멀티스텝 tool-use 검증 질문**이 된다.
- **q_rank_earliness(신규)**: "일평균 손실 영업시간 기준 매진 위험 상위 3개 품목은?" — 새 함수 단독 검증. 체인 실패 시 원인 분리(rank가 문제인지 explain이 문제인지). 기존 ranking 채점기(top-1 match) 재사용.
- 점수 정의를 질문 텍스트에 명시 — 양 arm에 동일 텍스트가 가므로 공정성 유지(rag-only도 정의는 알지만 수치는 못 구함).
- 기존 q_rank_top1/3/5는 MC 도구 검증으로 유지. `_ctx`의 `top_item`(매출 1위)은 q_order_top에서 더 이상 사용 안 함.

## 구현 세부

### functions.py
```
rank_stockout_earliness(daily, store_id, period, k=5, *, close_hour=22)
→ DataFrame[item_id, lost_hours_per_day, stockout_days, days]
```
- `_period_slice` 재사용. 품목별: `stockout_time` notna 행에서 `hour+minute/60` 추출 → `max(close_hour − t, 0)` → 전체 일수로 나눈 평균(매진 없는 날 0).
- FUNCTION_REGISTRY 등록 (`side=read`).

### grounding/tools.py
- ToolSpec 추가: 설명에 점수 의미 명시 — "Top-k items by observed stockout earliness: average selling-hours lost per day to stockouts (higher = stocks out earlier/more often). Historical observation, not a forecast." 인자 = store_id/period/k (모호 인자 없음).
- `_call` dispatch 1분기 추가.

### grounding/questions.py
- `q_order_top` 텍스트·gold 교체: `build_gold`가 `rank_stockout_earliness(k=1)`로 1위 품목 결정 → `explain_order(그 품목)` lineage 합 → `{"item_id": ..., "order_qty": ...}`.
- `q_rank_earliness` 추가 (RANKING, `rank_stockout_earliness`, k=3).
- `_ctx`는 유지하되 q_order_top 경로에서 `top_item` 미사용.

### grounding/arms.py + scorer.py
- `OUTPUT_SCHEMAS[DECOMPOSITION]`에 `item_id` 추가 (required 전키 — strict schema 규칙).
- `_grade_qty` → item_id exact match AND qty `abs ≤ 1e-6`.

## 테스트

- 새 함수: 결정론(동률 tie-break item_id asc), 매진 0 기간 가드 raise, 매진 없는 날 0 기여, close_hour 경계 clamp, 알려진 소형 fixture로 손계산 일치.
- build_gold: rank(k=1)→explain 체인이 결정론적으로 같은 품목·수량 반환.
- scorer: item 맞고 qty 틀림 / qty 맞고 item 틀림 / 둘 다 맞음 케이스.
- 기존 전체 테스트 무변경 통과 (additive 확인).
- **live smoke 필수** (schema 오류는 CI 못 잡음 — feedback 메모리): grounding-eval 1회 실행, q_order_top에서 grounded arm의 실제 2-step 체인 + q_rank_earliness 정답 확인.

## 범위 밖

- q_rank_top1/3/5의 관측 기반 전환 (MC 도구 검증 커버리지 유지).
- close_hour의 매장별 실측화 (실데이터 운영 시 store metadata로).
- MC 희소품목 P매진 100% 아티팩트 수정 (별건).

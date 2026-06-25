# Closed-Loop Order Recommendation (v7 S5) — Design

**Date**: 2026-06-25
**Branch**: feat/v7-ontology-cleanup (or new feat/v7-closed-loop)
**Depends on**: S1 ontology schema, S2 OntologyFunction registry, S3 grounding arms (tool-loop), S4 CUD writeback
**Status**: Design approved (2026-06-25)

## 목적

v7 AOS demonstrator의 closed-loop을 완성한다: **grounded 에이전트가 발주를 추천 → 사람 승인 게이트 → writeback commit → 확정 발주시트**. S3(grounding)와 S4(writeback)를 결합해 "온톨로지+AOS가 붙으면 에이전트가 *행동까지* 한다"는 thesis를 메커니즘으로 시연한다.

## 핵심 설계 결정 (LOCKED)

### D1. 쓰기 주도 = 하이브리드 (옵션 C)
LLM은 **read 도구만** 호출하고(state를 직접 mutate하지 않음), grounded 수치로 추론해 **구조화된 발주 제안**을 emit한다. 실제 쓰기(propose→commit)는 결정론 seam(`WritebackStore`)이 사람 승인 게이트를 통과한 뒤 수행한다.

- **AOS 구조와 동형**: 에이전트 read-only + 쓰기 app-api 위임 + commit 사람 게이트(완전 자율은 frontier).
- **leakage/안전 보존**: LLM이 상태를 직접 못 바꾼다.

### D2. LLM 수량 조정 = 암묵지 흡수 통로 (rationale)
LLM은 `rank_stockout_risk`의 모델 추천 수량을 **판단으로 조정**할 수 있다(예: 48→52). 이 조정 지점이 **feature화 불가능한 암묵지**(인근 행사, 발주 담당자 경험칙 등 모델이 모르는 맥락)가 시스템에 들어오는 통로다. kinetic layer 분석의 "암묵지 rule형" 갈래와 연결된다.

### D3. 정확도 주장 분리
closed-loop은 **메커니즘 시연**이지 정확도 측정이 아니다. 추천 수량의 비결정론에는 정확도 주장을 하지 않는다(정확도는 S3 grounding eval이 담당). 산출물에 "synthetic 메커니즘 시연" 라벨을 단다.

### D4. write OntologyFunction 등록 = `side` 마커
읽기 레지스트리는 `OntologyFunctionSpec(impl: Callable(df, args))` 형태로 stateless다. writeback은 stateful(`WritebackStore` 인스턴스 바인딩)이므로, `OntologyFunctionSpec`에 `side: "read" | "write"` 마커를 추가하고 write 함수(`propose_order`, `commit_order`)는 store를 주입받아 등록한다. **이들은 orchestrator가 호출하며 LLM 도구 surface에는 노출하지 않는다**(D1). 목적: 온톨로지의 함수 surface를 완성하고 lineage를 추적 가능하게 함.

### D5. 게이트 = 정책 주입 (인터랙티브 아님)
사람 승인 게이트는 `GatePolicy` 추상화(`(proposal) → GateDecision`)로 주입한다. 실시간 콘솔 prompt는 테스트성을 해치므로 기본은 정책. 내장 정책:
- `auto_approve` — 즉시 APPROVED (autonomous, frontier 연출; `require_approval=False` 경로)
- `approve_as_proposed` — 사람이 제안대로 승인 (rubber-stamp)
- `human_correct(corrections)` — 특정 품목 수량을 보정해 승인 (사람 보정 시연)

## 아키텍처

```
recommend agent  (LLM + read 도구, grounded)
      │  OrderProposal[]  (item_id, qty, rationale)   ← 새 OUTPUT 스키마
      ▼
run_closed_loop (loop.py 오케스트레이터)
      │  각 제안 검증 → WritebackStore.propose_order (PENDING)
      ▼
GatePolicy  approve / correct / reject  (require_approval 토글)
      │
      ▼
WritebackStore  APPROVED → confirmed_as_of → 확정 발주시트 (parquet 선택)
```

## 컴포넌트

| 컴포넌트 | 위치 | 책임 |
|---|---|---|
| `recommend_orders(client, dataset, store, period)` | `ontology/grounding/arms.py` (arm 추가) | read tool-loop 실행 → `OrderProposal[]` emit. 새 OUTPUT 스키마(proposals 배열). 기존 grounded tool-loop 인프라 재사용 |
| `OrderProposal` (frozen) | `ontology/loop.py` | `item_id: str, qty: float, rationale: str` |
| `GateDecision` (frozen) | `ontology/loop.py` | `action: APPROVE\|REJECT, approved_qty: float\|None, approver: str` |
| `GatePolicy` (Protocol/Callable) | `ontology/loop.py` | `(OrderProposal) → GateDecision`. 내장 3종(D5) |
| `run_closed_loop(client, dataset, store, period, writeback, gate, *, now)` | `ontology/loop.py` | 추천→검증→propose→게이트→commit. 확정 `WritebackStore` 상태 반환 |
| write 함수 등록 | `ontology/functions.py` | `OntologyFunctionSpec`에 `side` 필드 추가 + `propose_order`/`commit_order` write 스펙 등록 |
| CLI `closed-loop` | `cli` | end-to-end 실행, 제안·게이트 결정·확정시트 출력, `--out` parquet 저장, `--autonomous` 토글 |

## 데이터 흐름 (예시)

```
$ uv run bakery closed-loop --store 광교 --period 2026-07-06,2026-07-12 --policy approve_as_proposed
1. recommend_orders: LLM이 rank_stockout_risk/what_if/waste_cost 호출(grounded)
   → 제안 5건: [{P012, 52, "p_stockout 0.82, what_if +4 폐기 미미..."}, ...]
2. 각 제안 검증(qty 유한·비음수, item_id ∈ dataset) → propose_order(PENDING)
3. gate(approve_as_proposed) → 5건 APPROVED
4. confirmed_as_of(now) → 확정 발주시트 출력 + (--out 시) parquet
```

## 에러 / 엣지

- **무효 제안**(qty 비유한/음수, 미존재 item_id) → skip + log (S3 per-question skip 동형). 전량 무효면 빈 시트 + 경고.
- **게이트 거부** → REJECTED 레코드 (commit 안 됨).
- **LLM 도구 에러** → 기존 dispatch가 JSON error로 surface (S3 동일).
- **타임스탬프** → caller 주입(`now`), `WritebackStore`는 `datetime.now()` 안 부름 → 결정론.

## 테스트

- **결정론 (FakeLLM)**: canned `OrderProposal[]` → orchestrator 검증 → propose→gate→commit 상태전이, `confirmed_as_of` 스냅샷, parquet round-trip(closed-loop 레코드 포함).
- **게이트 정책 3종**: auto_approve / approve_as_proposed / human_correct 각각 결정 검증.
- **검증 가드**: 무효 제안 skip, 전량 무효 빈 시트.
- **live (skipif no key)**: 실제 gpt-5-mini로 1회 end-to-end smoke (blocker 아님).

## 범위 밖 (TODO)

- **상류 레버 `what_if_driver`** (날씨/캘린더 드라이버 가상변경 → forecast 재실행 → link 전파, 팔란티어 Scenario 동형). **⚠️ PoC 기간 내 필수** (Stretch 아님). forecast wiring 선행 필요 → 별도 작업으로 분리. TODO.md에 기록.
- 인터랙티브 콘솔 게이트 (필요 시 CLI 플래그로 후속).
- 실데이터 발주 포맷 통일(아티제 ① 포맷) — 별도 작업.

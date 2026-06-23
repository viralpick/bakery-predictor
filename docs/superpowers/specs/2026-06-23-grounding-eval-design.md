# Grounding Eval (S3) — with/without AOS 증분가치 측정 설계

> v7 PoC LayerB Flagship 지표(`docs/poc_scope_v7.md` §5.1, D2)의 구현 설계.
> 작성일: 2026-06-23. 선행: ontology mock(S1, `src/bakery/ontology/schema.py`) +
> OntologyFunction 래퍼(S2, `functions.py`)가 머지되어 있어야 한다.

---

## §1. 목적

PoC 회사 thesis = **"온톨로지 + AOS를 붙이면 안 붙였을 때보다 낫다"**. 이를
*예측 정확도(WAPE)*가 아니라 **grounding 정확도**로 측정한다(§0 함정: AOS는
예측 코어를 안 건드리므로 WAPE로 재면 "차이 없음"이 나와 thesis를 반증한다).

발주 분석 질문을 사전등록하고, 두 arm에 같은 질문을 던져 정답률을 비교한다:

- **Grounded arm** — 온톨로지의 OntologyFunction을 *도구(tools)*로 노출. 모델이
  함수를 호출해 **진짜 집계값**을 받아 답한다.
- **RAG-only arm (without-AOS)** — 같은 모델에 OntologyKnowledge 청크(개념·산식)만
  컨텍스트로 주고, 함수·관계는 주지 않는다. 모델은 **추측**으로 답한다.

**delta = grounded 정답률 − rag_only 정답률.** 이 delta가 "온톨로지+함수 레이어의
증분가치"다.

---

## §2. 핵심 결정 (LOCKED)

| # | 결정 | 근거 |
|---|---|---|
| D1 | LLM provider = **OpenAI(`gpt-5-mini`)**, 단 **provider 전환 가능 설계** | Bedrock 경유 Anthropic이 한국에서 현재 미발급. 회사는 양사 모두 사용 → 향후 Anthropic 도입 시 어댑터만 추가 |
| D2 | without-AOS baseline = **LLM + RAG**(OntologyKnowledge만, 함수·관계 없음) | 맨 LLM보다 공정·엄격. "관계+연산"이라는 AOS 고유가치만 고립해 측정 (`poc_scope_v7.md` Q2) |
| D3 | 답은 **structured JSON으로 강제**(자유텍스트 채점 아님) | 양 arm 동일 출력 스키마 → 파싱 없이 결정론 채점, 채점 신뢰성 확보 |
| D4 | gold(정답)는 **OntologyFunction 직접 호출로 결정론 생성** | 수동 라벨링 0, 체리피킹·조작 불가 (공정성 §7) |
| D5 | Q셋 규모 = **~12개**(함수 5개 × 2~3) Core 티어 | 라벨링 부담↓, 함수 커버리지 확보 |

---

## §3. 아키텍처

```
questions.py / arms.py / scorer.py   ← provider를 전혀 모름 (중립 타입만 사용)
        │ uses
LLMClient (Protocol)
   generate(messages, tools: list[ToolSpec], output_schema) → LLMResponse
        ├── OpenAIClient      gpt-5-mini   [지금 구현]
        └── AnthropicClient   claude-*     [미래: 어댑터만 추가]

Q셋(사전등록) ──┬─→ [Grounded arm]  모델 + OntologyFunction(tools) → 함수호출 → 진짜 수치 → 답
               ├─→ [RAG-only arm]  모델 + OntologyKnowledge 청크만 → 추측 → 답
               └─→ [Gold]          OntologyFunction 직접 호출 (결정론 정답)
                          ↓
                   scorer → arm별 정답률 → delta = grounded − rag_only
```

---

## §4. Provider 전환 설계 (1급 요구사항)

차이를 **어댑터 안에만** 가두고 상위 로직은 provider 무관하게 한다.

### 4.1 중립 타입 (`llm.py`)
- `ToolSpec(name, description, json_schema)` — provider 중립 도구 정의
- `Message(role, content, ...)` — 대화 메시지
- `ToolCall(id, name, arguments)` / `ToolResult(call_id, content)`
- `LLMResponse(text, tool_calls, parsed)` — `parsed`는 structured output 결과

### 4.2 어댑터가 흡수하는 차이
| 측면 | OpenAI | Anthropic(미래) |
|---|---|---|
| tool 정의 | `{type:"function", function:{...}}` | `{name, input_schema}` |
| structured output | `response_format`(json_schema) | `output_config.format` |
| 파라미터 | `temperature` 등 | `effort`/`thinking`(temperature 없음) |
| tool-call 파싱 | `tool_calls` 필드 | `content[].type=="tool_use"` |

### 4.3 공통 tool-loop
`arms.py`의 "함수 호출 → 결과 주입 → 재호출" 루프는 `LLMClient.generate`만
호출하고 provider 분기가 없다. 어댑터는 *한 턴*의 요청/응답 변환만 책임진다.

### 4.4 설정 / 팩토리
- `make_llm_client(provider, model) → LLMClient`
- env: `LLM_PROVIDER`(openai|anthropic), `LLM_MODEL`, `OPENAI_API_KEY`
- `OntologyFunctionSpec`(S2, 이미 존재) → `ToolSpec` 변환은 **한 곳**에만

---

## §5. Q셋 설계 (`questions.py`)

정답을 **집계로 자동검증 가능한 유형만** 1차 채택(NL 해석형은 범위 밖).

각 Q:
```python
@dataclass(frozen=True)
class Question:
    id: str
    text: str                 # 자연어 질문
    grader_type: str          # "numeric" | "ranking" | "decomposition"
    source_fn: str            # FUNCTION_REGISTRY 키
    fn_kwargs: dict           # gold 생성용 인자
    tolerance: float = 0.0    # numeric 허용오차(상대)
```

| grader_type | 출처 함수 | gold | 채점 |
|---|---|---|---|
| numeric | `demand_diff_by_condition`, `waste_cost` | 단일 수치 | `abs(pred−gold)/gold ≤ tolerance` |
| ranking | `rank_stockout_risk` | top-k 품목 리스트 | 집합 일치율(Jaccard) + top-1 일치 |
| decomposition | `explain_order` | 최종 발주량(+단계 존재) | 발주량 수치 일치, 단계명 포함 |

- **gold는 `source_fn`을 synthetic 데이터로 직접 호출**해 생성(결정론).
- 사전등록: Q셋을 `questions.py`에 **코드로 고정**(체리피킹 불가).
- 답 추출: 양 arm 모두 **structured output**으로 최종 답을 JSON 강제.
  - numeric → `{"answer_value": number}`
  - ranking → `{"top_items": [string]}`
  - decomposition → `{"order_qty": number}`

---

## §6. arms (`arms.py`)

- `run_grounded(client, question, dataset) → LLMResponse`
  - 시스템 프롬프트 + 온톨로지 스키마 요약 + OntologyFunction을 `ToolSpec`으로 노출
  - 모델이 함수 호출 → `arms`가 실제 `functions.py` 호출 → 결과 주입 → 최종 답
- `run_rag_only(client, question, dataset) → LLMResponse`
  - 시스템 프롬프트 + OntologyKnowledge 청크만. tools 없음. 추측으로 답
- 두 arm 모두 **동일 모델·동일 파라미터·동일 출력 스키마**. 도구 유무만 차이.

---

## §7. 채점 + 리포트 (`scorer.py`)

- `grade(question, response, gold) → bool`(유형별)
- `evaluate(client, questions, dataset) → EvalReport`
  - arm별 정답률, Q별 결과, **delta = grounded − rag_only**
- 리포트에 **"메커니즘 시연(measured on synthetic)"** 라벨 — 실데이터 측정 아님(§7 공정성).

---

## §8. 공정성 계약 (`poc_scope_v7.md` §7) 반영

1. **동일 모델·파라미터** — 양 arm 같은 client+model, 도구 유무만 차이.
2. **사전등록** — Q셋 코드에 고정, 실행 시 변경 불가.
3. **gold 결정론** — OntologyFunction 호출로 생성, 조작 불가.
4. **라벨 정직** — "synthetic 시연" 명시, 실데이터 *측정*과 구분.
5. **불변식** — provider 전환은 런 *간*에만. 한 런 내 양 arm은 동일 client·model.

---

## §9. 테스트 / 키 의존 분리

- **CI는 키 없이 통과해야 한다** → LLM 실호출 테스트는 mock 또는 skip.
- **키 없이 단위테스트 가능**(결정론이라): Q셋 로딩, gold 생성, `ToolSpec` 변환,
  채점기(유형별), 출력 스키마 검증.
- 실측 eval(키 필요)은 별도 run script / CLI로 분리. `OPENAI_API_KEY` 있을 때만.
- leakage 절대규칙 무관(이 레이어는 post-prediction read-only).

---

## §10. 파일 구조

```
src/bakery/ontology/grounding/
  __init__.py
  llm.py          # LLMClient Protocol + 중립 타입 + OpenAIClient + 팩토리
  questions.py    # Question dataclass + 사전등록 Q셋 + gold 생성기
  arms.py         # run_grounded / run_rag_only + 공통 tool-loop
  scorer.py       # 유형별 채점 + evaluate + EvalReport
  run.py          # 실측 eval 엔트리 (CLI 또는 함수, 키 필요)
tests/
  test_grounding_questions.py   # Q셋·gold·ToolSpec 변환 (키 불필요)
  test_grounding_scorer.py      # 유형별 채점 (키 불필요)
```

---

## §11. 범위 밖 (명시)

- NL 해석형 질문(집계 자동검증 불가) — 2차.
- Anthropic 어댑터 실구현 — 인터페이스만 맞춰두고 발급 후.
- 3-arm(맨 LLM 추가) — D2에서 LLM+RAG 단독 채택.
- what-if/CUD writeback(S4) — 별도.

---

## §12. Open Questions

| # | 질문 | 처리 |
|---|---|---|
| Q1 | `gpt-5-mini`의 tools + structured output 동시 사용 시그니처 | 구현 시 context7로 OpenAI SDK 확인 |
| Q2 | numeric tolerance 기본값(±%) | 구현 시 함수별 분포 보고 결정(초기 ±5%) |
| Q3 | ranking 채점을 Jaccard vs top-1 vs 둘 다 | 둘 다 리포트, 합격 기준은 top-1 우선 |

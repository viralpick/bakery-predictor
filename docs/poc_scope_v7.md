# PoC 범위 정의 v7 — AOS Value Demonstrator

> 내부 작업 기준 문서. v6(`docs/poc_scope_v6.md`) 이후, PoC의 *목적*을 재정의한다.
> 핵심 전환: **"수요를 잘 예측한다"가 아니라 "온톨로지 + AOS를 붙였을 때의 증분 가치를 보여준다".**
> 작성일: 2026-06-22. 선행 분석: `docs/kinetic_layer_fit_analysis.md` (§8 확률, §10 재중심화).

---

## §0. 왜 v7 — v6와의 misalignment

| | v6 (점추정 + 위험수치) | v7 (AOS Value Demonstrator) |
|---|---|---|
| 증명하려는 것 | "우리 ML이 기존 수기 발주보다 낫다" (ML vs 수기) | **"온톨로지 + AOS를 붙이면 안 붙였을 때보다 낫다" (AOS vs ML)** |
| 독립변수 | 우리 시스템 on/off | **AOS 레이어 on/off** |
| 위험 | 회사 thesis(AOS 증분가치)를 입증 못 함 | thesis 직격 |

**핵심 함정(반드시 회피)**: AOS의 결정/설명/시뮬 레이어는 예측 정확도를 거의 안 올린다(예측 코어가 양 arm 동일). 그러므로 **AOS를 WAPE로 평가하면 "차이 없음"이 나와 thesis를 *반증*한다.** → AOS가 *실제로 움직이는* 지표로 재야 한다 (§5).

---

## §1. AOS 온톨로지 실제 구조 (코드 확인된 사실)

"그래프 DB"가 아니라 **polyglot 실데이터(Mongo / Iceberg Lake / ClickHouse) 위에 얹은 의미·관계 메타데이터 그래프(Postgres)**. 3+1층:

| 층 | 무엇 | 역할 |
|---|---|---|
| 스키마/의미 | `OntologyObject` + `OntologyObjectStructure` | 객체(테이블)·필드의 뜻(displayName, description, PK/nameKey, 타입) |
| 관계(그래프) | `OntologyLink` (LinkType, Cardinality) | 객체↔객체 엣지: 방향·타입·**조인키·카디널리티(1:N)** |
| 지식(RAG) | `OntologyKnowledge` + `Chunk(pgvector)` | 개념·산식·SQL 지식 임베딩 검색 |
| 실데이터 | 객체 type별 분기 | Mongo/Iceberg/ClickHouse |
| 연산 | `OntologyFunction` | 파라미터화 재사용 쿼리 (에이전트가 데이터 위에 쌓는 안정적 API) |

- traversal = `OntologyLink` 조인키 → SQL JOIN / Mongo `$lookup` 컴파일.
- 에이전트는 온톨로지에 **read-only** (쓰기는 app-api 위임, CUD = frontier).
- **핵심 가치**: LLM이 추측(할루시네이션) 대신 *올바른 구조화 쿼리 → 진짜 값*을 받게 함 + 에이전트가 관계 그래프로 **자율 traversal/워크플로우**.
- **천장**: 메타데이터 큐레이션 품질 (description/Link가 부실하면 traversal 불가).

---

## §2. 통합 스택 — 네 조각이 어떻게 하나로

```
[Ontology]        의미·관계 지도 (item↔category↔store↔weather↔calendar↔stockout)
                  + Knowledge(산식) + OntologyFunction(재사용 연산)
      │  에이전트가 traverse·grounded 질의
[Forecast]        LightGBM 점추정 (학습 코어 — 온톨로지 무관)
      │
[Kinetic 효과]    예측→발주 derivation + decision lineage   ← 우리 decision layer (커밋 8d13157)
[Dynamic 효과]    scenario overlay + MC 위험                ← 우리 risk.py
      │  ↑ 이 둘을 OntologyFunction으로 등록
[AOS Agent]       온톨로지 traverse + Function 호출(=진짜 수치) + lineage/risk 읽고 자연어 reasoning
```

**관통 원칙**: "**수치 = 결정론 엔진, 해석 = LLM**". 이는 AOS 온톨로지 철학 = kinetic 철학(동일). 그래서 통합이 억지가 아니다. **우리가 만든 decision layer는 버려지지 않고 에이전트의 *호출 대상(OntologyFunction)*이 된다.**

---

## §3. PoC 구조 — 2층

두 종류의 증거를 *섞지 않고* 분리한다.

### Layer A — 하드 비즈니스 증거 (실데이터 전향)
- "ML이 수기보다 나은가" — **Arm 0 (기존 수기 발주) vs Arm 1 (ML 점추정 + 위험수치)**.
- 지표: 폐기비용↓ / 매진 median↑ / 매진률↓ (운영 KPI). = **기존 v6 그대로 유지.**
- 성격: 실세계 하드넘버. AOS와 무관하게 "예측의 가치"를 받친다.

### Layer B — AOS 증분가치 Demonstrator (자체 완결)
- 같은 예측 위에 **온톨로지 + 에이전트**를 얹어 **with/without AOS** 비교.
- 성격: 메커니즘·역량 *시연* (실세계 A/B보다 약한 증거 — 정직하게 라벨).

> 두 층을 명확히 구분해 발표: Layer A = "측정된 사실", Layer B = "시연된 메커니즘".

---

## §4. 결정사항 (LOCKED)

| # | 결정 | 근거 |
|---|---|---|
| D1 | **구현 = 베이커리 repo 안 self-contained mock** (실제 AOS 등록 아님) | 실제 AOS 등록은 객체등록·Gravitino 이슈·AOS팀 의존 → PoC 범위 초과. mock은 AOS와 **구조 동형**(OntologyObject/Link/Function 동일 형태)으로 만들어 반론 대응. |
| D2 | **Flagship 지표 = Grounding 정확도** | AOS 고유·하드넘버·thesis 직격·8주 현실적 (§5.1). |
| D3 | **2층 분리** (Layer A 실데이터 / Layer B 자체 demo) | 증거 종류 혼동 방지. |
| D4 | **decision layer(8d13157) = OntologyFunction 구현체로 재사용** | 기존 자산 활용, 통합 자연스러움. |

---

## §5. 지표 정의

### 5.1 Flagship — Grounding 정확도 (Layer B)
- 발주 분석 질문 **사전등록 Q셋(N개)** → **온톨로지 붙인 에이전트(진짜 집계값) vs 안 붙인 LLM(추측)** 정답률 비교.
- 예시 질문:
  - "지난 4주 광교에서 매진 위험 상위 5품목과 그 원인은?"
  - "비 오는 주말 평균 대비 빵 판매 차이는?"
  - "통팥빵 다음 주 발주 권장량과 그 근거(분해)는?"
- 채점: 정답(실데이터 집계로 검증 가능) 대비 정확/할루시네이션 여부.
- **delta = with-ontology 정답률 − without-ontology(맨 LLM) 정답률.**

### 5.2 결정 지원 (Layer B, Should)
- 역량 gap (Arm1엔 없음): "왜 이 발주량"(lineage) + "~면 어떻게"(what-if).
- **what-if 2단 레버**:
  - ① 하류(현재) — 발주량(결정변수) 변경 → 위험/비용 재계산. 운영자의 실제 결정 공간.
  - ② 상류 드라이버 — 날씨·캘린더 등 **모델 입력 feature를 가상 객체에 변경 → forecast 재실행 → 수요변화가 link(Weather→DailySales→Item→Category) 타고 전파**. 팔란티어 Scenario 동형. forecast 엔진 재사용(입력만 swap)이라 추가 비용 작음.
- **Scenario→commit closed-loop**: 여러 브랜치 시뮬 → 최선 선택 → CUD writeback(승인 게이트)으로 루프 닫기.
- 통제 시나리오에서 결정품질(기대비용) 정량화.

### 5.3 Layer A 운영 KPI
- 폐기비용 / 매진 median / 매진률 (v6과 동일, 아티제 합의).

---

## §6. 티어 (Layer B)

| 티어 | 내용 |
|---|---|
| **Core (필수)** | 베이커리 온톨로지 mock(객체 + OntologyLink) + decision/risk/what-if를 **OntologyFunction**으로 노출 + **grounded 에이전트** + 사전등록 Q셋 → grounding 정확도 eval |
| **Should** | what-if 시나리오 (드라이버 레버: 날씨·캘린더 가상변경 → forecast 재실행 → link 전파) + lineage "왜" 응답 + **CUD writeback (사람 승인 게이트)** + Batch/Live 실행모드 (§12) |
| **Stretch** | multi-hop traversal(매진위험 원인 거슬러 올라가기) / OntologyFunction 조합 / Scenario→commit closed-loop / Monitor 이상감지 |
| **범위 밖 (명시)** | **완전 자율 CUD**(승인 없는 자동 발주), 실데이터 전면 온톨로지화, Iceberg/Spark kinetic 인프라 |

---

## §7. 공정성 계약 (자체 구현이라 필수)

자체 데모는 마음먹으면 원하는 delta를 조작 가능 → *증거*가 되려면 4항 준수:
1. **Arm1 = 멀쩡한 baseline** — 실제 ML 예측 + 합리적 고정정책. 허수아비 금지.
2. **사전등록** — Q셋·시나리오를 미리 정의·고정. 체리피킹 금지.
3. **가정 동일** — 수요 cv·단가·비용을 양 arm 동일.
4. **라벨 정직** — "실데이터로 *측정*" vs "메커니즘 *시연*" 구분 표기.

---

## §8. 베이커리 Ontology mock 스키마 설계 (AOS 구조 동형)

repo 안에 AOS와 같은 형태로 가볍게 구현 (Postgres 불필요 — dataclass/JSON/dict로 충분).

### 8.1 OntologyObject (객체·필드 의미)
- `Store`, `Category`, `Item`, `DailySales`, `Weather`, `CalendarEvent`, `StockoutEvent`
- 각 필드: `displayName`, `description`, PK/nameKey, 타입 (= 에이전트가 읽는 의미)

### 8.2 OntologyLink (관계·조인키·카디널리티)
```
Item     ─belongs_to(N:1)→ Category
Item     ─sold_as(1:N)──→ DailySales
Store    ─has(1:N)─────→ DailySales
DailySales ─on(N:1)───→ Weather        (store_id, date)
DailySales ─on(N:1)───→ CalendarEvent  (date)
Item     ─had(1:N)────→ StockoutEvent
```

### 8.3 OntologyFunction (= 우리 decision layer 재사용)
| Function | 구현 | 반환 |
|---|---|---|
| `rank_stockout_risk(store, period, k)` | risk.py MC | 상위 k 품목 P(매진) |
| `explain_order(item, date)` | lineage.py | 발주량 분해 (base→안전→반올림) |
| `what_if(item, date, Δorder)` | risk.py + scenario | 발주 변경 시 위험 delta (하류 레버) |
| `what_if_driver(item, date, {weather/calendar})` | forecast 재실행 + risk.py | 드라이버 가상변경 시 수요·위험 delta (상류 레버, link 전파) |
| `commit_order(item, date, qty, approver)` | app-api mock | 승인된 발주 writeback (closed-loop, 사람 게이트) |
| `waste_cost(store, period)` | (기존 analysis/waste.py) | 폐기 비용 집계 |
| `demand_diff_by_condition(store, condition)` | 집계 쿼리 | 조건별 판매 차이 (예: 비/주말) |

### 8.4 OntologyKnowledge (RAG)
- 도메인 정의·산식 청크 (WAPE, adjusted_demand=α보정, 매진 censored 등) — 에이전트가 *의미* 공급용.

### 8.5 Agent
- 입력: 자연어 질의 → 온톨로지 traverse(어느 객체·링크) + Function 호출(진짜 수치) → 해석.
- without-AOS baseline: 동일 LLM에 스키마/관계/함수 없이 (맨 RAG 또는 맨 프롬프트).

---

## §9. 정직한 한계 / 포지셔닝

- **베이커리 규모선 조인이 단순** → 온톨로지가 "없으면 불가"는 아니다. **"스케일에서 복리로 커지는 메커니즘을 다루기 쉬운 예제로 시연"**으로 포지셔닝.
- **mock의 "진짜 AOS냐" 반론** → AOS와 **구조 동형**(OntologyObject/Link/Function 동일 개념)임을 명시해 대응.
- **에이전트 closed-loop (사람 게이트)** → 추천·설명·발주 writeback까지 시연하되 실제 commit은 사람 승인. **완전 자율 CUD는 frontier(범위 밖)**. (현재 실 AOS 에이전트는 read-only — v7 mock은 그 frontier 방향을 미리 시연하는 것임을 명시)
- **writeback = leakage 입구** → 에이전트가 쓴 발주가 다음 학습 feature로 들어가면 cutoff 위반 위험. writeback 객체에 valid-as-of 타임스탬프 필수 (절대규칙 #1).
- **grounding eval 부담** → Q셋 설계 + 정답 라벨링이 실제 작업량. Core의 핵심 리스크.

---

## §10. 기존 자산 재사용

- `src/bakery/decision/` (커밋 8d13157): lineage·policy·risk·pipeline → **OntologyFunction 구현체**.
- `bakery v6-predict` CLI: Arm1(ML 점추정+위험) 산출 경로.
- v4 3-stage(category_total·item_proportion): 점추정 공급 (수요이전 검증 통과 시).
- `analysis/waste.py` 등: waste_cost Function.

---

## §11. Open Questions / 다음 단계

| # | 질문 |
|---|---|
| Q1 | Grounding Q셋 규모·난이도·정답 라벨링 방법 (집계 자동검증 가능한 형태로) |
| Q2 | without-AOS baseline을 "맨 LLM"으로 할지 "LLM+RAG(의미만, 관계·함수 없음)"으로 할지 — 후자가 더 공정·엄격 |
| Q3 | Layer B 에이전트 LLM 선택 + 비용·평가 자동화 |
| Q4 | Should 티어 what-if 시나리오 사전등록 셋 |

**다음 first step**: 베이커리 ontology mock 스키마(§8)를 `src/bakery/ontology/`(가칭)에 dataclass로 구현 + OntologyFunction 래퍼(decision layer 재사용) → grounded 에이전트 → Q셋 eval. (Core 티어부터)

---

## §12. 실행 모드 — Batch materialize vs Live

같은 `OntologyFunction` 시그니처, **결과를 *언제* 계산·캐싱하느냐**만 다름.

| 모드 | 동작 | 우리 쓰임 |
|---|---|---|
| **Batch materialize** | 야간/주간 잡이 forecast+MC를 미리 돌려 **객체 property로 굳힘**(`DailyForecast.predicted_demand/.p_stockout/.recommended_order`). 에이전트는 모델 호출 없이 property 읽기 | 운영 발주 사이클 메인 산출물 + CUD writeback 대상. Layer A KPI가 여기서 나옴. `v6-predict` 배치 = 씨앗 |
| **Live / on-demand** | 함수가 쿼리 시점에 즉석 실행 | what-if(가상입력), ad-hoc 질의, 이벤트 재스코어링(매진 발생 → 즉시 재계산). `what_if`/risk.py 즉석 호출 = 씨앗 |

**왜 둘 다 보여주나**: 예측이 조회·조인 가능한 1급 객체 상태(batch)이면서 동시에 반사실로 심문 가능(live)함을 보임 → 단순 RAG와의 결정적 차이. batch property는 closed-loop writeback 대상, live는 Scenario 엔진.
**주의**: batch materialized property는 사용한 cutoff 타임스탬프 보존 — valid horizon 지난 값 silent 재사용 금지 (절대규칙 #1).

---

## §13. Modeling Objective 연계 (구현 범위 밖 — 설명용)

Foundry Modeling Objective = 모델 버전·eval·릴리스 게이트(라이프사이클 거버넌스). **구현 안 하되, closed-loop(CUD)을 넣는 순간 왜 필요한지를 설명할 수 있어야 함** (발표용 논거):

1. **closed-loop 안전 인터락** — writeback 켜지면 나쁜 모델 = 즉시 나쁜 발주(실제 돈). 챔피언/챌린저 + "holdout WAPE·폐기비용이 기존을 이겨야 승급" 게이트가 자동발주 신뢰의 잠금장치. **게이트 없는 closed-loop은 무모.**
2. **lineage 완성** — `explain_order`는 base→안전→반올림은 추적하나 "어느 모델 버전"이 빠짐. 버전 스탬프 → writeback 발주가 평가된 특정 모델까지 감사 가능.
3. **절반은 이미 있음** — rolling/expanding backtest + WAPE framework가 substrate. Modeling Objective는 이 backtest를 승급 게이트로 포장한 것(새 과학 아님).
4. **grounding 확장** — 에이전트가 "이 예측 믿을만해?"에 모델 eval 카드(최근 WAPE/coverage)를 OntologyKnowledge로 읽어 답 → grounding이 데이터 값에서 모델 신뢰도까지.

**한 줄**: "Modeling Objective는 범위 밖이지만, 우리 backtest+WAPE가 substrate고 빠진 조각(릴리스 게이트+버전 스탬프)이 정확히 closed-loop을 안전하게 만드는 인터락."

---

## §14. WBS + 일정 (AI Backend 구현분, 20 워킹데이)

> PoC 검증·운영(Layer A 전향실측·공정성 계약 등)은 별도 탭. 여기는 **광교 4주 실측 적용 *전까지*의 feature engineering + modeling 구현분만**.
> 전제: ① v0~v6 구현 완료(W0~W1은 신규개발 아닌 개선/통합), ② v6 decision layer(8d13157) 머지·안정화 완료.

### W0 (PoC 착수 前 — 토대 확정 / 게이트) — v0~v6 완료분 운영 데이터 기준 재검증·lock
- 공통 Feature Framework 및 학습구조 설계 (과적합 방지 구조 = time-split·rolling backtest·ensemble 정책 내장)
- 카테고리 단위 수요 예측 설계 (Stage 1)
- **수요 이전(흡수) 검증** ← ★Stage 2 진입 게이트 (절대규칙: "품목비율은 수요이전 검증 통과 시에만"). 통과해야 W1 진입

### W1 (Day 1–5) — 코어 모델 완성 + Feature
| 작업 | Day |
|---|---|
| 생산량 Multi-Window Feature 개발 (7/14/28일·동일요일 4주 rolling) | 1–2 |
| 제품 특성 Feature화 | 2–3 |
| 품목 비율 배분 모델 설계 (Stage 2 + 신제품 tracker) | 2–4 |
| 인터뷰 노하우 정량 Feature화 | 4–5 |
| 코어 보강 (calibration + Risk Estimation 모듈) | 5–7 |

### W2 (Day 6–10) — 결정 레이어 + 온톨로지 mock
| 작업 | Day |
|---|---|
| 코어 보강 마무리 (spillover) | 6–7 |
| 발주 derivation + decision lineage 온톨로지 연계 (기존 자산) | 7–8 |
| 베이커리 온톨로지 mock 설계 (Object/Link/Knowledge) | 7–9 |
| 결정·위험 함수 OntologyFunction화 | 9–11 |
| ▷ Grounding Q셋 설계·정답 라벨링 [병렬 착수, 최대 리스크] | 8–12 |

### W3 (Day 11–15) — 실행모드 + Scenario + 에이전트
| 작업 | Day |
|---|---|
| OntologyFunction화 마무리 | 11 |
| Batch materialize / Live 실행모드 분리 | 11–12 |
| What-if Scenario (드라이버 레버 → forecast 재실행 → 전파) | 12–14 |
| 운영 규칙(암묵지) 결정 rule화 | 13–14 |
| Grounded 에이전트 구현 + without-AOS baseline | 14–16 |

### W4 (Day 16–20) — Closed-loop + Grounding 측정
| 작업 | Day |
|---|---|
| 에이전트 마무리 | 16 |
| CUD writeback + 사람 승인 게이트 (유일 greenfield, 최대 흔들림) | 16–18 |
| Grounding eval 자동화 + delta 측정 (★Flagship) | 17–20 |
| 통합 / leakage 회귀 테스트 / 마감 버퍼 | 19–20 |

**리스크**: ① Grounding Q셋 라벨링(W2 조기착수 필수) ② CUD writeback(유일 신규 — 밀리면 Should로 내리고 Flagship 사수).

### §14.1 고객사 전달용 (plain 워딩)

| 그룹 | 고객 표현 | 핵심 작업 |
|---|---|---|
| 3 | **AI가 데이터를 이해하고 답하게 만들기** | 매장·상품·날씨·판매 데이터 관계도 구축 / 발주·품절위험 계산을 AI가 불러쓰는 표준 기능으로 / 실데이터 근거로 답하는 AI 어시스턴트(+대조군) / 발주 질문 세트·정답 기준 |
| 4 | **AI 추천을 실제 발주로 연결하기** | 권장 발주량 + "왜 이 수량" 근거 / 가정 시뮬레이션(비 예보·발주 조정 시) / AI 추천을 시스템에 반영(담당자 승인 후) / 야간 일괄·실시간 2방식 / 현장 노하우를 발주 규칙으로 |
| 5 | **마무리 및 신뢰성 검증** | AI 답변 정확도 최종 측정(데이터 연결 효과 입증) / 전체 통합·정합성·미래정보 누출 방지 확인 |

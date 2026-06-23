# Kinetic Layer × 빵 수요예측 — 적합성 분석

> 검토 대상: enhans Notion "kinetic layer 설계 제안 (WIP)" (38302490cbe080ba8413d0b4186f06ed)
> 검토 기준: 우리 PoC = `docs/poc_scope_v6.md` (260605 아티제 미팅 이후, 점추정+위험수치 / 광교 단독 / 4주+4주 전향적)
> 작성일: 2026-06-19 (개정 2026-06-20 — §8~§10 추가 / v6 정렬: 점추정·위험수치, Stage 2 조건부, 암묵지 demand·decision 구분)

---

## 0. 한 줄 결론

**코어 수요예측에는 못 쓴다(확률 영역을 설계가 의도적으로 배제). 그러나 예측 주변의 결정론적 가공·설명·시뮬레이션·이상감지 레이어에는 잘 맞는다.** 단, 지금 PoC(광교 단독, 4주+4주) 규모에 도입하는 것은 오버엔지니어링 — 진짜 가치는 **PoC 성공 후 production 플랫폼 + §7 장기 AI 발주비서 비전과의 정렬점**이다.

---

## 1. Kinetic Layer가 무엇인가

값이 여러 단계 규칙(변환·집계·분배)을 거쳐 파생되는 도메인에서:
- 계산하면서 **lineage(영수증)** 을 남기고 (drilldown — 원인 추적)
- 기간 간 **변동을 자동 감지**하고 (Monitor)
- 파라미터를 바꿨을 때 결과를 전체 재계산 없이 미리 산출한다 (**시뮬레이션** — Scenario overlay)

핵심 제약 (설계자 본인 명시):

> "kinetic layer는 ontology object의 property 변수의 조정, 즉 ontology로 컨트롤할 수 있는 변수로, **확률론적인 부분이 제외되고**, 정의된 계산 엔진으로부터 계산을 수행한다."

즉 **확률/통계 예측을 설계상 빼버린 결정론 엔진**이다 (원가 배부·정산처럼 규칙으로 값이 정해지는 도메인용).

### 계산 5개 type
| type | 모양 | 빵 도메인 대응 가능성 |
|---|---|---|
| FORMULA | 1:1 (행 단위) | `adjusted_demand = sold_normal + sold_closing×α` |
| ROLLUP | N:1 (집계) | 카테고리 합 = 품목들 합 |
| ALLOCATE | 1:N (분배) | Stage 2 품목 비율 배분 ★ |
| WINDOW | 누적 (시간축) | rolling/이동평균 feature |
| ITERATE | 순환 (고정점) | (해당 없음 — 빵엔 상호정산 같은 순환 없음) |

### 인프라
Spark(실행) + Iceberg/Gravitino(lineage 저장) + PostgreSQL(규칙 정의) + Calcite(rule→RelNode→SQL) + Kafka(이벤트) + LangGraph(LLM 해석). **엔터프라이즈급 lake 인프라 위에 얹는 설계.**

---

## 2. 우리가 하려는 것 (대조 기준)

PoC v6 방향:
- **산출물**: 점추정 + 품절/매진 위험 수치 (구간예측 폐기)
- **검증**: 광교 단독, 4주 구축 + 4주 전향적 실측, 기존 발주 시스템과 비교
- **모델**: LightGBM 코어 유지 + v4 3-stage (카테고리 합 → 품목 비율 → 신제품 tracker). ※ v6에선 **bulk→품목비율(Stage 2)은 수요이전 통계검증 통과 시에만 채택**(조건부)이고, **카테고리 정의는 아티제 제공**, **공통 backbone + 매장별 커스터마이징**(robust, 오버피팅 경계) 구조.
- **장기(§7)**: 온톨로지 + LLM 기반 발주 AI 비서 (feature 자동 선택, 발주량 reasoning)

우리 모델은 **"확률적 예측(Stage 1)" + "결정론적 가공(Stage 2/3, feature, sensitivity)"** 이 섞인 구조다. 이 분리가 적합성 판단의 핵심.

---

## 3. 어디에 쓸 수 있는가 / 못 쓰는가

| 우리 모델 컴포넌트 | 성격 | Kinetic 대응 | 적합 |
|---|---|---|---|
| **Stage 1 카테고리 수요 예측** (LightGBM **점추정**, v6) | 통계적 ML(학습) | — (해당 type 없음) | ❌ **못 씀** |
| 품절/매진 위험 수치 P(매진) — **v6 핵심 산출물** | 확률(점추정에 부속) | §8 **MC 껍질** (kinetic-native 아님) | △ kinetic 단독 ❌ / MC 껍질로 ✅ |
| `adjusted_demand = sold_normal + sold_closing×α` | 결정론 | FORMULA | ✅ |
| 카테고리 합 = 품목 합 | 결정론 | ROLLUP | ✅ |
| **Stage 2 품목 비율 배분** (총량 × 품목비율) | 결정론(분배) | ALLOCATE | ✅ ★ (단 아래 주의) |
| rolling/이동평균 feature | 결정론 | WINDOW | ✅ |
| α·정책 파라미터 **sensitivity sweep** | 시뮬레이션 | Scenario(Dynamic) overlay | ✅ |
| "이 품목 발주 N개가 왜 나왔나" 추적 | 설명가능성 | lineage drilldown | ✅ |
| 수요 drift / 매진 이상 감지 | 변동감지 | Monitor | ✅ |
| Stage 3 신제품 promote/fade 룰 | 결정론 룰 | (FORMULA+Monitor 조합) | △ 가능하나 단순 |

> **v6 정렬 메모 ①(산출물)**: v6 산출물은 *구간(quantile interval)이 아니라* **점추정 + 품절/매진 위험 수치**(v5 conformal 폐기). 그래도 결론은 같다 — Stage 1은 점추정이든 분위수든 **학습된 ML 예측**이라 kinetic이 표현 못 함. 위험 수치는 v6의 *핵심 산출물*이며, kinetic-native가 아니라 §8 MC 껍질로 낸다(수요 불확실성을 내부적으로 분포로 다루되 *deliverable*은 점추정+위험수치).
>
> **v6 정렬 메모 ②(Stage 2 주의)**: 위 ALLOCATE ★ 매핑엔 단서가 셋이다. **(ⅰ)** v6에선 **bulk→품목비율 구조 자체가 "수요이전 통계검증 통과 시에만" 채택**(조건부)이다 — 검증 실패 시 이 구조는 재검토되므로 ★ 매핑도 전제부 무효. **(ⅱ)** ALLOCATE에 매핑되는 건 **"총량을 비율로 분배"하는 *메커니즘*(+lineage+보존)**이지, 비율을 *만드는* 곱셈인자(`trend·stockout·new_boost` 등)가 아니다 — 그 인자들은 §9.2 정정대로 *demand-side라 피쳐화 대상*. **(ⅲ)** 카테고리 정의는 **아티제 제공**(우리가 임의로 안 묶음).

### ❌ 못 쓰는 영역 — 왜
1. **코어 예측이 곧 학습된 ML 추정**인데, kinetic은 확률/학습 부분을 설계상 배제. LightGBM 점추정을 kinetic FORMULA로 표현 불가.
2. **매진 확률 같은 위험 수치**(v6 핵심 산출물)도 확률 출력이라 kinetic-native로는 못 냄 — §8의 **MC 껍질**로 낸다(우리가 build). 그쪽 설계엔 미착수(Phase 4)라 소비가 아니라 공동개발.

### ✅ 쓸 수 있는 영역 — 본질
예측값이 *나온 다음*, 그것을 **규칙으로 가공·분배·집계하고, 설명하고, 파라미터를 흔들어보는** 모든 단계. 우리 v4의 Stage 2 이후가 전부 여기 해당.

---

## 4. 썼을 때 좋은 점 (우리 맥락에서)

1. **설명가능성 (lineage drilldown) — v4 가치 제안과 정확히 일치.**
   v4의 핵심 차별점이 "LGBM black box 대신 명시 룰 분배"였다. kinetic lineage는 "광교 통팥빵 23개 발주 = 카테고리총량 258 × 품목비율 8.9% → 정책보정 → 반올림"을 **클릭으로 끝까지 분해**해준다 (단 비율을 *만든* 수요신호는 §10.3대로 SHAP으로 설명 — derivation 아님). 아티제 실무자에게 "왜 이 수량인가"를 설명하는 데 직접적 가치. 발주 권장의 **신뢰·납득성**이 PoC 성패의 한 축이라면 강력.

2. **시뮬레이션 (Scenario overlay) — 우리가 이미 손으로 하는 일의 자동화.**
   우리는 이미 α{0.3,0.5,0.7,1.0}, boost ±10/20/30% 등 파라미터 sweep을 돌리고 있다. 이게 정확히 "파라미터 바꾸면 결과 어떻게 변하나"이고, overlay 방식은 **전체 재계산 없이 변경분만** 돌려 빠르다. "발주를 N개 늘리면 폐기·매진이 어떻게?"를 실시간 What-if로 제공 가능.

3. **변동감지 (Monitor) — 전향적 4주 검증과 궁합.**
   4주 prospective 동안 예측 vs 실측 drift, 매진 이상을 **데이터 적재 직후 자동 감지**. 수작업 모니터링 대체.

4. **보존 검증 공짜.**
   Stage 2 합 = Stage 1 total (우리가 sanity check로 diff 0.0000 확인하는 것)이 lineage 기여분 합산으로 **자동 보장**. 누락 시 어느 품목에서 깨졌는지까지 짚음.

5. **§7 장기 비전과 substrate 일치.**
   "온톨로지 + LLM 발주비서 — LLM은 reasoning/해석, 결정론 엔진이 수치"가 kinetic의 "수치는 엔진, 해석만 LLM" 원칙과 **그대로 포개진다**. LLM이 만든 규칙을 사람 승인 + backtesting 게이트로 검증하는 구조도 우리 §5.5(암묵지 피쳐화 + ablation)와 결이 같다.

---

## 5. 안 좋은 점 / 리스크

1. **PoC 규모에 과함 (가장 큰 문제).**
   광교 단독 5년치 daily를 uv 파이썬으로 돌리는 PoC에 Spark+Iceberg+Gravitino+Calcite를 얹는 건 명백한 오버엔지니어링. 4주 구축 일정을 인프라에 태워먹는다. lineage 행 폭증 대비책(grain 컷오프, 파티셔닝)도 우리 데이터 규모엔 불필요.

2. **코어를 못 덮는다 = "절반의 엔진".**
   가장 어렵고 가치 큰 부분(수요예측 자체)은 여전히 우리 LightGBM이 담당. kinetic은 그 *앞뒤* 결정론 파이프라인만 표준화. 도입해도 예측 정확도는 1pp도 안 오른다 — 좋아지는 건 **설명·시뮬·운영성**이지 정확도가 아니다.

3. **WIP + 미착수 영역이 우리 핵심과 겹침.**
   우리한테 가장 매력적인 "분포/확률 시뮬(매진확률)"과 "부분 재계산"이 이 설계에서 ❌/⚠️ 상태. 기대하고 붙으면 직접 구현 부담.

4. **규칙 표현 제약.**
   계산식은 5개 구조 + 정해진 산술만 (자유 SQL 불가). 우리 결정론 로직은 대부분 표현되지만, 비정형 휴리스틱이 생기면 새 type 추가 = 컴파일러(lower/codegen/provenance) 구현 부담.

---

## 6. 권고 (PM 전달용 포지셔닝)

| 시점 | kinetic layer의 위치 |
|---|---|
| **지금 PoC (4주+4주)** | ❌ 도입 안 함. 인프라 과함. 대신 v4 결정론 로직(Stage 2 등)을 **kinetic 5-type으로 표현 가능하다는 점만 의식하고 코드 구조를 깔끔하게** 유지 → 나중에 이식 쉽게. |
| **PoC 성공 후 production** | ✅ 강력 후보. 설명가능성(lineage) + What-if 시뮬 + 변동감지가 운영 발주 시스템의 실질 가치. |
| **§7 장기 (AI 발주비서)** | ✅ substrate 정렬. "LLM 해석 + 결정론 엔진 + lineage"가 정확히 우리 장기 그림. |

**한 줄 메시지**: kinetic layer는 우리 PoC를 *빠르게* 해주는 도구가 아니라, **PoC가 검증한 가치를 production·장기 비전에서 *설명가능하고 시뮬레이션 가능한* 발주 플랫폼으로 키울 때의 토대**다. 코어 예측(우리 ML)과 결정론 가공(kinetic)이 **상호보완**이지 대체 관계가 아니라는 점이 핵심.

---

## 7. 부록 — 구체 매핑 예시 (이식 시 참고)

```
[우리 v4]                                    [kinetic 표현]
sold_normal + sold_closing × α      →  FORMULA {op:add, args:[sold_normal,
                                                {op:mul, args:[sold_closing, α]}]}
Σ items → category_total            →  ROLLUP {group_by:[date,category],
                                                agg:sum, agg_input:adjusted_demand}
category_total × item_ratio[i]      →  ALLOCATE {amount:category_total,
                                                basis:item_ratio, ratio_partition:[item]}
rolling_28d_mean                    →  WINDOW {partition:[item], order_by:[date],
                                                frame:rows_28_preceding}
α sweep {0.3,0.5,0.7,1.0}           →  Scenario overlay (scenario_id별 α 딱지)
"품목 발주 N개 왜?"                  →  lineage drilldown (child_node=item)
Stage2 합 = Stage1 (diff 0)         →  보존 검증 (기여분 합산 자동)
```

> ⚠️ Stage 1 예측(LightGBM) 자체는 위 어느 type에도 안 들어간다 — 예측 결과가 kinetic 파이프라인의 *입력(원본 사실)*으로 들어오고, kinetic은 그 이후 결정론 가공만 담당한다. 이 경계가 분석의 핵심.

---

## 8. 확률론 가미 — 적용 방안

kinetic은 원래 제조 원가 배부·추적(결정론)용이지만, **확률을 어디에 꽂느냐**만 잘 정하면 우리 수요예측에 변형 적용할 수 있다. 실제로 원본 설계가 부록 "몬테카를로를 탭핑해본다면?"에서 그 길을 절반 그려놨다.

### 8.1 핵심 원칙 — 규칙을 확률화하지 마라

kinetic 규칙(FORMULA/ALLOCATE/…) 자체를 랜덤하게 만들면 안 된다. 이유:
1. kinetic의 핵심 보장(멱등·lineage 보존법칙·provenance semiring 정확성)이 **결정론 + 순수함수**에 의존. 규칙 랜덤화 시 재현성·보존검증이 깨진다.
2. 설계 본인도 MC 확장의 전제를 "Kinetic이 순수함수라 다른 입력으로 N번 재실행이 안전하다"로 명시 → **순수성은 확률화의 걸림돌이 아니라 enabler.**

표준 확률 모델링과 동일: 함수 f는 결정론, 불확실성은 **입력/파라미터**에 둔다.

### 8.2 우리에 맞춘 패턴 (= v6 "점추정 + 매진확률"과 구조 동일)

```
[우리 예측 모델]                    [kinetic — 결정론 그대로]        [출력]
LightGBM이 카테고리 수요의           ALLOCATE/ROLLUP 등
"분포"를 낸다 (quantile/residual) ─┐  Stage2 품목배분 파이프라인
                                  │        │
              ┌───────────────────┘        │ ×N (순수함수라 안전)
              │  Monte Carlo (dynamic layer)
              │  분포에서 N개 수요 시나리오 샘플 ──→ 매번 결정론 실행 ──→ 품목별 생산 분포
              │                                                          │
              └──────────────────────────────────────→ P50=점추정 / P(매진) / P(폐기) / 기대비용
```

- **점추정** = 중앙값(또는 mean) 수요로 결정론 1회 실행.
- **매진 위험 수치** = 수요 분포 N회 샘플 → 결정론 파이프라인 N회 → "발주 Q일 때 매진 횟수/N = P(매진)".
- 역할 분리: **LightGBM = 분포 생성기 / kinetic = 결정론 변환기 / MC = 껍질.** 6/5 산출물 형태가 이 구조에 그대로 떨어진다.

### 8.3 고유 차별화 — "위험 lineage"

provenance semiring은 추적 대상을 바꿀 수 있다 (설계 인용: "무엇을 추적할지만 바꾸면 다른 정보를 얻음"). 추적 대상을 **"금액 기여" → "분산(위험) 기여"**로 바꾸면:

> "광교 통팥빵 매진 위험의 60%는 카테고리 수요 불확실성에서, 40%는 비율 배분 불확실성에서 왔다"

를 **drilldown으로 분해** 가능. 일반 예측 도구엔 없는 기능 = "왜 이 품목이 위험한가"를 설명 가능한 발주 시스템.

### 8.4 함정 4개 (반드시 플래그)

1. **입력 분포 출처** — 우리 quantile은 카테고리 *주변* 분위수일 뿐 품목·날짜 *결합분포*가 아님. 특히 **품목 간 상관(=cannibalization)**을 독립 샘플링하면 위험 과소/과대 추정. 설계의 `correlations`(Cholesky)가 다루지만 상관 구조를 우리가 추정해야 함. (폐기한 conformal/quantile 작업이 여기 입력으로 부활 가능)
2. **Stage 2 비율 배분은 비선형** (분모 정규화) → 분산 lineage가 깔끔하게 안 더해짐. 설계의 linearity 판정이 "어디까지 분산전파, 어디부터 MC"를 자동 분기.
3. **PoC엔 풀 인프라 금지** — Spark/Iceberg 없이 부록의 벡터화 MC(`np.outer(ratio, pool_samples)` → 분위수)를 Stage 2 위에 numpy 몇 줄로 프로토타입. kinetic 본체는 production용.
4. **이 확률 dynamic layer는 그쪽 설계에서 ❌ 미착수(Phase 4)** — 소비가 아니라 로드맵 앞지른 공동개발. 협업이면 기회, 의존이면 리스크.

---

## 9. 암묵지(tacit knowledge) 피쳐화와의 연결

이 방향은 우리 PoC §5.5(운영 암묵지 피쳐화 + ablation)·§7(LLM 발주비서)과 **정면으로 맞닿는다.** kinetic이 그 둘을 묶는 지점이다.

### 9.1 설계가 이미 암묵지 경로를 갖고 있다

> "규칙은 소스 시스템 파싱(importer)과 **현업 암묵지(LLM)** 두 경로로 만들어지며, 둘 다 같은 승인 게이트를 거친다. **LLM이 만든 규칙은 사람 승인과 과거값 대조(backtesting)를 통과해야** 실제 계산에 쓰인다."

→ 이게 곧 **현업 암묵지 → LLM이 규칙으로 표현 → 승인 + backtesting → 계산 반영**. 우리 §7 그림 그대로.

### 9.2 단, 암묵지가 들어가는 길은 둘 — demand(학습) vs decision(선언)

핵심 구분은 암묵지가 **수요에 대한 것**이냐 **결정/제약에 대한 것**이냐다.

| 암묵지 종류 | 들어가는 곳 | 예시 | 성격 |
|---|---|---|---|
| **수요 지식** (뭐가 언제 왜 팔리나) | **피쳐화 → LightGBM** | `holiday_distance`, `residential_area_flag`, `holiday_distance × residential` 상호작용 | *학습됨* (데이터가 크기·곡선·유효성 결정) |
| **결정/제약 지식** (예측→발주 변환 정책) | **룰 → kinetic류** | 최소 진열량, 안전재고, 반올림, 묶음단위 | *선언됨* (학습 아닌 선택) |

→ **demand 암묵지는 피쳐화가 정답이다.** 빵 수요는 복잡계라 magnitude를 모르므로, 룰로 박는 게 아니라 모델이 데이터로 추정해야 한다. 전문가의 가치는 *"가중치"가 아니라 "어떤 신호를 만들지(=어디를 볼지)"*. kinetic류 룰은 **결정/제약(decision)** 에만 쓴다.

> ⚠️ **정정 (Q2 논의 반영, 2026-06-20)**: 이전 판에서 Stage 2 곱셈 인자(`trend × stockout × closing × new_boost`)를 "kinetic 룰 후보"로 적었으나, 이들 대부분은 *수요 패턴 보정*이라 **사실 피쳐화 대상**이다. 특히 `new_boost=1.2`처럼 *모르는 크기를 고정값으로 박는 것*은 정확히 피해야 할 안티패턴 — `new_product_age` 같은 feature로 바꿔 모델이 학습하게 하는 게 맞다. 진짜 kinetic 룰로 남길 것은 **순수 정책/제약**(최소생산·반올림·진열 하한·안전마진)뿐이다.

kinetic류로 옮겼을 때의 이점 (decision 룰에 한해):
- 선언적 + 버전 관리
- **lineage**: 어느 정책이 발주 권장에 얼마 기여했나 분해 (= §10의 Decision lineage)
- **What-if**: "안전마진 +1개면 폐기/매진 어떻게?"

### 9.3 §8(확률)과 §9(암묵지)가 만나는 지점

결정 정책의 *파라미터* 자체가 불확실할 때 (예: 마감할인 가중 `α`, 안전마진) → 그 파라미터에 분포를 얹는다 → MC → **"이 정책 선택의 불확실성이 매진/폐기 위험에 얼마나 기여하나"**.

즉 점추정 sensitivity(우리가 이미 α{0.3,0.5,0.7,1.0} 돌리는 것)를 **위험 기여도 분해**로 끌어올린다. (demand 암묵지는 이미 LGBM이 학습하므로, 분포를 얹는 대상은 *decision* 파라미터다 — demand가 아니라.)

### 9.4 정리 — 세 갈래 수렴

```
§5.5 암묵지 피쳐화 + ablation  ┐
§7  LLM 발주비서(규칙 생성·승인) ├─→ kinetic DerivationRule(암묵지→LLM 규칙→승인+backtesting+lineage)
§8  확률 위험 lineage          ┘     + 룰 계수에 분포 → 암묵지의 위험 기여도 평가
```

**경계만 지키면 된다**: demand(수요) 암묵지는 ML 코어에 feature로, decision(결정·제약) 암묵지는 kinetic류에 rule로. 둘은 대체가 아니라 분업.

---

## 10. 우리에 더 맞는 재중심화 — "Demand Intelligence / Forecast Reasoning Layer"

> GPT 교차검토 + Q1~Q3 논의 종합 (2026-06-20). kinetic을 *그대로* 가져오지 말고, 같은 철학을 공유하되 **이름·객체·무게중심을 아띠제용으로 재설계**하는 방향.

### 10.1 왜 이름부터 다르게 — 무게중심이 다르다

반도체 원가 kinetic은 **이미 아는 총액을 결정론 규칙으로 뒤로 분해(audit)**한다. 우리는 **모르는 값을 앞으로 예측하고 그 위에서 결정을 추적·시뮬**한다. 무게중심이 달라서, 그대로 가져오기보다 재명명·재중심화가 맞다.

- 이름: **Demand Intelligence Layer** (또는 **Forecast Reasoning Layer**)
- 파이프라인:
  ```
  Raw Data → Feature Store → ML Forecast → Decision Policy → Recommendation → Explanation / Simulation / Monitoring
  ```

### 10.2 kinetic 철학에서 가져올 것 = 3가지뿐

**A. Feature lineage** — "이 피쳐가 어떤 원천 데이터·암묵지에서 왔나" (설계시점 provenance)
```
holiday_distance ← 명절 캘린더 ← 매장 상권유형
                 ← 현업 가설: "주거지 매장은 명절 전후 수요↑"
```
나중에 LLM 발주비서가 "왜 이 feature를 보는가"를 설명할 근거. (이건 *정의* 계보. 한 예측에 *얼마 기여*했나는 SHAP — 10.3.)

**B. Decision lineage** — "예측값이 최종 발주량으로 어떻게 바뀌었나" (kinetic이 제일 잘하는 영역)
```
LGBM 예측 18.7 + 품절위험 보정 +2.1 + 최소 진열량 +1.0 + 반올림 = 권장 22개
```

**C. Experiment / ablation registry** — "암묵지를 feature로 넣었더니 진짜 KPI가 좋아졌나"
```
holiday_distance: WAPE 31.2→28.9, 매진률 14%→11%, 폐기비용 +2%
```
→ 아띠제에선 원가 배부 lineage보다 **이 ablation registry가 훨씬 중요**하다. §5.5 ablation을 일회성이 아니라 **영속 자산**으로 만든 것.

### 10.3 두 종류의 lineage를 구분 (Q1 결론)

| | Feature *contribution* lineage | Decision lineage |
|---|---|---|
| 어디 | ML Forecast 노드 | Decision Policy 노드 |
| 정체 | **SHAP 귀인** (학습된, 사후) | **derivation** (결정론, 실제 계산) |
| 예 | "수요 258 = base 220 + 주말 +25 + 비 −5 + 명절거리 +18" | "258 → 품목비율 → 정책보정 → 통팥빵 23" |
| 보장 | 가산성(합=예측) — 단 *귀인*이지 인과 아님 | 보존(합=총량) — 실제 유도 |

한 drilldown 트리에서 ML 노드는 SHAP으로, 정책 노드는 lineage로 explode. **UI에서 "귀인" vs "유도"를 라벨로 구분 필수** (혼동하면 SHAP을 인과로 오독).

### 10.4 객체 재명명

```
DerivationRule      → FeatureDefinition / DecisionPolicy
DerivationLineage   → FeatureLineage / DecisionLineage
Scenario            → What-if Order Scenario
Monitor             → Forecast Drift / Stockout-Waste Monitor
(신규)              → PredictNode (features → 수요분포, lineage = SHAP)
```

추적하는 질문 자체가 바뀐다:
- 원가 kinetic: **"값이 어떻게 계산됐나?"**
- Demand layer: **"예측에 어떤 정보가 들어갔고 / 그 예측이 어떤 정책으로 발주가 됐고 / 그 정책이 실제 운영 KPI를 개선했나?"**

### 10.5 가중치 역전 (Q3 핵심)

- 원가(총액 알려짐): **lineage가 주인공**, dynamic/MC는 곁다리(Phase-4).
- 수요(불확실): **What-if + 위험 시뮬이 주인공**, lineage는 보조.

→ 우리가 커스터마이징하면 **dynamic/MC/overlay를 1급 시민**으로 올리고, ITERATE(빵엔 상호정산 없음)는 삭제, Spark/Iceberg는 PoC 제외. 즉 원가 설계의 *역방향* 가중치.

### 10.6 리스크 (변치 않음)

- ⚠️ **PoC 범위 아님** — production / 장기 AI 발주비서 그림. PoC는 LightGBM 코어 + 3-stage + 점추정/위험수치(`docs/poc_scope_v6.md`).
- 이건 그 자체로 하나의 *제품*(forecast reasoning engine). "운영자가 실제로 *왜 이 수* / *바꾸면 어떻게*를 묻는가"가 PoC에서 질적으로 확인돼야 이 투자가 정당화된다. 안 물으면 과잉.

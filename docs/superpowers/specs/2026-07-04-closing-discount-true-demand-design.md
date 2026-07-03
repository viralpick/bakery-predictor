# 마감할인 실수요 검증 (α 실증 + 원가율 역추정) — Design

**Date**: 2026-07-04
**Depends on**: `analysis/discount.py`(분단위 라인아이템 + closing label), 폐기 실측 데이터, `analysis/waste.py`(WasteEstimator/실측 swap), bonavi 카테고리 룰, v4 `adjusted_demand = sold_normal + sold_closing×α`
**Status**: Design (프레이밍 승인 2026-07-04, 문서 리뷰 대기)

## 목적

메모리·v4 모델링에서 **카테고리 실수요 = 정상판매 + 마감할인판매 × α**로 역산하는데, α(0.5~0.7)는 근거 없는 추측값이다. 본 작업은 α를 **데이터로 실증**하고, 그 부산물로 **미지의 원가율 c를 역추정**한다.

α의 정의: 마감할인으로 팔린 물량 C 중 **진짜 수요 B의 비율** = B/C. C는 두 조각:
- **B (base)**: 제값에도 샀을 손님이 마침 늦게/할인시간에 온 것 → 진짜 수요
- **I (induced)**: 할인이라서만 산 딜헌터 → 제값이면 안 샀을, 과잉생산 떨이

검증 = 유도분 I를 **인과적으로 떼어내는 것**. α = 1 − I/C.

## 핵심 개념 (LOCKED)

### 두 개의 α는 서로 다른 질문의 답이다

| | 정의 | 원가율 c 필요? | 산출 |
|---|---|---|---|
| **α_A (descriptive)** | 마감 손님 중 제값에도 살 사람 비율 (손님에 대한 사실) | ❌ 불필요 | Phase A: 상수/구간 |
| **α\*(objective)** | 발주 파이프라인 총비용을 최소화하는 가중치 (비즈니스 최적) | ✅ 필요 | Phase B: c의 함수 곡선 |

이 둘은 **할인 마진이 양수냐**에 따라 갈린다. 떨이라도 마진이 남으면 유도분까지 생산하는 게 이득이라 α\* > α_A가 될 수 있다. 메모리의 "실수요=정상+마감α"는 이 둘을 한 글자로 쓰지만 다른 축이다. **둘을 독립 추정해 수렴을 보는 것이 최종 검증이다.**

### 원가율 c는 Phase B를 막지 않는다 — 파라미터로 돌린다

마감 한 잔이 손실인지 이익인지가 전부 c에 달렸다:
- 마감 판매 마진 = 정가 × (1 − 할인율 − c)  → 30% 할인 시 **c < 0.7면 아직 이익**
- 폐기비용 = 폐기수량 × 정가 × c
- 매진 기회손실 = 매진수량 × 정가 × (1 − c)

c를 미지수로 두고 **α\*(c) 곡선**을 뽑는다. 그러면 "α\*(c) = α_A가 되는 c는?"를 풀어 **원가율을 역추정**한다.

⚠️ 정직하게: 이건 구조적 항등식이 아니라 **정합성 렌즈**다. 교차점의 c는 "수요를 그대로 발주하는 게 비용최적이 되는 원가율"로 해석해야지 회계상 실제 원가율과 같다고 단정할 수 없다. 삼각측량·근사로 본다. 고객사가 실제 c를 제공하면 α\*(c)가 점으로 붕괴되며 α_A와 직접 대조된다(설계에 슬롯 유지).

## 데이터 (확인 완료)

- **라인아이템** (`discount.py::load_sales_with_discount`): `receipt_id, date, hour, minute, item_id, qty, unit_price, paid, discount_amt, discount_code, label, is_set` — **분(minute) 단위 타임스탬프**
- **마감 두 단계**: `0077`=30%(≈49,975건), `0069`=20%(≈9,070건) → **가격 depth 변동**
- **폐기 실측**: 존재 (per-day·per-item 수량). → 진짜 잔량 = 마감판매 + 폐기 계산 가능
- **없음**: 고객/회원 ID (receipt 단위까지만) → 재구매/충성도 방법 불가
- **c (원가율)**: 미지. 파라미터로 처리, 고객사 제공 시 대입.

## 범위 (LOCKED)

- **매장/카테고리**: 광교 bread/pastry. sandwich(단일품목)·cake(시즌)은 W0와 동일하게 제외.
- **레이어**: Stage 1 카테고리 총량만. 품목 비율(Stage 2)·double counting 회피는 기존 결정 유지 — 품목 potential을 카테고리 합에 넣지 않는다.
- **엣지**: 카테고리 전체 품절날(마감분 자체가 없음)은 α 추정 대상 아님. 별도 flag(기존 W0 후속 항목).

---

## Phase A — 구조적 α (cost-free, 3각 식별)

세 방법은 confound 방향이 서로 달라 삼각측량한다. 각각 α의 상·하한을 준다.

### A1. Kink / RD-in-time (마감할인 onset 불연속)

**아이디어**: 하루 안에서 판매율 q(t)(units/min). 마감할인 개시 시각 t0(그날 첫 closing 라인, 로버스트니스로 정책시각 20:00 고정) 직전은 제값, 직후는 할인. t0 직전 추세를 외삽한 것이 counterfactual 제값 판매율 → 마감창 관측율 − counterfactual = 유도분 I.

**식별 가정**: t0 전후 구매의향 프로세스의 연속성. 19:58 vs 20:02 도착 손님은 유사, 불연속적으로 바뀌는 건 가격뿐 → t0에서의 점프 = 가격효과.

**구현 스케치**:
- t0 이전(예: 17:00~t0) q(t) 회귀로 제값 추세 추정.
- t0 양쪽 local linear (RD). 마감창 전체에 대해 B = ∫counterfactual, C = ∫관측, I = C − B.
- 품목별 추정 후 카테고리 집계(마감 임박 품절로 인한 mix shift 통제).

**Confound → α 방향**:
- **저녁 commute 상승**: 할인 무관하게 19~20시 구매율이 오를 수 있음 → 점프 과대 → I 과대 → **α 하한**.
- onset 검출 노이즈(스태프가 몇 분에 걸쳐 적용) → 첫 closing 라인 + 정책시각 로버스트니스.

### A2. Depth elasticity (20% vs 30%)

**아이디어**: 두 depth로 마감 qty의 가격반응 추정 → discount=0으로 외삽 = base B. 기울기 = 유도분.

**구현 스케치**: 카테고리-day 단위 q(depth) 회귀, actual surplus(폐기+마감)·요일 FE 통제. 절편(discount=0) = B.

**Confound → α 방향**:
- **depth 내생성**: surplus 많은 날 깊게 할인 → depth와 판매가능량 양의 상관 → elasticity 상향편향 → I 과대 → **α 하한**. surplus·day-type 통제로 완화.
- 관측 depth 2점 → 0으로의 선형외삽은 가정(곡률 리스크 명시).
- **선행 점검**: 20% vs 30%가 시간 confound인지(20% 이른 저녁, 30% 마감 임박?) 데이터로 확인. 그렇다면 A1 confound와 겹침.

### A3. Surplus counterfactual (실측 폐기 활용)

**아이디어**: 진짜 잔량 S = 마감판매 + 폐기. 마감판매가 S에 비례(공급주도 떨이)하나, 아니면 포화(수요주도 고정 base)하나?

**구현 스케치**: 카테고리-day에서 closing_qty ~ S 회귀, 수요수준(정상판매)·요일·날씨 통제.
- d(closing)/d(S) ≈ 1 → 내놓는 만큼 팔림 = 공급주도 = 유도 큼 → **α 낮음**.
- closing_qty가 S에 무관하게 평평 → 고정 base = 진짜 수요 → **α 높음**.
- 포화 수준 = base B, S에 반응하는 몫 = 유도/떨이 capacity.

**Confound**: 고surplus날=저수요날 → 마감손님도 적음? 정상판매를 수요 proxy로 통제.

**A1/A2와 직교**: A1/A2는 가격(elasticity), A3는 가용성(공급주도 vs 수요주도). confound 방향이 달라 삼각측량 가치.

### Phase A 산출
- 세 α 추정 → 구간(점추정 아님, 프로젝트 관례). 어느 방법이 어느 쪽을 bound하는지 명시.
- 민감도. CI 겹침/구간 보고. (필요시 W0식 TOST 스타일.)
- 정직한 한계: A1/A2 모두 하한 쪽으로 편향 → α_A는 "적어도 이 이상" 성격. A3가 상한 균형.

---

## Phase B — 운영 α\*(c) + 원가율 역추정

### 목표
demand target D(α) = normal_qty + closing_qty×α (카테고리-day, v4 adjusted_demand와 동일). 이를 발주 정책에 넣어 총비용 최소 α\*(c)를 c 격자마다 산출 → α\*(c) 곡선.

**총비용** TC(α, c) = 폐기수량×정가×c + 마감손실(c) + 매진수량×정가×(1−c).

### ⚠️ 핵심 미해결 결정: counterfactual 발주 비용 산정

α는 **타깃(재구성 수요)**과 **비용 yardstick** 양쪽에 나타나 순진하게 하면 순환(α 상쇄)된다. 실제 발주는 과거 사실이라 바꿀 수 없고, "다르게 발주했다면"의 비용은 censoring(관측 안 된 초과수요) 때문에 수요분포 가정이 필요하다. 후보 두 개:

- **(B-1) Salvage-newsvendor (권장)**: 마감할인을 **salvage 채널**로 본다. 과잉생산이 전부 폐기가 아니라 일부는 마감으로 회수. newsvendor critical ratio CR = Cu/(Cu+Co), Cu=매진마진손실=정가×(1−c), Co=폐기비용이되 salvage(마감회수)로 경감. α(유도분)가 "salvage vs 진짜수요" 분할을 정한다. 관측된 폐기+마감+매진과 최적발주의 정합을 맞추는 α\*(c)를 푼다. 분석적이라 c 역추정에 직접 연결.
- **(B-2) Counterfactual 시뮬레이션 + censoring 모델**: 관측 위 수요분포를 모델링해 대체발주 비용 시뮬. 가정 많음 → 로버스트니스/폴백.

→ **이 formulation은 implementation plan(writing-plans)에서 스파이크로 확정.** 설계 단계에선 B-1을 primary, B-2를 fallback으로 고정.

### 원가율 역추정
- Phase A의 α_A(cost-free 구간)와 Phase B의 α\*(c) 곡선 교차 → **c_implied 구간**.
- c_implied가 베이커리 상식(대략 30~45%)에 들면 내부 정합성 확인.
- 고객사 c 제공 시: α\*(c_actual) 점 계산 → α_A와 직접 대조(설계 슬롯).

---

## 교차 검증 (최종 판정)

1. Phase A α_A (수요측, cost-free)
2. Phase B α\*(c) (비용측)
3. **수렴 여부**: α\*(c_implied) ≈ α_A → "떨이 해석이 경제적으로도 정합" 강한 결론. W0 흡수검증의 다중식별 삼각측량과 동일한 논리 구조.

## 절대규칙 준수

- **Time leakage 금지**: Phase B 백테스트는 rolling/expanding window, 미래 sales·weather·폐기 관측을 feature로 금지. lag 기반만.
- **품절 censored**: 마감분·품절 flag 보존. 판매모델과 위험모델 분리 유지.
- **Random split 금지**: 시간순 backtest.
- **MAPE 단독 금지**: Phase B 비용지표는 WAPE 보조 + 운영 cost.
- **Synthetic↔Real**: 실데이터 진입점 loader 경유.

## 산출물 (예정)

- `src/bakery/analysis/closing_demand.py` — Phase A 3방법 + α_A 집계 (기존 `_ols_fe` HC3 패턴 재사용, statsmodels 무의존)
- Phase B: 발주 백테스트 확장(기존 evaluation/backtest 활용) + α\*(c) 곡선 + c 역추정
- `reports/`: closing_alpha_estimates.csv, alpha_star_by_cost_rate.csv, cost_rate_implied.csv
- CLI: `closing-demand` (또는 기존 명령 확장)
- `docs/closing_discount_true_demand.md` — 고객사/PM 전달용 결과 리포트

## 테스트

- leakage 회귀 테스트(Phase B window).
- 분해 항등식 단위 테스트: normal + closing == total, B + I == C (정확값 `==`).
- A3 surplus 항등식: closing_qty + waste == surplus (실측 대조).
- degenerate 케이스(마감분 0인 날, 단일 depth만 있는 카테고리) 처리 — W0의 inconclusive false-pass 차단 교훈 적용.

## 정직한 한계

- 고객 ID 부재 → 딜헌터 식별은 행동(elasticity/가용성)에서만 간접. 개인 단위 검증 불가.
- A1/A2 하한 편향(commute/depth 내생성). A3로 균형하나 완전 무편향 아님.
- Phase B c 미지 → α\* 점 아닌 곡선. c_implied는 정합성 렌즈지 회계 원가율 아님.
- 광교 단독. 4매장 확장은 W0처럼 후속(일반화 근거).
- depth 2점 외삽·newsvendor 가정의 curvature/분포 리스크.

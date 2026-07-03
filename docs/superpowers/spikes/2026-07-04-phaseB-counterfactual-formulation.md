# Phase B counterfactual formulation 결정 (스파이크)

**Date**: 2026-07-04
**목적**: Phase B(운영 α\*(c) + 원가율 역추정) 본구현 전, counterfactual 발주 비용 산정 방식을 **B-1(salvage-newsvendor) vs B-2(시뮬레이션+censoring)** 중 확정한다. 코드 산출 아님 — 결정 + 근거.
**결론**: **B-1 salvage-newsvendor를 primary로 확정.** B-2는 로버스트니스/폴백.

---

## 0. 데이터 발견 (스파이크 중 확인 — 설계 전제 갱신)

`data/internal/v2/waste_alpha_4stores.parquet` (280,779행, 4매장, 2021~2025)이 예상보다 훨씬 풍부. **per-day-item로 다음이 이미 존재**:

| 컬럼 | 의미 | Phase B 역할 |
|---|---|---|
| `made` | 생산량 | 발주 Q (실제 발주 결정의 결과) |
| `out` | 실폐기량 | overage 실현분 (newsvendor 과잉비용) |
| `normal_qty` / `closing_qty` | 정상/마감 판매 (이미 분리) | 채널 분해 (Phase B 타깃 D(α) 재료) |
| `sold_total` | 총 판매 | 수요 하한 (out>0이면 uncensored) |
| `unit_price` | 정가 | 마진/비용 스케일 |
| `waste_cost` | = `out × unit_price` (**소매가**) | ⚠️ 원가 아님 |

**핵심 확인**: `waste_cost = out × unit_price` (row 검증: out=2, unit_price=4600 → waste_cost=9200). **원가율 미적용** → 사용자의 "원가율 모른다"와 정합. 진짜 폐기비용 = `out × unit_price × c`, c는 여전히 미지 파라미터. → **Phase B의 c=파라미터 + 역추정 설계 그대로 유효.**

이 데이터 덕분에 Phase B는 이전에 없던 것들을 갖는다: 발주 Q(made)·실현 폐기(out)·채널분해(normal/closing)가 **관측 물리량**으로 존재 → counterfactual 비용을 관측에 앵커링 가능(B-1의 핵심 전제 충족).

---

## 1. 순환 위험 (재현 확인)

Phase B 목표: `D(α) = normal + closing×α`를 발주 정책에 넣어 총비용 최소 α\*(c)를 찾기.

**순환**: α가 (i) 수요 타깃 D(α)와 (ii) 비용 yardstick 양쪽에 들어가면 상쇄된다. 순진하게 "재구성 수요 D(α)를 ground-truth로 두고 그에 맞춰 발주 Q(α)를 뽑아 D(α) 대비 비용" → Q도 D도 α로 스케일 → α가 소거되어 비용이 α에 무의미. **degenerate.**

**B-1이 끊는 법**: yardstick를 재구성 수요가 아니라 **관측 물리 결과**(made=Q, out=실폐기, sold_total=수요하한)로 둔다. 이건 과거 사실이라 α와 독립. α는 오직 "마감 물량 중 발주가 대비해야 할 진짜 수요 비율"로만 진입 → 소거 안 됨.

---

## 2. B-1 salvage-newsvendor — 성립조건 점검 (통과)

**아이디어**: 마감할인 = **salvage 채널**. 과잉생산이 전부 폐기가 아니라 일부는 마감으로 회수. α(유도분)가 "salvage로 흘러간 물량 중 진짜 수요 vs 순수 떨이" 분할을 정한다.

**Newsvendor critical ratio** CR = Cu / (Cu + Co):
- **Cu (underage, 부족)** = 매진 기회손실 = 정가 × (1 − c)
- **Co (overage, 과잉)** = 폐기비용 − salvage 회수. 순수폐기면 정가×c. 마감회수분은 salvage v=정가×(1−할인율) 회수 → 실효 Co 경감.

**필요 입력 — 전부 parquet에 존재**:
- 발주 Q = `made` ✓
- 과잉 실현 = `out` ✓
- 수요 하한 = `sold_total`; 채널 = `normal`/`closing` ✓
- 정가 = `unit_price` ✓
- c = 격자 파라미터 (미지, 역추정 대상)

**한 가지 모델링 선택 (censoring)**: 매진/절단 명시 플래그는 없음 → **`out==0`이면 잠재 매진(수요 절단), `out>0`이면 uncensored(수요=sold_total 확정)**로 추론. 이게 B-1이 필요로 하는 유일한 가정이며 관측 가능한 프록시로 충족된다. (W0의 "광교 무품절일 0" 문제와 달리, 여기선 out>0인 날이 다수라 uncensored 관측이 풍부 → 분포 식별 양호.)

→ **B-1 성립.** α–c–발주가 분석적으로 연결되고, c 역추정에 직접 쓰인다.

---

## 3. B-2 시뮬레이션 + censoring — 가정 비용 (폴백으로 강등)

**아이디어**: 관측 위 수요분포를 모델링해 "다르게 발주했다면"의 비용을 시뮬레이션.

**필요 가정**:
- 절단(매진)일의 수요 상방 분포 형태 가정 (parametric or 반모수).
- 발주 정책의 반사실 시나리오별 판매/폐기 시뮬 → 오차 누적.

**문제**: B-1이 관측 물리량으로 앵커링하는 데 비해 B-2는 미관측 상방을 추정 → 가정 많고 검증 어려움. parquet에 out(실폐기)가 있어 uncensored 관측이 풍부하므로 B-1의 앵커가 튼튼 → B-2를 primary로 쓸 이유 없음. **로버스트니스 체크로만 보관.**

---

## 4. 결정 + 원가율 역추정 스케치

**결정: B-1 primary, B-2 로버스트니스.**

**c 역추정 (Phase B 본구현에서)**:
1. Phase A α_A = **[0.85, 1.0]** (광교 bread/pastry, cost-free) — 이미 산출.
2. B-1로 α\*(c) 곡선 산출 (c 격자, 예 0.25~0.55).
3. **α\*(c) = α_A 교차 → c_implied**. c_implied가 베이커리 상식(≈0.30~0.45)에 들면 내부 정합성 확인.
4. **고객사 c 제공 시**: α\*(c_actual) 점 계산 → α_A와 직접 대조 (곡선이 점으로 붕괴). 설계 슬롯 유지.

**Phase A 결과와의 사전 정합성**: α_A≈[0.85,1.0]는 "마감물량 대부분이 진짜 수요"를 뜻함. B-1의 α\*(합리적 c)도 높게 나오면(마감회수 마진이 양수라 진짜수요로 취급) 수렴 → 강한 결론. 만약 α\*가 낮게(≪0.85) 나오면 "수요는 진짜지만 비용상으론 그만큼 생산할 가치 없음"을 의미 → descriptive vs objective α 괴리를 정량화 (이게 두-α 구분의 실증).

---

## 5. 범위 / 다음

- **Phase B 본구현 = 별도 플랜** (`docs/superpowers/plans/`에 후속 작성). B-1 salvage-newsvendor 발주 백테스트 + α\*(c) 곡선 + c 역추정 + 교차검증.
- 절대규칙: 시간순 rolling 백테스트(leakage 금지), WAPE 메인, 품절 censored 보존.
- 데이터: 위 parquet(광교 필터) 재사용. made/out/normal/closing/unit_price/sold_total.
- 다매장 확장은 W0처럼 후속(4매장 parquet 이미 있음 → 일반화 근거 쉬움).

## 6. 정직한 한계

- censoring을 out==0로 추론 → 마감 후 실제 매진 여부의 프록시(완벽 아님). 실 매진 로그 있으면 교체.
- salvage v(마감 회수가)는 할인율 depth(20/30%)별로 다름 — Phase A depth 분해 재사용.
- c_implied는 "수요 그대로 발주가 비용최적이 되는 원가율"이지 회계 원가율과 동일 보장 없음(정합성 렌즈). 고객사 c가 최종 검증.
- α\* 는 발주 정책(quantile) 선택에 의존 → 정책별 민감도 필요.

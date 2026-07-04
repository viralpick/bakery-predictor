# 전향적 KPI 비교 harness + WPE·Decoupling Score 진단 — 설계

작성 2026-07-05. 상태: **설계 제안(사용자 승인 대기)**. 관련 브랜치 `feat/lunar-holiday-leadup`(별도 브랜치 권장).

## 1. 목적

진짜 PoC(4주 구축 + 4주 전향적)의 채점표를 코드로 구현한다. 핵심 산출물 = **"우리 발주 추천 vs 아티제 현행 발주"를 합의 운영 KPI로 나란히 비교**하는 harness. 지금(데이터 수령 전) 만들어 두면 데이터 도착 시 loader만 바꿔 그날 돌아간다. 겸하여, harness의 수요 입력 품질을 재는 진단 지표(WPE·Decoupling Score)를 추가한다.

## 2. 핵심 설계 결정 (브레인스토밍)

- **반사실 처리 = 양쪽 동일 시뮬 + 실측 sanity (승인됨)**: 우리 발주량과 아티제 발주량을 *같은* 시뮬레이터(복원 수요 + 시간대 도착곡선)에 넣어 KPI 계산. 1차 보고 지표는 **Δ(우리 − 아티제)** — 시뮬 편향이 양쪽에 상쇄되므로 델타는 견고. 절대 KPI + 아티제 실측 KPI는 맥락/sanity로 병기.
- **시뮬 수요 입력 = potential_demand(복원값)** 필수: 두 발주량이 벌어지는 검열 구간(일찍 품절→이후 수요 미관측)의 tail이 필요하기 때문. → **③ WPE·ρ_DS는 별개 지표가 아니라 ①의 품질 게이트.**
- **검증 범위 = 회고 검증까지 지금 (사용자 자리비움 → 기본값 채택, 번복 가능)**: 재구성 생산(`생산 = 정상판매 + 마감판매 + 폐기`, 광교 항등식 90~93% 일치)을 아티제 발주 proxy로 써서 과거 광교 2021~25에 harness를 실제로 돌려 데모·검증 확보. 전향적 데이터 도착 시 baseline을 실제 발주로 교체.
- **granularity = item-day**: 매진시각이 본질적으로 item 단위. item-day에서 계산 → 카테고리·매장·기간으로 집계. 우리 모델은 카테고리 총량 예측→품목 배분(v4)이라 item-level 추천량이 나옴.

## 3. KPI 정의

합의 KPI 3종(1차) + 보조:

1. **폐기비용↓** — `waste_units = max(order − demand, 0)`, `waste_cost = waste_units × unit_price × cost_rate`. 기존 `business_metrics.simulate_profit` 재사용.
2. **매진시각 median↑** — item-day별 누적 도착곡선 `c(t)`에서 `soldout_time = min{t : c(t) ≥ order}`; 마감까지 미도달이면 "매진 없음"(censored). item-day median 보고. 늦을수록 좋음.
3. **매진률↓** — 마감 전 `c(close) ≥ order`인 item-day 비율.
4. (보조) WAPE / **WPE**(편향) / lost-sale units·cost / service level.

## 4. 반사실 매진시각 엔진 (새 핵심 로직)

주문량 `Q`와 복원 시간대 누적 도착곡선 `c(t)`가 주어지면 `soldout_time = min{t : c(t) ≥ Q}` (없으면 None).

- **도착곡선 재구성**: 비품절 item-day의 receipt hour/minute로 정규화 누적 프로필(요일·item/카테고리 조건부)을 만든 뒤, 그날의 **복원 일 총수요**로 스케일. 품절일의 검열된 tail은 이 프로필로 외삽(관측 판매만 쓰면 tail이 잘려 매진시각이 과대평가됨).
- 이 엔진이 "우리가 더 많이 발주했으면 몇 시에 팔렸을까"를 산출하는 부분.

## 5. 모듈 구조

기존 것 재사용, 신규 최소화:

- `evaluation/prospective.py` (신규):
  - `reconstruct_baseline_order(sales, waste)` — 생산 항등식 proxy (회고용).
  - `build_arrival_profile(receipts, *, by)` — item/카테고리·요일별 정규화 누적 도착 프로필.
  - `simulate_soldout(order_qty, arrival_curve)` — 반사실 매진시각/매진여부.
  - `simulate_item_day_kpis(orders, demand, profiles, params)` — item-day별 waste/lost/soldout_time/is_stockout.
  - `compare_policies(our_orders, baseline_orders, demand, ...)` — 나란히 KPI 표 + 델타 + 아티제 실측 sanity.
- `evaluation/metrics.py`: `wpe(y_true, y_pred)` 추가(`sum(yhat−y)/sum(y)`), `summarize`에 편입.
- `evaluation/diagnostics.py` (신규 or prospective 내): `decoupling_score(recovered_demand, stockout_rate, weights)` — **카테고리 레벨** 가중 Pearson(품절률, 복원수요). 0 근처=잘 분리, 강한 음수=복원 부족 경보. (품목 레벨은 흡수·검열 이중편향으로 식별불가 → 카테고리 한정.)
- `cli.py`: `prospective-eval` 커맨드.

demand 입력은 `potential_demand`(복원) 경로 사용. waste/lost 비용은 `business_metrics` 재사용(중복 금지).

## 6. 데이터 흐름

```
receipts(hour/minute) ─┬─ build_arrival_profile ──────────────┐
                       └─ potential_demand(복원 일총수요) ──┐  │
현행 발주(회고: 재구성생산 / 전향: 실제발주) ── baseline_orders │  │
우리 모델(category→item 배분) ──────────────── our_orders     │  │
                                                             ▼  ▼
                              simulate_item_day_kpis(orders, demand, profiles)
                                        │ (양쪽 각각)
                                        ▼
                              compare_policies → KPI 표(우리|아티제|Δ) + 실측 sanity
                                        │
                              diagnostics: WPE(예측편향), ρ_DS(복원 검열잔량)
```

## 7. 테스트

- `simulate_soldout`: 단조 도착곡선에 대해 Q↑ → soldout_time↑(또는 None); Q≥총수요면 None(매진 없음); Q=0이면 즉시 매진. 정확값 단언.
- `reconstruct_baseline_order`: 합성 (정상+마감+폐기) → 발주 proxy 정확 복원.
- `wpe`: 과대예측이면 양수, 과소예측이면 음수, 완벽예측이면 0 (정확값).
- `decoupling_score`: 복원=원판매(미복원)면 강한 음수, 완전복원면 ~0 (합성으로 부호·크기 검증).
- `compare_policies`: 동일 발주 두 개 넣으면 Δ=0; 우리가 정확히 더 나은 합성 케이스에서 폐기↓·매진시각↑ 방향 검증.
- leakage: 도착 프로필·복원이 미래 정보 안 쓰는지(전향 경로) — split 이후 계산 회귀 테스트.

## 8. 미해결/후속

- 재구성 생산 proxy는 근사(항등식 7~10% 갭). 전향 데이터의 실제 발주로 교체 시 확정.
- 도착 프로필의 품절 tail 외삽은 potential_demand 품질에 의존(WPE로 감시).
- ② 전향적 데이터 수급 계약(schema/loader)은 데이터 수령 시점에 진행(별도).
- ④ classification 트랙 백로그.

## 9. 승인 게이트

이 설계 승인 후 writing-plans로 구현 계획 작성. 그 전까지 코드 미작성.

# 명절 EventLevelPrior 확장 스캔 결과 (트랙2) — 광교

> ⚠️ **정정 배너 (2026-07-18)**: 이 스캔은 **깨진 공휴일 feature(calendar_raw 2021-23 누락)** 기반 base로
> 측정됨. 이후 feature 파이프라인 수리(`docs/holiday_premium_decomposition.md`)로 base가 크게 개선.
> **채택(어린이날) 결론은 유지**(수리 후 평일-lens에서도 base 0.118→blend 0.088로 prior 기여 확인).
> **기각(광복절·신정·설날·삼일절·부처님) 결론도 유지** — 단 실패 모드가 다름: 이 문서의 err25 수치는
> broken-feature base라 **stale**. 진짜 기각 사유는 **하락기 절대-median 앵커의 과대(decline-overshoot)로
> feature 수리와 무관**(수리는 base의 공휴일 blind만 고침, 앵커의 하락 미추종은 그대로). 수치는 참고만.

모델 개선 4트랙 중 **2번(특수일 prior 확장)** 검증. 이상치 분석에서 "크게 튄 날 12일 = 명절"이나
광교 EventLevelPrior엔 xmas+추석만 등록됨. 후보 명절 6종을 per-event OOS A/B(leave-past-out,
train window 730d, base LGBM vs blend)로 평가 → **prospective 순개선만 채택**.

## 결론: **어린이날만 채택**. 나머지 5종 기각.

`STORE_EVENT_PRIORS["광교"]` events에 `childrens (5,5)` 추가. (추가 커밋 없음.)

## 채택 게이트 (advisor)
prior는 두 조건을 **동시에** 만족해야 채택:
- **(a) base에 일관된 방향의 systematic bias** — 단일 레벨 앵커는 레벨만 이동시킴. 방향이 해마다
  뒤집히는 이벤트는 앵커로 못 고침(변동성만 추가).
- **(b) err25 작음** — 2025 fold(=2026 prospective proxy) blend 오차. pooled WAPE는 고수요 연도(2023)
  편중이라 낙관 편향 → 현재 레벨에서의 오차로 게이트.

## 스캔 결과 (full=전체 history median / recency2=최근 2회 median)
| 명절 | full WAPE→blend | err25 | recency2 err25 | base bias | 판정 |
|------|------|------|------|------|------|
| **어린이날** | 0.210→**0.179** | **−7%** | −17% | −0.185 (일관) | ✅ **채택** (full) |
| 광복절 | 0.212→0.148 | +19% | +19% | −0.170 | ❌ err25 과대 |
| 신정 | 0.153→0.129 | +26% | +21% | −0.153 | ❌ err25 과대 |
| 설날 | 0.248→0.233 | +14% | +3% | +0.041 (불일치) | ❌ 방향 불일치 |
| 삼일절 | 0.185→0.234 | +11% | +0% | −0.146 | ❌ WAPE 악화·불안정 |
| 부처님오신날 | 0.035→0.115 | n/a | n/a | ~0 | ❌ 이벤트효과 없음 |

- **어린이날**: base가 4/5년 과소예측(bias −0.185, 일관) + anchor median 317 ≈ 2025 actual 318 → 두 게이트 통과.
  유일하게 레벨이 안정(315/371/319/229/318). residual risk = 2024 blend +29% overshoot(actual 229 이상저조년).
- **설날**: base bias +0.041은 −25/+25/+38/−11%의 **상쇄**일 뿐(방향 불일치). recency2 err25 +3%는 3fold 중
  우연 — sign-flip은 레벨 앵커로 못 고침. 부처님(base 이미 무편향)과 같은 "prior가 변동성만 추가" 케이스.

## ★Throughline (트랙1과 동일한 원인)
광복절·신정이 (b) 실패 = **트랙1의 −26% 수요 하락이 이벤트 레벨에서 재출현**. 730d base는 이미 하락한
레벨을 추종하는데, historical-median prior가 **옛 (높은) 레벨로 도로 끌어올림** → 2025/2026에 과대예측(폐기).
- 광복절 base err25 −10% → blend +19%: prior가 하락을 되돌려 과대.
- recency-2로도 못 고침: 340→271(−20%) 급락이 2점 median엔 너무 가파름.
- 이벤트 *레벨*과 *lift* 둘 다 하락(광복절 lift 1.45→1.37→1.11)이라 **모든 backward 앵커가 prospective 과대**.
- 어린이날만 레벨이 진짜 안정이라 생존. **트랙2의 정직한 결과 = 채택 가능 명절 1종, 나머지는 하락에 막힘.**

## recency-limit: tested, not adopted
`EventLevelPrior(recency=N)` 추가(기본 None=전체 history, 기존 동작·13테스트 불변). 스캔서 검증:
어린이날 악화(−7%→−17%), 설날만 우연히 개선 → **일반 해법 아님, 채택 config 어디도 미사용**. dormant.

## 헤드라인 회귀 확인 (광교 총량, 730d·52folds)
| | WAPE | 매진율 | 폐기(surplus_rate) |
|---|---|---|---|
| before (xmas+chuseok) | 7.967% | 22.80% | 8.64% |
| after (+childrens) | **7.901%** | **22.53%** | 8.66% |

어린이날 연 1일이라 총량 aggregate는 미미 이동 — WAPE 소폭↓, **매진 −0.27pp**(과소예측 교정),
폐기 불변. 메커니즘 = 과소예측일 매진 방지(폐기 무영향). 회귀 없음. 486 pytest 통과.

## 재현
- `scripts/scan_holiday_priors.py` — 후보 6종 × recency(None/2) A/B. 결과 `reports/scan_holiday_priors*.log`.
- `scripts/verify_event_prior.py` — 등록 이벤트 회귀 검증(concerns=blend>base 감시). 광교 childrens/chuseok/xmas 전부 통과.
- 부처님오신날 양력 datemap은 스캔 스크립트 내 정의(기각돼 calendar.py 미반영).

## 다음 (모델 개선 4트랙)
- 트랙2 = 종료(어린이날 채택). 트랙1·2 공통 결론: **수요 −26% 하락이 recalibration/prior 양쪽의 상한**.
- 다음 = 트랙3(주말·여름 계절 과소예측). 트랙4(극한날씨)는 사용자 지시로 후순위.

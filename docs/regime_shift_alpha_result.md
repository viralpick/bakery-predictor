# 다중시각 재검증 ① — 2025-01-17 마감할인 depth cut 자연실험 (정직본)

**Date**: 2026-07-04
**매장/범위**: 광교(보나비), bread·pastry 카테고리, 2021~2025 item-day 패널
**질문**: 마감할인 depth를 30%→20%로 줄였을 때 마감판매(closing)가 변했나? 변화가 없으면 마감판매는 가격민감 떨이수요가 아니라 supply-driven 잔여물이고, 이는 α(마감판매 중 실수요 비율)가 높다는 방향의 증거다.
**배경**: Phase A가 structural α 식별 실패(α_A=NaN). Phase A2(depth elasticity)는 20% vs 30%를 "동시 존재하는 두 depth"로 오해해 degenerate(NaN) 처리했으나, 실제로는 **시간축으로 분리된 정책변경**이었다.

## 핵심 발견 — 메모리 전제 교정

메모리는 "광교 저녁 마감할인 5년 상시 → α 미식별"이라 기록했다. 실제로는 **할인의 *존재*가 상시일 뿐, depth는 2025-01-17에 정책변경**됐다:

| 할인코드 | 내용 | 유효기간 | 사용 |
|---|---|---|---|
| 0077 | 마감할인(30%) | 2012-01-01 ~ | **N (중단)** |
| 0069 | 마감할인(20%) | **2025-01-17 ~** | Y (현행) |

패널의 실효 depth로도 정확히 확인: 2024년 내내 **0.300**, 2025-02부터 **0.200**, 2025-01이 전환월(0.27, 부분). **4개 매장 전부 동시 전환**.

## 식별의 한계 — 왜 총수요로는 α를 못 구하나

이 정책변경은 "깨끗한" 자연실험이 **아니다**:
1. **대조군 부재** — 전 매장이 동시 전환. depth 30%를 유지한 control 매장이 없다.
2. **never-discounted 품목도 부적합** — 광교 185품목 중 23개뿐이며 정가판매량의 **0.9%**. 대조군으로 쓰기엔 너무 작다.
3. **총수요는 세속추세가 압도** — 4년 공통 balanced 품목이 16개뿐이고 YoY가 −38%(2024)→+34%(2025)로 요동. depth 효과를 이 소음에서 뽑을 수 없다.

→ 총수요(N+C) 채널에서 α를 점식별하는 것은 이 데이터로 불가능. **명시적으로 포기**한다.

## 식별 전략 — 내부통제되는 구성비

유일하게 robust한 신호는 **구성비**다:

    closing_share = closing_qty / (normal_qty + closing_qty)

매장 전체 수요충격은 normal·closing을 비례로 움직이지만, **할인 매력도 변화는 N/C 비율을 특정 방향으로 움직인다**. 마감구매자가 marginal bargain-hunter라면, 할인을 30→20%로 얕게 만들 때 closing_share가 **떨어져야** 한다.

**추정식** (`src/bakery/analysis/discount_regime.py`, CLI `regime-alpha`):
`closing_share ~ post_cut + trend + month FE + item FE`, item FE는 within-demeaning(FWL)으로 흡수, HC3 robust SE. 추가로 **placebo break-date 분포**(실제 cut 이전 가짜 전환일들)로 경험적 null 구성. 전환월(2025-01, 부분 depth)은 제외.

## 결과

| 카테고리 | n(item-day) | closing_share β (95%CI) | closing/made β (95%CI) | placebo max\|β\| | verdict |
|---|---|---|---|---|---|
| bread | 20,351 | **+0.0027** [−0.0083, +0.0138] | −0.0063 [−0.0155, +0.0029] | 0.0280 | **depth_invariant** |
| pastry | 39,406 | **−0.0145** [−0.0226, −0.0064] | −0.0204 [−0.0273, −0.0135] | 0.0424 | **depth_invariant** |

- **bread**: depth cut의 closing_share 효과 +0.27%p, CI가 0 포함. placebo(랜덤 전환일)의 β는 −2.8%p~+2.3%p로 실제 효과보다 **훨씬 큼** → 실제 정책변경이 랜덤 날짜보다도 non-event. 명확한 null.
- **pastry**: 효과 −1.45%p로 HC3 CI는 0을 배제(소폭 하락, 얕은 할인→약간 적은 마감판매, mild bargain-hunting 방향과 일치). **그러나 placebo에 −2.8%p, −4.2%p 규모의 break가 존재** → 랜덤 날짜도 이만한 구성비 이동을 만든다 → depth cut에 귀속 불가. placebo noise 대역 내 → depth_invariant.

## 결론

**depth를 33% 축소(30→20%)해도 마감판매 구성비가 랜덤 break 이상으로 움직이지 않는다.** 두 카테고리 일관. 이는:
- 마감판매가 **할인 깊이에 비탄력적** = 가격민감 떨이수요가 채우는 게 아니라 **supply-driven 잔여물**(그날 과잉생산분이 21시에 할인 소진). Phase B의 "made 근방 d 내생성" 위험이 독립 증거로 재확인됨.
- 마감구매자가 "30%라 억지로 온" marginal bargain-hunter였다면 20%에서 이탈해 closing_share가 떨어졌어야 하는데 안 떨어짐 → **높은 α 방향**(원하는 물건이라 할인 깊이와 무관하게 산다).

**단, 숫자 α는 여전히 점식별되지 않는다.** 이 재검증은 A1/A2/A3(Phase A)에 이은 **세 번째 독립 방법**으로, "자연실험으로도 depth는 α의 instrument가 못 된다 — 그 이유는 마감채널이 supply-driven이기 때문"임을 정직하게 규명한다. 방향(높은 α)은 강화됐다.

## 재현

```bash
uv run bakery regime-alpha           # reports/regime_shift_estimates.csv
uv run pytest tests/test_discount_regime.py
```

## 한계

- 구성비 신호는 store-level 수요충격엔 robust하나, **N/C 대체가 아닌 채널이동을 정밀 분해하지 못함**. "얼마가 정가로 이동했나"의 크기는 총수요 소음 때문에 미확정.
- placebo 5개는 정식 permutation 검정이 아닌 참고 분포. β의 크기 판단은 이 소규모 분포 기준.
- pastry의 mild 음(−1.45%p)은 "약한 bargain-hunting 존재" 가능성을 완전히 배제하진 않음 — 다만 랜덤 break와 구분 불가한 수준.

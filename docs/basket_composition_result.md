# 다중시각 재검증 ③ — 마감할인 basket 구성 (정직본)

**Date**: 2026-07-04
**매장/범위**: 광교(보나비), 영수증 line-item 2021~2025, basket=(판매일자·POS번호·영수증번호)
**질문**: 마감할인 품목을 산 고객이 같은 영수증에서 정가품도 사는가? 산다면 "실쇼핑"(마감품은 원래 살 물건 → 높은 α), 할인품만 담고 떠나면 "떨이 사냥"(할인이 만든 방문 → 낮은 α). ①(depth cut)이 supply-driven confound로 막혔으니, depth와 무관한 **행동**을 직접 본다.

## 데이터 정정 — basket key

공유 로더(`load_sales_with_discount`)는 영수증을 **영수증번호 단독**으로 묶는데, 이 번호는 매일 순환·재사용돼 661개로 충돌(영수증당 평균 681 line = 비현실). 진짜 basket = **(판매일자, POS번호, 영수증번호)** 복합키 → **242,506 baskets, 평균 1.86 line/basket**(median 1, max 28) = 정상 고객 basket. 단일매장(광교). ③ 전용 로더(`_load_basket_inputs`)로 복합키 재구성.

## 결과

`mixed_rate` = 마감 basket 중 정가 line도 담은 비율. `fp_value_share` = 마감 basket들의 총결제액 중 정가 line 금액 비중.

| scope | closing_cat | n_closing | mixed_rate | fp_value_share | size(closing/other) |
|---|---|---|---|---|---|
| all | any | 30,244 | 0.011 | 0.012 | 1.96 / 1.84 |
| pre_cut 30% | any | 25,564 | **0.007** | 0.006 | 1.96 / 1.85 |
| post_cut 20% | any | 4,680 | **0.031** | 0.034 | 1.99 / 1.82 |
| pre_cut 30% | bread | 12,578 | 0.007 | 0.005 | 2.18 / 1.84 |
| post_cut 20% | bread | 2,469 | 0.030 | 0.032 | 2.25 / 1.81 |

- 마감 basket이 정가품을 담는 비율은 **0.7~3.6%로 매우 낮다**. 마감 basket 크기(2.2)가 비마감(1.8)보다 크지만, 그 line들은 정가품이 아니라 **다른 할인품**이다.

## 핵심 — 이 신호는 저녁 timing에 confound된다

낮은 mixed_rate를 "떨이 사냥"으로 읽으면 **오답**이다. 마감할인은 저녁 time-lock이다:

- **마감 line의 시각 분포**: 20h 64.6% + 21h 25.6% = **90%가 20-21시**. (18h 2.6%, 19h 4.4%)
- **정가 line 비율(시각별)**: 주간(7-19h) 67~79% → **20h 6.0% / 21h 4.1% / 22h 4.2%로 붕괴**.

즉 마감구매가 일어나는 20-21시에는 매장 판매의 ~95%가 이미 할인품이다. 마감 basket에 정가품이 1-3%뿐인 것은 **그 시각에 정가 재고가 거의 없기 때문**이지, 고객이 "떨이만 노려서"가 아니다. → basket 구성은 **[[project-closing-discount-alpha]] Phase A의 저녁 미끼상품 잠식**을 재현할 뿐, 독립적인 α 판별자가 되지 못한다.

## pre/post 변화는 방향만, 크기는 confounded

depth cut 후 mixed_rate가 0.7%→3.1%로 올랐다(정가 동반이 늘어남 = 얕은 할인이 상대적으로 실쇼핑객을 더 남김 = 높은 α 방향). **그러나**:
- 같은 시기 저녁(20-21h) 정가 line 비율도 5.3%→6.5%로 올랐고 저녁 closing-share는 91.9%→90.6%로 내렸다 → mixed_rate 상승의 상당분이 **저녁 정가 재고 증가**로 설명됨.
- post 기간은 1년(2025)뿐이라 secular 추세와 분리 불가(① 총수요에서 본 것과 동일한 소음).

→ 방향은 높은 α와 모순되지 않으나, **크기를 α 증거로 쓸 수 없다**.

## 결론

**basket 행동 신호는 마감할인의 저녁 time-lock에 confound돼 α를 독립적으로 식별하지 못한다.** 사용자가 기대한 "③이 높은 α를 독립 확인" 은 나오지 않았다 — ③은 ①과 마찬가지로 confound에 막혔고, 그 이유가 구조적임을 정직하게 규명한다:
- **①**: 마감 volume이 supply-driven → depth가 instrument 못 됨.
- **③**: 마감이 저녁 time-lock → 그 시각 정가품 소진 → basket 구성이 timing에 종속.

**종합**: 광교에서 α는 A1/A2/A3(Phase A) + Phase B + 다중시각 ①③ 어느 경로로도 점식별되지 않는다. 이유가 이제 명확하다(상시 저녁 + supply-driven + time-lock). 약한 방향 증거(①의 depth-불변, ③의 pre/post)는 **높은 α와 모순되지 않으나 확증은 아니다**. 실무 함의는 Phase B의 operational implied-c(0.20/0.28, 매장 이미 cost-rational)가 여전히 가장 확실한 산출물이다.

## 재현

```bash
uv run bakery basket-alpha           # reports/basket_composition.csv
uv run pytest tests/test_basket_composition.py
```

## 한계

- 고객 식별자 부재 → **재구매(다음날 정가 복귀)** 분석 불가. 있었다면 timing confound를 우회할 수 있었다.
- 시각 통제 basket 비교(같은 20-21h 내 closing vs non-closing)는 비closing 저녁 basket이 희소(4-6%)해 검정력 없음.
- pre 4년 vs post 1년 비대칭.

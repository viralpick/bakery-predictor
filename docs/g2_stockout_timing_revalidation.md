# G2 stockout-timing 소비처 재검증 (하위2 ②)

**Date**: 2026-07-08
**배경**: `is_stockout` 92%→60% + `stockout_time`(첫 순간품절 이벤트 → 마지막 실판매) 재정의(하위1)가
stockout timing을 쓰는 소비처의 결론을 바꾸는지 검증.
**코드**: `scripts/revalidate_popularity_stockout.py`, 산출 `reports/revalidate_g2/`

## 소비처 분류 (stockout timing 직접 소비만 ② 대상)

| 소비처 | 신호 | 판정 |
|---|---|---|
| `models/item_proportion.py` | `avg_stockout_h`→`stockout_rank_pct`→`adj_stockout`(±20%), Stage2 품목비율 | **② 핵심 (라이브 배선)** |
| `analysis/popularity.py` | `avg_stockout_h` per-category 25/75분위 early/late 추천 | ② (advisory) |
| `models/stockout_classifier.py` | label=`is_stockout`(bool) | G3(bool)·반쯤 deprecated → 재학습은 별도 낮은 우선순위 |
| `evaluation/prospective.py` (retro 매진시각 KPI) | `soldout_median_h` = **potential_demand+도착곡선 시뮬** (stockout_time 직접 아님) | **③ 영역** (adjusted_demand 재설계와 합류) → ② 제외 |

## 방법 — 통제 비교

`build_store_daily`(옛 정의) 산출물의 `sold_units`·`closing_qty`(=회귀·비율 입력)를 고정하고,
`absorption_4stores.apply_fixed_stockout`로 처치변수(`is_stockout`/`stockout_time`)만 새 정의로
override → 같은 Y에서 **stockout_time 재정의 효과만 격리**. `item_proportion.compute_proportions`
(라이브 Stage2)를 옛/새로 각각 돌려 `avg_stockout_h`→`stockout_rank_pct`→`adj_stockout`→
`proportion` 차이 측정. cutoff=2025-12-31, 대상 bread/pastry.

## 결과 (bread/pastry)

| 매장 | 품목 | Spearman(rank_pct 옛 vs 새) | avg_h NaN 옛→새 | mean\|Δprop\| | max\|Δprop\| | mean\|Δadj\| |
|---|---|---|---|---|---|---|
| 광교 | 34 | +0.407 | 0→0 | 0.134%p | 0.411%p | 0.052 |
| 삼성 | 33 | +0.306 | 0→1 | 0.135%p | 0.447%p | 0.055 |
| 메세나 | 35 | +0.566 | 1→0 | 0.058%p | 0.308%p | 0.039 |
| 광화문 | 38 | +0.669 | 0→0 | 0.045%p | 0.149%p | 0.030 |

## 판정: Stage2 배분은 재정의에 **로버스트** (랭킹은 재정렬)

1. **인기 랭킹은 중간 정도 재정렬**(Spearman 0.31~0.67). 옛(아침 첫 순간품절)과 새(저녁 마지막 실판매)는
   서로 다른 물리량을 측정하므로 당연. **새 정의가 의미상 타당**(실제 완판 속도 = 진짜 인기).
2. **그러나 최종 Stage2 품목비율 영향은 무시 수준(<0.5%p/품목)**. `adj_stockout`이 `base_sold`(최근 28일
   점유율) 위에 얹히는 ±20% 유계 nudge이고, Σproportion=1 정규화를 거치면 랭킹 재정렬이
   비율로 거의 전파되지 않는다. base_sold가 지배적.
3. **avg_stockout_h NaN(매진 안 하는 품목→boost 0)은 거의 안 늘어남**(0~1건). is_stockout이 일단위
   60%라도 5년 history 집계에선 모든 품목이 매진일을 다수 가져 avg_h가 정의됨. "60%라 boost 상실" 우려는 기우.
4. **라이브 v4 스택**(`cli.py` v4-category-stack)은 `_load_real_daily`=fixed `bonavi_daily`를 쓰므로
   **광교 라이브 Stage2는 이미 새 정의로 동작 중**. 옛 정의는 4매장 `store_daily` 분석 스크립트에만 잔존(라이브 아님).

## 함의

- **배분(allocation)**: 재정의 무영향 → v4 Stage2 그대로 유효.
- **해석/설명 층**: 랭킹 자체를 노출·설명하는 곳(v7 온톨로지 "가장 인기 = 가장 이른 매진")은
  정정된 정의를 써야 정확(라이브 bonavi 경로는 이미 정정됨).
- **잔여**: `stockout_classifier` 재학습(label 92%→60% 재기반, G3·낮은 우선순위), retro 매진시각
  KPI 재측정(③ adjusted_demand 재설계와 합류).

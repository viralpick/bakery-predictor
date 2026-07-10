# conformal 잔차 진단 결과 (#2, diagnosis-only)

**날짜**: 2026-07-10
**대상**: 광교(store_gw01), item 경로 conformal 발주(PR#30)
**config**: s=0.74, n_folds=8, val_weeks=8, cal_fold_frac=0.5, α=0.5 (헤드라인 재현)
**스크립트**: `scripts/diagnose_conformal_residual.py`
**원본 로그**: `reports/log_conformal_residual_diagnosis.txt`, row-level `reports/conformal_residual_diagnosis.csv`

## 배경 / 질문

one-sided scale-정규화 split-conformal 발주(item 경로)의 실현 초과율이 nominal(1−s)보다
균일하게 +0.02~0.07 높다(s=0.74에서 +0.039). exchangeability 하에서 coverage가 맞아야
하는데 왜 under-cover 하는가? 메모리의 유력 가설은 **cal(과거)↔test(최근) 시간 드리프트**였다.

**진단만 한다** — 코드/모델 변경 없음. 결과로 #2 종료(PoC 충분) 여부를 판단한다.

## 재현

- cal folds=[4,5,6,7](과거) / test folds=[0,1,2,3](최근), cal Q_s(s=0.74, method=higher)=**0.992**
- test item-days=7,012 / cal=6,693
- **전체 실현 초과율 = 0.3056** (nominal 0.26, 잔차 **+0.0456**)
- CLI 헤드라인(0.299)과 미세하게 다른 이유: CLI `prospective-eval`은 `_fill_our_order`의
  inventory join·품절일 처리 등 downstream row 필터를 거친 population에서 초과율을 재고,
  이 스크립트는 conformal 출력 vs adjusted_demand를 **직접** 측정한다. 구조 진단 목적엔 직접
  측정이 더 정확하며(=conformal이 실제로 겨누는 잣대), 소수점 셋째 자리 차이는 population 차이다.

## 결과 — 3축 분해

### [A] 시간(test fold) — 부차적이나 실재하는 tail 차이

| fold | min_date | n | 초과율 | 잔차 | 필요 Q_s | (필요−cal) |
|---|---|---|---|---|---|---|
| 3 | 2025-05-22 | 1680 | 0.318 | +0.058 | 1.151 | +0.159 |
| 2 | 2025-07-17 | 1728 | 0.296 | +0.036 | 1.118 | +0.126 |
| 1 | 2025-09-11 | 1817 | 0.277 | +0.017 | 1.055 | +0.063 |
| 0 | 2025-11-06 | 1787 | 0.332 | +0.072 | 1.331 | +0.339 |

- normalized score 분포는 cal과 test의 **median이 거의 동일**(cal +0.595 vs test +0.593) —
  **location(위치) 드리프트는 없다**.
- 그러나 coverage를 지배하는 것은 0.74-분위이고, 그 값은 cal→test로 **상승**했다:
  **cal q74 = 0.992 → test q74 = 1.162**. 이 gap이 전체 test 초과율(0.306)이 nominal(0.26)을
  넘는 직접적 이유다. 즉 test 기간은 cal이 제공한 것보다 큰 마진을 실제로 필요로 했다.
- 단 시간순 단조는 아니다(fold1 최저, fold0/3 최고 = 추세가 아니라 계절성). → **깨끗한 시간
  추세는 없으나, 비단조적 cal↔test tail 차이(q74 0.99→1.16)는 실재**한다.
- **판정**: 시간 요인은 **부차적(secondary)**. 더 최근/가중 calibration 창은 **저우선·미검증
  레버**로 남는다("불필요"가 아니라 "우선순위 낮음"). 주 원인은 아래 [B].

### [B] volume tier — 지배적 원인 (volume 이질성)

item pre-cutoff 평균(scale)의 3분위. floor(1.0)에 걸린 품목 0개, 총 67품목.
tier 경계(scale): low 1.0~4.0 / mid 4.0~8.3 / high 8.3~19.8.

| tier | n | 초과율 | 잔차 | 필요 Q_s |
|---|---|---|---|---|
| **low** | 2404 | **0.629** | **+0.369** | **4.65** |
| mid | 2335 | 0.141 | −0.119 | 0.71 |
| high | 2273 | 0.133 | −0.127 | 0.80 |

- pooled Q_s=0.992가 **저volume 품목을 심하게 under-cover**(필요 4.65인데 0.99 부여),
  중/고volume은 **over-cover**(필요 0.71~0.80인데 0.99). 전체 잔차 +0.046은 사실상 전부
  저volume 품목에서 발생.
- 원인: scale=**평균** 정규화가 간헐적 slow-mover의 heavier-tail(높은 변동계수)을 못 맞춘다.
  간헐수요일수록 정규화 후에도 상대 잔차가 크다(Poisson-like). 평균으로 나눠도 tail이 남는다.

### [C] day_type — 2차 효과 (volume과 독립 주장 안 함)

| day_type | n | 초과율 | 잔차 |
|---|---|---|---|
| holiday | 304 | 0.365 | +0.105 |
| weekend | 1970 | 0.430 | +0.170 |
| **weekday** | 4738 | 0.250 | **−0.010** |

- **평일(전체의 68%)은 거의 정확**하게 calibrate. 주말/명절만 under-cover.
- ⚠️ [B]·[C]는 각각 marginal 분해라, 주말 +0.17이 **volume과 독립인 별개 분산 효과인지, 아니면
  저volume 품목이 주말에 몰려 나타난 것인지 이 표로는 구분 못 한다**(교차 분해 안 함). "2차"로만
  기술하고 직교성은 주장하지 않는다.
- median base는 요일·명절 피처로 평균 이동은 이미 잡으므로, 홀리데이 Mondrian은 median엔 이득이
  작다 — 원래의 "홀리데이 Mondrian 드롭" 판단과 부합.

### [운영 비용 측정] tier별 부족(shortfall) / 과잉(overage) 물량

⚠️ 이건 **frequency가 아니라 magnitude** — 초과 "빈도"만으론 운영 영향을 알 수 없어 물량(units)을
측정한다. 자기 발주 내 shortfall/overage는 baseline 스케일 오염 캐비엣과 무관(같은 잣대 안).

| tier | shortfall(u) | overage(u) | shortfall/day | overage/day | net(order−demand, u) |
|---|---|---|---|---|---|
| **low** | **5,879** | 1,951 | 2.4 | 0.8 | **−3,928** (과소) |
| mid | 744 | 8,201 | 0.3 | 3.5 | +7,457 (과잉) |
| high | 858 | 12,382 | 0.4 | 5.4 | +11,524 (과잉) |
| **합** | **7,480** | **22,534** | | | +15,054 |

- **저volume tier가 전체 shortfall의 79%**(5,879/7,480 units)를 차지 — "slow-mover under-cover는
  물량이 작다"는 직관은 **측정으로 반증**됐다. 저volume은 절대 units로도 과소발주의 주범이다.
- 반대로 중/고volume은 크게 **과잉발주**(net +7,457/+11,524 units), 전체 overage 22.5K ≫
  shortfall 7.5K(s=0.74가 median 위라 순 과잉발주는 정합).
- 즉 pooled conformal은 **저volume 과소 ↔ 중/고volume 과잉**으로 물량을 mis-allocate한다.

## 결론

1. **드리프트는 주 원인이 아니라 부차적** — location 이동은 없으나(cal/test median 동일),
   coverage를 지배하는 q74가 cal 0.99→test 1.16으로 실재 상승. 더 최근/가중 cal 창은 저우선 레버.
2. **잔차의 지배적 원인은 volume 이질성** — pooled scale-정규화 Q_s가 저volume slow-mover를
   under-cover(0.63, 필요 Q_s 4.65)하고 중/고volume을 over-cover(0.13). 평균-정규화로는
   간헐수요 tail을 못 맞춤.
3. 요일 효과는 2차(volume과의 교차는 미확인).
4. **운영적으로**: pooled 발주는 저volume 과소(전체 shortfall의 79%) ↔ 중/고volume 과잉
   (overage 22.5K)으로 물량을 잘못 배분한다.

## 종료 vs 보강 (architect 결정)

- (A) **#2 종료** — 전체 잔차 +0.046으로 작고, 운영 주력(평일 68% + 중/고volume)은 이미 잘
  calibrate. 동기였던 드리프트 가설이 주범 아님이 확인됨. 이미 0.679→0.30으로 개선된 경로의
  fine-tuning이라 PoC 충분.
- (B) **volume-Mondrian conformal** — tier별 Q_s 분리. **당초 예상(폐기 대가의 tradeoff)과 달리,
  측정 결과 Pareto 개선 여지**: 중/고volume 발주↓(overage 22.5K 축소=폐기↓) + 저volume 발주↑
  (shortfall 79% 완화=결품↓)을 동시에. 단 (i) 저volume tier의 실현 demand(6.4/day)가 historical
  scale(1~4)보다 커 **tier·scale 추정 자체가 stale**할 수 있고(성장/비정상성), (ii) tier당 표본이
  작아 tier별 Q_s 분산이 큼. PoC 필수는 아니나, 순가치가 음수라던 이전 판단은 측정으로 뒤집혔다.

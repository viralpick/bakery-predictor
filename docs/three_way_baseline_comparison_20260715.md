# 3-way baseline 비교 — naive vs 아띠제 vs 우리 모델 (2026-07-15, 광교)

**질문**: (a) daily 빵 판매 총량(카테고리 합) 점추정 정확도, (b) 운영 KPI(폐기/매진률/매진시각)를
현 상태에서 어느 수준까지 달성 가능한가. 세 비교군 = naive baseline / 아띠제 현행-로직
재구현 / 우리 최신 모델(카테고리-총량 + 특수일 prior).

**⚠️ 개념 구분**: (a)는 **수요예측 정확도** — naive·우리는 예측기, 아띠제는 발주정책이라
WAPE 대상 아님(발주=수요+안전마진). (b)는 **발주정책 KPI** — 세 정책(및 실제 생산량
reconstruction)을 같은 시뮬로 비교.

---

## (a) 카테고리-총량 점추정 정확도 (광교, 52 weekly folds, n_test=364, α=0.8)

| 모델 | WAPE | WPE(bias) | 출처 |
|---|---|---|---|
| moving_average(28d) | 14.84% | +0.50% | `naive_category_accuracy.py` |
| **seasonal_naive(4주 동일요일)** | **8.19%** | +0.50% | `naive_category_accuracy.py` |
| **우리 (카테고리-총량 + prior)** | **8.03%** | +0.70% | `store_predictive_power_summary.json` |

**핵심**: 광교 카테고리-총량에서 **4주 seasonal_naive(8.19%)가 이미 매우 강력** — 우리 모델
(8.03%)의 우위는 **0.16pp에 불과**. 빵 총량은 매끈·안정적이라 naive가 잡기 쉬운 series.

**4매장 맥락** (우리 우위는 어려운 매장에서 커짐):

| 매장 | naive WAPE | 우리 WAPE | Δ |
|---|---|---|---|
| 광교 | 8.19% | 8.03% | −0.16pp |
| 삼성타운 | 14.28% | 9.71% | −4.57pp |
| 메세나폴리스 | 9.64% | 9.14% | −0.50pp |
| 광화문 | 11.77% | 9.20% | −2.57pp |

MA(28d)는 전 매장에서 크게 뒤짐(10.9~30.4%). 즉 "의미있는 naive"는 seasonal_naive.

---

## ★(b-통합) 발주정책 KPI — 전 정책 vs 아띠제 실제 생산량 (authoritative, 2026-07-15)

**ground truth = 아띠제 실제 생산량(production_qty=QT_MADE)** 하나로 통일. 모든 정책을
**동일 population(13,522 item-day, item 경로 창)·동일 수요모델(adjusted_demand)·동일 arrival
profile**로 시뮬. `scripts/unified_policy_kpi.py`, `reports/unified_policy_kpi.csv`.
아래 (b)·(b-2)는 baseline이 섞여(재구현 vs 실생산) 혼란을 줬으므로 **이 표가 대표**다.

스코프(사용자 확정): 2번 = artisee_reimpl + our_category{quantile,nk,conformal}. actual_production은
기준. 매진을 **관점①(풀매진=전 품목 매진 / 총량소진=Σ수요>Σ발주)**과 **관점②(item-day 매진)**로 분리.
**동일 population = conformal test창 6,890 item-day**(conformal이 cal/test 분할이라 그 창에 전 정책 맞춤).

기준 actual_production: 폐기 **10.44M** · 일부매진② 0.022 · 풀매진① **0.000** · 총량소진① 0.000

| 정책 | 폐기(KRW) | vs 실생산 | 일부매진② | 풀매진① | 총량소진① |
|---|---|---|---|---|---|
| actual_production (기준) | 10.44M | — | 0.022 | 0.000 | 0.000 |
| **our_cat_quantile** q0.85 | 6.21M | **−40.5%** | 0.497 | 0.000 | 0.406 |
| **our_cat_conformal** s0.85 | 7.00M | **−32.9%** | 0.460 | 0.000 | 0.312 |
| our_cat_nk (q0.85+40) | 11.46M | +9.8% | 0.272 | 0.000 | 0.058 |
| artisee_reimpl | 9.23M | −11.6% | 0.311 | 0.000 | 0.246 |

*(폐기 음수=덜 버림=좋음. 매진 낮을수록 좋음, [[project_kpi_priority_framing]] 상 2차.)*

**해석(폐기 1차·매진 2차)**:
- **우리 총량 발주(quantile −40.5% / conformal −32.9%)가 실생산 대비 폐기 크게 절감.** 그리고
  **풀매진(전 품목 매진)은 전 정책 0** — 최악 케이스 없음.
- 대가 = 총량소진·일부매진 증가(quantile 총량소진 0.41). 흡수 가정상 매출영향 작으나 고객경험
  방향이라 **2차로 관리**(hard constraint 아님).
- **nk+40 버퍼는 과함**: 총량소진 0.06으로 매진 크게 줄이나 폐기 실생산 +9.8% 되돌아옴 → 폐기 1차 부적합.
- **artisee_reimpl은 열등**: 폐기 −12%인데 총량소진 0.25(실생산 0보다 나쁨) — 재구현 로직 < 실생산.
- **N,K vs conformal(이제 동일 population)**: conformal s0.85가 폐기 −33%·총량소진 0.31, nk+40이
  폐기 +10%·총량소진 0.06 — 서로 다른 서비스레벨의 프론티어 두 점(같은 s에서 폐기 비교는 s 정렬 필요).
- naive는 발주 아닌 예측 baseline이라 §(a) 정확도 표에.

---

## (b) [superseded — baseline=재구현] 발주정책 KPI — 우리 item vs 아띠제 재구현 (item-days 6,904)

아띠제 baseline(재구현: 3주평균 × sold-out지수 × 요일) 고정값:
**waste 14.60M · lost_margin 28.07M · stockout_rate 0.312 · soldout_median_h 19.04h**

| 우리 발주 (conformal s) | 실현 초과율 | stockout_rate | waste(KRW) | lost_margin(KRW) | soldout_median_h |
|---|---|---|---|---|---|
| s=0.70 | 0.333 | 0.333 | 26,986,490 | 36,292,940 | 17.08 |
| s=0.80 | 0.265 | 0.265 | 38,698,280 | 30,315,570 | 16.39 |
| s=0.90 | 0.183 | 0.183 | 86,982,100 | 21,949,350 | 16.11 |
| **아띠제** | — | **0.312** | **14,595,830** | **28,069,410** | **19.04** |

**핵심(item-level 기준, 잠정)**: 아띠제와 **동일 매진률(~0.31)**로 맞춘 지점(우리 s=0.70,
stockout 0.333)에서 **우리 폐기가 아띠제의 ~1.85배**(27.0M vs 14.6M)이고 lost_margin·soldout도
열위. 매진률을 더 낮추려(s=0.80/0.90) 하면 폐기가 폭증(38.7M→87.0M).

⚠️ **단 이 결과는 `our_order` = item-level LGBM production-quantile + conformal 경로**다 — 우리의
**약점 granularity**이고, 사용자가 물은 **카테고리-총량 모델(prior 포함)이 아니다.** 우리 강점은
카테고리-총량(§a에서 ~8% WAPE)이고, 폐기+매진 동시 발생(mis-allocation)은 카테고리 풀링이
공략하는 실패모드다. **카테고리-레벨 발주(총량→비율 배분) KPI를 현 regime(α=0.8·adjusted_demand·
bulk제외)에서 재측정하기 전까지 "아띠제를 못 이김"은 확정 아님.** (PR#27의 category-못이김은
target confound 시절 측정이라 이 regime엔 무효.) → **아래 (b-2) 참조.**

---

## (b-2) 카테고리-총량 발주(우리 강점 경로) vs reconstruction (2026-07-15 추가)

item-level(위)이 아니라 **총량→비율 배분** 경로. baseline = reconstruction(=아띠제 **실제
생산량**, 과잉+21%·매진 낮음). 총량 마진 3방식 구현·비교(`--category-margin`).

**8-fold (quantile·nk 동일 population, recon waste 31.20M·stockout 0.024):**

| 총량 마진 | 매진률 | 폐기(KRW) | vs recon 폐기 | Σ초과율 |
|---|---|---|---|---|
| quantile q0.85 | 0.477 | 20.72M | **−34%** | 0.362 |
| nk q0.85+40 | 0.265 | 37.05M | +19% | 0.045 |

**4-fold (conformal, cal/test 분할이라 별도 population, recon waste 16.83M):**

| conformal s=0.90 | 0.406 | 12.58M | −25%(delta) | 0.223 |

**해석(정직)**:
- **총량 경로는 아띠제 실제 생산량 대비 폐기를 이긴다** — quantile q0.85가 −34%(margin_optimize의
  −29~47%와 정합). item-level(위 b)에서 폐기가 +59%로 졌던 것과 **부호가 반대**. granularity가
  결정적이다.
- **N,K 버퍼는 매진↔폐기를 교환**: +40 버퍼로 매진 0.477→0.265이나 폐기가 recon을 넘어섬(+19%).
  버퍼가 공격적. **폐기 1차 KPI([[project_kpi_priority_framing]]) 기준으론 무버퍼/저버퍼(quantile)
  쪽이 유리**하고, 매진(2차)을 얼마나 사느냐로 버퍼 크기 결정.
- **conformal-on-total 정당성 확인**: q-스윕이 plateau(q0.85→0.95: Σ초과율 0.362→0.223)하는 걸
  conformal이 데이터-fit 마진으로 s=0.90에서 0.223 달성. 단 conformal은 cal/test 분할로 절반
  fold만 평가 → N,K와 절대 waste 직접비교는 **동일 population 재측정 필요**(현 수치는 각자
  baseline 대비 delta로만 비교 유효).

**남은 측정 과제**: N,K vs conformal을 **동일 population·동일 목표 매진률**에서 폐기 비교
(현재 conformal 4-fold / nk·quantile 8-fold라 절대비교 불가).

## 종합 — "어느 수준까지 가능한가" (현 상태)

1. **카테고리-총량 점추정**: WAPE ~8%로 양호하나, **광교에선 seasonal_naive와 동급**. 우리
   모델의 점추정 증분가치는 어려운 매장(삼성 −4.6pp)에서만 뚜렷.
2. **발주 KPI — granularity가 결정**:
   - **item-level(conformal)**: 아띠제 재구현(타이트 strawman) 대비 폐기 열위(동일 매진률 ~1.85배).
     우리 약점 경로.
   - **카테고리-총량(총량→배분, 우리 강점)**: 아띠제 **실제 생산량** 대비 폐기 **−34%**(quantile
     q0.85). 우리의 진짜 발주 경로는 이긴다.
   → "아띠제 이김"은 **어느 경로·어느 baseline이냐**에 달려 있다. 실무 보고 시 (총량 경로 · 실제
   생산량 대비)로 명시.
3. **총량 마진 = 폐기 1차 KPI로 선택**: quantile/nk/conformal 구현 완료. 폐기 최소는 저버퍼
   quantile, 매진(2차) 낮추려면 버퍼/conformal로 폐기 일부 희생. [[project_kpi_priority_framing]]
   따라 waste 우선.
4. **PoC 가치**: 총량 점추정(~8%, naive와 근접하나 어려운 매장 우위) + 총량 발주(실생산 대비
   폐기 −34%) + 설명/온톨로지 레이어. **item 매진(2차)은 배분 축 별도 과제**(총량과 무관).

## 캐비엣 (결론 신뢰도)

- **아띠제 = 재구현(ArtiseeBaseline), 실제 과거 발주 아님.** 그들도 매진타이밍 사용 → 우리
  잠재 우위(풀링·흡수배분·과잉 α보정)는 이 재구현엔 반영 안 됨. 실제 발주 데이터 수령 시 재검증.
- **우리 발주 = item-level LGBM production-quantile + conformal, prior·카테고리배분 미포함.**
  KPI 경로에 특수일 prior 미배선(eval 창에 크리스마스·추석 포함되나 소수일이라 aggregate
  영향 작음). 카테고리-총량→비율 배분 경로는 별도(PR#27서 item 못 이김).
- **naive는 발주정책 KPI 경로에 미배선** — (b)엔 naive-order 없음(수요예측만 비교).
- **KPI KRW 절대값**: waste sanity ratio 1.1~1.3배 과대·population 차이로 절대비교 제한,
  within-표 상대·매진률만 clean. soldout_median_h는 profile 정교화본(2026-07-15).

**원본**: `reports/naive_category_accuracy.csv`, `reports/store_predictive_power_summary.json`,
scratchpad `kpi_vs_artisee*.txt`.

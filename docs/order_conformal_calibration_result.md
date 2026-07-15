# 발주 conformal calibration 결과 — 진단 + 헤드라인 + 프론티어

**날짜**: 2026-07-09
**브랜치**: `feat/order-quantile-calibration`
**스펙**: `docs/superpowers/specs/2026-07-09-order-conformal-calibration-design.md`
**대상**: 광교(store_gw01), item-level 발주, `bakery prospective-eval --calibrate`
**원본 로그**: `reports/log_diag_item_q*.txt`, `reports/log_calib_item_s*.txt` (모두 실행 완료, 아래 숫자는 전부 이 로그에서 그대로 인용)

---

## ⚠️ 재측정 갱신 (2026-07-15) — α=0.8 · bulk 제외

아래 모든 표(1~3)의 수치는 **2026-07-15 재측정본**이다. 다음 두 변경으로 기존
(2026-07-09, α=0.5·bulk포함) 헤드라인이 무효화되어 재산출했다:

1. **마감할인 실수요 비율 α 0.5 → 0.8** (`category_aggregate.DEFAULT_ALPHA`, real 경로).
   demand target(`adjusted_demand`)을 상향 → 초과율이 전반적으로 소폭 상승하는 방향.
2. **bulk(예약) 검출 재설계 + `bonavi_loader` 배선** (`data/bulk.py` `flag_bulk_lines`).
   sold_units 510,585→509,520(−0.21%). item-day **카운트**는 거의 불변(bulk는 units만
   제거, 대부분 소수 spike day에 집중) → raw item-days 13,554→13,550.

**핵심 결론은 전부 유지된다** (수치만 소폭 이동):
- 표 1: q를 0.85→0.99로 올려도 raw 초과율 **0.741→0.435**로 완만하게만 감소(q0.99에서도
  nominal의 43배) → base quantile 구조적 under-calibration, conformal 도입 정당화 유지.
- 표 2: raw q0.85(동일 nominal 0.15) 초과율 **0.741** → conformal s=0.85 **0.227**, gap의
  **약 87% 닫힘**(기존과 동일). conformal 후 초과율이 α/bulk 변화에도 s=0.74에서 ~0.30으로
  **안정** — post-hoc adaptive margin이 demand 레벨 이동을 흡수함을 보여줌.
- 표 3: 잔차 전 구간 양수(+0.033~+0.083), s=0.74~0.85 사용 가능 프론티어, s=0.95 운영 불가
  (WPE 3.59) — 정성 결론 불변.

**✅ receipts arrival profile 정교화 (2026-07-15, 이 세션 후속 반영)**: 매진 시각
`soldout_median_h`는 arrival profile(시간대 분포)로 반사실 역산되는데, 그 profile 입력
(`_load_real_receipts`)이 기존엔 **qty=1.0(방문 건수)** + **bulk 미제외**였다. KPI 소스별
노출도를 코드로 확인(`evaluation/prospective.py`)하면:
- **초과율·WPE·stockout_rate = demand 기반** (`is_stockout = order_qty < adjusted_demand`,
  `simulate_soldout` line 26) → profile과 무관·불변. bulk/α로 움직이는 것도 이 demand 경로.
- **soldout_median_h만 profile(receipts) 기반** (매진 시각은 arrival profile로 역산,
  line 112~116) → 유일하게 profile 정교화의 영향을 받는 지표.

**변경**: profile 입력을 (1) **수량-가중**(qty=판매수량, 방문 건수 아님) + (2) **bulk 국소
제외**(is_bulk 라인 제외)로 바꿨다. profile이 분배하는 daily 수요(`adjusted_demand`)가
bulk 제외본이므로 shape도 walk-in 수량으로 population을 맞춘 것이다. `load_receipts_with_time`가
qty·is_bulk 컬럼을 추가 산출(행·기존 receipt_id 보존 → substitution 등 다른 소비처 불변),
`_load_real_receipts`가 opt-in으로 필터·가중. 표 3의 `soldout_median_h`는 이 정교화본이다.

**캐비엣 (의도된 선택)**:
- **profile 타이밍 vs α**: profile은 마감할인 판매의 *전량*을 싣는다(α 0.8 미적용). α는
  counterfactual 수요 *크기*의 보정이지 재고가 *언제* 빠졌는가(타이밍)가 아니고, receipts에
  할인 flag가 없어 α-가중 자체가 불가하다 → 타이밍 shape엔 전량 반영이 타당. (광교 저녁
  상시할인 특성상 soldout_median_h 이동이 크면 이 저녁 집중이 원인일 수 있음.)
- **artisee baseline 공유**: `_load_real_receipts`는 `--baseline artisee` residual curve도
  공유하므로 그쪽도 qty-가중으로 이동한다(PR#41 real 경로 수치 재실행 필요). 본 문서의
  기본(현행) baseline 프론티어는 이미 반영됨.

**원본 로그(재측정)**: `reports/remeasure_diag_q0.90.txt`~`q0.99.txt`(표1),
`reports/remeasure_conf_s0.70.txt`~`s0.95.txt`(표2·3),
`reports/remeasure_{summary,frontier_summary}.txt`(요약). raw q0.85는 파일 미저장 —
재현: `uv run bakery prospective-eval --source real --store-id store_gw01 --n-folds 8
--production-quantile 0.85`.

---

## 배경

`project_prospective_derisk_retro`에서 확인된 문제 ③: full-window de-risk 회고 결과 raw
production-quantile(q0.85) 기반 발주가 **genuine under-calibration**을 보였다 —
`P(demand>order)` 초과율이 nominal(1−α=0.15)의 4배가 넘는 0.636~0.679 수준. 단순히 q를
올리는 것으로 해결되는지, 아니면 base quantile 추정 자체가 구조적으로 편향돼 있어 별도의
보정이 필요한지를 먼저 확인해야 conformal 계층 도입이 정당화된다.

이 문서는 (1) q 단독 상향의 한계를 진단하고, (2) cross-fold half-split 기반 scale-정규화
pooled conformal 보정(base=median, Task 1~4 구현)이 그 gap을 얼마나 닫는지, (3) 서비스레벨
s를 0.70~0.95로 스윙했을 때 실현 초과율과 waste/lost/stockout 트레이드오프가 어떻게
움직이는지를 실제 실행 결과로 기록한다.

## 표 1 — 진단: raw production-quantile 스윙 (무보정)

`--n-folds 8 --alpha 0.8`, item-level, 보정 없음. item-days = 13,550 (전체 8-fold 윈도우).

| q (production_quantile) | nominal 초과율 목표 (1−q) | 실현 초과율 P(demand>order) | WPE |
|---|---|---|---|
| 0.85 | 0.15 | **0.741** | −0.383 |
| 0.90 | 0.10 | **0.683** | −0.305 |
| 0.95 | 0.05 | **0.615** | −0.214 |
| 0.99 | 0.01 | **0.435** | −0.014 |

**판정**: q를 0.85→0.99로 올려도(nominal 목표를 0.15→0.01로 15배 좁혀도) 실현 초과율은
0.741→0.435로 완만하게만 줄어든다. q=0.99에서도 초과율이 nominal의 43배(0.435 vs 0.01)다.
**base median/quantile 추정 자체가 구조적으로 under-calibrated**되어 있고, q 상향만으로는
어떤 실행 가능한 범위에서도 nominal에 도달하지 못한다 — conformal 보정 계층 도입이
정당화된다.

## 표 2 — 헤드라인: raw q0.85 vs conformal 보정

`--calibrate --n-folds 8 --our-order-val-weeks 8`(cal 4-fold / test 4-fold, cal-fold-frac
기본값 0.5). 보정 경로는 base=median(q0.5) + item별 scale 정규화 + cross-fold half-split
conformal margin. item-days = 6,904 (test-fold 절반만, diag의 절반 population).

| 설정 | nominal 목표 (1−s) | 실현 초과율 | 잔차(실현−목표) |
|---|---|---|---|
| raw q0.85 (무보정) | 0.15 | 0.741 | +0.591 |
| conformal s=0.74 | 0.26 | **0.308** | +0.048 |
| conformal s=0.85 (동일 nominal 0.15와 비교) | 0.15 | **0.227** | +0.077 |

- brief가 지정한 1차 비교(raw q0.85 vs s=0.74)는 population과 nominal이 다르므로
  참고용: raw 0.741 → conformal 0.308로 초과율이 절반 이하로 줄었다.
- **동일 nominal(0.15)로 맞춘 apples-to-apples 비교가 더 정직한 신호**다: raw q0.85(0.15
  목표)는 0.741, conformal s=0.85(같은 0.15 목표)는 0.227. 같은 목표에서 잔차가
  0.591 → 0.077로, gap의 **약 87%가 닫힌다**.
- WPE(s=0.74)=0.270 — 여전히 over-forecast 방향 편향이지만 raw q0.85의 −0.383(반대
  방향, under-forecast)보다 부호가 뒤집혔을 뿐 크기는 비슷한 수준.
  이 WPE 비교는 서로 다른 baseline population(conformal item-days=6,904 vs raw
  q0.85 진단 item-days=13,550) 간 비교라 방향(부호 반전)만 참고할 수 있고 절대
  크기 비교는 부적절하다 — 아래 캐비엣의 "오직 초과율/방향 신호만 정성적으로
  비교 가능" 원칙과 동일하게 적용된다.

## 표 3 — 프론티어: s ∈ {0.70, 0.74, 0.80, 0.85, 0.90, 0.95}

모든 행이 동일 population(item-days=6,904, cal 4-fold/test 4-fold)과 동일 baseline
(waste=16,945,650 KRW, lost_margin=1,693,693 KRW, stockout_rate=0.022306,
soldout_median_h=19.634100)을 공유하므로 **행 간(within-table) 비교는 유효**하다.
(soldout_median_h는 2026-07-15 profile 정교화본 — 수량-가중·bulk 제외. footfall 근사
대비 이동은 ~1분 수준으로 미미해 기존 근사가 이미 양호했음을 확인.)
(⚠️ baseline KPI가 기존 α=0.5·bulk포함본 대비 크게 이동했다 — 특히 stockout_rate
0.085→0.022, lost_margin 3.97M→1.69M. baseline 재구성 로직(`reconstruct_baseline_order`
= normal+closing+waste)은 07-09 이후 **코드 변경 없음**(git 확인 — artisee PR은 별도
`--baseline artisee` 경로만 추가)이므로 이 이동은 코드 drift가 아니라 순수 데이터 regime
효과다: bulk spike day가 demand에서 빠지면서 현행 발주의 매진/손실(demand 기반 KPI)이
줄었다. **버전 간 KPI Δ 절대 비교는 무의미**하고 within-table 상대 비교와 초과율만
유효하다는 기존 원칙이 그대로 적용된다.)

| s | nominal(1−s) | 실현 초과율 | 잔차 | Δwaste(KRW, our−baseline) | Δlost_margin(KRW) | Δstockout_rate | Δsoldout_median_h | WPE |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 0.30 | 0.333 | +0.033 | +10,040,840 | +34,599,250 | +0.3103 | −2.559 | 0.208 |
| 0.74 | 0.26 | 0.308 | +0.048 | +13,766,970 | +32,260,370 | +0.2855 | −2.834 | 0.270 |
| 0.80 | 0.20 | 0.265 | +0.065 | +21,752,630 | +28,621,880 | +0.2423 | −3.248 | 0.394 |
| 0.85 | 0.15 | 0.227 | +0.077 | +33,013,010 | +25,566,560 | +0.2050 | −3.664 | 0.561 |
| 0.90 | 0.10 | 0.183 | +0.083 | +70,036,450 | +20,255,660 | +0.1602 | −3.522 | 1.085 |
| 0.95 | 0.05 | 0.086 | +0.036 | +251,238,200 | +7,238,964 | +0.0642 | −1.318 | 3.590 |

트레이드오프 방향은 기대대로: s↑ → 초과율↓, waste Δ↑(더 많이 발주), lost_margin
Δ↓(매진 손실 감소), stockout_rate Δ↓. 단 s=0.90→0.95에서 waste Δ가 7,000만→2억 5,100만
KRW로 급등하는데 stockout_rate Δ 개선폭은 +0.160→+0.064로 절반 이하로 꺾인다 — **s=0.95는
운영적으로 쓸 수 없는 지점**(대량 폐기로 매진 개선 실익이 급감, WPE=3.590으로 base 대비
3.6배 과다발주).

## 판정 (정직)

**부분 성공 — gap의 대부분은 닫히지만 일관된 양(+)의 잔차가 남는다.**

- 모든 s에서 실현 초과율이 nominal보다 **높은 방향으로만** 벗어난다(잔차 +0.033~+0.083,
  전 구간 양수). 즉 conformal 보정 후에도 여전히 소폭 under-order 방향 편향이 남아있다 —
  대부분(raw 대비 gap의 ~87%)은 닫혔지만 "conformal이 calibration gap을 완전히
  닫았다"고 말할 수는 없다.
- 잔차는 s=0.70(+0.033)에서 s=0.90(+0.083)까지 커지다가 s=0.95(+0.036)에서 다시
  줄어든다 — s에 대해 단조가 아니다. 상대적으로는 더 뚜렷하다: nominal 대비 잔차 비율이
  0.70→0.90 구간에서 11%→83%로 커진다.
- **추정 원인** (크기 확인, 원인은 정황 근거):
  1. **chronological cal/test split의 exchangeability 위반**: calibration fold(이른
     시기)와 test fold(늦은 시기)가 시간순으로 분리되어 있어, 두 구간 사이에 트렌드나
     명절 편차가 있으면 conformal의 exchangeability 가정이 깨진다. 광교는 5년치 데이터에
     설/추석 lead-up 시프트가 이미 확인된 바 있다(`project_external_benchmark_research`).
  2. **소표본 극단 quantile 불안정**: `--our-order-val-weeks 8`로 cal fold가 4주(=4
     fold)로 제한되어 있어, s가 높아질수록(0.90, 0.95) conformal margin이 분포의 꼬리
     끝단 quantile에 의존하게 되고 이 추정이 표본 부족으로 불안정해진다. s=0.95에서
     waste가 약 16배 튀는 것이 이 불안정성의 직접적 증거다.
  3. **item별 scale 정규화 잔차**: pooled conformal이 item별 scale로 정규화하지만, scale
     추정 자체(before-cutoff 데이터 기반)가 완벽하지 않으면 정규화 후에도 item 간
     이분산성이 일부 남아 pooled quantile이 일부 item에는 과소, 다른 item에는 과다 보정될
     수 있다.
- **실무 결론**: s=0.74~0.85 구간이 실현 초과율과 waste 증가 속도의 균형이 맞는
  "사용 가능한" 프론티어다. s=0.90 이상은 waste 급등 대비 stockout 개선이 급격히
  줄어들어 권장하지 않는다.

## 캐비엣

- **광교 단독**: 이 결과는 store_gw01 하나에서만 검증됐다. 다매장 일반화는 미검증.
- **test fold 절반 표본**: cal-fold-frac=0.5로 인해 프론티어/헤드라인의 실현 초과율은
  8-fold 중 절반(4-fold, item-days=6,904)에서만 계산된다. 진단표(표 1)는 8-fold
  전체(item-days=13,550)를 쓰므로 **표 1과 표 2/3의 population이 다르다** — 두 표 사이의
  절대 KPI(KRW) 비교는 부적절하고, 오직 "초과율이라는 방향/크기 신호"만 정성적으로
  비교 가능하다(그래서 표 2에 동일 nominal(0.15) apples-to-apples 행을 별도로 추가함).
- **exchangeability 위반 가능성(명절 shift)**: calibration fold가 test fold보다 항상
  이른 시기이므로, conformal의 표준 가정(exchangeable scores)이 시간 트렌드나 명절
  lead-up 효과 아래에서 정확히 성립하지 않는다. 위 판정의 "잔차" 원인 후보 1번.
- **KPI Δ 스케일 오염**: waste/lost_margin/stockout Δ는 baseline(현행 발주 시스템) 대비
  값인데, baseline 자체가 실행별로 스코어링된 population(진단 8-fold 전체 vs
  헤드라인/프론티어 test-4-fold)에 따라 달라진다(표1 진단 baseline waste=31,365,470 vs
  표2/3 프론티어 baseline waste=16,945,650 — 서로 다른 population이라 직접 비교 불가).
  프론티어 표(표 3) **내부**는 baseline이 고정돼 있어 상대 비교가 유효하지만, 진단표와는
  비교하면 안 된다. 이 오염 때문에 **이 문서에서 신뢰할 수 있는 유일한 순수(clean) 신호는
  초과율(calibration 초과율)이고, KRW 단위 waste/lost 절대값은 방향성 참고용**이다
  (waste sanity ratio도 실측 대비 시뮬레이션이 약 1.1~1.3배로 항상 부풀려져 있음을
  각 로그가 자체 보고). **soldout_median_h**(매진 시각)는 arrival profile 기반이라
  profile 정교화(수량-가중·bulk 제외, 2026-07-15)의 영향을 받는 유일한 지표다 —
  stockout_rate·초과율은 demand 기반이라 무관(상단 배너 참조).
- **후속 범위 아님**: category-level 발주 재보정, 다매장 conformal, exchangeability를
  깨지 않는 cal/test split(예: 블록 랜덤화) 검토는 이 태스크 범위 밖이며 후속 과제로
  남긴다.

## 원본 데이터

| 파일 | 내용 |
|---|---|
| `reports/log_diag_item_q0.85.txt` ~ `q0.99.txt` | 표 1 원본 |
| `reports/log_calib_item_s0.70.txt` ~ `s0.95.txt` | 표 2·표 3 원본 (s=0.74 파일이 헤드라인) |
| `reports/diag_item_q*.csv`, `reports/calib_item_s*.csv` | 각 run의 `--out-csv` 산출물(row-level) |

`reports/`는 `.gitignore`에 포함되어 있어 CSV/로그는 git에 커밋되지 않는다(재현하려면 위
brief의 명령을 그대로 재실행).

## 후속 #1 (category 경로 conformal) — 결정: 미실행 (2026-07-10)

원래 후속 백로그 #1은 "`ConformalOrderCalibrator`(path-agnostic)를 카테고리-총합에 재사용해
category under-calibration을 교정"이었다. 착수 단계에서 재검토한 결과 **conformal은 item
경로 전용으로 확정하고 카테고리 경로에는 적용하지 않기로 결정**했다.

**근거**
- conformal이 item 경로에 필요했던 이유는 item quantile 모델이 **구조적으로
  under-dispersed**이기 때문이다. 위 진단표(표 1)대로 q를 0.85→0.99로 15배 좁혀도 실현
  초과율은 0.741→0.435까지만 내려가 q-tuning만으로는 nominal에 못 닿았다. 그래서 잔차
  마진(conformal)이 필요했다.
- 반면 **카테고리 총합은 매끈한 aggregate 시계열**이라 sparse/under-dispersed 문제가
  약하고, LGBM production_q(quantile 회귀)가 총합 분위를 직접 예측하는 게 자연스럽다.
  그 위에 conformal 잔차를 덧씌우는 것은 "보정의 보정"으로 설계상 지저분하고, 애초에
  필요성도 불분명하다.

**남기는 loose end — 카테고리 총합 초과율 0.346**
- `target_unification_remeasure_result.md` 기준, 통일 target(adjusted_demand)·q=0.85·
  full-window n_folds=8·광교에서 **카테고리-총합** 초과율 `P(Σdemand>Σorder)=0.346`
  (nominal 0.15). 배분 **후** item-day 초과율은 0.458이지만, conformal이 직접 손댈 잣대는
  배분 **전** 총합값 0.346이다.
- 이 총합 under-calibration은 교정하지 않고 남긴다. 카테고리 경로는 PR#27 재측정에서
  item-level을 못 이긴 **non-primary 경로**이므로(운영 경로 = item + conformal), 쓰지 않는
  경로를 calibrate하는 데 추가 투자하지 않는다. 향후 카테고리 총합을 nominal에 맞춰야 할
  일이 생기면, 우선 검토할 레버는 conformal이 아니라 **LGBM production_q 스윕**(0.85→0.90/
  0.95 — 매끈한 aggregate라 q-tuning으로 잡힐 가능성이 item보다 높다)이다.

**코드 변경 없음.** 이 태스크는 설계 결정으로 종료한다.

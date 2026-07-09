# 발주 conformal calibration 결과 — 진단 + 헤드라인 + 프론티어

**날짜**: 2026-07-09
**브랜치**: `feat/order-quantile-calibration`
**스펙**: `docs/superpowers/specs/2026-07-09-order-conformal-calibration-design.md`
**대상**: 광교(store_gw01), item-level 발주, `bakery prospective-eval --calibrate`
**원본 로그**: `reports/log_diag_item_q*.txt`, `reports/log_calib_item_s*.txt` (모두 실행 완료, 아래 숫자는 전부 이 로그에서 그대로 인용)

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

`--n-folds 8 --alpha 0.5`, item-level, 보정 없음. item-days = 13,554 (전체 8-fold 윈도우).

| q (production_quantile) | nominal 초과율 목표 (1−q) | 실현 초과율 P(demand>order) | WPE |
|---|---|---|---|
| 0.85 | 0.15 | **0.679** | −0.301 |
| 0.90 | 0.10 | **0.651** | −0.266 |
| 0.95 | 0.05 | **0.606** | −0.199 |
| 0.99 | 0.01 | **0.421** | 0.018 |

**판정**: q를 0.85→0.99로 올려도(nominal 목표를 0.15→0.01로 15배 좁혀도) 실현 초과율은
0.679→0.421로 완만하게만 줄어든다. q=0.99에서도 초과율이 nominal의 42배(0.421 vs 0.01)다.
**base median/quantile 추정 자체가 구조적으로 under-calibrated**되어 있고, q 상향만으로는
어떤 실행 가능한 범위에서도 nominal에 도달하지 못한다 — conformal 보정 계층 도입이
정당화된다.

## 표 2 — 헤드라인: raw q0.85 vs conformal 보정

`--calibrate --n-folds 8 --our-order-val-weeks 8`(cal 4-fold / test 4-fold, cal-fold-frac
기본값 0.5). 보정 경로는 base=median(q0.5) + item별 scale 정규화 + cross-fold half-split
conformal margin. item-days = 6,904 (test-fold 절반만, diag의 절반 population).

| 설정 | nominal 목표 (1−s) | 실현 초과율 | 잔차(실현−목표) |
|---|---|---|---|
| raw q0.85 (무보정) | 0.15 | 0.679 | +0.529 |
| conformal s=0.74 | 0.26 | **0.299** | +0.039 |
| conformal s=0.85 (동일 nominal 0.15와 비교) | 0.15 | **0.220** | +0.070 |

- brief가 지정한 1차 비교(raw q0.85 vs s=0.74)는 population과 nominal이 다르므로
  참고용: raw 0.679 → conformal 0.299로 초과율이 절반 이하로 줄었다.
- **동일 nominal(0.15)로 맞춘 apples-to-apples 비교가 더 정직한 신호**다: raw q0.85(0.15
  목표)는 0.679, conformal s=0.85(같은 0.15 목표)는 0.220. 같은 목표에서 잔차가
  0.529 → 0.070으로, gap의 **약 87%가 닫힌다**.
- WPE(s=0.74)=0.280 — 여전히 over-forecast 방향 편향이지만 raw q0.85의 −0.301(반대
  방향, under-forecast)보다 부호가 뒤집혔을 뿐 크기는 비슷한 수준.

## 표 3 — 프론티어: s ∈ {0.70, 0.74, 0.80, 0.85, 0.90, 0.95}

모든 행이 동일 population(item-days=6,904, cal 4-fold/test 4-fold)과 동일 baseline
(waste=18,476,490 KRW, lost_margin=3,972,432 KRW, stockout_rate=0.084733,
soldout_median_h=20.397664)을 공유하므로 **행 간(within-table) 비교는 유효**하다.

| s | nominal(1−s) | 실현 초과율 | 잔차 | Δwaste(KRW, our−baseline) | Δlost_margin(KRW) | Δstockout_rate | Δsoldout_median_h | WPE |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 0.30 | 0.324 | +0.024 | +8,691,587 | +31,280,210 | +0.2390 | −3.065 | 0.217 |
| 0.74 | 0.26 | 0.299 | +0.039 | +12,395,960 | +28,837,180 | +0.2147 | −3.212 | 0.280 |
| 0.80 | 0.20 | 0.256 | +0.056 | +20,544,930 | +24,979,900 | +0.1716 | −3.526 | 0.409 |
| 0.85 | 0.15 | 0.220 | +0.070 | +30,096,130 | +22,135,730 | +0.1357 | −4.031 | 0.553 |
| 0.90 | 0.10 | 0.175 | +0.075 | +62,044,250 | +17,236,760 | +0.0898 | −4.038 | 1.012 |
| 0.95 | 0.05 | 0.085 | +0.035 | +235,676,800 | +5,179,929 | +0.0001 | −1.962 | 3.438 |

트레이드오프 방향은 기대대로: s↑ → 초과율↓, waste Δ↑(더 많이 발주), lost_margin
Δ↓(매진 손실 감소), stockout_rate Δ↓. 단 s=0.90→0.95에서 waste Δ가 6,200만→2억 3,600만
KRW로 급등하는데 stockout_rate Δ 개선은 +0.090→+0.0001로 거의 사라진다 — **s=0.95는
운영적으로 쓸 수 없는 지점**(대량 폐기로 매진 개선 실익이 없음, WPE=3.438로 base 대비
3.4배 과다발주).

## 판정 (정직)

**부분 성공 — gap의 대부분은 닫히지만 일관된 양(+)의 잔차가 남는다.**

- 모든 s에서 실현 초과율이 nominal보다 **높은 방향으로만** 벗어난다(잔차 +0.024~+0.075,
  전 구간 양수). 즉 conformal 보정 후에도 여전히 소폭 under-order 방향 편향이 남아있다 —
  대부분(raw 대비 gap의 ~85%+)은 닫혔지만 "conformal이 calibration gap을 완전히
  닫았다"고 말할 수는 없다.
- 잔차는 s=0.70(+0.024)에서 s=0.90(+0.075)까지 커지다가 s=0.95(+0.035)에서 다시
  줄어든다 — s에 대해 단조가 아니다. 상대적으로는 더 뚜렷하다: nominal 대비 잔차 비율이
  0.70→0.90 구간에서 8%→75%로 커진다.
- **추정 원인** (크기 확인, 원인은 정황 근거):
  1. **chronological cal/test split의 exchangeability 위반**: calibration fold(이른
     시기)와 test fold(늦은 시기)가 시간순으로 분리되어 있어, 두 구간 사이에 트렌드나
     명절 편차가 있으면 conformal의 exchangeability 가정이 깨진다. 광교는 5년치 데이터에
     설/추석 lead-up 시프트가 이미 확인된 바 있다(`project_external_benchmark_research`).
  2. **소표본 극단 quantile 불안정**: `--our-order-val-weeks 8`로 cal fold가 4주(=4
     fold)로 제한되어 있어, s가 높아질수록(0.90, 0.95) conformal margin이 분포의 꼬리
     끝단 quantile에 의존하게 되고 이 추정이 표본 부족으로 불안정해진다. s=0.95에서
     waste가 13배 튀는 것이 이 불안정성의 직접적 증거다.
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
  전체(item-days=13,554)를 쓰므로 **표 1과 표 2/3의 population이 다르다** — 두 표 사이의
  절대 KPI(KRW) 비교는 부적절하고, 오직 "초과율이라는 방향/크기 신호"만 정성적으로
  비교 가능하다(그래서 표 2에 동일 nominal(0.15) apples-to-apples 행을 별도로 추가함).
- **exchangeability 위반 가능성(명절 shift)**: calibration fold가 test fold보다 항상
  이른 시기이므로, conformal의 표준 가정(exchangeable scores)이 시간 트렌드나 명절
  lead-up 효과 아래에서 정확히 성립하지 않는다. 위 판정의 "잔차" 원인 후보 1번.
- **KPI Δ 스케일 오염**: waste/lost_margin/stockout Δ는 baseline(현행 발주 시스템) 대비
  값인데, baseline 자체가 실행별로 스코어링된 population(진단 8-fold 전체 vs
  헤드라인/프론티어 test-4-fold)에 따라 달라진다(표1 진단 baseline waste=35,000,960 vs
  표2/3 프론티어 baseline waste=18,476,490 — 서로 다른 population이라 직접 비교 불가).
  프론티어 표(표 3) **내부**는 baseline이 고정돼 있어 상대 비교가 유효하지만, 진단표와는
  비교하면 안 된다. 이 오염 때문에 **이 문서에서 신뢰할 수 있는 유일한 순수(clean) 신호는
  초과율(calibration 초과율)이고, KRW 단위 waste/lost 절대값은 방향성 참고용**이다
  (waste sanity ratio도 실측 대비 시뮬레이션이 약 1.23~1.27배로 항상 부풀려져 있음을
  각 로그가 자체 보고).
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

# Phase B — 발주 최적화 + implied c 갭 (원가율 파라미터) — Design

**Date**: 2026-07-04 (Fable 5 검토 반영 rev)
**Depends on**: Phase A(α_A=NaN), 실측 폐기 `data/internal/v2/waste_alpha_4stores.parquet`, W0 흡수검증([[project-demand-absorption-w0]]), v4 스택(`models/category_total.py`, `models/lightgbm_regressor.GlobalLGBM` quantile, `evaluation/business_metrics.py`), Phase B 스파이크
**Status**: Design rev (프레이밍+Fable검토 반영, 문서 리뷰 대기)

## 목적

Phase A가 광교 structural α 식별 실패(저녁 상시 마감할인 confound, α_A=NaN). Phase B는 **물리적 발주/폐기 기반 operational 접근**:
1. **주 산출물 — implied c 갭**: 매장 현행 발주가 마치 원가율 얼마인 것처럼 행동하는가(implied c) → 실제 원가율과의 갭 = 과잉생산 여지. (실측 매장 service level 94.6%/91.4% → implied c ≈ 0.05~0.09.)
2. **보조 — 비용최적 발주 Q\*(c)** + 절감액. **정직하게 하면 작거나 음수 가능**(Fable 실측: DOW-조건부에서 bread +0.6%/pastry −24%). fallback 내러티브 = implied c 갭.
3. c(원가율)=미지 파라미터. 고객사 c 제공 시 점으로 확정.

발주는 α 아닌 c+수요분포로 결정(newsvendor). 이 전환이 Phase A의 잠식/식별 confound를 우회한다.

## Fable 5 검토 반영 (핵심 수정)

- **C1 two-class newsvendor**: 수요 D=정상+마감이면 out=마감으로도 못 판 것 → Co=c·price 정확. 오차는 **Cu에 있음** — Q\*가 마감 밴드에 걸리면 한계 미달분은 정가마진 아닌 **할인마진 (1−δ−c)p**로 팔림. 순수 CR=1−c(Level 1)는 Q\*를 **체계적 과대생산** → 상한으로만. 올바른 조건 D3 참조.
- **C2 implied α 폐기**: (Q\*−normal_med)/closing_med는 전 c에서 >1(꼬리 분위수÷중심경향)+행동 α와 다른 service-level 숫자 → 오독. **implied c of current policy로 대체**(주 산출물).
- **C3 placebo arm 필수**: 카테고리 (Q\*−D)⁺는 품목배분 완벽 가정 → Q\*=made로도 유령 절감 10~14%. **Q=made placebo arm**으로 총량효과 분리.
- **IMPORTANT-4 조건화**: 무조건 경험분포 분위수는 매장에 대패 → **v4 quantile 스택(요일/캘린더 조건화) 재사용**, "동일 조건화 수준" 공정비교.
- **IMPORTANT-5 d 내생성**: 축소 반사실은 수요분포 붕괴 → Q\*는 **made 근방 한계 재최적화**로 스코핑, A3 slope로 대역.
- **IMPORTANT-6 항등식**: demand>made가 5~9%일(out<0 등) → 명시 제외규칙.

## 핵심 결정 (LOCKED)

### D1. 카테고리 단위 (광교 bread/pastry)
품목-일 censoring 심함(43.9%/38.3% out>0)+음수 out → 부적합. 카테고리-일 out==0 ~1.5~1.9%(깨끗). W0가 "품목 품절→카테고리 내 흡수→총량 보존" 입증 → **현행 정책·현행 공급 하** 카테고리 sold_total을 카테고리 실수요로 사용(이 한정 명시). 품목 배분=Stage-2(범위 밖).

### D2. 수요 분포 = v4 quantile 스택 재사용 (조건화 필수)
- 카테고리-일 수요 = Σ(품목 sold_total). target = adjusted/sold 계열(category_total 관례).
- **`models/category_total.fit_category_total` / `CategoryTotalModel.predict_production`(LGBM quantile) 재사용** — 요일·캘린더·트렌드 조건화. 무조건 경험분위수 금지(매장이 이미 조건화 발주 → 불공정).
- censored(카테고리 out==0) 비율 보고, 보수 처리(수요≥made 하한). uncensored=out>0.
- **왜 재발명 안 하나**: Fable 지적 — v4에 이미 조건화 quantile 스택 존재. 재사용이 공정비교의 전제.

### D3. two-class newsvendor (원가율 c)
정상수요 N, 마감밴드 폭 C(=마감 물량), 할인깊이 δ(Phase A depth 분해: 0077=30%,0069=20% 가중). 최적 Q\*의 1계 조건:
```
P(N > Q) + (1−δ)·P(N ≤ Q < N+C) = c
```
- **Level 1 (상한)**: 순수 newsvendor CR=1−c → Q\*=수요 (1−c)분위수. 마감밴드를 정가마진으로 평가 → **Q\* 과대(상한)**. `business_metrics.simulate_profit`(단일클래스 cost) 재사용.
- **Level 2 (주)**: 위 two-class 조건. 마감밴드 한계손실을 할인마진으로 → Q\* 하향(정직). 입력(정상/마감 결합분포, δ) 보유.
- price = 카테고리 총매출/총수량(수량가중 단가), **t-이전 데이터로 계산**(leakage 일관). 폐기물량-가중 단가로 cross-check(M1).
- c 그리드 0.25~0.55 step 0.05 상수화. c=**한계 생산원가율**(회계원가율 아닐 수 있음, M2).

### D4. implied c of current policy (주 산출물, α 대체)
현행 발주(made)의 카테고리 service level = P(demand ≤ made) 추정 → 그 service level을 주는 c: **implied_c = 1 − service_level**(Level 1 기준) 또는 two-class 역산(Level 2). 실측 매장 SL 94.6%/91.4% → **implied c ≈ 0.05~0.09**.
- 메시지: "매장은 원가율 5~9%인 것처럼 발주(거의 매진 우선) → 실제 c가 그보다 높으면 그 갭 = 과잉생산·폐기 여지." 오독불가·항상 유의미.
- **implied α는 산출 안 함**(폐기; 원하면 dimension-consistent service-level α를 별도 라벨로만, 기본 제외).

### D5. 백테스트 / 절감액 (leakage 금지, placebo arm 필수)
- v4 rolling/expanding backtest(`evaluation/backtest.py` 관례) — 수요모델은 t-이전만 학습.
- 각 c: `simulate_profit`(CostParams.cost_rate=c) 재사용해 3개 arm 비용 비교:
  1. **Q\*(c)** (Level 2 최적)
  2. **Q=made** (placebo — 매장 실발주, 동일 반사실 산식)
  3. (참고) Q\* Level 1 상한
- **절감액 = cost(Q=made) − cost(Q\*(c))** — placebo 대비라 유령 배분효과 분리. 카테고리 배분 완벽 가정분은 "Stage-2 조건부 상한"으로 별도 표기(convexity상 실제 out ≥ 카테고리 overage).
- **d 내생성**: Q\*를 made 근방으로 스코핑, A3 slope(0.29/0.38)로 "마감 유도분 최대 s·잔량" 시나리오 대역 병기.
- WAPE 보조 + cost KPI 메인.

### D6. 산출물
- `src/bakery/analysis/order_optimization.py` (Phase B 모듈) — 수요분포(v4 재사용 래퍼) + two-class Q\*(c) + implied c + 백테스트(placebo arm).
- CLI `phaseb-order`.
- `reports/phaseB_implied_c.csv`(주: 카테고리 → service_level, implied_c), `reports/phaseB_order_savings.csv`(c → Q\*, made, Δ폐기/Δ매진, 절감액 Level1/2, placebo 대비, 배분상한).
- 정직한 결과 문서: **주=implied c 갭**, 보조=절감액(작/음수여도 정직), α_A=NaN이라 교차검증 없음, c 오면 점.

## 아키텍처 / 데이터 흐름
1. `load_category_daily(store="광교")` — parquet 필터 → 카테고리-일(demand=Σsold_total, made=Σmade, out=Σout, normal, closing, price 대표, censored flag). **항등식 제외규칙 적용(D 아래)**.
2. 수요분포 = v4 `fit_category_total`(quantile) 래핑, t-이전 학습.
3. `two_class_order(dist_or_model, N, C, delta, c)` → Q\*(c).
4. `implied_c(made, demand_dist)` → 현행 정책 implied c.
5. `backtest(category_daily, c_grid)` — rolling, 3 arm(Q\*, made, Level1), simulate_profit 재사용.
6. CLI → CSV + print(implied c 갭 강조).

각 함수 단일책임·독립테스트. v4/business_metrics/category_total 최대 재사용, 재발명 금지.

## 항등식/데이터 규칙 (LOCKED — 구현자 재량 금지)
- 항등식 `made = normal+closing+out`, `sold_total=normal+closing`. **|identity_diff| > IDENTITY_TOL(상수) 카테고리-일 = 수요분포·백테스트 양쪽 제외 + 제외율 보고.**
- `out < 0` → 0으로 clip, **clip 물량·행수 보고**.
- demand>made(5~9%) 케이스: uncensored지만 항등식 밖 → 위 규칙으로 처리, 잔여는 censored 취급.

## 절대규칙 준수
Leakage 금지(수요모델 t-이전, price도 t-이전) / 품절 censored 보존 / 시간순 rolling / MAPE 단독 금지(cost 메인) / parquet 진입점.

## 테스트
- 항등식/제외규칙: 규칙대로 제외·clip·보고(정확값).
- two-class newsvendor known-answer: 알려진 N,C,δ,c → Q\* = 이론값(수식 직접 검증).
- Level 1 = 상한: Level 1 Q\* ≥ Level 2 Q\* (동일 입력).
- implied c known-answer: made·분포 주면 SL·implied c 정확.
- placebo arm: Q=made arm이 실제 out 재현(반사실 산식 일관).
- leakage 회귀: t 이후 데이터 주입해도 t 시점 Q\* 불변.
- degenerate: 빈 히스토리·단일값·c=0/1 → 자신만만한 값 금지(NaN+note).

## 정직한 한계
- **절감액이 작거나 음수일 수 있음**(DOW-조건부 실측). 그 경우 가치는 implied c 갭.
- α_A 부재 → α↔c 동시식별 불가, c 확정은 고객사 대기.
- d가 Q에 내생(축소 반사실 분포붕괴) → made 근방 한계 재최적화로 한정, A3 대역.
- 카테고리 sold_total=참수요는 **현행 정책·공급 하** W0 흡수 가정(광교 한정, 삼성/광화문 유보).
- Cu=(1−c)p는 상한(매진 시 타카테고리 흡수/재방문 가능, M3).
- 진열 충만도(수요 견인) 등 미모델 편익 — made 대비 절감 주장 시 각주(M4).
- price 품목 이질성(pastry 11배) → 수량+폐기물량 가중 cross-check.
- COVID 연도(2021~22) 히스토리 → rolling window로 완화(M5).

## 다음 (Phase B 이후)
다중시각 재검증: ①할인정책 변경/중단 자연실험 ②never-discounted 품목 저녁 대조 ③depth별 basket/재구매 ④(향후 식별 α)∩ c 교차.

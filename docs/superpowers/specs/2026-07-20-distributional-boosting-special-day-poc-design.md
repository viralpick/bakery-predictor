# 분포 부스팅 특수일 PoC 설계

날짜: 2026-07-20
상태: 승인됨 (architect, "A로 가자")
관련 메모리: project_distributional_forecasting_stack, project_margin_buffer_optimization,
project_special_day_feature_spec, project_order_conformal_calibration

## 배경 / 가치 가설

발주 안전마진의 정석 = μ(x)뿐 아니라 **σ(x)도 feature 함수로 학습**하는 분포 회귀.
현 스택은 LightGBM q0.5/q0.85 독립 fit → **conditional spread mis-calibration**(저수요일=명절에
마진이 오히려 넓어짐; 광화문 마진-레벨 상관 −0.81, 광교 −0.05).

architect 가치 가설: **전체 WAPE는 크게 안 올라도, 특수일(급등·급락) 예측력이 오르면 의미 있다.**
갑자기 피크치는 날/급락하는 날을 못 맞추는 게 큰 운영 리스크이기 때문. → 성공 지표를
전체 평균이 아니라 **이벤트일 서브셋의 분포/마진 커버리지**로 잡는다.

## 현 상태 (착수 전 사실)

- LSS/NGBoost 의존성 **없음** → NGBoost 신규 추가.
- 특수일 처리기 **이미 존재**: `EventLevelPrior`(models/event_prior.py) — post-model 레벨 앵커
  블렌드, xmas 배포됨. 분포모델과 경쟁/보완 관계 → ablation으로 다룸.
- 다매장 데이터: `data/internal/v2/`(sales/inventory) + `scripts/store_daily.py`
  `build_store_daily(store_cd, store_id, exclude_bulk=True)`. STORE_MAP에 광교(store_gw01)·
  광화문(store_gh01) 존재. spread 병리도 이 경로(`scripts/store_predictive_power.py`)에서 측정됨.
- 카테고리 총합 stack = `category_aggregate.build_category_daily`/`build_features`(패키지 CLI 경로,
  광교 단독 parquet). 다매장은 scripts 경로. **PoC는 scripts 다매장 경로 사용**(2매장 필요).

## 접근 결정

**A. NGBoost (LogNormal)** 채택. 근거: torch 없는 sklearn 기반이라 가볍게 가치 가설 검증,
카테고리 총합은 매끈한 양수·우편향 aggregate라 LogNormal이 μ·σ 동시추정에 자연스럽고
"×K(레벨비례 마진)" 직관과 부합. 대안 LightGBMLSS(torch 무거움)는 이기면 그때 풀버전 고려.
2-quantile scale-coupling(신규 의존성 0)은 proper 분포·σ(x) 학습이 아니라 반쪽 → 제외.

## 설계

### 범위
- 매장: 광교(store_gw01) + 광화문(store_gh01).
- 레벨: **category-total**(bread/pastry 등 TARGET_CATEGORIES). spread 병리·재측정 KPI 모두 이 레벨.
- 격리: `scripts/distributional_boosting_poc.py` (코어 src 무변경; 이기면 그때 src 승격).

### 모델
- NGBoost(Dist=LogNormal), base learner=기본(sklearn tree). feature = 현 카테고리 stack feature
  (date/lag/rolling + cyclic_calendar/holiday/event/weather). μ(x)·σ(x) 동시추정.
- 발주량 = 적합 분포의 분위수(q0.85 기본; q sweep 부차).
- 학습 target = adjusted_demand(정상+0.8×마감, 헌장). LogNormal이므로 log 변환/양수 보장 처리.

### 비교 baseline (동일 fold·feature)
- 현 스택: LightGBM q0.5 + conformal, 그리고 q0.85 직접발주. (category_total 경로 재사용)

### 평가 (walk-forward expanding, n_folds=8 × 8주 — 재측정과 동일 프로토콜)
1. **이벤트일 서브셋** (★주 지표) — EventLevelPrior 이벤트 목록(설/추석/xmas/어린이날/발렌타인/
   화이트데이 등): coverage@0.85, pinball loss, 폐기/매진 trade-off(헌장 KPI).
2. **전체·평일 서브셋** (품질 게이트) — 크게 안 올라도 되나 **퇴행 없어야**.
3. **통계적 극단값** — dow 통제 잔차 상/하위 X%: **검출·리포트만**(성공 판정 제외, 정의는 추후 고객사).
4. **spread 진단** — 마진-레벨 상관(광화문 −0.81 개선되는가) 재측정.

### EventLevelPrior 관계 (ablation)
분포모델을 event_prior 블렌드 **有/無 둘 다** 평가 → σ(x) 학습이 이벤트 앵커를 대체하는지 보완하는지 판정.

### 산출물
- `scripts/distributional_boosting_poc.py`
- `docs/distributional_boosting_poc_result.md` (2매장 × 4서브셋 표 + spread 진단 + ablation)

## 성공 판정

이벤트일에서 (a) 커버리지가 nominal(0.85)에 **더 근접** AND (b) 폐기-매진 프론티어가 현 스택을
**지배(최소 비열위)**, 그리고 전체 **퇴행 없음**. 광화문에서 spread 병리(마진-레벨 상관) 완화가
실측되면 강한 확증. 세 조건 미충족 시 → LogNormal/분포족·feature 강화 재검토 후 drop 판단.

## 리스크 / 열린 문제

- rare event σ 추정 데이터 부족(5년에 추석 5번뿐) → 이벤트 강도/연휴위치 feature가 informative해야.
  conformal이 그 한계에도 커버리지 보장(잠정). PoC에서 이벤트일 σ 추정 신뢰구간 함께 리포트.
- NGBoost가 LightGBM 대비 feature 처리(categorical) 약할 수 있음 → 동일 feature·fold로 공정 비교,
  point WAPE 퇴행 시 원인 분리(모델 엔진 vs 분포 접근).
- 광화문 데이터 품질/기간이 광교와 다를 수 있음 → 착수 시 행 수·기간·결측 sanity check 선행.

## Non-goals
- item-level 분포 예측(희소 count, 별도 트랙).
- LightGBMLSS 풀버전(A가 이기면 후속).
- 통계적 극단값의 성공지표화(검출만, 정의는 고객사 협의 후).
- src 코어 승격(PoC 격리; 승리 시 별도 작업).

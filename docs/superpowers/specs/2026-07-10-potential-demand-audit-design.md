# potential_demand 전역 감사 — real 소비처 adjusted_demand 전환 (설계)

- 날짜: 2026-07-10
- 백로그: 발주 calibration 후속 #3 (TODO.md, 2026-07-09 합의 순서 3→4→1→2 중 첫째)
- 브랜치 기준: `main` @ 81490b0 (PR#30 conformal 머지 후)
- 관련 메모리: `project_stockout_time_bug_and_adjusted_demand`(potential_demand 폐기 확정),
  `project_order_conformal_calibration`(후속 백로그), `project_poc_v7_aos_demonstrator`(ontology=현재 기준)

## 배경 / 문제

`potential_demand`(censoring-corrected 수요, `features/potential_demand.py`)는 **real 데이터에서 폐기 확정**됐다. 3중 근거:
1. 수요 흡수(W0) 하 double-count.
2. 기반 `stockout_time`이 로더 버그(하루 다중 품절 이벤트 중 첫 것만 취함)로 손상.
3. 애초에 광교는 진짜 카테고리 품절이 거의 없음(과잉생산, 조기 전체소진 0.2%).

수요 target은 `adjusted_demand`(= `sold_units − closing_qty × (1−α)`, `features/category_aggregate.py:build_item_adjusted_demand`)로 확정됐다. PR#29/#30에서 **prospective-eval 경로**는 이미 adjusted_demand로 통일됐으나, 다음 경로들이 아직 real에서 `potential_demand`를 소비한다:

| 경로 | 소비 형태 | 기본 source |
|---|---|---|
| **ontology `rank_stockout_risk` / `_item_demand_points`** (v7 AOS 데모, ★현재 기준) | 수요 점추정 프록시 기본값 `DEMAND_PROXY_COL` | grounding-eval `--source` |
| **backtest** (`cmd_backtest`, v2/v3) | v2/v3 **학습 target만** (평가는 sold_units 잣대) | synthetic |
| **predict-next-week** (`cmd_predict_next_week`, v2/v3) | 학습 target + `yhat_potential_demand` 출력 컬럼 | (인자) |
| **alpha-sweep** (`cmd_alpha_sweep`, v2) | 학습 target + profit-sim "true demand" 주입(cli:583) | **real** |
| **business-report** (`cmd_business_report`, v0~v3) | 학습 target + profit-sim + 사업 KPI 주입(cli:817) | **real** |
| prospective-eval | — | ✅ 이미 adjusted_demand (PR#29/#30) |

**주입(profit-sim/KPI) 지점은 `alpha-sweep`·`business-report`** 두 커맨드다(spec 초안이 backtest로 오기). `backtest`는 v2/v3를 potential_demand로 학습만 하고 fold WAPE는 sold_units로 평가한다.

## 목표 / 비목표

**목표**
- real 데이터 경로(ontology + backtest + predict-next-week + alpha-sweep + business-report)가 `potential_demand`를 소비하지 않게 한다.
- 소스별 단일 결정 규칙으로 통일한다.
- **국소화 원칙**: 모델 레벨 `_default_target`(v2/v3→potential_demand)은 **변경하지 않는다**(synthetic 기본값 + 기존 pin 테스트 보존). 전환은 CLI/ontology 레이어에서 real일 때 `y_col`/`demand_col`을 `adjusted_demand`로 **명시 전달**하는 방식으로만 한다.

**비목표 (YAGNI)**
- `potential_demand` 필드·모듈의 물리적 제거 (synthetic ground-truth + schema 정합성 때문에 불가·불필요).
- 데이터 레이어(`bonavi_loader.attach_potential_demand`) 구조 변경.
- stockout-classifier 재학습 (별건 = 백로그 #4).
- 신규 모델링 로직.

## 핵심 결정 (사용자 승인 완료)

1. **real 수요 컬럼 = `adjusted_demand`** (sold_units 폴백 아님). 확정된 PoC target이고 prospective-eval과 일관.
2. **범위 = ontology + backtest + predict-next-week** (+ business_metrics 잣대). 데이터 레이어는 손대지 않음.
3. **"제거"의 실질 의미**: real 경로가 소비를 멈춤. synthetic 생성기 + arrival-curve 헬퍼는 유지. 물리 삭제 아님.

## 설계

### 1. 수요 컬럼 결정 규칙 (공유 원칙)

소스별 단일 규칙을 모든 경로에 동일 적용한다 (드리프트 방지).

- `source == "real"` → `build_item_adjusted_demand(daily, alpha)`로 enrich, 컬럼 `"adjusted_demand"`.
- `source == "synthetic"` → 그대로, 컬럼 `"potential_demand"`.

`build_item_adjusted_demand`는 **자립적**이다(내부에서 real 마감할인 데이터를 `load_sales_with_discount().closing_discount()`로 로드, `sold_units`/`item_id`/`date`만 있으면 동작, 입력 프레임 비변형). 따라서 각 경로에 한 줄 enrich 호출로 충분하다.

**공유 헬퍼** (`cli.py`):
```
def _resolve_demand_col(daily, source, alpha) -> (daily, col):
    """real → adjusted_demand enrich, synthetic → potential_demand 그대로."""
```
- `--alpha`(기본 `DEFAULT_ALPHA` = 0.5) 옵션을 `backtest` / `predict-next-week`에 추가. ontology 경로는 동일 기본값 상속.
- **α는 경로 간 단일 기본값 고정** → 경로별 수치 비교 가능성 유지. (prospective-eval의 `--alpha`와 같은 기본값.)

### 2. 경로별 변경

공유 메커니즘: `_build_forecasters(variants, *, v23_target=...)` / 개별 `GlobalLGBM(..., y_col=demand_col)` 로 v2/v3 target을 CLI에서 주입. v0/v1은 항상 `sold_units`. `--closing-alpha`(기본 `DEFAULT_ALPHA`) 옵션 추가(alpha-sweep의 quantile `alphas`와 이름 충돌 회피 위해 `closing_alpha` 명명).

**backtest (`cmd_backtest`)**
- source==real이면 `_resolve_demand_col`로 daily enrich.
- `_build_forecasters(..., v23_target=demand_col)` → v2/v3가 real에서 `adjusted_demand` 학습(배관은 PR#29 검증). fold 평가 잣대(run_backtest y_col=sold_units)는 operations view라 그대로.

**predict-next-week (`cmd_predict_next_week`)**
- 동일 enrich.
- 점추정 모델 + production-quantile 모델(cli.py:180) 둘 다 `y_col=demand_col` 명시.
- 출력 컬럼: `yhat_adjusted_demand`(real) / `yhat_potential_demand`(synthetic). (`demand_col` 변수, cli.py:175.)

**alpha-sweep (`cmd_alpha_sweep`)**
- 동일 enrich. 루프의 `GlobalLGBM(feature_set=variant, ...)`에 `y_col=demand_col`.
- profit-sim 주입부(cli:583) → `demand_col` 사용. `simulate_profit(..., potential_col=demand_col)`.

**business-report (`cmd_business_report`)**
- 동일 enrich. `_build_forecasters(..., v23_target=demand_col)`.
- profit-sim + 사업 KPI 주입부(cli:817) → `demand_col`. `simulate_profit(..., potential_col=demand_col)`.

**ontology grounding (`rank_stockout_risk` / `_item_demand_points`, v7 AOS)**
- `run_eval(source)`(grounding/run.py)가 `load_dataset(source)` 단일 지점 → real이면 여기서 `dataset.daily`를 `build_item_adjusted_demand`로 enrich (frozen dataclass이므로 `dataclasses.replace`).
- `functions.py`에 `_resolve_demand_proxy(daily)` 추가: 프레임에 `adjusted_demand` 있으면 그것, 없으면 `potential_demand`. rank/point 함수 기본 `demand_col`을 이 리졸버로. → **source를 tools/arms까지 스레딩할 필요 없이** enrich된 프레임이 자동으로 adjusted 채택(real), synthetic은 potential 유지.
- `DEMAND_PROXY_COL`은 fallback 상수로 유지. `functions.py` docstring "라이브 forecast 아직 안 wired" 문구를 real=adjusted 전환 반영으로 갱신.

### 3. 유지 (변경 없음)

- `data/synthetic.py`의 `potential_demand` 생성 — PoC ground-truth, 로더 버그 없음.
- `StoreHours` / `bakery_hour_profile` (arrival-curve 헬퍼) — 범용, prospective가 재사용. `potential_demand` 모듈에 있지만 censoring 교정과 무관.
- `data/schema.py`의 `potential_demand` 필드 — synthetic이 계속 채움.

### 4. Deprecation 마커 (문서화만)

데이터 레이어 로직은 안 건드리되, 오염된 필드가 real에서 미소비임을 명시:
- `bonavi_loader.py` attach 지점: "real에서 stockout_time 버그로 오염, 더 이상 소비 안 됨" 주석.
- `schema.py` `potential_demand` 필드 설명: 위 경고 추가.
- `ontology/functions.py` `DEMAND_PROXY_COL` docstring: real은 adjusted_demand 사용으로 갱신.

## 데이터 흐름 (변경 후)

```
real:  loader.daily ──_resolve_demand_col──> +adjusted_demand ──> {backtest 학습/잣대, predict target/출력, ontology 수요점}
       (potential_demand 컬럼은 프레임에 남되 real 경로에서 미참조)
synth: synthetic.daily ────────────────────> potential_demand ──> (동일 경로, 기존 동작 보존)
```

## 테스트 / 마이그레이션

**기존 pin 테스트는 그대로 통과 (국소화 효과)**
- `test_v2_pipeline::test_v2_default_target_is_potential_demand`, `test_v3_pipeline::test_v3_default_target_is_potential_demand`, `test_backtest_clone`(y_col=="potential_demand") — 모두 `_default_target` **미변경**이라 green 유지. (모델 기본값은 potential_demand 그대로, CLI만 real에서 override.)
- `test_potential_demand`, `test_discount_analysis`, `test_prospective*`, `test_ontology_functions` — 기존 동작 보존, 회귀 방지 유지.

**신규/보강 테스트 (helper 단위 — real 파일 의존 회피)**
- `_resolve_demand_col`: `discount_rows` 주입해 결정론 테스트 — synthetic→(frame 불변, "potential_demand"), real→("adjusted_demand" 컬럼 존재).
- `_build_forecasters(v23_target=...)`: v2/v3 forecaster.y_col == 전달값, v0/v1 == "sold_units".
- `_resolve_demand_proxy(daily)`: adjusted_demand 있는 프레임→"adjusted_demand", 없으면 "potential_demand".
- `run_eval` real enrich: `load_dataset` monkeypatch로 real 소스 시 dataset.daily에 adjusted_demand 부착 확인.

**필수 통과 (회귀 게이트)**
- `test_split_leakage` / `test_features_leakage` — lag/rolling이 `adjusted_demand`로 재구성돼도 leakage-safe (PR#29에서 검증됨). 반드시 green.

**신규 단언**
- `_resolve_demand_col`: source별 반환 컬럼 정확값 비교.
- real backtest/predict target == `adjusted_demand`, synthetic == `potential_demand`.
- real ontology 수요 점추정 == adjusted 기반.

## 캐비엣

- `predict-next-week` 출력 의미 변화: potential이 아닌 adjusted 수요 예측. **의도된 교정**이며 컬럼명 변경으로 정직하게 노출. `next_week_predictions.csv` 자동 소비처 없음(수동 리포트용).
- real 광교 ontology `p_stockout`은 여전히 거의 degenerate(진짜 품절 0.2%). 이 감사는 **수요 컬럼**만 고침 — 품절 확률 아티팩트는 백로그 #4(stockout-classifier 재학습) 소관.
- 광교 단독. 타매장(삼성 등)은 lost-sales 신호 있어 품절 프로파일 다를 수 있음(현 PoC 범위 밖).

## 리스크

- **낮음**: 데이터 레이어/schema 구조 변경 없음, 신규 모델링 없음. y_col 배관은 PR#29에서 검증됨.
- 주의: 3개 경로에 α 옵션 추가 시 CLI 시그니처 변경 → 호출부/문서(CLAUDE.md 실행 예시) 동기화 필요.

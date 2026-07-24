# Harness distributional 배선 설계 (Phase 2-1)

**날짜**: 2026-07-25
**브랜치**: feat/harness-distributional
**선행**: Phase 1 스파인 (PR#53, main `3fe50b0`). 마스터 로드맵 1단계.

## 목표

harness `windowed_backtest` 코어를 forecaster 종류로 **일반화**하여, 실험 1개(YAML 1파일)가 `spec.forecaster` 리스트 전부(category_total + distributional_total)를 실행하고 **forecaster별 결과 + 비교표**를 산출한다. category_total 엔진 동등성 게이트(Phase 1 Task 3)는 절대 깨지지 않는다.

## 배경: 두 forecaster의 관계

canonical 경로의 `feat`은 하루 **1행 = 빵 전체 총량**(bread+pastry+sandwich 합, cake 제외; 1826일). `category_total`은 카테고리를 따로 예측하는 게 아니라 **총량 시계열 1개를 모델링**한다. distributional_total은 **동일 타깃·동일 granularity**를 예측하되, 불확실성 모델링 방식만 다르다:

| | category_total | distributional_total |
|---|---|---|
| 엔진 | LightGBM | NGBoost (LogNormal) |
| 점추정 predict_expected | L1 median | 분포 median |
| 발주 predict_production | 독립 분위수 q0.85 (**fit에 고정**) | 분포 분위수 q0.85 (**predict에 지정**) |
| 불확실성 | 분위수 독립 학습 → 저수요일 spread 병리 | μ·σ 동시추정 → 분위수 일관 |
| fit 시그니처 | alpha_demand, production_q | n_estimators, lr, random_state |
| feature | 범주형 허용 | numeric-only (NGBoost) |
| 타깃 제약 | 없음 | 양수-only (LogNormal) |

distributional은 category_total의 spread 병리를 고치려 만든 대안이나, 광교 full-window선 둘이 거의 패리티(광교는 병리 −0.05로 미미). 배선 목적 = **매장·변동성에 따라 어느 엔진이 나은지 같은 표면에서 나란히 비교.**

## 아키텍처 (승인된 Approach A: Forecaster 어댑터)

fold 루프 + event_prior 블렌드는 forecaster 무관하게 동일하고, **fit 호출과 predict_production 호출 규약만 다르다.** 이 차이를 어댑터에 격리하고 windowed_backtest는 순수 오케스트레이션으로 유지한다(게이트가 지키는 부분).

### 1. Forecaster 어댑터 레이어 — `src/bakery/harness/forecasters.py` (신규)

```python
class FittedForecaster(Protocol):
    def predict_expected(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_production(self, df: pd.DataFrame) -> np.ndarray: ...   # q 바인딩됨

class Forecaster(Protocol):
    name: str
    def fit(self, train: pd.DataFrame, *, target_col: str, alpha: float,
            production_q: float) -> FittedForecaster: ...
```

- **CategoryTotalForecaster** (`name="category_total"`): `fit` → `fit_category_total(train, target_col=target_col, alpha_demand=alpha, production_q=production_q)`. 반환 `CategoryTotalModel`이 이미 `predict_expected`/`predict_production`(q fit-고정) 계약 만족 → 그대로 반환(무손상).
- **DistributionalTotalForecaster** (`name="distributional_total"`): `fit` → `fit_distributional_total(train, target_col=target_col)` (alpha 무시) 후 **얇은 wrapper**(`_ProdQBound`)로 감싸 `predict_production(df)`가 인자 없이 `model.predict_production(df, production_q=production_q)`를 호출하도록 바인딩. **ngboost는 fit 내부 lazy import**(category 전용 실행 시 무거운 import 회피).
  - **★결정성 (hermetic seed)**: NGBoost 0.5.11은 `NGBRegressor(random_state=...)` 인자만으론 비결정적(fit마다 상대오차 ~0.8%) — **전역 numpy RNG**를 사용하기 때문. 실증 완료: `random_state=42`만으론 diff 2.4단위, `np.random.seed(42)` 후 fit → diff 0.0. 어댑터 `fit`에서 **save/restore로 hermetic하게** 시드(전역 RNG 누수 없음, code-quality "숨은 글로벌 변경" 회피). `fit_distributional_total`(공유 src)은 건드리지 않음(기존 distributional 스택 수치 shift 방지). 시드는 `42` 하드코딩(fit_distributional_total 기본값과 일치, 설정화는 YAGNI).
    ```python
    state = np.random.get_state()
    np.random.seed(42)
    try:
        model = fit_distributional_total(train, target_col=target_col)
    finally:
        np.random.set_state(state)
    ```
    fold 루프 8-fold 2회 실행 run-to-run 정확일치·루프 전후 RNG 상태 불변 실증 완료.

### 2. `backtest_core.windowed_backtest` 일반화 — 한 곳만 변경

시그니처에 `forecaster: Forecaster | None = None` 추가. 함수 진입부에서 `fc = forecaster or CategoryTotalForecaster()`. fold 루프의
```python
model = fit_category_total(train_df, target_col=target_col, alpha_demand=alpha, production_q=production_q)
```
한 줄을
```python
model = fc.fit(train_df, target_col=target_col, alpha=alpha, production_q=production_q)
```
로 교체. **나머지(fold 경계·`predict_expected`/`predict_production` 호출·event_prior 블렌드·folds/preds 조립·metrics_from_preds)는 바이트 불변.** default=None→CategoryTotalForecaster이므로 기존 호출부(weekly_overlay 등 sibling 스크립트)·엔진 동등성 게이트 무손상. event_prior 블렌드는 예측 배열에 씌우는 post-model이라 두 엔진 균일 적용.

import 방향: `forecasters` → `models`(category_total/distributional_total). `backtest_core` → `forecasters`(default). 순환 없음.

### 3. Runner 다중 forecaster — `ExperimentResult` 신규

```python
@dataclass
class ExperimentResult:
    name: str
    runs: dict[str, RunResult]      # forecaster명 → RunResult (기존 RunResult 재사용)
    comparison: pd.DataFrame        # forecaster당 1행: forecaster, wape, wpe, stockout_risk, surplus_mean_units, surplus_rate, n_test
```

`run_experiment(spec, *, out_dir, cache_dir=None, _trace=None) -> ExperimentResult`:
1. runnable = `[f for f in spec.forecaster if is_runnable(f)]`; point/composite는 경고 후 스킵. runnable 없으면 ValueError.
2. `feat`을 **1회만** 빌드(캐시 공유; 키 = source/store/target/alpha — Phase 1과 동일).
3. events/lunar resolve (event_prior layer 있을 때).
4. runnable 각 forecaster: `fc = build_forecaster(fname)` → `windowed_backtest(feat, ..., forecaster=fc)` → `metrics_from_preds` → `out/<exp>/<fname>/{predictions.csv, fold_results.csv, metrics.json}` 기록 → `RunResult` 수집.
5. `comparison` DataFrame 조립 → `out/<exp>/comparison.csv`, `config_resolved.yaml` 기록.

산출물 레이아웃:
```
out/<exp>/
  config_resolved.yaml
  comparison.csv
  <forecaster>/{predictions.csv, fold_results.csv, metrics.json}
```

### 4. Registry + CLI

- `registry.build_forecaster(name) -> Forecaster` 팩토리(category_total/distributional_total 인스턴스화; 그 외 KeyError).
- `is_supported_phase1` → **`is_runnable`**(category_total + distributional_total = True; point/composite = False). Phase 1 test의 이름 변경 반영.
- `harness-run`: `ExperimentResult.comparison`을 rich 표로 콘솔 출력.

## Acceptance (검증 기준)

1. **★category_total 엔진 동등성 게이트(기존 `test_backtest_core_equivalence`) 계속 통과** — 어댑터 리팩토링이 category 경로를 바꾸지 않았음을 증명(회귀 방지 핵심 hard gate).
2. **distributional 신규 test** (`test_distributional_wiring`): `windowed_backtest(feat, forecaster=DistributionalTotalForecaster())`가 예측 산출 + **결정성**(hermetic seed 덕분에 2회 실행 `expected`/`production` `assert_array_equal` 정확일치 — random_state 인자만으론 불충분, §1 hermetic seed 참조) + WAPE 유한·정상 범위(0<wape<1). harness에 distributional "원본"이 없으므로 정확일치 대신 결정성+sanity. **주의**: 이 결정적 WAPE는 한 realization일 뿐, 과거 발행된(비결정적) full-window distributional 수치와 같을 필요 없음 → 특정 앵커 등호 아닌 sane 범위로만 단언.
3. **runner test**: `ExperimentResult` 반환, `runs`에 두 forecaster 키, `comparison` 2행, 산출물 파일 존재. distributional이 느리므로 **소 n_folds(8)** 사용.
4. 전체 스위트 통과(사전존재 `test_store_daily_redefine` 실패는 무관, 별도).

## 마이그레이션 스텝 (기존 심볼 변경 → 호출부 갱신)

- `run_experiment` 반환 `RunResult` → `ExperimentResult`: `test_runner.py`(반환 타입 단언), `cli.py cmd_harness_run`(result.metrics → comparison 출력) 갱신. **grep 결과 소비처는 이 둘 + `__init__` re-export뿐**(sibling 스크립트는 windowed_backtest만 사용, 영향 없음).
- `is_supported_phase1` → `is_runnable`: `test_registry.py`, `runner.py` 갱신. `__init__` re-export 갱신.
- `windowed_backtest`에 `forecaster` 인자 추가는 **후방호환**(default=None) — 기존 호출부 무변경.

## Open risks

1. distributional numeric-only feature / 양수-only 타깃 — 기존 CLI full-window([[project_distributional_forecasting_stack]] c-2b)서 동일 `feat`에 이미 동작 → 저위험. 계획서 첫 스텝서 `fit_distributional_total(feat 서브셋)` 스모크로 확인.
2. `ExperimentResult` 도입 = `run_experiment` 반환 타입 변경 → 위 마이그레이션 스텝으로 소비처 전부 갱신.
3. distributional 52-fold는 수 분 소요 → 테스트는 8-fold, 게이트는 category 52 유지.

## 범위 밖 (다음 로드맵 단계)

- report/viz 승격(2단계) — comparison을 시각화/HTML로. 이번엔 comparison.csv까지만.
- DEFAULT_METRICS(6종) ↔ metrics_from_preds(5종 실산출) 이름 정합 — report 단계서 reconcile.
- point/composite forecaster 실행(현재 registry 등록만, 경고 스킵 유지).
- event_priors 완전 단일화.

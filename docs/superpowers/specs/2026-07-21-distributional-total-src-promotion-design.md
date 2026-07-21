# 분포모델 src 승격 설계 (DistributionalTotalModel)

날짜: 2026-07-21
상태: 설계 승인됨 (architect: "alias 고고")
관련 메모리: project_distributional_forecasting_stack
선행 PoC: `docs/distributional_boosting_poc_result.md` (NGBoost LogNormal 채택 방향 확정)
관련 스펙: `docs/superpowers/specs/2026-07-20-distributional-boosting-special-day-poc-design.md`

## 배경

NGBoost(LogNormal) 분포 PoC가 채택 방향으로 확정됨:
- point-WAPE 패리티(전체 0.086 ≈ LightGBM 0.087) + 광화문 spread 병리 반전(−0.55→+0.17)
- ★log-변환 shortcut 반증: 곱셈 공간(log target)≠곱셈 결합(공유 μ). NGBoost 결합 적합이 핵심.
- conformal은 이득 미검증(분포모델이 최근 OOS서 near-calibrated) → **raw 분포 분위수 그대로 발주 권장**.

지금은 PoC가 `scripts/distributional_boosting_poc.py`에 격리돼 `--with ngboost` 임시 실행 상태.
이 작업 = 이긴 분포모델을 코어 `src/bakery/models/`로 승격해 재사용 가능한 프로덕션 단위로 만든다.

## 범위 (architect 결정: 최소)

**포함**: 분포모델 클래스 + fit 함수 + 계약/leakage 테스트, ngboost 정식 의존성.
**비포함 (명시적 non-goal)**: CLI 배선(prospective-eval), decision 레이어, predict-next-week,
walk-forward conformal, item-level 분포, 다분포족(NegBin). 전부 후속 별도 작업.

이유: no-yolo 원칙(한 번에 여러 파일 대규모 수정 금지). 코어 유닛을 안전히 안착시키고,
소비자 배선은 각각 독립 검증하며 진행한다.

## 아키텍처

### 모듈
- 신규 `src/bakery/models/distributional_total.py` (신규 파일, 단일 책임 = 분포추정)
- 기존 `CategoryTotalModel`(LightGBM 경로)은 **무손상** — 별도 경로로 공존
- `select_feature_cols`·`LEAK_COLS`는 `category_total`에서 import (leak 컬럼 단일 출처 유지)

### 의존성
- `pyproject.toml` `dependencies`에 `ngboost>=0.5` 추가 (설치 확인: 0.5.11, sklearn 1.8.0 호환, ~5.4s/fit)

### 공개 API

```python
def fit_distributional_total(
    train: pd.DataFrame,
    target_col: str = "adjusted_demand_unit",
    n_estimators: int = 500,
    learning_rate: float = 0.02,
    random_state: int = 42,
) -> "DistributionalTotalModel":
    """NGBoost(Dist=LogNormal)로 category-total 수요의 μ(x)·σ(x)를 동시 추정."""


@dataclass
class DistributionalTotalModel:
    model: NGBRegressor           # 적합된 NGBoost(LogNormal)
    feature_cols: list[str]       # select_feature_cols(train, target_col) 결과
    target_col: str

    # --- 분포 네이티브 인터페이스 ---
    def predict_dist(self, df):                      # scipy frozen lognorm 배열
    def predict_quantile(self, df, q: float):        # 임의 분위수 (0<q<1)
    def predict_median(self, df):                    # q0.5 (= scale)
    def predict_sigma(self, df):                     # σ(x) log-space (진단·coupling)

    # --- CategoryTotalModel drop-in 호환 alias ---
    def predict_expected(self, df):                  # = predict_median(df)
    def predict_production(self, df, production_q: float = 0.85):  # = predict_quantile(df, production_q)
```

- alias 근거: 향후 CLI에서 `fit_category_total(...)`↔`fit_distributional_total(...)` 무손상 swap.
  두 모델이 동일 계약(`predict_expected`/`predict_production`)을 만족 → 비교·교체 깔끔.
- ⚠️ `predict_expected` = **median**(점추정), LogNormal의 통계적 기댓값(mean=exp(μ+σ²/2))이 **아님**.
  이는 의도적 — 기존 `CategoryTotalModel.predict_expected`(L1=median) 의미와 일치시켜 drop-in 성립.
  (발주 base가 median이므로 운영상으로도 median이 맞음.)
- **event_prior / conformal은 이 클래스에 넣지 않음**: 기존 post-model 레이어
  (`EventLevelPrior.blend`, `ConformalOrderCalibrator`)를 소비자가 씌우는 구조 유지.
  모델은 순수 분포추정 단위 (단일 책임).

### 구현 노트
- fit: `X = train[feature_cols]`, `y = train[target_col]`. NGBoost는 numpy 입력
  (`X.to_numpy()`, `y.to_numpy()`).
- `predict_dist`: `model.pred_dist(df[feature_cols].to_numpy()).dist` (scipy frozen lognorm).
  분위수는 `.ppf(q)`, median은 `.ppf(0.5)`, σ는 `pred_dist(...).params["s"]`.
- 예측 시 df는 feature_cols만 있으면 됨 (target 불필요) — leakage-safe 계약.

## 안전장치

- **LogNormal 양수 전용**: `fit`에서 train target에 y≤0 존재 시 `ValueError`(명확 실패).
  휴무/0 수요일 처리는 호출자 책임 (docstring 명시). category-total은 구조적 양수라
  정상 경로에선 발생 안 함 (PoC서 광교 min 148·광화문 min 76).
- **재현성**: `random_state` 고정 → 동일 입력·seed 동일 예측.
- **feature 누수 차단**: `select_feature_cols` 재사용 → target·LEAK_COLS 자동 제외.

## 테스트 (`tests/test_distributional_total.py`)

계약·leakage·결정성을 정확값/속성으로 검증 (truthy·부분문자열 금지):

1. **fit→predict shape**: `predict_quantile(test, 0.85)` 길이 == len(test).
2. **분위수 단조성**: 모든 행에서 `q0.5 ≤ q0.85 ≤ q0.95` (LogNormal ppf 단조).
3. **median 일관성**: `predict_median` == `predict_dist().ppf(0.5)` (allclose).
4. **alias 동등**: `predict_expected` == `predict_median`,
   `predict_production(q)` == `predict_quantile(q)` (정확 동일 배열).
5. **예측 양수**: 모든 분위수 예측 > 0 (LogNormal 정의역).
6. **feature_cols 계약**: target·LEAK_COLS ∉ feature_cols.
7. **leakage 계약**: target 컬럼을 뺀 df로 predict 정상 동작 (피처만 필요).
8. **양수 가드**: train에 y≤0 섞으면 `fit`가 `ValueError`.
9. **결정성**: 동일 random_state 두 번 fit → 예측 allclose.

테스트 데이터 = 소형 합성 프레임(양수 target + 몇 개 feature + LEAK_COLS 일부 컬럼).
NGBoost fit이 테스트당 수 초라 `n_estimators`는 작게(예: 50) 오버라이드.

## 검증 절차

1. `uv run pytest tests/test_distributional_total.py` (신규 계약 테스트)
2. `uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py` (회귀 무손상)
3. `uv run pytest` (전체 스위트 — ngboost 의존성 추가가 기존 import 안 깨는지)
4. 3축 리뷰(재사용성·품질·효율)

## 리스크 / 열린 문제

- ngboost 의존성 추가가 `uv sync`·CI 환경에서 해석되는지 (로컬 `--with` 확인됨, `uv add`는 미실행).
- 승격 후 CLI 배선(후속)까지 실제 파이프라인엔 미투입 — 이 작업 단독으론 프로덕션 예측 미변경.
- 2매장(광교·광화문) PoC 기반 — 4매장 일반화·타분포족은 후속에서.

## 후속 (이 스펙 밖)

CLI `--model distributional` 배선 → event_prior 블렌드 → (필요시)walk-forward conformal
→ decision 레이어 → predict-next-week → 4매장/NegBin 확장. 각각 독립 스펙.

# DistributionalTotalModel src 승격 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NGBoost(LogNormal) 분포모델을 `src/bakery/models/distributional_total.py`로 승격해 μ(x)·σ(x) 동시추정 발주 모델을 재사용 가능한 코어 유닛으로 만든다.

**Architecture:** 신규 파일에 `fit_distributional_total()` + `DistributionalTotalModel`(dataclass)을 둔다. NGBoost `Dist=LogNormal`로 적합하고 scipy frozen lognorm으로 임의 분위수를 낸다. 기존 `CategoryTotalModel`(LightGBM)은 무손상 공존하며, drop-in alias(`predict_expected`/`predict_production`)로 동일 계약을 만족시켜 향후 CLI swap을 가능케 한다. event_prior·conformal은 소비자 레이어로 남긴다.

**Tech Stack:** Python, NGBoost(LogNormal), scipy.stats, pandas, numpy, pytest, uv.

## Global Constraints

- 의존성: `ngboost>=0.5` (설치 확인 0.5.11, scikit-learn 1.8.0 호환).
- leak 컬럼 단일 출처: `select_feature_cols`·`LEAK_COLS`는 `bakery.models.category_total`에서 import (재정의 금지).
- target 기본값: `adjusted_demand_unit` (헌장).
- 재현성: `random_state=42` 기본.
- LogNormal 양수 전용: train target에 y≤0 있으면 `ValueError`.
- 테스트 단언은 정확값/속성 비교(truthy·부분문자열 금지). NGBoost fit 비용 절감 위해 테스트는 `n_estimators=50`.
- `CategoryTotalModel`(LightGBM) 경로 무손상 — 기존 파일 수정 금지(신규 파일만).

---

## File Structure

- **Create** `src/bakery/models/distributional_total.py` — 분포모델 (fit + 클래스). 단일 책임 = category-total 수요의 분포추정.
- **Create** `tests/test_distributional_total.py` — 계약·leakage·결정성 테스트.
- **Modify** `pyproject.toml` — `dependencies`에 `ngboost>=0.5` 추가 (`uv add`가 `uv.lock`도 갱신).

---

### Task 1: 의존성 추가 + 분포모델 코어

**Files:**
- Modify: `pyproject.toml` (dependencies에 ngboost 추가)
- Create: `src/bakery/models/distributional_total.py`
- Test: `tests/test_distributional_total.py`

**Interfaces:**
- Consumes: `bakery.models.category_total.select_feature_cols(df, target_col) -> list[str]` (기존; date·target·LEAK_COLS 제외한 feature 컬럼).
- Produces:
  - `fit_distributional_total(train: pd.DataFrame, target_col: str = "adjusted_demand_unit", n_estimators: int = 500, learning_rate: float = 0.02, random_state: int = 42) -> DistributionalTotalModel`
  - `DistributionalTotalModel(model: NGBRegressor, feature_cols: list[str], target_col: str)` with:
    - `predict_dist(df) -> scipy frozen lognorm 배열`
    - `predict_quantile(df, q: float) -> np.ndarray`
    - `predict_median(df) -> np.ndarray`
    - `predict_sigma(df) -> np.ndarray`

- [ ] **Step 1: ngboost 의존성 추가**

Run:
```bash
uv add "ngboost>=0.5"
```
Expected: `pyproject.toml` `dependencies`에 ngboost 추가, `uv.lock` 갱신, 설치 성공.
Verify:
```bash
uv run python -c "from ngboost import NGBRegressor; from ngboost.distns import LogNormal; print('ok')"
```
Expected: `ok`

- [ ] **Step 2: 실패 테스트 작성 (코어 계약)**

Create `tests/test_distributional_total.py`:
```python
import numpy as np
import pandas as pd
import pytest

from bakery.models.distributional_total import (
    DistributionalTotalModel,
    fit_distributional_total,
)

TARGET = "adjusted_demand_unit"


def _synth(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """양수 target + feature 2개 + LEAK_COL 1개(sold_total_unit)."""
    rng = np.random.RandomState(seed)
    x1, x2 = rng.rand(n), rng.rand(n)
    y = np.exp(5.0 + 0.5 * x1 + 0.3 * rng.randn(n))  # 양수, lognormal-ish
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "f1": x1, "f2": x2,
        "sold_total_unit": y * 1.1,   # LEAK_COL — feature에서 제외돼야
        TARGET: y,
    })


def _fit(df: pd.DataFrame | None = None) -> DistributionalTotalModel:
    df = _synth() if df is None else df
    return fit_distributional_total(df, target_col=TARGET, n_estimators=50)


def test_predict_quantile_shape():
    m, df = _fit(), _synth()
    assert len(m.predict_quantile(df, 0.85)) == len(df)


def test_quantile_monotonic():
    m, df = _fit(), _synth()
    q50 = m.predict_quantile(df, 0.5)
    q85 = m.predict_quantile(df, 0.85)
    q95 = m.predict_quantile(df, 0.95)
    assert np.all(q50 <= q85)
    assert np.all(q85 <= q95)


def test_median_matches_dist():
    m, df = _fit(), _synth()
    assert np.allclose(m.predict_median(df), np.ravel(m.predict_dist(df).ppf(0.5)))


def test_predictions_positive():
    m, df = _fit(), _synth()
    assert np.all(m.predict_quantile(df, 0.85) > 0)


def test_sigma_shape_and_positive():
    m, df = _fit(), _synth()
    sigma = m.predict_sigma(df)
    assert len(sigma) == len(df)
    assert np.all(sigma > 0)


def test_feature_cols_exclude_leak_and_target():
    m = _fit()
    assert TARGET not in m.feature_cols
    assert "sold_total_unit" not in m.feature_cols
    assert "date" not in m.feature_cols
    assert "f1" in m.feature_cols
    assert "f2" in m.feature_cols


def test_predict_without_target_column():
    m = _fit()
    df_no_target = _synth().drop(columns=[TARGET])
    assert len(m.predict_quantile(df_no_target, 0.85)) == len(df_no_target)


def test_nonpositive_target_raises():
    df = _synth()
    df.loc[0, TARGET] = 0.0
    with pytest.raises(ValueError):
        fit_distributional_total(df, target_col=TARGET, n_estimators=50)


def test_deterministic():
    df = _synth()
    m1 = fit_distributional_total(df, target_col=TARGET, n_estimators=50)
    m2 = fit_distributional_total(df, target_col=TARGET, n_estimators=50)
    assert np.allclose(m1.predict_quantile(df, 0.85), m2.predict_quantile(df, 0.85))
```

- [ ] **Step 3: 테스트 실행 → 실패 확인**

Run: `uv run pytest tests/test_distributional_total.py -x --color=no`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.models.distributional_total'` (collection error).

- [ ] **Step 4: 분포모델 구현**

Create `src/bakery/models/distributional_total.py`:
```python
"""category-total 수요의 분포회귀 발주 모델 (NGBoost LogNormal, μ·σ 동시추정).

PoC(docs/distributional_boosting_poc_result.md)서 채택 방향 확정:
LightGBM 독립분위수의 spread 병리(저수요일 과대마진)를 공유-μ 결합(LogNormal)으로 해소.
발주 = 적합 분포의 분위수(기본 q0.85). CategoryTotalModel(LightGBM)과 무손상 공존하며
drop-in alias(predict_expected/predict_production)로 동일 계약을 만족한다.

event_prior 블렌드·conformal 보정은 이 클래스에 넣지 않는다 — 소비자가 씌우는 post-model 레이어.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import LogNormal

from bakery.models.category_total import select_feature_cols


def fit_distributional_total(
    train: pd.DataFrame,
    target_col: str = "adjusted_demand_unit",
    n_estimators: int = 500,
    learning_rate: float = 0.02,
    random_state: int = 42,
) -> "DistributionalTotalModel":
    """NGBoost(Dist=LogNormal)로 μ(x)·σ(x)를 동시 추정.

    LogNormal은 양수 전용 → train target에 y≤0 있으면 ValueError.
    휴무/0 수요일 처리는 호출자 책임(category-total은 구조적 양수).
    """
    feat_cols = select_feature_cols(train, target_col)
    y = train[target_col].to_numpy()
    n_bad = int((y <= 0).sum())
    if n_bad:
        raise ValueError(
            f"LogNormal requires positive target; found {n_bad} non-positive rows in '{target_col}'"
        )
    model = NGBRegressor(
        Dist=LogNormal,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=False,
    ).fit(train[feat_cols].to_numpy(), y)
    return DistributionalTotalModel(model=model, feature_cols=feat_cols, target_col=target_col)


@dataclass
class DistributionalTotalModel:
    model: NGBRegressor
    feature_cols: list[str]
    target_col: str

    def _pred_dist(self, df: pd.DataFrame):
        return self.model.pred_dist(df[self.feature_cols].to_numpy())

    def predict_dist(self, df: pd.DataFrame):
        """적합 LogNormal의 scipy frozen 분포(배열). 임의 분위수·전체 분포용."""
        return self._pred_dist(df).dist

    def predict_quantile(self, df: pd.DataFrame, q: float) -> np.ndarray:
        return np.ravel(self._pred_dist(df).dist.ppf(q))

    def predict_median(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_quantile(df, 0.5)

    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """σ(x) log-space (LogNormal shape). 진단·coupling용."""
        return np.ravel(self._pred_dist(df).params["s"])
```

- [ ] **Step 5: 테스트 실행 → 통과 확인**

Run: `uv run pytest tests/test_distributional_total.py --color=no`
Expected: 9 passed (test_predict_quantile_shape, test_quantile_monotonic, test_median_matches_dist, test_predictions_positive, test_sigma_shape_and_positive, test_feature_cols_exclude_leak_and_target, test_predict_without_target_column, test_nonpositive_target_raises, test_deterministic).

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml uv.lock src/bakery/models/distributional_total.py tests/test_distributional_total.py
git commit -m "feat: DistributionalTotalModel src 승격 (NGBoost LogNormal μ·σ)"
```

---

### Task 2: CategoryTotalModel drop-in alias

**Files:**
- Modify: `src/bakery/models/distributional_total.py` (DistributionalTotalModel에 alias 메서드 추가)
- Test: `tests/test_distributional_total.py` (alias 테스트 추가)

**Interfaces:**
- Consumes: Task 1의 `DistributionalTotalModel.predict_median` / `predict_quantile`.
- Produces:
  - `DistributionalTotalModel.predict_expected(df) -> np.ndarray` (= predict_median)
  - `DistributionalTotalModel.predict_production(df, production_q: float = 0.85) -> np.ndarray` (= predict_quantile)
  - 계약: 기존 `CategoryTotalModel.predict_expected(df)` / `predict_production(df)`와 동일 시그니처 → 무손상 swap.

- [ ] **Step 1: 실패 테스트 작성 (alias 동등)**

Append to `tests/test_distributional_total.py`:
```python
def test_alias_expected_is_median():
    m, df = _fit(), _synth()
    assert np.array_equal(m.predict_expected(df), m.predict_median(df))


def test_alias_production_is_quantile():
    m, df = _fit(), _synth()
    assert np.array_equal(m.predict_production(df, 0.85), m.predict_quantile(df, 0.85))
    # 기본 production_q=0.85 확인
    assert np.array_equal(m.predict_production(df), m.predict_quantile(df, 0.85))
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `uv run pytest tests/test_distributional_total.py::test_alias_expected_is_median tests/test_distributional_total.py::test_alias_production_is_quantile --color=no`
Expected: FAIL — `AttributeError: 'DistributionalTotalModel' object has no attribute 'predict_expected'`.

- [ ] **Step 3: alias 메서드 구현**

Append to `DistributionalTotalModel` (in `src/bakery/models/distributional_total.py`), after `predict_sigma`:
```python
    # --- CategoryTotalModel drop-in 호환 alias ---
    # predict_expected = median(점추정), LogNormal 통계적 기댓값(mean)이 아님 —
    # 기존 CategoryTotalModel(L1=median) 계약과 일치시켜 무손상 swap.
    def predict_expected(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_median(df)

    def predict_production(self, df: pd.DataFrame, production_q: float = 0.85) -> np.ndarray:
        return self.predict_quantile(df, production_q)
```

- [ ] **Step 4: 신규 테스트 실행 → 통과 확인**

Run: `uv run pytest tests/test_distributional_total.py --color=no`
Expected: 11 passed (Task 1의 9 + alias 2).

- [ ] **Step 5: 회귀·전체 스위트 확인**

Run leakage 회귀:
```bash
uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py --color=no
```
Expected: 12 passed (무손상).

Run 전체 스위트 (ngboost 의존성 추가가 기존 import 안 깨는지):
```bash
uv run pytest --color=no
```
Expected: 전체 통과 (신규 11 포함, 실패 0).

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/models/distributional_total.py tests/test_distributional_total.py
git commit -m "feat: DistributionalTotalModel drop-in alias (predict_expected/predict_production)"
```

---

## Post-Plan (이 플랜 밖, 별도 수행)

- 3축 리뷰(재사용성·품질·효율) — `/review-triple`.
- 메모리 갱신 (project_distributional_forecasting_stack: src 승격 완료).
- 후속 스펙: CLI `--model distributional` 배선 → event_prior 블렌드 → walk-forward conformal → decision → predict-next-week → 4매장/NegBin.

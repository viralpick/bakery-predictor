# 발주 quantile conformal calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** item 발주가 목표 서비스레벨(기본 0.74)을 실현 초과율로 달성하도록, base median 예측 위에 scale-정규화 one-sided split-conformal 잔차 마진을 씌우는 path-agnostic calibration 레이어를 만든다.

**Architecture:** ③에서 통일한 adjusted_demand 잣대 위에서, v2 LGBM을 median(q0.5)으로 학습(base). expanding backtest fold들을 시간순으로 앞쪽(cal)/뒤쪽(test)으로 half-split → cal folds 잔차 `E=(y−ŷ)/scale`의 s-분위 `Q_s`를 pooled로 산출 → test folds 발주 = `ŷ + Q_s×scale`. base를 target과 분리(median)해 프론티어 스윙이 재학습 없이 저렴.

**Tech Stack:** Python, pandas, numpy, LightGBM(quantile), typer CLI, pytest, uv.

## Global Constraints

- **Time leakage 금지**: scale는 첫 val 창 이전 이력에서만, cal folds는 test folds보다 시간상 앞. calibrator는 test 데이터를 절대 안 본다. `test_split_leakage.py`/`test_features_leakage.py` 통과.
- **테스트 단언 강도**: 기대값 아는 단언은 정확값(`==`/`pytest.approx`). 통계적 coverage 단언만 허용오차(`abs(...) < tol`) + 이유 주석.
- **매직값 금지**: 서비스레벨 기본 0.74, cal_fold_frac 기본 0.5 — 모듈 상수로.
- **하위호환**: `--calibrate` 기본 off. synthetic 경로·기존 CLI 옵션·order_level 기본 item 불변.
- **base=median**: v2 LGBM `objective="quantile", alpha=0.5`. target 서비스레벨은 conformal 마진이 담당(base 재학습 없이 s 스윙).
- **잣대=adjusted_demand**(③). coverage(초과율)만이 순 신호 — KPI Δ vs baseline은 스케일 오염 상존.
- **광교 단독**(`_load_real_daily` n_stores≠1 raise). category·다매장은 비목표.
- **커밋 꼬리말**:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD
  ```

---

## File Structure

- `src/bakery/models/conformal_order.py` (신규) — `ConformalOrderCalibrator` 순수 클래스. 상태=Q_s 하나.
- `src/bakery/features/scale.py` (신규) — `compute_item_scale` leakage-safe per-item 규모.
- `src/bakery/cli.py` (수정) — `_conformal_order_predictions` 조립 + `prospective-eval` 옵션 배선.
- `tests/test_conformal_order.py`, `tests/test_item_scale.py`, `tests/test_conformal_order_predictions.py` (신규).
- `docs/order_conformal_calibration_result.md` (신규, Task 5).

---

### Task 1: ConformalOrderCalibrator (순수 모듈)

**Files:**
- Create: `src/bakery/models/conformal_order.py`
- Test: `tests/test_conformal_order.py`

**Interfaces:**
- Produces: `ConformalOrderCalibrator` — `fit(scores: np.ndarray, service_level: float) -> ConformalOrderCalibrator` (normalized score의 s-분위 `self.q_s` 저장), `apply(base_pred: np.ndarray, scales: np.ndarray) -> np.ndarray` (`clip(base + q_s*scale, 0, None)`). 상수 `DEFAULT_SERVICE_LEVEL = 0.74`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conformal_order.py
import numpy as np
import pytest

from bakery.models.conformal_order import ConformalOrderCalibrator, DEFAULT_SERVICE_LEVEL


def test_default_service_level_is_cost_optimal():
    assert DEFAULT_SERVICE_LEVEL == 0.74


def test_q_s_is_higher_quantile_of_scores():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    calib = ConformalOrderCalibrator().fit(scores, service_level=0.8)
    # method="higher" 0.8-quantile of 5 sorted values → index ceil(0.8*4)=4 → 4.0... use np ref
    assert calib.q_s == pytest.approx(np.quantile(scores, 0.8, method="higher"))


def test_apply_adds_margin_scaled_and_clips_negative():
    calib = ConformalOrderCalibrator().fit(np.array([-5.0, -5.0]), service_level=0.5)  # q_s=-5
    base = np.array([10.0, 3.0])
    scales = np.array([2.0, 1.0])
    # order = base + q_s*scale = [10-10, 3-5] = [0, -2] → clip → [0, 0]
    got = calib.apply(base, scales)
    assert got.tolist() == [0.0, 0.0]


def test_apply_positive_margin_exact():
    calib = ConformalOrderCalibrator().fit(np.array([1.0, 1.0]), service_level=0.5)  # q_s=1.0
    got = calib.apply(np.array([10.0]), np.array([3.0]))
    assert got.tolist() == [13.0]  # 10 + 1.0*3


def test_coverage_contract_on_exchangeable_synthetic():
    # 이유: conformal coverage는 통계적 보장이라 정확값 불가 → 허용오차 단언.
    rng = np.random.default_rng(42)
    n = 4000
    scale = np.full(n, 5.0)
    resid = rng.normal(0, 5.0, size=n)      # y - base, exchangeable
    scores = resid / scale
    half = n // 2
    calib = ConformalOrderCalibrator().fit(scores[:half], service_level=0.8)
    base = np.zeros(half)
    order = calib.apply(base, scale[half:])
    y = resid[half:]                         # base=0 → y == resid
    exceed = float((y > order).mean())
    assert abs(exceed - 0.2) < 0.03          # nominal miss = 1 - 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conformal_order.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.models.conformal_order'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/models/conformal_order.py
"""One-sided, scale-normalized split-conformal 발주 보정.

base median 예측 위에 잔차 마진을 씌워 목표 서비스레벨 s의 coverage
(P(demand > order) ≈ 1−s)를 달성한다. 잔차는 item scale로 정규화해 pooled로
분위를 구하므로 희소 품목도 강건. path-agnostic — 배열만 받는다.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SERVICE_LEVEL = 0.74  # cost-optimal Cu/(Cu+Co) = 0.85/(0.85+0.30)


class ConformalOrderCalibrator:
    q_s: float

    def fit(self, scores: np.ndarray, service_level: float) -> "ConformalOrderCalibrator":
        """normalized conformity score E=(y−ŷ)/scale 의 s-분위를 저장.

        method="higher"로 유한표본서 약간 보수적(coverage 하한 보호).
        """
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]
        if scores.size == 0:
            raise ValueError("scores is empty after NaN drop")
        self.q_s = float(np.quantile(scores, service_level, method="higher"))
        return self

    def apply(self, base_pred: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """order = clip(base + q_s × scale, 0, None)."""
        base_pred = np.asarray(base_pred, dtype=float)
        scales = np.asarray(scales, dtype=float)
        return np.clip(base_pred + self.q_s * scales, 0.0, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conformal_order.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/models/conformal_order.py tests/test_conformal_order.py
git commit -m "feat: ConformalOrderCalibrator — one-sided scale-정규화 split-conformal 발주 보정

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 2: compute_item_scale (leakage-safe per-item 규모)

**Files:**
- Create: `src/bakery/features/scale.py`
- Test: `tests/test_item_scale.py`

**Interfaces:**
- Consumes: daily DataFrame (`item_id`, `date`, `y_col`).
- Produces: `compute_item_scale(daily: pd.DataFrame, before_date, y_col: str = "adjusted_demand", floor: float = 1.0) -> dict[str, float]` — `date < before_date` 행만으로 item별 `y_col` 평균, `max(mean, floor)`. 반환 dict의 키는 str item_id.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_item_scale.py
import pandas as pd
import pytest

from bakery.features.scale import compute_item_scale


def _daily():
    return pd.DataFrame({
        "item_id": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01",
                                 "2021-01-01", "2021-01-02"]),
        "adjusted_demand": [10.0, 20.0, 999.0, 0.0, 0.0],
    })


def test_mean_over_pre_cutoff_only():
    # cutoff 2021-01-15 → A: mean(10,20)=15 (999 제외); B: mean(0,0)=0 → floor 1.0
    scale = compute_item_scale(_daily(), before_date=pd.Timestamp("2021-01-15"))
    assert scale["A"] == pytest.approx(15.0)
    assert scale["B"] == pytest.approx(1.0)  # floor


def test_leakage_rows_on_or_after_cutoff_excluded():
    # cutoff 2021-01-02 (strict <) → A: only 2021-01-01=10.0
    scale = compute_item_scale(_daily(), before_date=pd.Timestamp("2021-01-02"))
    assert scale["A"] == pytest.approx(10.0)


def test_floor_applies_to_custom_value():
    scale = compute_item_scale(_daily(), before_date=pd.Timestamp("2021-01-15"), floor=5.0)
    assert scale["B"] == pytest.approx(5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_item_scale.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bakery.features.scale'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bakery/features/scale.py
"""Leakage-safe per-item 규모 — conformal 마진 정규화용.

scale_i = cutoff 이전 이력의 item별 target 평균(floor 적용). 예측 시점 이후
데이터를 안 보므로 conformal calibration의 normalized score에 안전하게 쓴다.
"""
from __future__ import annotations

import pandas as pd


def compute_item_scale(
    daily: pd.DataFrame,
    before_date,
    y_col: str = "adjusted_demand",
    floor: float = 1.0,
) -> dict[str, float]:
    before = pd.to_datetime(before_date)
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    hist = d[d["date"] < before]
    means = hist.groupby(hist["item_id"].astype(str))[y_col].mean()
    return {item: max(float(m), floor) for item, m in means.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_item_scale.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/features/scale.py tests/test_item_scale.py
git commit -m "feat: compute_item_scale — leakage-safe per-item 규모 (conformal 정규화용)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 3: _conformal_order_predictions 조립 (median base + cross-fold half-split)

**Files:**
- Modify: `src/bakery/cli.py` (신규 함수 추가; `_our_order_predictions` 근처)
- Test: `tests/test_conformal_order_predictions.py` (create)

**Interfaces:**
- Consumes: `ConformalOrderCalibrator`(Task 1), `compute_item_scale`(Task 2), `build_item_adjusted_demand`/`DEFAULT_ALPHA`(cli 기존 import), `GlobalLGBM`/`LGBMParams`/`generate_time_splits`/`run_backtest`(cli 기존 import), `_load_dataset`/`_enrich_if_needed`(cli 기존).
- Produces:
  - `_median_base_fold_predictions(daily: pd.DataFrame, *, val_weeks: int, n_folds: int) -> tuple[pd.DataFrame, list]` — v2 LGBM q0.5로 expanding backtest, 반환 pred_df 컬럼 `[item_id, date, fold, adjusted_demand, yhat]` + windows.
  - `_conformal_order_predictions(store_id: str, *, service_level: float = DEFAULT_SERVICE_LEVEL, val_weeks: int = 8, n_folds: int = 8, cal_fold_frac: float = 0.5, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame` — `[item_id, date, fold, our_order]` (test folds만).

- [ ] **Step 1: Write the failing test (구조·leakage 계약)**

`_median_base_fold_predictions`는 실 LGBM·실데이터라 단위테스트 대신 `_conformal_order_predictions`의 순수 조립 로직을 검증하기 위해, 조립 핵심을 **순수 헬퍼로 분리**해 테스트한다. 아래 테스트는 그 순수 헬퍼 `_apply_conformal_to_folds`를 고정한다(Step 3에서 함께 정의).

```python
# tests/test_conformal_order_predictions.py
import numpy as np
import pandas as pd
import pytest

from bakery.cli import _apply_conformal_to_folds


def _preds():
    # 2 folds × 2 items. fold 0 = cal, fold 1 = test (frac 0.5, 2 folds → n_cal=1).
    return pd.DataFrame({
        "item_id": ["A", "B", "A", "B"],
        "date": pd.to_datetime(["2021-03-01", "2021-03-01", "2021-04-01", "2021-04-01"]),
        "fold": [0, 0, 1, 1],
        "adjusted_demand": [12.0, 6.0, 10.0, 5.0],
        "yhat": [10.0, 5.0, 10.0, 5.0],
    })


def test_only_test_folds_returned_and_schema():
    scale = {"A": 4.0, "B": 2.0}
    out = _apply_conformal_to_folds(_preds(), scale, service_level=0.5, cal_fold_frac=0.5)
    assert list(out.columns) == ["item_id", "date", "fold", "our_order"]
    assert set(out["fold"].unique()) == {1}          # cal fold 0 제외
    assert len(out) == 2


def test_our_order_equals_base_plus_qs_times_scale():
    # cal fold0 scores: A (12-10)/4=0.5, B (6-5)/2=0.5 → q_s(s=0.5, higher)=0.5
    # test fold1: A 10+0.5*4=12, B 5+0.5*2=6
    scale = {"A": 4.0, "B": 2.0}
    out = _apply_conformal_to_folds(_preds(), scale, service_level=0.5, cal_fold_frac=0.5)
    got = dict(zip(out["item_id"], out["our_order"]))
    assert got["A"] == pytest.approx(12.0)
    assert got["B"] == pytest.approx(6.0)


def test_missing_item_scale_defaults_to_floor_one():
    preds = _preds().assign(item_id=["A", "C", "A", "C"])  # C not in scale dict
    scale = {"A": 4.0}
    out = _apply_conformal_to_folds(preds, scale, service_level=0.5, cal_fold_frac=0.5)
    # cal scores: A 0.5, C (6-5)/1=1.0 → q_s higher@0.5 of [0.5,1.0]=1.0
    # test C: 5 + 1.0*1.0 = 6.0
    got = dict(zip(out["item_id"], out["our_order"]))
    assert got["C"] == pytest.approx(6.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conformal_order_predictions.py -v`
Expected: FAIL — `ImportError: cannot import name '_apply_conformal_to_folds'`

- [ ] **Step 3: Write minimal implementation**

cli.py 상단 import에 추가 (기존 import 블록):
```python
from .models.conformal_order import ConformalOrderCalibrator, DEFAULT_SERVICE_LEVEL
from .features.scale import compute_item_scale
```

`_our_order_predictions` 정의 근처에 3개 함수 추가:
```python
def _apply_conformal_to_folds(
    pred_df: pd.DataFrame, scale: dict[str, float], *,
    service_level: float, cal_fold_frac: float,
) -> pd.DataFrame:
    """fold별 base 예측을 앞쪽(cal)/뒤쪽(test)로 half-split → conformal 보정.

    순수 함수(실 LGBM 무관): pred_df[item_id,date,fold,adjusted_demand,yhat] +
    item scale dict → test folds의 [item_id,date,fold,our_order].
    """
    folds = sorted(pred_df["fold"].unique())
    n_cal = max(1, int(len(folds) * cal_fold_frac))
    cal_folds, test_folds = set(folds[:n_cal]), set(folds[n_cal:])
    cal = pred_df[pred_df["fold"].isin(cal_folds)]
    test = pred_df[pred_df["fold"].isin(test_folds)].copy()

    def _scale_of(items: pd.Series) -> np.ndarray:
        return items.astype(str).map(scale).fillna(1.0).to_numpy()

    cal_scale = _scale_of(cal["item_id"])
    scores = ((cal["adjusted_demand"].to_numpy() - cal["yhat"].to_numpy()) / cal_scale)
    calib = ConformalOrderCalibrator().fit(scores, service_level)
    test["our_order"] = calib.apply(test["yhat"].to_numpy(), _scale_of(test["item_id"]))
    return test[["item_id", "date", "fold", "our_order"]].reset_index(drop=True)


def _median_base_fold_predictions(
    daily: pd.DataFrame, *, val_weeks: int, n_folds: int,
) -> tuple[pd.DataFrame, list]:
    """v2 LGBM q0.5(median) base로 expanding backtest. pred_df에 actual+yhat+fold 보존."""
    windows = generate_time_splits(
        daily["date"], n_splits=n_folds,
        val_horizon_days=val_weeks * 7, step_days=val_weeks * 7,
    )
    forecaster = GlobalLGBM(
        feature_set="v2", y_col="adjusted_demand",
        params=LGBMParams(objective="quantile", alpha=0.5),
    )
    _, pred_df = run_backtest(daily, [forecaster], windows, y_col="adjusted_demand")
    pred_df["item_id"] = pred_df["item_id"].astype(str)
    return pred_df[["item_id", "date", "fold", "adjusted_demand", "yhat"]], windows


def _conformal_order_predictions(
    store_id: str, *, service_level: float = DEFAULT_SERVICE_LEVEL,
    val_weeks: int = 8, n_folds: int = 8, cal_fold_frac: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """base median 발주 + cross-fold half-split conformal 보정 → test folds our_order."""
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    daily = build_item_adjusted_demand(daily, alpha=alpha)
    pred_df, windows = _median_base_fold_predictions(daily, val_weeks=val_weeks, n_folds=n_folds)
    first_val_start = min(w.val_start for w in windows)
    scale = compute_item_scale(daily, before_date=first_val_start, y_col="adjusted_demand")
    out = _apply_conformal_to_folds(
        pred_df, scale, service_level=service_level, cal_fold_frac=cal_fold_frac
    )
    console.print(
        f"[cyan]conformal our_order[/] {n_folds} fold(s)(cal {int(n_folds*cal_fold_frac)}/"
        f"test {n_folds-int(n_folds*cal_fold_frac)}), s={service_level}, "
        f"{out['date'].nunique()} dates × {out['item_id'].nunique()} items"
    )
    return out
```

주의: `console`, `generate_time_splits`, `GlobalLGBM`, `LGBMParams`, `run_backtest`, `build_item_adjusted_demand`, `_load_dataset`, `_enrich_if_needed`는 cli.py에 이미 import/정의됨(기존 `_our_order_predictions`/`_quantile_backtest_predictions`가 사용). `np`/`pd`도 상단에 있음.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conformal_order_predictions.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Leakage 회귀 확인**

Run: `uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py --color=no`
Expected: 통과 (기존 그대로 — 신규 함수는 leakage-safe 조립).

- [ ] **Step 6: Commit**

```bash
git add src/bakery/cli.py tests/test_conformal_order_predictions.py
git commit -m "feat: _conformal_order_predictions — median base + cross-fold half-split conformal 보정

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 4: prospective-eval CLI 배선 (--calibrate)

**Files:**
- Modify: `src/bakery/cli.py` (`_real_prospective_inputs`, `_load_prospective_inputs`, `cmd_prospective_eval`)

**Interfaces:**
- Consumes: `_conformal_order_predictions`(Task 3).
- Produces: `prospective-eval`가 `--calibrate` 시 item 발주를 conformal 보정값으로 채우고 test folds에서만 평가.

- [ ] **Step 1: `_real_prospective_inputs`에 calibrate 라우팅**

기존 `_real_prospective_inputs`의 order 예측 분기(`if order_level == "category": ... else: _our_order_predictions(...)`)를 다음으로 확장. 시그니처에 `calibrate: bool = False, service_level: float = DEFAULT_SERVICE_LEVEL, cal_fold_frac: float = 0.5` 추가:
```python
    if order_level == "category":
        predictions = _category_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, alpha=alpha,
        )
    elif calibrate:
        predictions = _conformal_order_predictions(
            store_id, service_level=service_level, val_weeks=val_weeks,
            n_folds=n_folds, cal_fold_frac=cal_fold_frac, alpha=alpha,
        )
    else:
        predictions = _our_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, alpha=alpha,
        )
```
(calibrate + order_level=="category"는 category 우선 — 이번 스코프 item만이므로 category 분기가 calibrate를 무시하는 현 순서로 충분. 필요 시 후속.)

- [ ] **Step 2: `_load_prospective_inputs` thread**

`_load_prospective_inputs`에 `calibrate: bool = False, service_level: float = DEFAULT_SERVICE_LEVEL, cal_fold_frac: float = 0.5` 추가하고 `_real_prospective_inputs(..., calibrate=calibrate, service_level=service_level, cal_fold_frac=cal_fold_frac)` 전달.

- [ ] **Step 3: `cmd_prospective_eval`에 옵션 추가**

typer.Option 3개 추가:
```python
    calibrate: bool = typer.Option(
        False, help="conformal 보정 발주 사용(real+item 경로). base median + cross-fold half-split"
    ),
    target_service_level: float = typer.Option(
        DEFAULT_SERVICE_LEVEL, help="conformal 목표 서비스레벨 s (초과율 목표=1−s, 기본 0.74)"
    ),
    cal_fold_frac: float = typer.Option(
        0.5, help="앞쪽 folds 중 calibration 비율(나머지=test)"
    ),
```
그리고 `_load_prospective_inputs(...)` 호출에 `calibrate=calibrate, service_level=target_service_level, cal_fold_frac=cal_fold_frac` 전달.

- [ ] **Step 4: Smoke test (item calibrate 경로)**

Run: `uv run bakery prospective-eval --source real --order-level item --calibrate --n-folds 8 --our-order-val-weeks 8 --target-service-level 0.74 --out-csv reports/_smoke_calib.csv`
Expected: 크래시 없이 완주. 콘솔에 `conformal our_order 8 fold(s)(cal 4/test 4)...` + `calibration 초과율 P(demand>order)=...` 출력. 초과율이 raw(0.679)보다 목표(≈0.26) 쪽으로 이동했는지 확인(정확값은 Task 5 헤드라인).

- [ ] **Step 5: 하위호환 회귀**

Run: `uv run bakery prospective-eval --source synthetic` (calibrate 기본 off → 불변)
Run: `uv run pytest tests/ --color=no`
Expected: synthetic 정상, full-suite 통과.

- [ ] **Step 6: Commit**

```bash
git add src/bakery/cli.py
git commit -m "feat: prospective-eval --calibrate (conformal 발주, target-service-level, cal-fold-frac)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

### Task 5: 진단 + 프론티어 스윙 + 결과 문서

**Files:**
- Create: `docs/order_conformal_calibration_result.md`
- (읽기) `reports/` 산출물

**Interfaces:**
- Consumes: Task 4 완료된 harness.

- [ ] **Step 1: 진단 — production_quantile 스윙 (무보정 base 미스칼 특성화)**

무보정(raw q) 초과율이 q를 올려도 nominal에 못 가는지 기록:
```bash
for q in 0.85 0.90 0.95 0.99; do
  uv run bakery prospective-eval --source real --order-level item \
    --n-folds 8 --production-quantile $q --alpha 0.5 \
    --out-csv reports/diag_item_q$q.csv 2>&1 | tee reports/log_diag_item_q$q.txt
done
```
각 로그의 `calibration 초과율` 기록. (base quantile 상향의 한계 = conformal 정당화.)

- [ ] **Step 2: 헤드라인 — conformal @ s=0.74**

```bash
uv run bakery prospective-eval --source real --order-level item --calibrate \
  --n-folds 8 --our-order-val-weeks 8 --target-service-level 0.74 \
  --out-csv reports/calib_item_s0.74.csv 2>&1 | tee reports/log_calib_item_s0.74.txt
```
초과율(목표 0.26)·WPE·waste/lost/stockout Δ(test folds) 기록.

- [ ] **Step 3: 프론티어 스윙**

```bash
for s in 0.70 0.74 0.80 0.85 0.90 0.95; do
  uv run bakery prospective-eval --source real --order-level item --calibrate \
    --n-folds 8 --our-order-val-weeks 8 --target-service-level $s \
    --out-csv reports/calib_item_s$s.csv 2>&1 | tee reports/log_calib_item_s$s.txt
done
```
s별 실현 초과율(≈1−s 수렴 확인) + waste/lost/stockout Δ 수집.

- [ ] **Step 4: 결과 문서 작성**

`docs/order_conformal_calibration_result.md`:
- 배경(③ genuine under-calibration → conformal, spec 링크).
- **표 1 진단**: raw q {0.85,0.9,0.95,0.99} 초과율 — q 상향만으로 nominal 미도달 확인.
- **표 2 헤드라인**: raw q0.85(0.679) vs conformal s=0.74 — 실현 초과율이 목표 0.26에 수렴했는지.
- **표 3 프론티어**: s 0.70~0.95 → 실현 초과율 + waste↔stockout Δ 트레이드오프.
- **판정(정직)**: 초과율이 목표에 수렴하면 conformal이 calibration gap을 닫음(성공). 여전히 미달이면 그 크기·원인(scale 추정·분포 shift) 보고.
- 캐비엣: 광교 단독 / test folds 절반 표본 / exchangeability(명절 shift) / KPI Δ 스케일 오염 상존 / category·다매장 후속.

- [ ] **Step 5: Commit**

```bash
git add docs/order_conformal_calibration_result.md
git commit -m "docs: 발주 conformal calibration 결과 — 진단+헤드라인+프론티어

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VvS4NZ5XaNXt19z13bUSkD"
```

---

## Self-Review (작성자 체크)

- **Spec coverage**: 방법 1(base median)→Task 3 / 2(scale)→Task 2 / 3(cross-fold half-split)→Task 3 `_apply_conformal_to_folds` / 4~6(score·Q_s·apply)→Task 1 / 7(test-only 평가)→Task 3 (test folds만 반환)·Task 4. 컴포넌트 conformal_order.py→T1, scale→T2, cli 배선→T3·T4, 결과문서→T5. 진단·프론티어→T5. ✅
- **Placeholder scan**: 모든 코드 스텝 실제 코드. 진단/프론티어는 실행 명령 명시. TBD 없음. ✅
- **Type consistency**: `ConformalOrderCalibrator.fit(scores, service_level)`/`apply(base, scales)`(T1↔T3 동일), `compute_item_scale(daily, before_date, y_col, floor)->dict`(T2↔T3), `_apply_conformal_to_folds(pred_df, scale, service_level, cal_fold_frac)`(T3 정의↔테스트), `_conformal_order_predictions(...)`(T3↔T4), `DEFAULT_SERVICE_LEVEL`(T1↔T3↔T4). ✅
- **주의**: `np.quantile(..., method="higher")`는 numpy≥1.22 필요(이 repo 충족). 테스트 기대값도 method="higher" 기준으로 계산됨(T1 test_q_s, T3 scores 예시).

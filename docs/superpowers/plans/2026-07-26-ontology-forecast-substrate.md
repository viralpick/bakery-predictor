# 온톨로지 Forecast Substrate (5a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** forward 2층 예측(총량 → event_prior → 품목 배분)을 `cli.py` 밖 공유 seam으로 추출하고 중간값을 구조화 반환하여, 온톨로지가 단일 예측 엔진을 소비하도록 배선한다.

**Architecture:** cli에 갇힌 `_category_future_order_predictions` 합성과 그 private 의존을 신규 `src/bakery/forecast/` 패키지로 이동(단일 출처, cli는 import back). 신규 `forecast_forward()`가 `ForwardForecast`(base/prior 총량 + 비중 factor + 품목 수량)를 반환. `functions.py`의 demand_point는 컬럼 평균 → 이 forward 예측으로 재배선. scenario 재배선·설명 함수(5b)는 범위 밖.

**Tech Stack:** Python, pandas, LightGBM, pytest, uv. 대상 데이터 = real 소스(광교, `store_gw01`).

## Global Constraints

- **Time leakage 금지**: fit은 `date < 첫 미래일`(관측 history)만. 미래 행 target=NaN append로 lag가 seam 너머 계산. `test_split_leakage.py`/`test_features_leakage.py`/`test_event_prior_leakage.py` 통과 유지.
- **byte-equal 게이트**: seam 추출·cli 래퍼화 전후 `_category_future_order_predictions("store_gw01")` 출력과 `next_week_predictions.csv`가 동일 인자에서 정확 일치.
- **coherence invariant**: date당 `Σ item_quantities.our_order == category_totals.prior_prod`, `Σ demand_point == prior_median`.
- **faithfulness**: `ForwardForecast.base_*`/`prior_*`/`proportions` factor값 == 엔진 실제 중간 산출(이상화된 "룰" 라벨 금지).
- **의도적 변경(라벨)**: 온톨로지 demand_point = 과거 평균 → forward 예측. docstring·리뷰에 명시.
- **커밋**: 각 태스크 끝에 커밋. 브랜치 `spec/ontology-forecast-substrate-5a`에서 작업.
- **pytest 실행**: `uv run pytest`. 카운트 필요 시 `--color=no`(이 repo는 addopts에 `-q` 있음, `-q` 추가 금지).
- **코드 이동 원칙**: "verbatim 이동"은 함수 본문을 바꾸지 않는다. import 오류는 함수가 참조하는 심볼을 신규 모듈에 import해 해소한다. 로직 변경 발견 시 중단·원인 규명.

---

### Task 1: `forecast/` 패키지 + 데이터 로더 이동 (`forecast/loaders.py`)

`_category_future_order_predictions`가 의존하는 데이터 로더 2종을 cli 밖으로 빼 순환의존을 끊는다. 이 둘은 cli의 item·decision 경로와도 공유되므로 cli는 alias로 import back(호출부·monkeypatch 테스트 무변경).

**Files:**
- Create: `src/bakery/forecast/__init__.py`
- Create: `src/bakery/forecast/loaders.py`
- Modify: `src/bakery/cli.py` (defs 삭제 + import back)

**Interfaces:**
- Produces: `forecast.loaders.load_real_daily(store_id: str) -> pd.DataFrame`, `forecast.loaders.load_forecast_weather(horizon: pd.DatetimeIndex) -> pd.DataFrame | None`

- [ ] **Step 1: 패키지 생성**

`src/bakery/forecast/__init__.py` (빈 파일).

- [ ] **Step 2: 로더 이동**

`cli.py`의 `_load_forecast_weather`(라인 499–523)와 `_load_real_daily`(라인 2152–2163) 본문을 **verbatim**으로 `forecast/loaders.py`에 옮기고 함수명만 `load_forecast_weather`, `load_real_daily`로 변경. 두 함수가 참조하는 심볼(pandas 및 파일 경로/로더 유틸 — cli 상단 import에서 해당 심볼을 찾아 `loaders.py` 상단에 동일하게 import)을 추가.

- [ ] **Step 3: cli에서 import back**

`cli.py`에서 두 def를 삭제하고 상단 import 블록(다른 `from .` import 근처)에 추가:

```python
from .forecast.loaders import (
    load_forecast_weather as _load_forecast_weather,
    load_real_daily as _load_real_daily,
)
```

alias(`_load_...`)를 유지하는 이유: 호출부(cli 287·350·2378·2463·2482·2675·2698)와 `test_artisee_baseline.py:255`의 `monkeypatch.setattr("bakery.cli._load_real_daily", ...)`가 그대로 동작한다.

- [ ] **Step 4: 회귀 테스트 (behavior 불변)**

Run: `uv run pytest tests/test_predict_next_week.py tests/test_artisee_baseline.py --color=no`
Expected: PASS (순수 이동, 동작 불변)

- [ ] **Step 5: 전체 스위트**

Run: `uv run pytest --color=no`
Expected: PASS (ImportError 발생 시 `loaders.py` import 누락 → 추가)

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/forecast/__init__.py src/bakery/forecast/loaders.py src/bakery/cli.py
git commit -m "refactor(forecast): 데이터 로더를 forecast/loaders.py로 이동 (5a Task1)"
```

---

### Task 2: forward 합성 헬퍼 이동 (`forecast/forward.py`)

`_category_future_order_predictions`의 전용/공유 헬퍼를 `forecast/forward.py`로 이동. `_category_base_predict`·`_blend_event_prior`는 cli의 category backtest 경로(`_category_total_fold_predictions`, 라인 2241)와 공유되므로 cli는 import back.

**Files:**
- Create: `src/bakery/forecast/forward.py`
- Modify: `src/bakery/cli.py` (defs 삭제 + import back)

**Interfaces:**
- Consumes: `forecast.loaders.load_forecast_weather` (Task 1)
- Produces (모두 시그니처 불변, verbatim 이동):
  - `_category_base_predict(train, test, *, target_col, total_model, production_quantile) -> tuple`
  - `_blend_event_prior(train, dates, base_median, base_prod, *, target_col) -> tuple`
  - `_forecast_to_category_weather(forecast_weather, store_id) -> pd.DataFrame | None`
  - `_extend_category_features(hist, *, horizon_days, alpha, target_col) -> tuple[pd.DataFrame, pd.DatetimeIndex]`

- [ ] **Step 1: 헬퍼 이동**

`cli.py`에서 아래 4개 def를 **verbatim**으로 `forecast/forward.py`에 이동:
- `_category_base_predict` (라인 2206–2227)
- `_blend_event_prior` (라인 2228–2240)
- `_forecast_to_category_weather` (라인 2407–2425)
- `_extend_category_features` (라인 2426–2440)

`forward.py` 상단에 이 함수들이 참조하는 심볼을 import (cli 상단 import에서 확인):
```python
from __future__ import annotations
import numpy as np
import pandas as pd
from .loaders import load_forecast_weather
from ..features.category_aggregate import (
    DEFAULT_ALPHA, EVENTS, LUNAR_EVENTS, build_category_daily, build_features, fill_forecast_weather,
)
from ..models.category_total import fit_category_total
from ..models.distributional_total import fit_distributional_total
from ..models.event_prior import EventLevelPrior
from ..models.item_proportion import distribute_total
```
(위 목록 중 Task 2 이동 함수가 실제로 참조하지 않는 심볼은 제외 가능. Task 3에서 나머지가 쓰인다.)

- [ ] **Step 2: cli에서 import back**

`cli.py`에서 4개 def 삭제, 상단에 추가:
```python
from .forecast.forward import (
    _blend_event_prior,
    _category_base_predict,
    _extend_category_features,
    _forecast_to_category_weather,
)
```
(alias 불필요 — 동일 이름 유지. cli 내부 소비처 2272·2277이 그대로 동작.)

- [ ] **Step 3: 회귀 — category backtest + forward 경로**

Run: `uv run pytest tests/test_predict_next_week.py tests/test_category_future_forecast.py tests/test_category_total.py tests/test_event_prior.py --color=no`
Expected: PASS

- [ ] **Step 4: 전체 스위트**

Run: `uv run pytest --color=no`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/forecast/forward.py src/bakery/cli.py
git commit -m "refactor(forecast): forward 합성 헬퍼를 forecast/forward.py로 이동 (5a Task2)"
```

---

### Task 3: `ForwardForecast` + `forecast_forward` (중간값 노출 seam)

현 `_category_future_order_predictions` 합성을 재현하되 **중간값(base/prior 총량 + 비중 factor)을 구조화 반환**하는 신규 seam. 프레임 주입(`daily`)으로 I/O 없이 테스트 가능하게 설계.

**Files:**
- Modify: `src/bakery/forecast/forward.py` (신규 코드 추가)
- Create: `tests/test_forecast_forward.py`

**Interfaces:**
- Consumes: `_extend_category_features`, `_forecast_to_category_weather`, `_category_base_predict`, `_blend_event_prior` (Task 2), `load_real_daily`/`load_forecast_weather` (Task 1), `build_category_daily`, `fill_forecast_weather`, `distribute_total`
- Produces:
  - `class ForwardForecast` (frozen dataclass): `.category_totals`, `.proportions`, `.item_quantities` (모두 `pd.DataFrame`)
  - `forecast_forward(store_id: str, *, daily: pd.DataFrame | None = None, horizon_days: int = 7, total_model: str = "lightgbm", event_prior: bool = True, production_quantile: float = 0.85, alpha: float = DEFAULT_ALPHA, use_forecast: bool = True) -> ForwardForecast`
  - `category_totals` 컬럼: `[date, base_median, base_prod, prior_median, prior_prod]`
  - `item_quantities` 컬럼: `[store_id, item_id, category_id, date, demand_point, our_order]` (현 `_category_future_order_predictions` 반환과 동일)

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_forecast_forward.py`:

```python
import numpy as np
import pandas as pd
import pytest

from bakery.cli import _category_future_order_predictions
from bakery.forecast.forward import ForwardForecast, forecast_forward

STORE = "store_gw01"
KW = dict(horizon_days=7, total_model="lightgbm", event_prior=True,
          production_quantile=0.85, use_forecast=False)


@pytest.fixture(scope="module")
def result() -> ForwardForecast:
    return forecast_forward(STORE, **KW)


def test_item_quantities_match_cli(result):
    """seam의 item_quantities == 현 cli private 함수 출력(golden)."""
    golden = _category_future_order_predictions(
        STORE, horizon_days=7, production_quantile=0.85,
        total_model="lightgbm", event_prior=True, use_forecast=False,
    ).reset_index(drop=True)
    got = result.item_quantities.reset_index(drop=True)
    assert list(got.columns) == list(golden.columns)
    pd.testing.assert_frame_equal(got, golden, check_dtype=False)


def test_coherence_sum_equals_total(result):
    """date당 Σ our_order == prior_prod, Σ demand_point == prior_median."""
    ct = result.category_totals.set_index("date")
    by_date = result.item_quantities.groupby("date")[["our_order", "demand_point"]].sum()
    for d, row in by_date.iterrows():
        assert row["our_order"] == pytest.approx(ct.loc[d, "prior_prod"], rel=1e-9)
        assert row["demand_point"] == pytest.approx(ct.loc[d, "prior_median"], rel=1e-9)


def test_faithfulness_base_vs_prior(result):
    """event_prior on → prior_*가 base_*와 달라질 수 있고(특수일), 아니면 동일.
    base_*는 blend 이전 Stage1 예측 그대로."""
    ct = result.category_totals
    assert set(["base_median", "base_prod", "prior_median", "prior_prod"]).issubset(ct.columns)
    assert len(ct) == result.item_quantities["date"].nunique()
    # 비특수일은 prior==base (blend가 앵커 없는 날은 항등)
    assert (ct["prior_prod"] >= 0).all() and (ct["base_prod"] >= 0).all()


def test_proportions_factor_columns(result):
    """5b explain_item_order가 소비할 factor 컬럼 존재 + 정규화."""
    p = result.proportions
    for col in ["date", "item_id", "proportion", "base_sold",
                "adj_trend", "adj_stockout", "adj_closing", "adj_new"]:
        assert col in p.columns
    for _, g in p.groupby("date"):
        assert g["proportion"].sum() == pytest.approx(1.0, rel=1e-9)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_forecast_forward.py --color=no`
Expected: FAIL (`ImportError: cannot import name 'forecast_forward'`)

- [ ] **Step 3: `ForwardForecast` + `forecast_forward` 구현**

`forecast/forward.py` 상단 import에 추가: `from dataclasses import dataclass`, 그리고 `from .loaders import load_real_daily` (Task 2에서 `load_forecast_weather`만 import했다면 `load_real_daily` 병기). 이어서 아래 코드 추가:

```python
@dataclass(frozen=True)
class ForwardForecast:
    """forward 2층 예측 + 중간값. 5b 설명 함수가 재계산 없이 소비.

    category_totals: [date, base_median, base_prod, prior_median, prior_prod]
        base_* = Stage1 예측(event_prior blend 전), prior_* = blend 후.
    proportions: compute_proportions 출력(target_date→date), factor 컬럼 포함.
    item_quantities: [store_id, item_id, category_id, date, demand_point, our_order].
    """
    category_totals: pd.DataFrame
    proportions: pd.DataFrame
    item_quantities: pd.DataFrame


def forecast_forward(
    store_id: str, *, daily: pd.DataFrame | None = None, horizon_days: int = 7,
    total_model: str = "lightgbm", event_prior: bool = True,
    production_quantile: float = 0.85, alpha: float = DEFAULT_ALPHA,
    use_forecast: bool = True,
) -> ForwardForecast:
    """마지막 관측일 다음 horizon_days일의 카테고리 총량 예측 → 품목 배분.

    forward-only, leakage-safe(fit은 관측 history만). daily=None이면 real 소스
    로드(cli byte-equal 경로), 아니면 주입 프레임으로 build_category_daily(테스트).
    중간값(base/prior 총량, 비중 factor)을 ForwardForecast로 노출한다.
    """
    target_col = "adjusted_demand_unit"
    if daily is None:
        daily = load_real_daily(store_id)
        hist = build_category_daily(alpha=alpha).df
    else:
        hist = build_category_daily(daily_raw=daily, alpha=alpha).df
    feats, horizon = _extend_category_features(
        hist, horizon_days=horizon_days, alpha=alpha, target_col=target_col,
    )
    if use_forecast:
        fw = load_forecast_weather(horizon)
        cat_fw = _forecast_to_category_weather(fw, store_id) if fw is not None else None
        if cat_fw is not None:
            feats = fill_forecast_weather(feats, cat_fw)
    feats = feats.sort_values("date").reset_index(drop=True)
    is_future = feats["date"].isin(horizon)
    train = feats[~is_future].dropna(subset=[target_col])
    test = feats[is_future]
    base_median, base_prod = _category_base_predict(
        train, test, target_col=target_col,
        total_model=total_model, production_quantile=production_quantile,
    )
    pre_median = np.asarray(base_median, dtype=float)   # blend 이전 스냅샷
    pre_prod = np.asarray(base_prod, dtype=float)
    if event_prior:
        base_median, base_prod = _blend_event_prior(
            train, test["date"], base_median, base_prod, target_col=target_col,
        )
    dates = test["date"].to_numpy()
    prop_result = distribute_total(daily, pd.Series(np.asarray(base_prod, dtype=float), index=dates))
    order = prop_result.quantities.rename(columns={"qty": "our_order"})
    point = distribute_total(daily, pd.Series(np.asarray(base_median, dtype=float), index=dates)) \
        .quantities.rename(columns={"qty": "demand_point"})
    preds = order.merge(point, on=["item_id", "date"], how="left")
    preds["item_id"] = preds["item_id"].astype(str)
    cat_src = daily.drop_duplicates("item_id").assign(item_id=lambda d: d["item_id"].astype(str))
    cat_map = cat_src.set_index("item_id")["category_id"]
    preds["store_id"] = store_id
    preds["category_id"] = preds["item_id"].map(cat_map)
    item_quantities = preds[
        ["store_id", "item_id", "category_id", "date", "demand_point", "our_order"]
    ].reset_index(drop=True)
    category_totals = pd.DataFrame({
        "date": dates,
        "base_median": pre_median, "base_prod": pre_prod,
        "prior_median": np.asarray(base_median, dtype=float),
        "prior_prod": np.asarray(base_prod, dtype=float),
    })
    proportions = prop_result.proportions.rename(columns={"target_date": "date"})
    return ForwardForecast(
        category_totals=category_totals, proportions=proportions,
        item_quantities=item_quantities,
    )
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_forecast_forward.py --color=no`
Expected: PASS (4 tests). 실패 시: `test_item_quantities_match_cli`가 어긋나면 로직 미세 차이 → 중단하고 `_category_future_order_predictions`와 라인 대조.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/forecast/forward.py tests/test_forecast_forward.py
git commit -m "feat(forecast): forecast_forward seam + ForwardForecast 중간값 노출 (5a Task3)"
```

---

### Task 4: cli forward 경로를 seam 소비로 전환 (byte-equal)

`_category_future_order_predictions`의 중복 합성을 삭제하고 `forecast_forward`를 호출. cli 출력 스키마·CSV 불변.

**Files:**
- Modify: `src/bakery/cli.py` (`_category_future_order_predictions` 본문 교체)

**Interfaces:**
- Consumes: `forecast_forward` (Task 3)

- [ ] **Step 1: byte-equal 골든 캡처 테스트 작성**

`tests/test_forecast_forward.py`에 추가:

```python
def test_cli_wrapper_byte_equal():
    """cli 래퍼화 후에도 _category_future_order_predictions 출력 불변."""
    got = _category_future_order_predictions(
        STORE, horizon_days=7, production_quantile=0.85,
        total_model="lightgbm", event_prior=True, use_forecast=False,
    ).reset_index(drop=True)
    ref = forecast_forward(STORE, **KW).item_quantities.reset_index(drop=True)
    pd.testing.assert_frame_equal(got[ref.columns], ref, check_dtype=False)
```

- [ ] **Step 2: 실패 확인 (교체 전)**

Run: `uv run pytest tests/test_forecast_forward.py::test_cli_wrapper_byte_equal --color=no`
Expected: PASS (교체 전에도 성립 — Task 3가 재현했으므로). 이 테스트는 Step 4 교체가 회귀 아님을 보장하는 안전망.

- [ ] **Step 3: cli 래퍼화**

`cli.py::_category_future_order_predictions`(라인 2441–2501) 본문을 아래로 교체:

```python
def _category_future_order_predictions(
    store_id: str, *, horizon_days: int = 7, production_quantile: float = 0.85,
    total_model: str = "lightgbm", event_prior: bool = True,
    alpha: float = DEFAULT_ALPHA, use_forecast: bool = True,
) -> pd.DataFrame:
    """미래 horizon_days일 카테고리 총량 예측 → item 배분.

    forecast_forward seam을 소비(단일 출처). 반환 스키마·leakage 규칙은 seam이 보장.
    [store_id, item_id, category_id, date, demand_point, our_order] 반환.
    """
    ff = forecast_forward(
        store_id, horizon_days=horizon_days, total_model=total_model,
        event_prior=event_prior, production_quantile=production_quantile,
        alpha=alpha, use_forecast=use_forecast,
    )
    preds = ff.item_quantities
    console.print(
        f"[cyan]category future order[/] "
        f"{pd.Timestamp(preds['date'].min()).date()}~{pd.Timestamp(preds['date'].max()).date()}, "
        f"model={total_model}, event_prior={'on' if event_prior else 'off'}, "
        f"forecast={'on' if use_forecast else 'off'}, "
        f"{preds['date'].nunique()} dates × {preds['item_id'].nunique()} items"
    )
    return preds
```

cli 상단 import에 추가: `from .forecast.forward import forecast_forward` (기존 forward import 라인에 병합).

- [ ] **Step 4: byte-equal + predict-next-week 회귀**

Run: `uv run pytest tests/test_forecast_forward.py tests/test_predict_next_week.py --color=no`
Expected: PASS

- [ ] **Step 5: 전체 스위트**

Run: `uv run pytest --color=no`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/cli.py tests/test_forecast_forward.py
git commit -m "refactor(cli): forward 발주 예측을 forecast_forward seam 소비로 전환 (5a Task4)"
```

---

### Task 5: `functions.py` demand_point를 forward 예측으로 재배선

온톨로지 함수의 수요를 컬럼 평균 → forward 예측으로 교체. 이것이 architect의 "왜 K개 생산" 질문을 가능케 하는 핵심 배선.

**Files:**
- Modify: `src/bakery/ontology/functions.py`
- Modify: `tests/test_ontology_functions.py`, `tests/test_ontology_demand_proxy.py` (fixture forward 마이그레이션)

**Interfaces:**
- Consumes: `forecast_forward` (Task 3, `daily` 주입 + `use_forecast=False`로 결정론)
- Produces: `functions._forward_demand_points(daily, store_id, period, *, horizon_days) -> pd.DataFrame` ([item_id, demand_point])

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_ontology_functions.py`에 추가 (forward 예측 기반 demand 검증):

```python
def test_rank_stockout_risk_uses_forward_forecast(real_daily_fixture):
    """demand_point가 과거 평균이 아니라 forward 예측이어야 한다(의도적 변경).
    forecast_forward로 계산한 값과 일치."""
    from bakery.forecast.forward import forecast_forward
    daily, store_id, period = real_daily_fixture
    ff = forecast_forward(store_id, daily=daily, use_forecast=False,
                          horizon_days=7).item_quantities
    ranked = rank_stockout_risk(daily, store_id, period, k=3)
    # 예측 기반 demand_point가 rank 입력으로 흘러갔는지: 상위 item이 예측 프레임에 존재
    assert set(ranked["item_id"]).issubset(set(ff["item_id"].astype(str)))
    assert len(ranked) == 3
```

`real_daily_fixture`는 real 소스 단일매장 daily + forward horizon 안에 드는 period를 반환(기존 ontology 테스트가 쓰는 real fixture 패턴 재사용; period = 마지막 관측일 다음 3일).

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_ontology_functions.py::test_rank_stockout_risk_uses_forward_forecast --color=no`
Expected: FAIL (`rank_stockout_risk`가 아직 컬럼 평균 사용 → item 집합 불일치 또는 fixture 미정의)

- [ ] **Step 3: `_forward_demand_points` 구현 + 소비처 교체**

`functions.py`에 추가:

```python
def _forward_demand_points(
    daily: pd.DataFrame, store_id: str, period: tuple[str, str], *, horizon_days: int = 7,
) -> pd.DataFrame:
    """forward 예측 demand_point (과거 평균 _item_demand_points 대체).

    의도적 변경(5a): 온톨로지 수요 = adjusted_demand 컬럼 평균 → forecast_forward.
    period는 다가오는 horizon 내 대상으로 슬라이스. daily 주입·use_forecast=False로 결정론.
    """
    from ..forecast.forward import forecast_forward
    ff = forecast_forward(store_id, daily=daily, horizon_days=horizon_days,
                          use_forecast=False).item_quantities
    dates = pd.to_datetime(ff["date"])
    mask = (dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1]))
    sliced = ff.loc[mask]
    if sliced.empty:
        raise ValueError(f"forward 예측에 period {period} 대상 없음 (horizon 밖)")
    return (sliced.groupby("item_id", observed=True)["demand_point"].mean()
            .reset_index(name="demand_point"))
```

`rank_stockout_risk`(라인 78–79)와 `explain_order`(라인 130–131)에서:
```python
    items = _item_demand_points(_period_slice(daily, store_id, *period), demand_col)
```
를 다음으로 교체:
```python
    items = _forward_demand_points(daily, store_id, period)
```
`demand_col`/`_resolve_demand_proxy` 인자 경로는 synthetic fallback용으로 남기되, forward 경로에선 미사용(주석 라벨). `rank_stockout_earliness`(observed stockout_time)·`what_if`(demand_point 인자 수령)는 무변경.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_ontology_functions.py::test_rank_stockout_risk_uses_forward_forecast --color=no`
Expected: PASS

- [ ] **Step 5: 기존 ontology 테스트 마이그레이션**

`tests/test_ontology_functions.py`·`tests/test_ontology_demand_proxy.py`에서 `rank_stockout_risk`/`explain_order`가 **과거 period + 컬럼평균 demand**를 기대하던 단언을 forward 기반으로 갱신:
- 과거 period 슬라이스 fixture → forward 대상(마지막 관측일 다음 며칠)으로 교체.
- demand_point 정확값 단언은 `forecast_forward(...).item_quantities`에서 파생한 기대값으로 교체(정확값 비교 유지, code-quality 규칙 8).
- `what_if`/`rank_stockout_earliness` 테스트는 무변경.

Run: `uv run pytest tests/test_ontology_functions.py tests/test_ontology_demand_proxy.py --color=no`
Expected: PASS

- [ ] **Step 6: 전체 스위트 (scenario 무영향 확인)**

Run: `uv run pytest --color=no`
Expected: PASS. `tests/test_scenario.py`는 이 스펙에서 안 건드리므로 그대로 통과해야 한다(scenario 재배선 = 범위 밖).

- [ ] **Step 7: 커밋**

```bash
git add src/bakery/ontology/functions.py tests/test_ontology_functions.py tests/test_ontology_demand_proxy.py
git commit -m "feat(ontology): demand_point를 forward 예측으로 재배선 (5a Task5)"
```

---

## Self-Review 체크

- **Spec coverage**: §3(seam 추출)=Task1–3, §4(cli 래퍼)=Task4, §5(functions.py 재배선)=Task5, §6 acceptance=Task3(coherence/faithfulness)+Task4(byte-equal)+각 태스크 전체 스위트. §2 비목표(scenario)=Task5 Step6에서 무영향 확인. ✅ 갭 없음.
- **Placeholder scan**: 이동은 라인범위+verbatim, 신규 코드는 전체 표기. ✅
- **Type consistency**: `forecast_forward`/`ForwardForecast`/`_forward_demand_points` 시그니처가 Task3 정의 ↔ Task4·5 소비에서 일치. `category_totals` 컬럼(base_*/prior_*)·`proportions`(adj_* factor)·`item_quantities`(6컬럼)이 테스트와 일치. ✅

## Execution Handoff

두 실행 옵션:
1. **Subagent-Driven (권장)** — 태스크별 fresh subagent + 태스크 간 리뷰(2단계). 이동/byte-equal의 회귀 위험을 태스크 경계에서 잡기 좋음.
2. **Inline Execution** — 이 세션에서 executing-plans로 배치 실행 + 체크포인트.

어느 방식으로 진행할까요?

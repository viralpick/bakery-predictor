# potential_demand 전역 감사 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** real 데이터 경로(backtest / predict-next-week / alpha-sweep / business-report / ontology)가 오염된 `potential_demand` 대신 `adjusted_demand`를 소비하도록 전환한다.

**Architecture:** 모델 레벨 `_default_target`(v2/v3→potential_demand)은 **변경하지 않는다**. 전환은 CLI/ontology 레이어에서 source가 real일 때 `y_col`/`demand_col`을 `adjusted_demand`로 명시 전달하는 방식. 소스별 결정은 공유 헬퍼 `_resolve_demand_col`(real→adjusted enrich, synthetic→potential 그대로) 하나로 통일한다. synthetic 생성기·schema 필드·arrival-curve 헬퍼는 유지.

**Tech Stack:** Python 3, pandas, LightGBM, typer(CLI), pytest. `uv run` 실행.

## Global Constraints

- Time leakage 금지: lag/rolling은 split 이후 계산. `test_split_leakage.py` / `test_features_leakage.py` 반드시 통과.
- 모델 레벨 `_default_target` / `GlobalLGBM` 기본 y_col은 변경 금지 (기존 pin 테스트 보존).
- 데이터 레이어(`bonavi_loader.attach_potential_demand`) / `schema.py` 컬럼 구조 변경 금지 (주석만).
- closing-discount α 파라미터명은 `closing_alpha`(기본 `DEFAULT_ALPHA`) — alpha-sweep의 quantile `alphas`와 충돌 회피.
- pytest 카운트 필요 시 `-q` 추가 금지, `--color=no` 사용 (repo addopts에 -q 있음).
- 커밋 메시지 한국어. 각 태스크 끝 커밋.

## 테스트 전략 (왜 real e2e가 아닌가)

real 소스는 로컬 parquet/xlsx 파일에 의존(CI/테스트 환경에 없음). 따라서:
- **real 분기 로직**은 `discount_rows` 주입 등 **단위 테스트**로 결정론 검증.
- **synthetic 회귀**는 `_load_dataset` monkeypatch로 in-memory `DailyDataset` 주입해 동작 불변 확인.
- **주입 스레딩**(profit-sim)은 `simulate_profit` monkeypatch로 `potential_col` 인자 캡처.

## File Structure

- `src/bakery/cli.py` — `_resolve_demand_col` 신설, `_build_forecasters(v23_target=)`, 4개 커맨드 wiring.
- `src/bakery/ontology/functions.py` — `_resolve_demand_proxy` 신설, 2개 함수 기본 demand_col 리졸버화.
- `src/bakery/ontology/grounding/run.py` — real일 때 `dataset.daily` enrich.
- `src/bakery/data/bonavi_loader.py`, `src/bakery/data/schema.py` — deprecation 주석.
- `tests/test_cli_demand_resolver.py` (신규), `tests/test_ontology_demand_proxy.py` (신규), 기존 테스트 보강.
- `.claude/CLAUDE.md` — 실행 예시 동기화(옵션·출력 컬럼명).

---

### Task 1: 공유 수요 컬럼 리졸버 `_resolve_demand_col`

**Files:**
- Modify: `src/bakery/cli.py` (import 확인 + 헬퍼 추가)
- Test: `tests/test_cli_demand_resolver.py` (신규)

**Interfaces:**
- Produces: `_resolve_demand_col(daily: pd.DataFrame, source: str, closing_alpha: float, discount_rows: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_cli_demand_resolver.py`:
```python
import pandas as pd
from bakery.cli import _resolve_demand_col


def _daily():
    return pd.DataFrame({
        "store_id": ["S", "S"],
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "sold_units": [10, 20],
        "potential_demand": [10.0, 20.0],
    })


def test_synthetic_returns_potential_and_same_frame():
    df = _daily()
    out, col = _resolve_demand_col(df, "synthetic", 0.5)
    assert col == "potential_demand"
    assert out is df  # synthetic은 프레임 비변형


def test_real_attaches_adjusted_demand():
    df = _daily()
    discount = pd.DataFrame({
        "item_id": ["A"],
        "date": pd.to_datetime(["2026-01-02"]),
        "qty": [4],
    })
    out, col = _resolve_demand_col(df, "real", 0.5, discount_rows=discount)
    assert col == "adjusted_demand"
    row = out.set_index("date").loc["2026-01-02"]
    # adjusted = sold - closing*(1-alpha) = 20 - 4*0.5 = 18
    assert float(row["adjusted_demand"]) == 18.0
    # closing 매칭 없는 날은 adjusted == sold
    assert float(out.set_index("date").loc["2026-01-01"]["adjusted_demand"]) == 10.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_cli_demand_resolver.py --color=no`
Expected: FAIL — `ImportError: cannot import name '_resolve_demand_col'`

- [ ] **Step 3: 헬퍼 구현**

`src/bakery/cli.py` — 먼저 import 확인. 상단에 `build_item_adjusted_demand`(이미 line 36 근처 import됨)와 `DEFAULT_ALPHA`가 import돼 있는지 확인하고, `DEFAULT_ALPHA`가 없으면 category_aggregate import 줄에 추가:
```python
from .features.category_aggregate import (
    build_features, build_item_adjusted_demand, DEFAULT_ALPHA,
)
```
(이미 `DEFAULT_ALPHA`를 cli.py:2119에서 쓰므로 대개 import돼 있음 — 중복 추가 금지, grep로 확인 후 없을 때만.)

`_parse_variants` 근처(모듈 헬퍼 영역)에 추가:
```python
def _resolve_demand_col(
    daily: pd.DataFrame,
    source: str,
    closing_alpha: float,
    discount_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """소스별 수요 컬럼 결정.

    real  → build_item_adjusted_demand로 adjusted_demand 부착 후 컬럼명 반환.
    synth → 입력 프레임 그대로, 'potential_demand'.

    potential_demand는 real에서 stockout_time 로더 버그로 오염돼 소비 금지
    (docs/superpowers/specs/2026-07-10-potential-demand-audit-design.md).
    """
    if source == "real":
        enriched = build_item_adjusted_demand(
            daily, discount_rows=discount_rows, alpha=closing_alpha
        )
        return enriched, "adjusted_demand"
    return daily, "potential_demand"
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_cli_demand_resolver.py --color=no`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/cli.py tests/test_cli_demand_resolver.py
git commit -m "feat: 소스별 수요 컬럼 리졸버 _resolve_demand_col (real→adjusted_demand)"
```

---

### Task 2: `_build_forecasters` v23_target + backtest wiring

**Files:**
- Modify: `src/bakery/cli.py` (`_build_forecasters`, `cmd_backtest`)
- Test: `tests/test_cli_demand_resolver.py` (추가)

**Interfaces:**
- Consumes: `_resolve_demand_col` (Task 1)
- Produces: `_build_forecasters(variants, *, include_production=False, production_quantile=0.85, v23_target: str | None = None)`

- [ ] **Step 1: 실패 테스트 추가**

`tests/test_cli_demand_resolver.py` 하단에:
```python
from bakery.cli import _build_forecasters
from bakery.models.lightgbm_regressor import GlobalLGBM


def _lgbm_by_fs(forecasters):
    return {f.feature_set: f for f in forecasters if isinstance(f, GlobalLGBM)}


def test_build_forecasters_default_keeps_potential():
    fs = _build_forecasters(["v0", "v2"])
    lg = _lgbm_by_fs(fs)
    assert lg["v0"].y_col == "sold_units"
    assert lg["v2"].y_col == "potential_demand"


def test_build_forecasters_v23_target_override():
    fs = _build_forecasters(["v0", "v1", "v2", "v3"], v23_target="adjusted_demand")
    lg = _lgbm_by_fs(fs)
    assert lg["v0"].y_col == "sold_units"
    assert lg["v1"].y_col == "sold_units"
    assert lg["v2"].y_col == "adjusted_demand"
    assert lg["v3"].y_col == "adjusted_demand"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_cli_demand_resolver.py::test_build_forecasters_v23_target_override --color=no`
Expected: FAIL — `TypeError: _build_forecasters() got an unexpected keyword argument 'v23_target'`

- [ ] **Step 3: 구현**

`src/bakery/cli.py` `_build_forecasters` 교체:
```python
def _build_forecasters(variants: list[str], *, include_production: bool = False,
                       production_quantile: float = 0.85,
                       v23_target: str | None = None):
    """Build baseline + LightGBM-per-variant list, optionally adding quantile
    production models for v2/v3 (lightgbm_v2_q85 etc.).

    v23_target: v2/v3의 학습 target을 명시(예: real→'adjusted_demand'). None이면
    모델 기본값(_default_target=potential_demand). v0/v1은 항상 sold_units.
    """
    forecasters = [SeasonalNaive(n_weeks=4), MovingAverage(window=28)]
    for v in variants:
        y = v23_target if (v23_target and v in {"v2", "v3"}) else None
        forecasters.append(GlobalLGBM(feature_set=v, y_col=y))  # demand (median) model
        if include_production and v in {"v2", "v3"}:
            prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
            forecasters.append(GlobalLGBM(feature_set=v, params=prod_params, y_col=y))
    return forecasters
```

`cmd_backtest` 시그니처에 `closing_alpha: float = DEFAULT_ALPHA` 추가(마지막 파라미터, `out_dir` 뒤). 본문에서 daily enrich + forecaster 구성 교체:
```python
    variant_list = _parse_variants(variants)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, variant_list)
    daily, demand_col = _resolve_demand_col(daily, source, closing_alpha)
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )
    forecasters = _build_forecasters(variant_list, include_production=include_production,
                                     v23_target=demand_col)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_cli_demand_resolver.py --color=no`
Expected: PASS (4 passed)

- [ ] **Step 5: 회귀 확인 (모델 pin 테스트 불변)**

Run: `uv run pytest tests/test_v2_pipeline.py tests/test_v3_pipeline.py tests/test_backtest_clone.py --color=no`
Expected: PASS (모두 통과 — `_default_target` 미변경 확인)

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/cli.py tests/test_cli_demand_resolver.py
git commit -m "feat: backtest v2/v3 real target을 adjusted_demand로 (_build_forecasters v23_target)"
```

---

### Task 3: predict-next-week — target + 출력 컬럼

**Files:**
- Modify: `src/bakery/cli.py` (`cmd_predict_next_week`, line 140~199)
- Test: `tests/test_predict_next_week.py` (신규)

**Interfaces:**
- Consumes: `_resolve_demand_col` (Task 1)

- [ ] **Step 1: 실패 테스트 작성 (synthetic 회귀 — 출력 컬럼 보존)**

`tests/test_predict_next_week.py`:
```python
import pandas as pd
import pytest
from typer.testing import CliRunner

from bakery.cli import app
from bakery.data.loader import DailyDataset


def _dataset(daily):
    """DailyDataset은 7개 프레임 전부 필수(frozen dataclass). 미사용은 빈 프레임."""
    empty = pd.DataFrame()
    return DailyDataset(daily=daily, weather=empty, calendar=empty, competitor=empty,
                        living_population=empty, population=empty, consumption=empty)


def _synthetic_dataset(n_days=140):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for item in ("A", "B"):
            rows.append({
                "store_id": "store_A", "item_id": item, "category_id": "bread",
                "date": d, "sold_units": 10 + (item == "B") * 5,
                "is_stockout": False, "stockout_time": pd.NaT,
                "open_hours": 13, "capacity": 100,
                "potential_demand": 10.0 + (item == "B") * 5,
            })
    return _dataset(pd.DataFrame(rows))


@pytest.fixture
def patch_v0_dataset(monkeypatch):
    monkeypatch.setattr("bakery.cli._load_dataset",
                        lambda source, data_dir: _synthetic_dataset())


def test_predict_v0_synthetic_writes_sold_units_col(patch_v0_dataset, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "predict-next-week", "--source", "synthetic",
        "--model", "lightgbm", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(tmp_path / "next_week_predictions.csv")
    assert "yhat_sold_units" in out.columns
```

> 주: v0 모델은 캘린더/날씨 불필요 — 빈 프레임으로 충분.

- [ ] **Step 2: 실패/기준선 확인**

Run: `uv run pytest tests/test_predict_next_week.py --color=no`
Expected: 현재 코드에서 PASS일 수도 있음(v0는 sold_units 기존 동작). 이 테스트는 **회귀 가드** — Step 3 편집 후에도 green이어야 한다. FAIL이면 fixture(DailyDataset 필드) 교정.

- [ ] **Step 3: 구현**

`cmd_predict_next_week` 시그니처에 `closing_alpha: float = DEFAULT_ALPHA` 추가. 본문 160~184 교체:
```python
    feature_set = _model_to_feature_set(model)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, [feature_set]) if feature_set else ds.daily
    if feature_set in {"v2", "v3"}:
        daily, target_col = _resolve_demand_col(daily, source, closing_alpha)
    else:
        target_col = None
    last = daily["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=7, freq="D")
    forecaster = _pick_model(model)
    if feature_set in {"v2", "v3"}:
        forecaster = GlobalLGBM(feature_set=feature_set, y_col=target_col)
    forecaster.fit(daily)
    pairs = daily[["store_id", "item_id", "category_id"]].drop_duplicates()
    target = pairs.merge(pd.DataFrame({"date": horizon}), how="cross")
    if feature_set in {"v1", "v2", "v3"}:
        forecast_weather = _load_forecast_weather(horizon) if use_forecast else None
        target = _enrich_target(
            target, ds, forecast_weather=forecast_weather, include_external=(feature_set == "v3"),
        )
    yhat = forecaster.predict(target)
    demand_col = f"yhat_{target_col}" if feature_set in {"v2", "v3"} else "yhat_sold_units"
    target = target.assign(**{demand_col: yhat.round(2).to_numpy(), "model": forecaster.name})

    if feature_set in {"v2", "v3"}:
        prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
        prod_model = GlobalLGBM(feature_set=feature_set, params=prod_params,
                                y_col=target_col).fit(daily)
```
(이후 `recommended_production` 등 나머지 로직은 `demand_col` 변수를 그대로 참조하므로 불변.)

- [ ] **Step 4: 회귀 통과 확인**

Run: `uv run pytest tests/test_predict_next_week.py --color=no`
Expected: PASS — synthetic v0 출력 컬럼 `yhat_sold_units` 유지.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/cli.py tests/test_predict_next_week.py
git commit -m "feat: predict-next-week v2/v3 real target·출력컬럼을 adjusted_demand로"
```

---

### Task 4: alpha-sweep — target + profit-sim 주입

**Files:**
- Modify: `src/bakery/cli.py` (`cmd_alpha_sweep`, line 526~595)
- Test: `tests/test_business_sim_injection.py` (신규)

**Interfaces:**
- Consumes: `_resolve_demand_col` (Task 1)

- [ ] **Step 1: 실패 테스트 작성 (simulate_profit potential_col 캡처)**

`tests/test_business_sim_injection.py`:
```python
import pandas as pd
import pytest
from typer.testing import CliRunner

from bakery.cli import app
from bakery.data.loader import DailyDataset


def _dataset(daily):
    empty = pd.DataFrame()
    return DailyDataset(daily=daily, weather=empty, calendar=empty, competitor=empty,
                        living_population=empty, population=empty, consumption=empty)


def _real_like_dataset(n_days=140):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        rows.append({
            "store_id": "store_A", "item_id": "A", "category_id": "bread",
            "date": d, "sold_units": 12, "is_stockout": False,
            "stockout_time": pd.NaT, "open_hours": 13, "capacity": 100,
            "potential_demand": 12.0,
        })
    return _dataset(pd.DataFrame(rows))


@pytest.fixture
def capture_profit(monkeypatch):
    """simulate_profit가 받은 potential_col을 기록. 실 모델·xlsx·discount 우회."""
    seen = {}

    def fake_profit(pred_df, *, potential_col=None, **kw):
        seen["potential_col"] = potential_col
        out = pred_df.copy()
        for c in ("revenue_krw", "waste_cost_krw", "lost_margin_krw", "net_profit_krw"):
            out[c] = 0.0
        return out

    monkeypatch.setattr("bakery.cli.simulate_profit", fake_profit)
    monkeypatch.setattr("bakery.cli._load_dataset",
                        lambda source, data_dir: _real_like_dataset())
    # adjusted_demand 부착 (real closing 파일 우회)
    monkeypatch.setattr(
        "bakery.cli.build_item_adjusted_demand",
        lambda daily, discount_rows=None, alpha=0.5: daily.assign(
            adjusted_demand=daily["sold_units"].astype(float)),
    )
    monkeypatch.setattr("bakery.cli.pd.read_excel",
                        lambda *a, **k: pd.DataFrame({"품목코드": ["A"], "상품구분": ["SS"],
                                                      "판매단가": [3000]}))
    return seen


def test_alpha_sweep_real_uses_adjusted_demand(capture_profit, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "alpha-sweep", "--source", "real", "--variant", "v0",
        "--alphas", "0.5", "--n-splits", "2", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert capture_profit["potential_col"] == "adjusted_demand"
```

> 주: `cmd_alpha_sweep` 인자명(`--out-dir`, `--n-splits`, `--variant`, `--alphas`, `--item-master`)을 실제 시그니처로 확인해 맞춘다. `--variant v0`로 quantile 분기를 피해 median 경로만 태운다. item_master 기본 경로 존재 여부와 무관하도록 `pd.read_excel` monkeypatch.

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_business_sim_injection.py::test_alpha_sweep_real_uses_adjusted_demand --color=no`
Expected: FAIL — 현재 `potential_col` 미전달(기본 "potential_demand") → assert 실패.

- [ ] **Step 3: 구현**

`cmd_alpha_sweep` 시그니처에 `closing_alpha: float = DEFAULT_ALPHA` 추가. `ds = _load_dataset(...)` / `daily = _enrich_if_needed(...)` 직후:
```python
    ds = _load_dataset(source, None)
    daily = _enrich_if_needed(ds, [variant])
    daily, demand_col = _resolve_demand_col(daily, source, closing_alpha)
```
forecaster 루프에서 `GlobalLGBM(feature_set=variant, ...)` 두 곳에 `y_col=demand_col` 추가:
```python
    for a in alpha_list:
        if a == 0.5:
            forecasters.append(GlobalLGBM(feature_set=variant, y_col=demand_col))
        else:
            params = LGBMParams(objective="quantile", alpha=a)
            forecasters.append(GlobalLGBM(feature_set=variant, params=params, y_col=demand_col))
```
주입부(현 583~590) 교체:
```python
    # Inject demand column for true-demand-aware profit simulation
    if demand_col in daily.columns:
        d_lookup = daily.set_index(["store_id", "item_id", "date"])[demand_col]
        pred_df = pred_df.copy()
        pred_df[demand_col] = pred_df.set_index(
            ["store_id", "item_id", "date"]
        ).index.map(d_lookup)
```
`simulate_profit(...)` 호출에 `potential_col=demand_col` 추가:
```python
        profit = simulate_profit(sub, unit_prices=unit_prices, params=cost_params,
                                 potential_col=demand_col)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_business_sim_injection.py::test_alpha_sweep_real_uses_adjusted_demand --color=no`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/cli.py tests/test_business_sim_injection.py
git commit -m "feat: alpha-sweep real target·profit-sim 잣대를 adjusted_demand로"
```

---

### Task 5: business-report — targets + profit-sim/KPI 주입

**Files:**
- Modify: `src/bakery/cli.py` (`cmd_business_report`, line 628~830)
- Test: `tests/test_business_sim_injection.py` (추가)

**Interfaces:**
- Consumes: `_resolve_demand_col` (Task 1), `_build_forecasters(v23_target=)` (Task 2)

- [ ] **Step 1: 실패 테스트 추가**

`tests/test_business_sim_injection.py` 하단에 (fixture 재사용):
```python
def test_business_report_real_uses_adjusted_demand(capture_profit, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "business-report", "--source", "real", "--variants", "v0",
        "--n-splits", "2", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert capture_profit["potential_col"] == "adjusted_demand"
```

> 주: `cmd_business_report`의 실제 인자명(`--variants`, `--out-dir`, `--item-master`, `--n-splits`)을 확인. business-report도 `pd.read_excel`로 item_master를 읽으므로 fixture의 read_excel monkeypatch가 필요(이미 capture_profit fixture에 포함). 출력 파일 경로 인자가 `out_dir`가 아니면 맞춘다.

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_business_sim_injection.py::test_business_report_real_uses_adjusted_demand --color=no`
Expected: FAIL — 기본 potential_demand.

- [ ] **Step 3: 구현**

`cmd_business_report` 시그니처에 `closing_alpha: float = DEFAULT_ALPHA` 추가. `daily = _enrich_if_needed(...)`(현 688) 직후:
```python
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, variant_list)
    daily, demand_col = _resolve_demand_col(daily, source, closing_alpha)
```
`_build_forecasters(...)` 호출(현 797)에 `v23_target=demand_col` 추가:
```python
    forecasters = _build_forecasters(
        variant_list, include_production=..., production_quantile=production_quantile,
        v23_target=demand_col,
    )
```
(기존 인자는 그대로 두고 v23_target만 추가.)
주입부(현 815~822) 교체 — Task 4 Step 3과 동일 패턴:
```python
    # Inject demand column so simulate_profit sees the censoring-corrected target.
    if demand_col in daily.columns:
        d_lookup = daily.set_index(["store_id", "item_id", "date"])[demand_col]
        pred_df = pred_df.copy()
        pred_df[demand_col] = pred_df.set_index(
            ["store_id", "item_id", "date"]
        ).index.map(d_lookup)
```
이 커맨드 안의 **모든** `simulate_profit(...)` 호출에 `potential_col=demand_col` 추가(grep `simulate_profit` in 628~860로 전수 확인).

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_business_sim_injection.py --color=no`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/cli.py tests/test_business_sim_injection.py
git commit -m "feat: business-report real target·KPI 잣대를 adjusted_demand로"
```

---

### Task 6: ontology — 수요 프록시 자동 리졸버 + run_eval real enrich

**Files:**
- Modify: `src/bakery/ontology/functions.py` (`_resolve_demand_proxy` 신설, `rank_stockout_risk`·`explain_order` 기본값)
- Modify: `src/bakery/ontology/grounding/run.py` (`run_eval` real enrich)
- Test: `tests/test_ontology_demand_proxy.py` (신규)

**Interfaces:**
- Produces: `_resolve_demand_proxy(daily: pd.DataFrame) -> str`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_ontology_demand_proxy.py`:
```python
import pandas as pd
from bakery.ontology.functions import _resolve_demand_proxy


def _frame(cols):
    base = {"store_id": ["S"], "item_id": ["A"],
            "date": pd.to_datetime(["2026-01-01"]), "sold_units": [10]}
    base.update({c: [10.0] for c in cols})
    return pd.DataFrame(base)


def test_proxy_prefers_adjusted_when_present():
    df = _frame(["potential_demand", "adjusted_demand"])
    assert _resolve_demand_proxy(df) == "adjusted_demand"


def test_proxy_falls_back_to_potential():
    df = _frame(["potential_demand"])
    assert _resolve_demand_proxy(df) == "potential_demand"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_ontology_demand_proxy.py --color=no`
Expected: FAIL — `ImportError: cannot import name '_resolve_demand_proxy'`

- [ ] **Step 3: functions.py 구현**

`DEMAND_PROXY_COL` 정의 아래에 추가:
```python
def _resolve_demand_proxy(daily: pd.DataFrame) -> str:
    """수요 점추정 컬럼 결정: adjusted_demand 있으면 그것(real), 없으면 potential_demand.

    real 데이터의 potential_demand는 stockout_time 로더 버그로 오염 →
    grounding/run.py가 real일 때 adjusted_demand를 부착하면 자동 채택된다.
    """
    return "adjusted_demand" if "adjusted_demand" in daily.columns else DEMAND_PROXY_COL
```
`rank_stockout_risk`·`explain_order` 시그니처의 `demand_col: str = DEMAND_PROXY_COL` → `demand_col: str | None = None`. 각 함수 본문 첫 줄(`items = _item_demand_points(...)`) 앞에:
```python
    demand_col = demand_col or _resolve_demand_proxy(daily)
```
`DEMAND_PROXY_COL` docstring/모듈 docstring(line 9~12)의 "live 예측 미wired" 문구를 real=adjusted_demand 자동 채택 반영으로 갱신.

- [ ] **Step 4: functions 테스트 통과 + 회귀**

Run: `uv run pytest tests/test_ontology_demand_proxy.py tests/test_ontology_functions.py --color=no`
Expected: PASS (기존 test_ontology_functions는 synthetic potential_demand 프레임 → 리졸버가 potential_demand 반환하므로 불변)

- [ ] **Step 5: run_eval real enrich 테스트 작성**

`tests/test_ontology_demand_proxy.py` 하단에:
```python
from bakery.data.loader import DailyDataset


def test_run_eval_enriches_real(monkeypatch):
    import bakery.ontology.grounding.run as run_mod

    captured = {}

    def fake_load(source):
        daily = pd.DataFrame({
            "store_id": ["S"], "item_id": ["A"],
            "date": pd.to_datetime(["2026-01-01"]), "sold_units": [10],
            "potential_demand": [10.0],
        })
        empty = pd.DataFrame()
        return DailyDataset(daily=daily, weather=empty, calendar=empty, competitor=empty,
                            living_population=empty, population=empty, consumption=empty)

    def fake_client(provider, model):
        return object()

    def fake_eval_with_client(client, dataset):
        captured["cols"] = list(dataset.daily.columns)
        from bakery.ontology.grounding.scorer import EvalReport
        return EvalReport(results=[], grounded_accuracy=0.0, rag_accuracy=0.0, delta=0.0)

    monkeypatch.setattr(run_mod, "load_dataset", fake_load)
    monkeypatch.setattr(run_mod, "make_llm_client", fake_client)
    monkeypatch.setattr(run_mod, "run_eval_with_client", fake_eval_with_client)
    monkeypatch.setattr(
        "bakery.ontology.grounding.run.build_item_adjusted_demand",
        lambda daily, alpha=0.5: daily.assign(adjusted_demand=daily["sold_units"].astype(float)),
    )

    run_mod.run_eval(source="real")
    assert "adjusted_demand" in captured["cols"]
```

> 주: `EvalReport(results, grounded_accuracy, rag_accuracy, delta)` — scorer.py:27 확정. run.py의 `load_dataset`/`make_llm_client`/`run_eval_with_client`/`build_item_adjusted_demand`는 모두 `run` 모듈 네임스페이스로 monkeypatch(import된 심볼).

- [ ] **Step 6: run.py 구현**

`src/bakery/ontology/grounding/run.py` import에 추가:
```python
from ...features.category_aggregate import build_item_adjusted_demand, DEFAULT_ALPHA
import dataclasses
```
`run_eval` 교체:
```python
def run_eval(provider: str = "auto", model: str = "gpt-5-mini",
             source: str = "synthetic") -> EvalReport:
    client = make_llm_client(provider, model)
    dataset = load_dataset(source)
    if source == "real":
        enriched = build_item_adjusted_demand(dataset.daily, alpha=DEFAULT_ALPHA)
        dataset = dataclasses.replace(dataset, daily=enriched)
    return run_eval_with_client(client, dataset)
```

- [ ] **Step 7: 통과 확인**

Run: `uv run pytest tests/test_ontology_demand_proxy.py --color=no`
Expected: PASS (3 passed)

- [ ] **Step 8: 커밋**

```bash
git add src/bakery/ontology/functions.py src/bakery/ontology/grounding/run.py tests/test_ontology_demand_proxy.py
git commit -m "feat: ontology 수요 프록시 adjusted_demand 자동 채택 + run_eval real enrich"
```

---

### Task 7: deprecation 마커 + 문서 동기화 + 전체 회귀

**Files:**
- Modify: `src/bakery/data/bonavi_loader.py` (attach 지점 주석)
- Modify: `src/bakery/data/schema.py` (potential_demand 필드 주석)
- Modify: `.claude/CLAUDE.md` (실행 예시)

- [ ] **Step 1: deprecation 주석 추가**

`bonavi_loader.py`의 `attach_potential_demand` 호출부(현 307/404 근처)에 주석:
```python
    # NOTE: potential_demand는 real에서 stockout_time 로더 버그로 오염됨(하루 다중
    # 품절 이벤트 중 첫 것만 취함). real 소비 경로는 adjusted_demand로 전환됨
    # (#3 감사, 2026-07-10). 이 컬럼은 schema 정합성·synthetic 패리티 위해서만 유지.
```
`schema.py`의 `potential_demand` 필드 주석(현 21~23) 끝에 추가:
```python
# ⚠️ real에서는 stockout_time 버그로 오염 — real 경로는 adjusted_demand 사용(#3 감사).
```

- [ ] **Step 2: CLAUDE.md 실행 예시 동기화**

`.claude/CLAUDE.md`의 실행 예시에서 `predict-next-week` / `backtest` 예시에 real일 때 출력이 `yhat_adjusted_demand`이고 `--closing-alpha` 옵션(기본 0.5)이 생겼음을 한 줄 주석으로 반영. (명령 자체 형식은 유지.)

- [ ] **Step 3: 전체 테스트 통과 확인**

Run: `uv run pytest --color=no`
Expected: PASS (신규 테스트 포함 전부 통과, leakage 테스트 포함)

- [ ] **Step 4: leakage 게이트 명시 확인**

Run: `uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py --color=no`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/bonavi_loader.py src/bakery/data/schema.py .claude/CLAUDE.md
git commit -m "docs: potential_demand real-오염 deprecation 마커 + 실행 예시 동기화 (#3)"
```

---

## Self-Review 체크리스트 (계획 작성자용, 실행 전 확인)

1. **Spec coverage**: 5개 real 경로(backtest T2 / predict T3 / alpha-sweep T4 / business-report T5 / ontology T6) + deprecation 마커(T7) 모두 태스크로 커버. ✅
2. **_default_target 불변**: 모델 레벨 미변경 — T2 Step 5에서 pin 테스트 회귀 확인. ✅
3. **Type 일관성**: `_resolve_demand_col`(T1) 반환 col을 T2~T5가 `y_col`/`demand_col`/`potential_col`로 일관 사용. `_build_forecasters(v23_target=)`(T2)를 T5가 재사용. `_resolve_demand_proxy`(T6) 별도. ✅
4. **확정됨**: `DailyDataset`(7필드), `EvalReport`(4필드) — fixture 반영 완료.
5. **실행 착수 시 확인(코드 대조)**: 각 커맨드 CLI 인자명(`--out-dir`/`--n-splits`/`--variant(s)`/`--item-master`)을 typer 시그니처로 대조, `DEFAULT_ALPHA` import 중복 여부(cli.py는 대개 기존 import 재사용). 불일치 시 fixture 인자만 조정.

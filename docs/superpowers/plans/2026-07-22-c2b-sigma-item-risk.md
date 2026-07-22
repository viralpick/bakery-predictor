# c-2b: category σ(x) → item-risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 분포모델 category σ(x)를 item 위험으로 내려, `decision/risk.py`의 하드코딩 `cv=0.30` placeholder를 대체하고 predict-next-week category 출력의 `stockout_prob`(현 NaN)을 채운다.

**Architecture:** risk 셸에 LogNormal 샘플링 경로를 추가하되 σ 미제공 시 기존 Normal cv 경로를 유지(backward-compat). c-1 함수(`_category_future_order_predictions`)가 날짜별 σ를 산출해 item에 broadcast(store-total이 LogNormal이고 item=고정비율×total이면 σ 불변 — 수학적으로 정확). predict-next-week 핸들러가 `(demand_point, our_order, σ)`로 `simulate_item_risk`를 호출해 `p_stockout`을 채운다. σ는 post-prediction 입력이라 leakage와 구조적으로 무관.

**Tech Stack:** Python, numpy, pandas, LightGBM, NGBoost(LogNormal), typer CLI, pytest.

## Global Constraints

- 발주 base·점추정 = **median**(LogNormal mean 아님) — 기존 `predict_expected` 계약 유지.
- σ 분배 = **broadcast(공유-μ)**: item σ_log = category σ_log, 그날 모든 item 동일. item-level σ 미검증 → 인플레이션 금지(overclaim 회피).
- predict-next-week 출력 **8-col 스키마 불변**: `[store_id, item_id, category_id, date, demand_col, stockout_prob, recommended_production, model]`. p_waste/expected_cost는 CSV에 추가하지 않는다(v6-predict 경로에서만 노출).
- backward-compat 절대 규칙: σ 미전달/미존재 시 기존 동작·값 **정확 동일**(기존 v6 호출·테스트 무회귀).
- `test_split_leakage.py`·`test_features_leakage.py` 무회귀.
- 커밋 메시지 말미:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` /
  `Claude-Session: https://claude.ai/code/session_01DtB1FdcnGrW4fRxTGjgHWz`
- pytest 카운트 필요 시 `-q` 추가 금지(addopts에 `-q` 있음) — `--color=no` 사용.

## File Structure

- `src/bakery/decision/risk.py` (수정): LogNormal 샘플러 + `simulate_item_risk`에 `demand_sigma_log` 키워드 인자.
- `src/bakery/decision/pipeline.py` (수정): `items`의 optional `demand_sigma_log` 컬럼을 per-row로 risk에 스루.
- `src/bakery/cli.py` (수정): `_category_base_predict` σ 반환(3-tuple), `_category_future_order_predictions` σ broadcast 컬럼, `_predict_next_week_category` risk 채우기.
- `tests/test_decision_layer.py` (수정): risk LogNormal 경로 + pipeline σ 스루 테스트.
- `tests/test_predict_next_week.py` (수정): distributional σ→finite stockout_prob, lightgbm→NaN, 8-col 핀.

---

### Task 1: risk.py LogNormal 경로 (backward-compat)

**Files:**
- Modify: `src/bakery/decision/risk.py` (`_sample_demand` 아래 신규 함수 + `simulate_item_risk` 46-69)
- Test: `tests/test_decision_layer.py`

**Interfaces:**
- Consumes: 기존 `RiskParams`, `RiskResult`.
- Produces: `simulate_item_risk(demand_point, order_qty, params=RiskParams(), rng=None, *, demand_sigma_log: float | None = None) -> RiskResult`. `demand_sigma_log`가 유효(not None, `>0`, `demand_point>0`)하면 demand~LogNormal(median=demand_point, shape=demand_sigma_log), 아니면 기존 Normal cv 경로.

- [ ] **Step 1: 실패 테스트 4개 작성**

`tests/test_decision_layer.py`의 risk 테스트 블록(약 51-77행 근처, `test_expected_cost...` 뒤)에 추가:

```python
def test_lognormal_median_gives_half_stockout_at_order_equals_point():
    # LogNormal: P(X > median) = 0.5 exactly. order=point=median → p_stockout≈0.5.
    p = RiskParams(n_samples=60000, seed=4)
    res = simulate_item_risk(20.0, 20.0, p, demand_sigma_log=0.4)
    assert abs(res.p_stockout - 0.5) < 0.02


def test_higher_sigma_raises_stockout_when_order_above_point():
    # order(25) > point(20): 분산 클수록 상단 tail이 order를 더 자주 넘음.
    p = RiskParams(n_samples=60000, seed=8)
    lo = simulate_item_risk(20.0, 25.0, p, demand_sigma_log=0.15)
    hi = simulate_item_risk(20.0, 25.0, p, demand_sigma_log=0.55)
    assert hi.p_stockout > lo.p_stockout


def test_sigma_none_matches_normal_default_exactly():
    # backward-compat: None 전달 == 미전달 (기존 Normal cv 경로, 동일 seed).
    p = RiskParams(n_samples=5000, seed=7)
    a = simulate_item_risk(20.0, 23.0, p)
    b = simulate_item_risk(20.0, 23.0, p, demand_sigma_log=None)
    assert a == b


def test_nonpositive_point_with_sigma_falls_back_to_normal():
    # demand_point<=0 → LogNormal 불가 → Normal fallback(예외 없이 all-waste).
    res = simulate_item_risk(0.0, 5.0, RiskParams(n_samples=5000, seed=10), demand_sigma_log=0.3)
    assert res.p_stockout == 0.0
    assert res.p_waste == 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_decision_layer.py::test_lognormal_median_gives_half_stockout_at_order_equals_point --color=no`
Expected: FAIL — `simulate_item_risk() got an unexpected keyword argument 'demand_sigma_log'`.

- [ ] **Step 3: 구현**

`src/bakery/decision/risk.py`에서 `_sample_demand`(46-48행) 바로 아래에 추가:

```python
def _sample_demand_lognormal(
    point: float, sigma_log: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    # median=point → mu=log(point). 비율 스케일 하에서 σ 불변(store-total LogNormal 상속).
    return np.exp(rng.normal(np.log(point), sigma_log, n))
```

`simulate_item_risk`(51-69행) 시그니처·샘플링 분기 교체:

```python
def simulate_item_risk(
    demand_point: float,
    order_qty: float,
    params: RiskParams = RiskParams(),
    rng: np.random.Generator | None = None,
    *,
    demand_sigma_log: float | None = None,
) -> RiskResult:
    """Monte-Carlo P(stockout)/P(waste)/expected cost for one item's order.

    demand_sigma_log가 주어지면(>0, demand_point>0) demand~LogNormal(median=demand_point,
    shape=demand_sigma_log)로 샘플(분포모델 category σ 상속). 그 외에는 기존
    Normal(point, cv·point) placeholder 경로(backward-compat).
    """
    rng = rng if rng is not None else np.random.default_rng(params.seed)
    if demand_sigma_log is not None and demand_sigma_log > 0 and demand_point > 0:
        demand = _sample_demand_lognormal(demand_point, demand_sigma_log, params.n_samples, rng)
    else:
        demand = _sample_demand(demand_point, params.demand_cv, params.n_samples, rng)
    short = np.clip(demand - order_qty, 0.0, None)
    leftover = np.clip(order_qty - demand, 0.0, None)
    cost = params.unit_margin * short + params.unit_cost * leftover
    return RiskResult(
        p_stockout=float((short > 0).mean()),
        p_waste=float((leftover > 0).mean()),
        expected_short=float(short.mean()),
        expected_waste=float(leftover.mean()),
        expected_cost=float(cost.mean()),
    )
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_decision_layer.py --color=no`
Expected: 기존 + 신규 4개 모두 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/decision/risk.py tests/test_decision_layer.py
git commit -m "feat(decision): risk 셸 LogNormal 경로 추가 (σ 주입, backward-compat)

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01DtB1FdcnGrW4fRxTGjgHWz')"
```

---

### Task 2: pipeline.py σ 컬럼 스루

**Files:**
- Modify: `src/bakery/decision/pipeline.py` (`_recommend_one` 47-56, `build_recommendation` 57-84)
- Test: `tests/test_decision_layer.py`

**Interfaces:**
- Consumes: Task 1의 `simulate_item_risk(..., demand_sigma_log=...)`.
- Produces: `build_recommendation`이 `items`에 `demand_sigma_log` 컬럼이 있으면 per-row로 risk에 전달, 없거나 NaN이면 None(기존 cv 경로). 출력 스키마·`METRIC_COLS` 불변.

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_decision_layer.py` pipeline 테스트 블록 끝(약 137행 뒤)에 추가:

```python
def test_pipeline_threads_demand_sigma_log():
    # 정책상 order>demand_point(안전마진) → σ 클수록 p_stockout 커야 함.
    base = _items()
    risk = RiskParams(n_samples=30000, seed=3)
    r_lo = build_recommendation(base.assign(demand_sigma_log=[0.1, 0.1]), PolicyParams(), risk)
    r_hi = build_recommendation(base.assign(demand_sigma_log=[0.6, 0.6]), PolicyParams(), risk)
    assert (r_hi.table["p_stockout"].to_numpy() > r_lo.table["p_stockout"].to_numpy()).all()


def test_pipeline_nan_sigma_falls_back_to_cv():
    # demand_sigma_log 컬럼이 NaN이면 None 취급 → 기존 cv 경로와 동일 결과.
    base = _items()
    risk = RiskParams(n_samples=4000, seed=9)
    with_nan = base.assign(demand_sigma_log=[float("nan"), float("nan")])
    a = build_recommendation(base, PolicyParams(), risk)
    b = build_recommendation(with_nan, PolicyParams(), risk)
    assert a.table["p_stockout"].tolist() == b.table["p_stockout"].tolist()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_decision_layer.py::test_pipeline_threads_demand_sigma_log --color=no`
Expected: FAIL — σ 컬럼이 무시돼 r_lo/r_hi p_stockout이 (cv 경로라) 동일 → assert 실패.

- [ ] **Step 3: 구현**

`src/bakery/decision/pipeline.py` `_recommend_one`(47-56행)에 σ 인자 추가:

```python
def _recommend_one(item_id: str, demand_point: float, policy: PolicyParams,
                   risk: RiskParams, rng: np.random.Generator,
                   demand_sigma_log: float | None = None) -> tuple[dict, DecisionLineage]:
    order, lineage = apply_policy(item_id, demand_point, policy)
    res = simulate_item_risk(demand_point, order, risk, rng, demand_sigma_log=demand_sigma_log)
    row = dict(
        item_id=item_id, demand_point=float(demand_point), order_qty=order,
        p_stockout=res.p_stockout, p_waste=res.p_waste,
        expected_short=res.expected_short, expected_waste=res.expected_waste,
        expected_cost=res.expected_cost,
    )
    return row, lineage
```

`build_recommendation`(74-82행)의 루프를 σ 스루로 교체:

```python
    _validate(items)
    seeds = np.random.SeedSequence(risk.seed).spawn(len(items))
    carry = [c for c in CARRY_COLS if c in items.columns]
    has_sigma = "demand_sigma_log" in items.columns
    rows, lineages = [], []
    for seed, record in zip(seeds, items.itertuples(index=False)):
        item_rng = np.random.default_rng(seed)
        sigma = getattr(record, "demand_sigma_log", None) if has_sigma else None
        if sigma is not None and pd.isna(sigma):
            sigma = None
        row, lineage = _recommend_one(
            record.item_id, record.demand_point, policy, risk, item_rng, sigma)
        for col in carry:
            row[col] = getattr(record, col)
        rows.append(row)
        lineages.append(lineage)
    table = pd.DataFrame(rows, columns=[*carry, *METRIC_COLS])
    return Recommendation(table=table, lineages=lineages)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_decision_layer.py --color=no`
Expected: 신규 2개 포함 전부 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/decision/pipeline.py tests/test_decision_layer.py
git commit -m "feat(decision): pipeline이 demand_sigma_log 컬럼을 risk에 스루

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01DtB1FdcnGrW4fRxTGjgHWz')"
```

---

### Task 3: cli.py — `_category_base_predict` σ 반환 + `_category_future_order_predictions` σ broadcast

**Files:**
- Modify: `src/bakery/cli.py` (`_category_base_predict` 2035-2058, 호출부 2101·2302, `_category_future_order_predictions` 2270-2328)
- Test: `tests/test_predict_next_week.py`

**Interfaces:**
- Consumes: `fit_distributional_total(...).predict_sigma(test)` (날짜별 σ_log ndarray), `fit_category_total`(σ 없음).
- Produces:
  - `_category_base_predict(...) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]` = `(base_median, base_prod, base_sigma)`. distributional → σ ndarray, lightgbm → None.
  - `_category_future_order_predictions(...)` 반환 컬럼에 `demand_sigma_log` 추가(그날 모든 item broadcast; lightgbm이면 NaN). 최종 7-col: `[store_id, item_id, category_id, date, demand_point, our_order, demand_sigma_log]`.

- [ ] **Step 1: 실패 테스트 작성 (lightgbm base_predict → σ None 계약)**

`tests/test_predict_next_week.py` 끝에 추가(경량 — LightGBM만, NGBoost 회피):

```python
def test_category_base_predict_returns_none_sigma_for_lightgbm():
    """_category_base_predict는 (median, prod, sigma) 3-tuple. lightgbm은 sigma=None."""
    import numpy as np
    from bakery.cli import _category_base_predict

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "adjusted_demand_unit": rng.uniform(100, 300, n),
        "dow": rng.integers(0, 7, n).astype(float),
        "is_holiday": rng.integers(0, 2, n).astype(float),
    })
    train, test = df.iloc[:50], df.iloc[50:]
    median, prod, sigma = _category_base_predict(
        train, test, target_col="adjusted_demand_unit",
        total_model="lightgbm", production_quantile=0.85,
    )
    assert len(median) == len(test)
    assert len(prod) == len(test)
    assert sigma is None
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_predict_next_week.py::test_category_base_predict_returns_none_sigma_for_lightgbm --color=no`
Expected: FAIL — `ValueError: not enough values to unpack (expected 3, got 2)`.

- [ ] **Step 3: 구현 — base_predict 3-tuple**

`src/bakery/cli.py` `_category_base_predict`(2035-2058) 본문 교체(docstring σ 반환 추가):

```python
    import numpy as np

    if total_model == "distributional":
        model = fit_distributional_total(train, target_col=target_col)
        base_prod = np.clip(model.predict_production(test, production_q=production_quantile), 0.0, None)
        base_sigma = np.asarray(model.predict_sigma(test), dtype=float)
    elif total_model == "lightgbm":
        model = fit_category_total(train, target_col=target_col, production_q=production_quantile)
        base_prod = np.clip(model.predict_production(test), 0.0, None)
        base_sigma = None
    else:
        raise ValueError(f"unknown total_model: {total_model!r} (expected 'lightgbm' or 'distributional')")
    base_median = np.clip(model.predict_expected(test), 0.0, None)
    return base_median, base_prod, base_sigma
```

docstring 첫 줄 아래에 한 줄 추가:
```
    base_sigma는 distributional일 때 날짜별 σ(log-space) ndarray, lightgbm이면 None.
```

- [ ] **Step 4: 호출부 2곳 갱신**

fold 경로(약 2101행) — σ 무시:
```python
        base_median, base_prod, _ = _category_base_predict(
            train_df, test_df, target_col=target_col,
            total_model=total_model, production_quantile=production_quantile,
        )
```

future 경로(약 2302행) — σ 수신:
```python
    base_median, base_prod, base_sigma = _category_base_predict(
        train, test, target_col=target_col,
        total_model=total_model, production_quantile=production_quantile,
    )
```

- [ ] **Step 5: 구현 — future func σ broadcast**

`_category_future_order_predictions`에서 `preds`에 `category_id`·`store_id`를 넣은 직후(약 2321행, `preds["category_id"] = ...` 다음 줄)에 σ broadcast 추가:

```python
    if base_sigma is not None:
        sigma_by_date = dict(zip(pd.to_datetime(dates), base_sigma))
        preds["demand_sigma_log"] = pd.to_datetime(preds["date"]).map(sigma_by_date).astype(float)
    else:
        preds["demand_sigma_log"] = float("nan")
```

반환문(약 2328행) 컬럼 목록에 `demand_sigma_log` 추가:
```python
    return preds[["store_id", "item_id", "category_id", "date",
                  "demand_point", "our_order", "demand_sigma_log"]]
```

- [ ] **Step 6: 통과 확인 + 회귀**

Run: `uv run pytest tests/test_predict_next_week.py --color=no`
Expected: 신규 lightgbm σ=None 테스트 PASS. 기존 `test_predict_category_handler_schema`는 Task 4에서 스텁을 7-col로 갱신하기 전까지 통과 유지(핸들러가 아직 σ 컬럼 미참조) — 이 시점엔 PASS.

Run: `uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py --color=no`
Expected: PASS (회귀 없음).

- [ ] **Step 7: 커밋**

```bash
git add src/bakery/cli.py tests/test_predict_next_week.py
git commit -m "feat(cli): c-1 카테고리 예측이 날짜별 σ(x)를 산출해 item에 broadcast

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01DtB1FdcnGrW4fRxTGjgHWz')"
```

---

### Task 4: cli.py — `_predict_next_week_category` risk 채우기

**Files:**
- Modify: `src/bakery/cli.py` (`_predict_next_week_category` 159-190)
- Test: `tests/test_predict_next_week.py` (`_stub_future_orders` + `test_predict_category_handler_schema` 갱신 + 신규 finite-σ 테스트)

**Interfaces:**
- Consumes: Task 1 `simulate_item_risk(demand_point, order, demand_sigma_log=σ)`, Task 3 `_category_future_order_predictions`의 `demand_sigma_log` 컬럼.
- Produces: predict-next-week category 출력 `stockout_prob` = σ 있는 행은 `p_stockout`(0<p<1), σ NaN 행은 NaN. 8-col 스키마 불변.

- [ ] **Step 1: 실패 테스트 작성 (스텁 7-col화 + finite-σ + lightgbm NaN)**

`tests/test_predict_next_week.py`에서 `_stub_future_orders`에 σ 컬럼 추가:

```python
def _stub_future_orders(sigma=(0.25, 0.25)):
    """_category_future_order_predictions 반환 스키마(7 cols)의 최소 2행."""
    return pd.DataFrame({
        "store_id": ["store_gw01", "store_gw01"],
        "item_id": ["A", "B"],
        "category_id": ["bread", "pastry"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "demand_point": [8.7, 14.2],
        "our_order": [9.7, 14.9],
        "demand_sigma_log": list(sigma),
    })
```

기존 `test_predict_category_handler_schema`는 lightgbm(σ 없음) 시나리오로 유지하되 스텁을 NaN σ로:
```python
def test_predict_category_handler_schema(monkeypatch, tmp_path):
    """category 모드 출력이 item 경로와 동일 8-col 스키마 + 규약을 지키는지 pin.

    lightgbm(σ 없음) → stockout_prob NaN. real 데이터 없이 c-1 함수를 스텁."""
    monkeypatch.setattr(
        "bakery.cli._category_future_order_predictions",
        lambda *a, **k: _stub_future_orders(sigma=(float("nan"), float("nan"))),
    )
    result = CliRunner().invoke(app, [
        "predict-next-week", "--source", "real", "--order-level", "category",
        "--total-model", "lightgbm", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(tmp_path / "next_week_predictions.csv")
    assert list(out.columns) == [
        "store_id", "item_id", "category_id", "date",
        "yhat_adjusted_demand_unit", "stockout_prob", "recommended_production", "model",
    ]
    assert out["recommended_production"].tolist() == [10.0, 15.0]   # our_order.round(0)
    assert out["yhat_adjusted_demand_unit"].tolist() == [8.7, 14.2]  # demand_point
    assert out["stockout_prob"].isna().all()
    assert out["model"].unique().tolist() == ["category_total:lightgbm"]
```

신규 finite-σ 테스트 추가:
```python
def test_predict_category_fills_stockout_prob_from_sigma(monkeypatch, tmp_path):
    """distributional σ 있는 행 → stockout_prob 유한(0<p<1), 8-col 스키마 유지."""
    monkeypatch.setattr(
        "bakery.cli._category_future_order_predictions",
        lambda *a, **k: _stub_future_orders(sigma=(0.3, 0.3)),
    )
    result = CliRunner().invoke(app, [
        "predict-next-week", "--source", "real", "--order-level", "category",
        "--total-model", "distributional", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(tmp_path / "next_week_predictions.csv")
    assert list(out.columns) == [
        "store_id", "item_id", "category_id", "date",
        "yhat_adjusted_demand_unit", "stockout_prob", "recommended_production", "model",
    ]
    assert out["stockout_prob"].notna().all()
    assert ((out["stockout_prob"] > 0.0) & (out["stockout_prob"] < 1.0)).all()
    assert out["model"].unique().tolist() == ["category_total:distributional"]
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_predict_next_week.py::test_predict_category_fills_stockout_prob_from_sigma --color=no`
Expected: FAIL — 핸들러가 아직 `stockout_prob=nan` 고정 → `notna().all()` 실패.

- [ ] **Step 3: 구현 — 핸들러 risk 채우기**

`src/bakery/cli.py` `_predict_next_week_category`(159-190). 먼저 파일 상단 import에 risk 셸이 없으면 추가(함수 내 지역 import로도 가능):

함수 본문에서 `out = preds.rename(...)` 블록을 교체. `stockout_prob`을 σ 기반 risk로 계산.
`np`는 cli.py 모듈 레벨에 없으므로(함수 지역 import 패턴) 지역 import를 넣는다:

```python
    import numpy as np

    from bakery.decision.risk import RiskParams, simulate_item_risk

    demand_col = "yhat_adjusted_demand_unit"
    model_name = f"category_total:{total_model}"
    risk_params = RiskParams()

    def _row_stockout(row) -> float:
        sigma = row["demand_sigma_log"]
        if pd.isna(sigma):
            return float("nan")
        rng = np.random.default_rng(risk_params.seed)
        return simulate_item_risk(
            float(row["demand_point"]), float(row["our_order"]),
            risk_params, rng, demand_sigma_log=float(sigma),
        ).p_stockout

    out = preds.rename(columns={"demand_point": demand_col}).assign(
        stockout_prob=preds.apply(_row_stockout, axis=1).to_numpy(),
        recommended_production=preds["our_order"].round(0).to_numpy(),
        model=model_name,
    )
    out[demand_col] = out[demand_col].round(2)
    out["stockout_prob"] = out["stockout_prob"].round(4)
    cols = [
        "store_id", "item_id", "category_id", "date",
        demand_col, "stockout_prob", "recommended_production", "model",
    ]
```

주의: `_row_stockout`이 `preds`(rename 전, `demand_point`/`our_order`/`demand_sigma_log` 컬럼 보유)를 참조하도록 `out` assign에서 `preds.apply(...)`로 계산한다(`out = preds.rename(...)`은 새 프레임이라 `preds`는 원본 컬럼 유지).

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_predict_next_week.py --color=no`
Expected: finite-σ 신규 테스트 + 갱신된 lightgbm-NaN 테스트 모두 PASS.

- [ ] **Step 5: 전체 스위트 회귀**

Run: `uv run pytest --color=no`
Expected: 전체 PASS (516+ 신규 추가분). backward-compat로 기존 decision/pipeline/prospective 테스트 무회귀.

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/cli.py tests/test_predict_next_week.py
git commit -m "feat(cli): predict-next-week category가 σ(x)로 stockout_prob 채움 (c-2b)

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01DtB1FdcnGrW4fRxTGjgHWz')"
```

---

### Task 5: 광교 실측 smoke + 3축 리뷰

**Files:** (검증 전용, 코드 변경 없음)

- [ ] **Step 1: 실측 smoke (distributional σ 배열 end-to-end)**

Run:
```bash
uv run bakery predict-next-week --source real --order-level category \
  --total-model distributional --out-dir reports
```
Expected: 231행(7일×33 item) 내외, `stockout_prob` 유한(0<p<1), `model=category_total:distributional`.
확인:
```bash
uv run python -c "import pandas as pd; d=pd.read_csv('reports/next_week_predictions.csv'); print(d['stockout_prob'].describe()); print(d['stockout_prob'].isna().sum(), 'NaN')"
```
Expected: NaN 0개, min>0, max<1.

- [ ] **Step 2: lightgbm smoke (σ 없음 → NaN 유지)**

Run:
```bash
uv run bakery predict-next-week --source real --order-level category \
  --total-model lightgbm --out-dir reports
```
Expected: `stockout_prob` 전부 NaN(σ 없음), 정상 종료.

- [ ] **Step 3: 3축 리뷰**

`/review-triple`로 변경분(risk.py / pipeline.py / cli.py) 재사용성·품질·효율 점검. 핵심 확인:
- backward-compat 경로가 정말 값 불변인지(σ=None 분기).
- broadcast σ의 저해상도 한계가 docstring/spec에 정직히 남았는지.
- magic 값 없는지(RiskParams 기본값 사용).

- [ ] **Step 4: 결과 보고**

smoke 수치(stockout_prob 분포)와 리뷰 요약을 사용자에게 보고. PR 생성 여부 확인.

---

## Self-Review

**1. Spec coverage:**
- risk.py LogNormal 경로 → Task 1 ✅
- pipeline.py σ 스루 → Task 2 ✅
- `_category_base_predict` σ 반환 → Task 3 ✅
- `_category_future_order_predictions` σ broadcast → Task 3 ✅
- `_predict_next_week_category` stockout_prob 채우기 → Task 4 ✅
- 8-col 스키마 불변 → Task 4 테스트 핀 ✅
- backward-compat → Task 1(None==default)·Task 2(NaN fallback) 테스트 ✅
- leakage 무회귀 → Task 3 Step 6·Task 4 Step 5 ✅
- 정직한 한계(저해상도) → spec + Task 5 리뷰 확인 ✅
- 비목표(p_waste CSV 미추가·item σ 미학습·conformal 미적용) → 플랜에 미포함 ✅

**2. Placeholder scan:** 모든 step에 실제 코드/명령/기대출력 포함. TODO/TBD 없음.

**3. Type consistency:**
- `demand_sigma_log: float | None` — Task 1(인자)·Task 2(컬럼→인자)·Task 3(컬럼 산출)·Task 4(소비) 전부 동일 이름·타입.
- `_category_base_predict` 3-tuple `(base_median, base_prod, base_sigma)` — Task 3 정의·호출부 2곳(fold `_`, future 수신) 일치.
- `simulate_item_risk` 시그니처 — Task 1 정의, Task 2·Task 4 호출 시 keyword-only `demand_sigma_log` 일치.

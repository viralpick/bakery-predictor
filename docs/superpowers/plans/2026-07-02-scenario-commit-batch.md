# 다품목 배치 scenario-commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 여러 품목에 같은 드라이버 시나리오를 한 번에 재예측→조정발주→게이트→커밋하되, 모델 fit을 1회만 공유한다.

**Architecture:** `what_if_driver`의 품목별 뒷단을 `_what_if_for_item` 코어로 추출하고, `what_if_driver_batch`가 validate/enriched/fit을 1회 공유한 뒤 품목마다 코어 호출(실패 skip). `loop.py`에 `run_scenario_commit_batch`, `cli.py`에 `scenario-commit-batch` 커맨드. 기존 단일 경로(`what_if_driver`, `run_scenario_commit`)는 동작 불변.

**Tech Stack:** Python 3.12, pandas, LightGBM, typer, pytest, uv. 스펙: `docs/superpowers/specs/2026-07-02-scenario-commit-batch-design.md`

## Global Constraints

- fit 공유: `what_if_driver_batch`는 N품목이어도 `_fit_demand_model`을 **1회만** 호출 (스펙 D1)
- 배치=단일 결과 동일: 배치의 각 품목 결과 == 단일 `what_if_driver` 결과 (스펙 D3)
- 품목 실패 = skip + `log.warning`, 배치 중단 안 함 (스펙 D4)
- 품목 지정: CLI `--items "a,b"` 명시 or 생략 시 매장 전 품목; 매장 품목 추출은 CLI에서, `run_scenario_commit_batch`는 순수 리스트만 받음 (스펙 D5)
- 결정론(LLM 미개입), `train_cutoff` caller 주입(fit은 cutoff 이전만), 단일 store (스펙 D6)
- 기존 `what_if_driver`/`run_scenario_commit` 동작 불변 (스펙 D2) — 기존 테스트가 green이어야 함
- 함수 30줄 이내, guard clause 우선 (글로벌 code-quality 규칙)
- WhatIfDriverResult 필드: store_id, item_id, driver_overrides, before_demand, after_demand, demand_delta, before_p_stockout, after_p_stockout, before_expected_cost, after_expected_cost, out_of_support, propagation_path

---

### Task 1: scenario.py — 코어 추출 + `what_if_driver_batch` (fit 공유)

**Files:**
- Modify: `src/bakery/ontology/scenario.py` (로거 추가, `_what_if_for_item` 추출, `what_if_driver` 리팩토링, `what_if_driver_batch` 신규)
- Test: `tests/test_scenario.py` (append)

**Interfaces:**
- Consumes: `_build_enriched`, `_fit_demand_model`, `_period_item_rows`, `_predict_demand`, `_count_support`, `_propagation_path`, `_validate_drivers`, `simulate_item_risk`, `apply_policy`, `WhatIfDriverResult` (all existing in scenario.py)
- Produces:
  - `_what_if_for_item(enriched, model, store_id, item_id, period, driver_overrides, *, base_order, risk, policy) -> WhatIfDriverResult`
  - `what_if_driver_batch(daily, calendar, weather, store_id, item_ids, period, driver_overrides, *, base_order=None, train_cutoff, feature_set="v2", risk=RiskParams(), policy=PolicyParams()) -> list[WhatIfDriverResult]` — Task 2가 소비

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scenario.py` 하단에 추가. 상단 import는 이미 `from bakery.ontology import scenario as sc`, `_build_enriched`, `_fit_demand_model`, `_cutoff_and_period` 헬퍼, `dataset` fixture가 있으므로 그대로 사용:

```python
def _two_items(dataset, store_id):
    sub = dataset.daily[dataset.daily["store_id"] == store_id]
    return list(sub["item_id"].drop_duplicates())[:2]


def test_batch_fits_model_once(dataset, monkeypatch):
    """fit 공유: N품목이어도 _fit_demand_model 1회만."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = sorted(dataset.daily["store_id"].unique())[0]
    items = _two_items(dataset, store)

    calls = {"n": 0}
    real_fit = sc._fit_demand_model

    def counting_fit(*a, **k):
        calls["n"] += 1
        return real_fit(*a, **k)

    monkeypatch.setattr(sc, "_fit_demand_model", counting_fit)
    results = sc.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store, items, period,
        {"is_rain": 1}, train_cutoff=cutoff)
    assert calls["n"] == 1                    # ← fit shared, not per-item
    assert len(results) == len(items)


def test_batch_matches_single(dataset):
    """배치=단일: 각 품목 결과가 단일 what_if_driver와 동일."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = sorted(dataset.daily["store_id"].unique())[0]
    items = _two_items(dataset, store)

    batch = {r.item_id: r for r in sc.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store, items, period,
        {"is_rain": 1}, train_cutoff=cutoff)}
    for item in items:
        single = sc.what_if_driver(
            dataset.daily, dataset.calendar, dataset.weather, store, item, period,
            {"is_rain": 1}, train_cutoff=cutoff)
        assert batch[item].before_demand == single.before_demand
        assert batch[item].after_demand == single.after_demand


def test_batch_skips_unknown_item(dataset):
    """품목 실패는 skip, 나머지 정상."""
    enriched = _build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = sorted(dataset.daily["store_id"].unique())[0]
    good = _two_items(dataset, store)[0]

    results = sc.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store,
        [good, "NONEXISTENT_ITEM"], period, {"is_rain": 1}, train_cutoff=cutoff)
    ids = {r.item_id for r in results}
    assert good in ids
    assert "NONEXISTENT_ITEM" not in ids       # skipped, no crash
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_scenario.py -q -k "batch"`
Expected: FAIL — `AttributeError: module ... has no attribute 'what_if_driver_batch'`

- [ ] **Step 3: 구현**

`src/bakery/ontology/scenario.py` 상단 import 블록(라인 19 `from ..models.lightgbm_regressor import GlobalLGBM` 아래)에 로거 추가:

```python
import logging

log = logging.getLogger(__name__)
```
(`import logging`은 `from __future__` 아래, 기존 `from dataclasses import dataclass` 위 stdlib 위치에 두고, `log = ...`는 import 블록 끝에.)

기존 `what_if_driver` 함수(현재 body: validate→enriched→fit→base_rows→...→return)를 아래 3개로 교체. `_what_if_for_item`은 기존 base_rows 이후 로직을 그대로 옮긴 것:

```python
def _what_if_for_item(enriched, model, store_id, item_id, period, driver_overrides,
                      *, base_order, risk, policy) -> WhatIfDriverResult:
    """Per-item what-if on a pre-fitted model (batch shares one fit). Ceteris paribus:
    only driver columns are overridden; lag/rolling recomputed from history."""
    base_rows = _period_item_rows(enriched, store_id, item_id, period)
    before_demand = _predict_demand(model, base_rows)
    if base_order is None:
        base_order = apply_policy(item_id, before_demand, policy)[0]
    pert_rows = base_rows.copy()
    for col, val in driver_overrides.items():
        pert_rows[col] = val
    after_demand = _predict_demand(model, pert_rows)
    before = simulate_item_risk(before_demand, base_order, risk)
    after = simulate_item_risk(after_demand, base_order, risk)
    return WhatIfDriverResult(
        store_id=store_id, item_id=item_id, driver_overrides=dict(driver_overrides),
        before_demand=before_demand, after_demand=after_demand,
        demand_delta=after_demand - before_demand,
        before_p_stockout=before.p_stockout, after_p_stockout=after.p_stockout,
        before_expected_cost=before.expected_cost, after_expected_cost=after.expected_cost,
        out_of_support=_count_support(enriched, store_id, driver_overrides) == 0,
        propagation_path=_propagation_path(driver_overrides),
    )


def what_if_driver(daily, calendar, weather, store_id, item_id, period,
                   driver_overrides, *, base_order: float | None = None, train_cutoff,
                   feature_set: str = "v2", risk: RiskParams = RiskParams(),
                   policy: PolicyParams = PolicyParams()) -> WhatIfDriverResult:
    """Upstream Scenario lever: perturb driver(s) → re-forecast demand → propagate
    to stockout risk/cost. Read-only. before/after share one fitted model; only the
    driver columns differ (ceteris paribus). train_cutoff is caller-injected (leakage).
    base_order=None derives the order internally via apply_policy(before_demand, policy)
    — pass the same policy used downstream (e.g. run_scenario_commit) so the whatif
    risk numbers stay consistent with the eventually committed order."""
    _validate_drivers(driver_overrides)
    enriched = _build_enriched(daily, calendar, weather)
    model = _fit_demand_model(enriched, train_cutoff, feature_set)
    return _what_if_for_item(enriched, model, store_id, item_id, period,
                             driver_overrides, base_order=base_order, risk=risk, policy=policy)


def what_if_driver_batch(daily, calendar, weather, store_id, item_ids, period,
                         driver_overrides, *, base_order: float | None = None,
                         train_cutoff, feature_set: str = "v2",
                         risk: RiskParams = RiskParams(),
                         policy: PolicyParams = PolicyParams()) -> list[WhatIfDriverResult]:
    """Multi-item what-if sharing ONE fit (fit is item-independent). Failing items
    (e.g. no rows in period) are logged and skipped; only successes returned."""
    _validate_drivers(driver_overrides)
    enriched = _build_enriched(daily, calendar, weather)
    model = _fit_demand_model(enriched, train_cutoff, feature_set)
    out: list[WhatIfDriverResult] = []
    for item_id in item_ids:
        try:
            out.append(_what_if_for_item(enriched, model, store_id, item_id, period,
                                         driver_overrides, base_order=base_order,
                                         risk=risk, policy=policy))
        except Exception as exc:                     # per-item guard: keep the batch alive
            log.warning("scenario batch: skip item %s (%s)", item_id, exc)
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_scenario.py -q`
Expected: 전부 PASS (기존 what_if_driver 테스트 포함 — 리팩토링 후 동작 불변 증명)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/scenario.py tests/test_scenario.py
git commit -m "feat: what_if_driver_batch — 다품목 fit 공유 (코어 _what_if_for_item 추출)"
```

---

### Task 2: loop.py — `run_scenario_commit_batch`

**Files:**
- Modify: `src/bakery/ontology/loop.py` (`run_scenario_commit_batch` 추가; `run_scenario_commit` 무변경)
- Test: `tests/test_closed_loop.py` (append)

**Interfaces:**
- Consumes: `scenario.what_if_driver_batch` (Task 1), `apply_policy`, `OrderProposal`, `ScenarioCommitResult`, `APPROVE`, `WritebackStore`, `GatePolicy` (existing in loop.py)
- Produces: `run_scenario_commit_batch(dataset, store_id, item_ids, period, driver_overrides, writeback, gate, *, now, train_cutoff, policy=PolicyParams(), risk=RiskParams()) -> list[ScenarioCommitResult]` — Task 3가 소비

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py` 하단에 추가. import 라인(현재 `from bakery.ontology.loop import run_scenario_commit, ScenarioCommitResult, auto_approve, human_correct`)에 `run_scenario_commit_batch` 추가:

```python
from bakery.ontology.loop import (
    run_scenario_commit, run_scenario_commit_batch, ScenarioCommitResult,
    auto_approve, human_correct,
)
```

테스트 (기존 test_scenario_commit 패턴의 store/item/period 준비 방식을 따름 — dataset fixture 재사용):

```python
def _batch_ctx(dataset):
    import pandas as pd
    store = sorted(dataset.daily["store_id"].unique())[0]
    sub = dataset.daily[dataset.daily["store_id"] == store]
    items = list(sub["item_id"].drop_duplicates())[:2]
    dates = pd.to_datetime(sub["date"]).sort_values().unique()
    cutoff = str(pd.Timestamp(dates[-3]).date())
    period = (str(pd.Timestamp(dates[-2]).date()), str(pd.Timestamp(dates[-1]).date()))
    return store, items, period, cutoff


def test_scenario_commit_batch_commits_each_item(dataset):
    from bakery.ontology.writeback import WritebackStore
    store, items, period, cutoff = _batch_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    results = run_scenario_commit_batch(
        dataset, store, items, period, {"is_rain": 1}, wb, auto_approve,
        now=f"{period[0]}T09:00:00", train_cutoff=cutoff)
    assert len(results) == len(items)
    assert all(isinstance(r, ScenarioCommitResult) for r in results)
    assert all(r.committed.status == "APPROVED" for r in results)
    assert len(wb.records) == len(items)


def test_scenario_commit_batch_requires_approval_gate(dataset):
    from bakery.ontology.writeback import WritebackStore
    store, items, period, cutoff = _batch_ctx(dataset)
    wb = WritebackStore(require_approval=False)
    with pytest.raises(ValueError):
        run_scenario_commit_batch(
            dataset, store, items, period, {"is_rain": 1}, wb, auto_approve,
            now=f"{period[0]}T09:00:00", train_cutoff=cutoff)
```

(`pytest`는 test_closed_loop.py 상단에 이미 import돼 있음; 없으면 추가.)

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py -q -k "batch"`
Expected: FAIL — `ImportError: cannot import name 'run_scenario_commit_batch'`

- [ ] **Step 3: 구현**

`src/bakery/ontology/loop.py`의 `run_scenario_commit` 함수 바로 뒤에 추가:

```python
def run_scenario_commit_batch(dataset: DailyDataset, store_id: str, item_ids: list,
                              period: tuple[str, str], driver_overrides: dict,
                              writeback: WritebackStore, gate: GatePolicy, *, now: str,
                              train_cutoff: str, policy: PolicyParams = PolicyParams(),
                              risk: RiskParams = RiskParams()) -> list[ScenarioCommitResult]:
    """Multi-item scenario → adjusted order → gate → commit, sharing ONE model fit.
    Deterministic (no LLM). Failing items skipped by what_if_driver_batch."""
    if not writeback.require_approval:
        raise ValueError(
            "scenario-commit-batch drives approval via the GatePolicy; "
            "pass WritebackStore(require_approval=True)")
    wifs = scenario.what_if_driver_batch(
        dataset.daily, dataset.calendar, dataset.weather, store_id, item_ids, period,
        driver_overrides, base_order=None, train_cutoff=train_cutoff, risk=risk,
        policy=policy)
    out: list[ScenarioCommitResult] = []
    drivers_str = ", ".join(f"{k}={v}" for k, v in driver_overrides.items())
    for wif in wifs:
        base_order = apply_policy(wif.item_id, wif.before_demand, policy)[0]
        scenario_order = apply_policy(wif.item_id, wif.after_demand, policy)[0]
        rationale = (f"scenario [{drivers_str}]: demand {wif.before_demand:.1f}→"
                     f"{wif.after_demand:.1f}, order {base_order:.0f}→{scenario_order:.0f}")
        proposal = OrderProposal(wif.item_id, scenario_order, rationale)
        rec = writeback.propose_order(store_id, wif.item_id, period[0], scenario_order,
                                      proposed_at=now)
        decision = gate(proposal)
        if decision.action == APPROVE:
            rec = writeback.approve(rec.record_id, decision.approver,
                                    approved_at=now, approved_qty=decision.approved_qty)
        else:
            rec = writeback.reject(rec.record_id, decision.approver)
        out.append(ScenarioCommitResult(whatif=wif, base_order=base_order, committed=rec))
    return out
```

Note: 커밋 블록(apply_policy→propose→gate→commit)이 단일 `run_scenario_commit`과 유사하다. 스펙 D2가 단일 무변경을 요구하므로 배치는 자체 루프로 구현한다. `_commit_one` 헬퍼 추출은 후속 리팩토링 후보(양쪽 공유). 기존 test_scenario_commit 테스트가 단일 동작 불변을 보장한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -q`
Expected: 전부 PASS (기존 run_scenario_commit 테스트 포함)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/loop.py tests/test_closed_loop.py
git commit -m "feat: run_scenario_commit_batch — 다품목 시나리오→게이트→커밋 (fit 공유)"
```

---

### Task 3: cli.py — `scenario-commit-batch` 커맨드

**Files:**
- Modify: `src/bakery/cli.py` (`scenario-commit` 커맨드 뒤에 추가)
- Test: `tests/test_cli_helpers.py` (append — typer 인트로스펙션)

**Interfaces:**
- Consumes: `run_scenario_commit_batch` (Task 2), `_parse_period`, `_parse_drivers`, `_write_and_label`, `_select_gate_policy` (existing in cli.py), `load_dataset`, `WritebackStore`
- Produces: typer command `scenario-commit-batch` with params store/period/drivers/items/gate/source/now/out

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_helpers.py` 하단에 추가:

```python
def test_scenario_commit_batch_command_registered():
    import typer
    import bakery.cli as c
    group = typer.main.get_group(c.app)
    cmd = group.get_command(None, "scenario-commit-batch")
    assert cmd is not None
    opts = [p.name for p in cmd.params]
    assert "items" in opts
    assert "gate" in opts
    assert "policy" not in opts
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_cli_helpers.py -q -k "batch_command"`
Expected: FAIL — `cmd is None` (커맨드 미등록)

- [ ] **Step 3: 구현**

`src/bakery/cli.py`의 `cmd_scenario_commit` 함수 전체 뒤(다음 `def _` 또는 `if __name__` 앞)에 추가:

```python
@app.command("scenario-commit-batch")
def cmd_scenario_commit_batch(
    store: str,
    period: str,                        # "YYYY-MM-DD,YYYY-MM-DD"
    drivers: str,                       # "is_rain=1,is_snow=0"
    items: str = "",                    # "a,b,c"; 비면 매장 전 품목
    gate: str = "human",               # auto(frontier) | human(rubber-stamp)
    source: str = "synthetic",
    now: str = "",                      # ISO; 비면 period start의 09:00
    out: str = "",                      # parquet 경로(옵션)
) -> None:
    """v7 다품목 Scenario→commit: 여러 품목에 같은 드라이버 시나리오를 배치 커밋.

    모델 fit 1회 공유. --items 생략 시 해당 매장 전 품목. 결정론(LLM 미개입).
    --source synthetic 이면 메커니즘 시연.
    """
    from .data.loader import load_dataset
    from .ontology.loop import run_scenario_commit_batch
    from .ontology.writeback import WritebackStore

    start, end, stamp = _parse_period(period, now)
    gate_policy = _select_gate_policy(gate)
    driver_overrides = _parse_drivers(drivers)
    dataset = load_dataset(source)
    item_ids = ([s.strip() for s in items.split(",")] if items
                else sorted(dataset.daily.loc[dataset.daily["store_id"] == store,
                                              "item_id"].unique()))
    wb = WritebackStore(require_approval=True)
    results = run_scenario_commit_batch(
        dataset, store, item_ids, (start, end), driver_overrides, wb, gate_policy,
        now=stamp, train_cutoff=start)

    console.print(f"[bold]scenario-commit-batch[/] store={store} "
                  f"items={len(item_ids)} drivers={driver_overrides}")
    for res in results:
        w = res.whatif
        console.print(f"  {w.item_id}: demand {w.before_demand:.1f}→{w.after_demand:.1f} "
                      f"order {res.base_order:.0f}→{res.committed.proposed_qty:.0f} "
                      f"{res.committed.status} by={res.committed.approver}")
    if not results:
        console.print("[yellow]no committed items (all skipped?)[/]")
    _write_and_label(wb, out, source)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_cli_helpers.py -q`
Expected: 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/cli.py tests/test_cli_helpers.py
git commit -m "feat: scenario-commit-batch CLI — 다품목 배치 커맨드 (--items or 전체)"
```

---

### Task 4: 전체 회귀 + CLI 스모크 + TODO

**Files:**
- Modify: `TODO.md`
- Test: 전체 스위트 + CLI 실행

**Interfaces:**
- Consumes: Task 1~3 전체

- [ ] **Step 1: 전체 테스트**

Run: `uv run pytest -q`
Expected: 전부 PASS (기준선 295 + 신규 ~8). 실패 시 실패 테스트 단위로 원인 수정 (3회 내 미해결 시 사용자 보고).

- [ ] **Step 2: CLI 스모크 — 명시 품목**

먼저 synthetic 품목 2개 확보:
```bash
ITEMS=$(uv run python -c "from bakery.data.loader import load_dataset; d=load_dataset('synthetic').daily; s=sorted(d.store_id.unique())[0]; print(','.join(list(d[d.store_id==s]['item_id'].drop_duplicates())[:2]))")
uv run bakery scenario-commit-batch store_A 2024-06-01,2024-06-07 "is_rain=1" --items "$ITEMS" --gate human
```
Expected: 헤더 + 2개 품목 각 한 줄(demand/order/APPROVED) + synthetic 라벨. 에러 없음.

- [ ] **Step 3: CLI 스모크 — 전체 품목(생략)**

```bash
uv run bakery scenario-commit-batch store_A 2024-06-01,2024-06-07 "is_rain=1" --gate human
```
Expected: `items=N`(매장 전 품목 수) + 품목별 라인들. 에러 없음.

- [ ] **Step 4: TODO 갱신**

`TODO.md`의 "다품목 배치 scenario-commit" 항목을 완료로:
```markdown
- [x] 다품목 배치 scenario-commit — what_if_driver_batch(fit 공유) + run_scenario_commit_batch + CLI scenario-commit-batch.
```

- [ ] **Step 5: Commit**

```bash
git add TODO.md
git commit -m "docs: TODO — 다품목 배치 scenario-commit 완료 체크"
```

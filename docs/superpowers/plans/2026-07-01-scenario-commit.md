# Scenario→commit closed-loop (v7 S7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 가상 시나리오(드라이버) 하에서 조정된 발주량을 계산해 사람 승인 게이트를 통과시켜 확정하는 `run_scenario_commit` orchestrator + `scenario-commit` CLI를 구현한다 (상류 S6 + 하류 S5 결합).

**Architecture:** 결정론 orchestrator(LLM 미개입). `scenario.what_if_driver`(S6)로 before/after demand를 재예측 → `apply_policy`로 base/scenario 발주량 산출 → `OrderProposal`을 `WritebackStore`+`GatePolicy`(S5)로 propose→gate→commit. 순환 제거를 위해 `what_if_driver`의 `base_order`를 optional(None→내부 `apply_policy(before)`)로 확장한다.

**Tech Stack:** Python, pandas, dataclasses, typer(CLI), pytest. 재사용: `ontology/scenario.what_if_driver`, `decision.apply_policy`/`PolicyParams`/`RiskParams`, `ontology/writeback`(WritebackStore/OrderRecord), `ontology/loop`(OrderProposal/GatePolicy/APPROVE/gate 정책), `cli._select_gate_policy`.

## Global Constraints

- **결정론 / leakage 계승**: 타임스탬프는 caller 주입(`now`), `train_cutoff`도 caller 주입(fit은 cutoff 이전만 — S6 규칙). LGBM seed 고정. `datetime.now()` 호출 금지.
- **LLM 미개입 / 쓰기 게이트**: 이 경로는 결정론이며 LLM 도구 surface에 노출하지 않는다(S5 "LLM read-only, 쓰기는 게이트된 결정론" 계승). 게이트 단일 레버: `WritebackStore(require_approval=True)` + `GatePolicy`.
- **재사용만**: 신규 예측·정책 로직 없음. what_if_driver(S6) + apply_policy + writeback/gate(S5) 조립.
- **하위호환**: `what_if_driver`의 `base_order`를 optional로 바꿔도 기존 호출(base_order 명시)과 S6 테스트가 그대로 통과해야 한다.
- 기존 전체 테스트 불변(`uv run pytest`, 현재 main 기준 통과). 새 의존성 금지.
- 커밋 메시지 말미 트레일러:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01MkjMzDP7i4zJ1HJFUqbBcR`

---

### Task 1: what_if_driver `base_order` optional 확장 (D5)

순환 제거. `base_order`를 `float | None = None`로 바꾸고, None이면 `before_demand` 계산 직후 `apply_policy(item_id, before_demand)[0]`로 자동 산출한다.

**Files:**
- Modify: `src/bakery/ontology/scenario.py` (what_if_driver 시그니처 103-105, 본문 118 앞)
- Test: `tests/test_scenario.py`

**Interfaces:**
- Consumes: `apply_policy`(decision) — `apply_policy(item_id: str, demand_point: float, params=PolicyParams()) -> tuple[float, DecisionLineage]`.
- Produces: `what_if_driver(..., base_order: float | None = None, ...)` — None이면 내부에서 `apply_policy(item_id, before_demand)[0]`를 base_order로 사용. 반환 타입·필드 불변.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_scenario.py`에 추가 (기존 `sc`, `dataset`, `_cutoff_and_period`, `_StubModel` 재사용):

```python
def test_what_if_driver_base_order_none_uses_policy(dataset, monkeypatch):
    """base_order=None → 내부에서 apply_policy(before_demand) 사용; 명시 호출과 동일 위험."""
    from bakery.decision import apply_policy
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _StubModel())
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    cutoff, period = _cutoff_and_period(enriched)
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    auto = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                             store, item, period, {"is_rain": 1}, base_order=None,
                             train_cutoff=cutoff)
    # _StubModel: before demand = 10.0 → policy order
    policy_order = apply_policy(item, 10.0)[0]
    explicit = sc.what_if_driver(dataset.daily, dataset.calendar, dataset.weather,
                                 store, item, period, {"is_rain": 1}, base_order=policy_order,
                                 train_cutoff=cutoff)
    assert auto.before_p_stockout == explicit.before_p_stockout
    assert auto.after_expected_cost == explicit.after_expected_cost
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_scenario.py::test_what_if_driver_base_order_none_uses_policy -v`
Expected: FAIL — `base_order` is required (TypeError: missing keyword-only argument) 또는 None에 대한 산술 오류.

- [ ] **Step 3: 최소 구현**

`scenario.py` import에 `apply_policy` 추가 (기존 `from ..decision import RiskParams, simulate_item_risk` 줄을 확장):

```python
from ..decision import RiskParams, apply_policy, simulate_item_risk
```

what_if_driver 시그니처의 `base_order` 를 optional로:

```python
def what_if_driver(daily, calendar, weather, store_id, item_id, period,
                   driver_overrides, *, base_order: float | None = None, train_cutoff,
                   feature_set: str = "v2", risk: RiskParams = RiskParams()) -> WhatIfDriverResult:
```

`before_demand = _predict_demand(model, base_rows)` 다음 줄에 삽입 (기존 line 113 뒤):

```python
    if base_order is None:
        base_order = apply_policy(item_id, before_demand)[0]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: PASS (신규 + 기존 전부 — 기존 테스트는 base_order를 명시하므로 불변).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/scenario.py tests/test_scenario.py
git commit -m "feat: what_if_driver base_order optional (None → apply_policy(before_demand))"
```

---

### Task 2: `run_scenario_commit` orchestrator + `ScenarioCommitResult`

시나리오 → 조정 발주량 → 게이트 → writeback commit.

**Files:**
- Modify: `src/bakery/ontology/loop.py`
- Test: `tests/test_closed_loop.py`

**Interfaces:**
- Consumes: `scenario.what_if_driver`(Task 1, base_order optional); `apply_policy`/`PolicyParams`/`RiskParams`(decision); `WritebackStore`/`OrderRecord`(writeback); `OrderProposal`/`GatePolicy`/`APPROVE`(loop, 기존); `WhatIfDriverResult`(scenario).
- Produces:
  - `ScenarioCommitResult` (frozen): `whatif: WhatIfDriverResult`, `base_order: float`, `committed: OrderRecord`.
  - `run_scenario_commit(dataset, store_id, item_id, period, driver_overrides, writeback, gate, *, now, train_cutoff, policy=PolicyParams(), risk=RiskParams()) -> ScenarioCommitResult`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py`에 추가 (기존 `dataset` fixture, `RecommendFakeLLM` 있음; scenario stub는 신규):

```python
from bakery.ontology import scenario as sc
from bakery.ontology.loop import run_scenario_commit, ScenarioCommitResult, auto_approve, human_correct
from bakery.ontology.writeback import WritebackStore, APPROVED, REJECTED
from bakery.ontology.loop import APPROVE, REJECT, GateDecision


class _ScenarioStubModel:
    """before demand=10, after=7 when is_rain flipped to 1 (deterministic)."""
    def predict(self, target):
        import pandas as pd
        rain = float(target["is_rain"].iloc[0]) if "is_rain" in target.columns else 0.0
        return pd.Series([10.0 - 3.0 * rain] * len(target))


def _sc_ctx(dataset):
    import pandas as pd
    enriched = sc._build_enriched(dataset.daily, dataset.calendar, dataset.weather)
    dates = sorted(pd.to_datetime(enriched["date"]).dt.date.unique())
    cutoff = str(dates[-3]); period = (str(dates[-2]), str(dates[-1]))
    store = enriched["store_id"].iloc[0]
    item = enriched.loc[enriched["store_id"] == store, "item_id"].iloc[0]
    return store, item, period, cutoff


def test_scenario_commit_commits_adjusted_order(dataset, monkeypatch):
    from bakery.decision import apply_policy
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    res = run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, auto_approve,
                              now="2024-01-01T09:00:00", train_cutoff=cutoff)
    assert isinstance(res, ScenarioCommitResult)
    assert res.whatif.before_demand == 10.0 and res.whatif.after_demand == 7.0
    assert res.base_order == apply_policy(item, 10.0)[0]
    scenario_order = apply_policy(item, 7.0)[0]
    assert res.committed.status == APPROVED
    assert res.committed.approved_qty == scenario_order          # auto_approve → 제안대로


def test_scenario_commit_human_correction(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    res = run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb,
                              human_correct({item: 99.0}), now="2024-01-01T09:00:00",
                              train_cutoff=cutoff)
    assert res.committed.approved_qty == 99.0


def test_scenario_commit_reject(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    reject_gate = lambda proposal: GateDecision(REJECT, None, "human")
    res = run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, reject_gate,
                              now="2024-01-01T09:00:00", train_cutoff=cutoff)
    assert res.committed.status == REJECTED


def test_scenario_commit_rejects_autonomous_store(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=False)
    with pytest.raises(ValueError):
        run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, auto_approve,
                            now="2024-01-01T09:00:00", train_cutoff=cutoff)


def test_scenario_commit_rationale_describes_scenario(dataset, monkeypatch):
    monkeypatch.setattr(sc, "_fit_demand_model", lambda *a, **k: _ScenarioStubModel())
    store, item, period, cutoff = _sc_ctx(dataset)
    wb = WritebackStore(require_approval=True)
    run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb, auto_approve,
                        now="2024-01-01T09:00:00", train_cutoff=cutoff)
    # rationale lives on the proposal, not persisted; assert via a capturing gate
    captured = {}
    def cap_gate(p):
        captured["rationale"] = p.rationale
        return GateDecision(APPROVE, None, "human")
    wb2 = WritebackStore(require_approval=True)
    run_scenario_commit(dataset, store, item, period, {"is_rain": 1}, wb2, cap_gate,
                        now="2024-01-01T09:00:00", train_cutoff=cutoff)
    assert "scenario" in captured["rationale"] and "is_rain" in captured["rationale"]
```

(`pytest` import는 파일 상단에 이미 있음; 없으면 추가.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py::test_scenario_commit_commits_adjusted_order -v`
Expected: FAIL — `ImportError: cannot import name 'run_scenario_commit'`.

- [ ] **Step 3: 최소 구현**

`loop.py` import 영역에 추가:

```python
from . import scenario
from ..decision import PolicyParams, RiskParams, apply_policy
```

파일 끝에 추가:

```python
@dataclass(frozen=True)
class ScenarioCommitResult:
    whatif: "scenario.WhatIfDriverResult"
    base_order: float
    committed: OrderRecord


def run_scenario_commit(dataset: DailyDataset, store_id: str, item_id: str,
                        period: tuple[str, str], driver_overrides: dict,
                        writeback: WritebackStore, gate: GatePolicy, *, now: str,
                        train_cutoff: str, policy: PolicyParams = PolicyParams(),
                        risk: RiskParams = RiskParams()) -> ScenarioCommitResult:
    """Upstream scenario → adjusted order → human gate → writeback commit.
    Deterministic (no LLM). Reuses what_if_driver(S6) + apply_policy + writeback(S5)."""
    if not writeback.require_approval:
        raise ValueError(
            "scenario-commit drives approval via the GatePolicy; "
            "pass WritebackStore(require_approval=True)")
    wif = scenario.what_if_driver(
        dataset.daily, dataset.calendar, dataset.weather, store_id, item_id, period,
        driver_overrides, base_order=None, train_cutoff=train_cutoff, risk=risk)
    base_order = apply_policy(item_id, wif.before_demand, policy)[0]
    scenario_order = apply_policy(item_id, wif.after_demand, policy)[0]
    drivers_str = ", ".join(f"{k}={v}" for k, v in driver_overrides.items())
    rationale = (f"scenario [{drivers_str}]: demand {wif.before_demand:.1f}→{wif.after_demand:.1f}, "
                 f"order {base_order:.0f}→{scenario_order:.0f}")
    proposal = OrderProposal(item_id, scenario_order, rationale)
    rec = writeback.propose_order(store_id, item_id, period[0], scenario_order, proposed_at=now)
    decision = gate(proposal)
    if decision.action == APPROVE:
        rec = writeback.approve(rec.record_id, decision.approver,
                                approved_at=now, approved_qty=decision.approved_qty)
    else:
        rec = writeback.reject(rec.record_id, decision.approver)
    return ScenarioCommitResult(whatif=wif, base_order=base_order, committed=rec)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: PASS (신규 5개 + 기존 전부).

순환 import 점검 — Run: `uv run python -c "import bakery.ontology.loop; import bakery.ontology.scenario; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/ontology/loop.py tests/test_closed_loop.py
git commit -m "feat: run_scenario_commit orchestrator — scenario re-forecast → adjusted order → gate → writeback"
```

---

### Task 3: CLI `scenario-commit` 명령 + 드라이버 파서

**Files:**
- Modify: `src/bakery/cli.py` (`_select_gate_policy` 1187 근처, `if __name__` 1238 앞)
- Test: `tests/test_closed_loop.py`

**Interfaces:**
- Consumes: `run_scenario_commit`(Task 2); `_select_gate_policy`(cli, 기존); `WritebackStore`(writeback); `load_dataset`(loader).
- Produces:
  - `_parse_drivers(spec: str) -> dict[str, float]` — `"is_rain=1,is_snow=1"` → `{"is_rain":1.0,"is_snow":1.0}`; 형식 오류/빈 값 → ValueError.
  - typer 명령 `scenario-commit` — 옵션 `--store --item --period "start,end" --drivers "k=v,k=v" --policy auto|human --source(=synthetic) --now("") --out("")`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_closed_loop.py`에 추가:

```python
def test_cli_registers_scenario_commit_command():
    from bakery.cli import app
    names = {c.name for c in app.registered_commands}
    assert "scenario-commit" in names


def test_parse_drivers_maps_pairs():
    from bakery.cli import _parse_drivers
    assert _parse_drivers("is_rain=1,is_snow=0") == {"is_rain": 1.0, "is_snow": 0.0}
    assert _parse_drivers(" is_public_holiday = 1 ") == {"is_public_holiday": 1.0}
    import pytest
    with pytest.raises(ValueError):
        _parse_drivers("is_rain")        # no '='
    with pytest.raises(ValueError):
        _parse_drivers("")               # empty
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_closed_loop.py::test_cli_registers_scenario_commit_command -v`
Expected: FAIL — `assert 'scenario-commit' in names`.

- [ ] **Step 3: 최소 구현**

`src/bakery/cli.py`의 `closed-loop` 명령 뒤, `if __name__` 앞에 추가:

```python
def _parse_drivers(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad driver spec: {part!r} (expected key=value)")
        key, val = part.split("=", 1)
        out[key.strip()] = float(val.strip())
    if not out:
        raise ValueError("no drivers parsed; expected e.g. 'is_rain=1,is_snow=0'")
    return out


@app.command("scenario-commit")
def cmd_scenario_commit(
    store: str,
    item: str,
    period: str,                        # "YYYY-MM-DD,YYYY-MM-DD"
    drivers: str,                       # "is_rain=1,is_snow=0"
    policy: str = "human",              # auto(frontier) | human(rubber-stamp)
    source: str = "synthetic",
    now: str = "",                      # ISO; 비면 period start의 09:00
    out: str = "",                      # parquet 경로(옵션)
) -> None:
    """v7 Scenario→commit: 가상 드라이버 시나리오 하 조정 발주량을 사람 게이트 통과해 확정.

    상류 what_if_driver(재예측) + 하류 writeback(게이트)를 잇는 결정론 closed-loop.
    --source synthetic 이면 시연용. 정확도 주장이 아니라 메커니즘 시연.
    """
    from .data.loader import load_dataset
    from .ontology.loop import run_scenario_commit
    from .ontology.writeback import WritebackStore

    start, end = (s.strip() for s in period.split(","))
    stamp = now or f"{start}T09:00:00"
    gate = _select_gate_policy(policy)
    driver_overrides = _parse_drivers(drivers)
    dataset = load_dataset(source)
    wb = WritebackStore(require_approval=True)
    res = run_scenario_commit(dataset, store, item, (start, end), driver_overrides,
                              wb, gate, now=stamp, train_cutoff=start)

    w = res.whatif
    console.print(f"[bold]scenario-commit[/] store={store} item={item} drivers={driver_overrides}")
    console.print(f"  demand {w.before_demand:.1f} → {w.after_demand:.1f} (Δ{w.demand_delta:+.1f})"
                  + ("  [yellow]out-of-support[/]" if w.out_of_support else ""))
    console.print(f"  order {res.base_order:.0f} → {res.committed.proposed_qty:.0f}  "
                  f"{res.committed.status} qty={res.committed.approved_qty} by={res.committed.approver}")
    if out:
        wb.to_parquet(out)
        console.print(f"[green]wrote[/] {out} ({len(wb.records)} records)")
    console.print(
        "[yellow]synthetic 메커니즘 시연 (mechanism demo, not accuracy)[/]"
        if source == "synthetic" else f"[cyan]source={source}[/]")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_closed_loop.py -v`
Expected: PASS (신규 2개 + 기존 전부).

전체 회귀 — Run: `uv run pytest -q`
Expected: 전부 통과, 실패 0 (live 1 skip).

- [ ] **Step 5: 수동 smoke (선택, 키 불필요 — LLM 미개입)**

Run: `uv run bakery scenario-commit --store 광교 --item <실item> --period 2024-01-01,2024-01-07 --drivers "is_rain=1" --policy human --out /tmp/sc.parquet`
Expected: demand before→after, order base→scenario, APPROVED 출력 + parquet. (period가 데이터 마지막 구간이 아니면 lag 미해결로 demand 0 가능 — 설계 기인지.)

- [ ] **Step 6: 커밋**

```bash
git add src/bakery/cli.py tests/test_closed_loop.py
git commit -m "feat: scenario-commit CLI — end-to-end scenario → gated writeback commit"
```

---

## Self-Review

**Spec coverage:**
- D1(결정론 orchestrator, LLM 미노출) → Task 2 (LLM 없음), CLI도 LLM 없음. ✓
- D2(재사용만) → Task 2가 what_if_driver+apply_policy+writeback 조립, 신규 모델링 0. ✓
- D3(결정론·leakage·게이트 단일레버) → now/train_cutoff 주입, require_approval=True 가드(Task 2). ✓
- D4(발주량=apply_policy(after), base=apply_policy(before)) → Task 2. ✓
- D5(base_order optional) → Task 1. ✓
- 컴포넌트(ScenarioCommitResult, run_scenario_commit, CLI) → Task 2·3. ✓
- 에러/엣지(require_approval=False, 거부, 드라이버 파싱) → Task 2·3 테스트. ✓
- 테스트(stub 모델 결정론, 게이트 3종, rationale) → Task 2·3. ✓
- 범위 밖(LLM 자율선택/다품목/세션캐시) → 구현 안 함. ✓

**Placeholder scan:** 없음 — 모든 step 실제 코드/명령/기대출력.

**Type consistency:** `run_scenario_commit(dataset, store_id, item_id, period, driver_overrides, writeback, gate, *, now, train_cutoff, policy, risk) -> ScenarioCommitResult(whatif, base_order, committed)` — Task 2 정의 ↔ Task 3 호출 일치. `what_if_driver(..., base_order: float|None=None, ...)` Task 1 ↔ Task 2 `base_order=None` 호출 일치. `apply_policy(item_id, demand, policy)->(qty, lineage)` 실제 시그니처 일치. `_select_gate_policy`/`_parse_drivers`/`GateDecision(action, approved_qty, approver)`/`OrderProposal(item_id, qty, rationale)`/writeback 메서드 시그니처 기존과 일치.

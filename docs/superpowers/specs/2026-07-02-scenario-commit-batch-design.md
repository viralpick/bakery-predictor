# 다품목 배치 scenario-commit (v7) — Design

**Date**: 2026-07-02
**Depends on**: S6 what_if_driver (scenario.what_if_driver, WhatIfDriverResult, _build_enriched/_fit_demand_model/_period_item_rows/_predict_demand), S7 run_scenario_commit (loop.py, ScenarioCommitResult), decision.apply_policy, S5 writeback/GatePolicy, CLI 헬퍼(_parse_period/_parse_drivers/_write_and_label)
**Status**: Design approved (2026-07-02)

## 목적

현재 `run_scenario_commit`은 단일 `(store, item)`만 처리한다. 여러 품목에 대해 같은 드라이버 시나리오(예: "비 오는 날")를 한 번에 재예측→조정발주→게이트→커밋한다. "비 오는 날 전 품목 발주를 어떻게 조정할지"라는 실무 배치 시나리오에 직결된다.

핵심은 **fit 공유**다. `what_if_driver`는 내부에서 `_fit_demand_model`을 매번 호출하므로, 품목 N개를 순진하게 루프 돌리면 모델을 N번 재학습해 배치가 느려진다. 배치의 진짜 일은 "모델 1회 fit → 품목마다 predict"로 재조립하는 것이다.

## 핵심 설계 결정 (LOCKED)

### D1. 접근 A — 코어 추출 + 배치 헬퍼 (fit 공유)
`what_if_driver`의 품목별 뒷단을 `_what_if_for_item` 코어로 추출하고, 배치 진입점 `what_if_driver_batch`가 `_validate_drivers`/`_build_enriched`/`_fit_demand_model`을 **1회** 수행한 뒤 품목마다 코어를 호출한다. 기존 `what_if_driver`(단일)도 코어를 재사용하는 얇은 래퍼로 리팩토링 — 동작 불변, 하위호환. 배치 전용 predict 복제(접근 B)나 what_if_driver에 optional model/enriched 주입(접근 C)은 로직 중복·인자 오염으로 기각.

### D2. 레이어 분리 유지
- **scenario.py** (재예측): `_what_if_for_item`, `what_if_driver`(단일), `what_if_driver_batch`(다품목).
- **loop.py** (게이트 커밋): `run_scenario_commit`(단일), `run_scenario_commit_batch`(다품목).
- **cli.py**: 새 커맨드 `scenario-commit-batch`. 기존 단일 `scenario-commit`은 무변경.

### D3. 배치=단일 결과 동일 (불변식)
배치의 품목별 결과는 단일 `what_if_driver`를 그 품목에 대해 부른 것과 **동일**해야 한다(fit seed 고정, 같은 코어 공유이므로 자동 보장). 테스트로 못박는다. 이것이 fit 공유가 정확성을 해치지 않는다는 보증이다.

### D4. 품목 실패 = skip + 경고 (배치 강건성)
품목 하나가 실패(예: 해당 기간 데이터 없음 → `_period_item_rows`가 raise)해도 배치를 중단하지 않는다. `what_if_driver_batch`가 품목별로 try/except하여 실패 품목은 `log.warning` + skip, 성공 결과만 반환한다. `run_closed_loop`의 skip 패턴과 동일.

### D5. 품목 지정 = 명시 리스트 + 전체 기본
CLI `--items "a,b"` 주면 해당 품목만, 생략하면 해당 매장 전 품목(dataset.daily에서 추출). 매장 품목 추출은 CLI에서 1회 수행해 `run_scenario_commit_batch`에 명시 리스트로 전달한다(loop 함수는 순수 리스트만 받음 — 데이터 접근 책임 분리).

### D6. 결정론·leakage 계승
LLM 미개입(결정론, S5/S7 원칙). `train_cutoff`는 caller 주입(S6 leakage 규칙 — fit은 cutoff 이전만). 단일 store 유지(multi-store 배치는 범위 밖).

## 아키텍처

```
CLI scenario-commit-batch
  --items 파싱 or 매장 전 품목 추출 → item_ids
  → run_scenario_commit_batch(dataset, store, item_ids, period, drivers, wb, gate, now, train_cutoff)
        require_approval 가드
        → what_if_driver_batch(...)                     # scenario.py
              _validate_drivers 1회
              enriched = _build_enriched 1회
              model = _fit_demand_model(enriched, cutoff) 1회   # ← fit 공유
              for item in item_ids:
                  try: results.append(_what_if_for_item(enriched, model, ...))
                  except: log.warning + skip
              return list[WhatIfDriverResult]
        → for wif in results:
              base_order = apply_policy(item, wif.before_demand, policy)[0]
              scenario_order = apply_policy(item, wif.after_demand, policy)[0]
              propose → gate → approve/reject (writeback)
        → list[ScenarioCommitResult]
  → _write_and_label + 품목별 요약 출력
```

## 구현 세부

### scenario.py
```python
def _what_if_for_item(enriched, model, store_id, item_id, period, driver_overrides,
                      *, base_order, risk, policy) -> WhatIfDriverResult:
    # 기존 what_if_driver의 base_rows→predict→override→after→risk→result 로직 그대로.
    # base_order is None이면 apply_policy(item_id, before_demand, policy)[0].

def what_if_driver(daily, calendar, weather, store_id, item_id, period, driver_overrides,
                   *, base_order=None, train_cutoff, feature_set="v2",
                   risk=RiskParams(), policy=PolicyParams()) -> WhatIfDriverResult:
    _validate_drivers(driver_overrides)
    enriched = _build_enriched(daily, calendar, weather)
    model = _fit_demand_model(enriched, train_cutoff, feature_set)
    return _what_if_for_item(enriched, model, store_id, item_id, period,
                             driver_overrides, base_order=base_order, risk=risk, policy=policy)

def what_if_driver_batch(daily, calendar, weather, store_id, item_ids, period,
                         driver_overrides, *, base_order=None, train_cutoff,
                         feature_set="v2", risk=RiskParams(),
                         policy=PolicyParams()) -> list[WhatIfDriverResult]:
    _validate_drivers(driver_overrides)
    enriched = _build_enriched(daily, calendar, weather)
    model = _fit_demand_model(enriched, train_cutoff, feature_set)
    out = []
    for item_id in item_ids:
        try:
            out.append(_what_if_for_item(enriched, model, store_id, item_id, period,
                                         driver_overrides, base_order=base_order,
                                         risk=risk, policy=policy))
        except Exception as exc:
            log.warning("scenario batch: skip item %s (%s)", item_id, exc)
    return out
```
- `log`은 scenario.py 모듈 로거(없으면 추가).

### loop.py
```python
def run_scenario_commit_batch(dataset, store_id, item_ids, period, driver_overrides,
                              writeback, gate, *, now, train_cutoff,
                              policy=PolicyParams(), risk=RiskParams()
                              ) -> list[ScenarioCommitResult]:
    # require_approval 가드(단일과 동일 메시지 패턴)
    # wifs = scenario.what_if_driver_batch(...)
    # for wif in wifs: base/scenario order = apply_policy(...); propose→gate→commit
    # return [ScenarioCommitResult(whatif=wif, base_order=..., committed=rec), ...]
```
- 단일 `run_scenario_commit`은 무변경(what_if_driver 계속 사용, fit 1회라 문제없음).

### cli.py
```python
@app.command("scenario-commit-batch")
def cmd_scenario_commit_batch(store, period, drivers, items="", gate="human",
                              source="synthetic", now="", out=""):
    start, end, stamp = _parse_period(period, now)
    gate_policy = _select_gate_policy(gate)
    driver_overrides = _parse_drivers(drivers)
    dataset = load_dataset(source)
    item_ids = ([s.strip() for s in items.split(",")] if items
                else sorted(dataset.daily.loc[dataset.daily["store_id"] == store, "item_id"].unique()))
    wb = WritebackStore(require_approval=True)
    results = run_scenario_commit_batch(dataset, store, item_ids, (start, end),
                                        driver_overrides, wb, gate_policy,
                                        now=stamp, train_cutoff=start)
    # 헤더 + 품목별 한 줄 요약 + _write_and_label
```

## 테스트

- **fit 1회** (핵심): `what_if_driver_batch`가 N품목이어도 fit을 1회만 — `_fit_demand_model` 또는 `GlobalLGBM.fit`을 spy(monkeypatch counter)로 카운트.
- **배치=단일 동등**: 배치 결과의 각 품목 before/after demand == 단일 `what_if_driver` 결과.
- **skip**: 존재하지 않는 품목을 섞으면 그 품목만 빠지고 나머지 정상, 로그 발생.
- **run_scenario_commit_batch**: 품목별 APPROVED 레코드가 writeback에 생성, ScenarioCommitResult 리스트 반환. require_approval=False면 ValueError.
- **CLI 스모크**: synthetic, `--items` 지정 + 생략(전체) 둘 다.
- 전체 회귀 무변경 통과 (기존 what_if_driver/scenario-commit 테스트가 리팩토링 후에도 green — 하위호환 증거).

## 범위 밖

- LLM 자율 시나리오 선택(다음 frontier — 이 배치가 실행 기반).
- multi-store 배치.
- 배치 결과 parquet 이외 포맷/집계 리포트.

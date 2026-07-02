# Scenario→commit closed-loop (v7 S7) — Design

**Date**: 2026-07-01
**Depends on**: S5 closed-loop (WritebackStore, GatePolicy, OrderProposal, run_closed_loop, `_select_gate_policy` CLI helper), S6 what_if_driver (scenario.what_if_driver, WhatIfDriverResult), decision.apply_policy
**Status**: Design approved (2026-07-01)

## 목적

상류 레버(S6 what_if_driver)와 하류 commit(S5 writeback 게이트)을 하나로 잇는 **닫힌 루프**를 구현한다: 가상 시나리오(날씨/캘린더 드라이버) 하에서 조정된 발주량을 계산해 사람 승인 게이트를 통과시켜 확정한다. "드라이버 가정 변경 → 모델 재예측 → 조정 발주 → 사람 승인 → writeback"의 end-to-end를 시연한다.

## 핵심 설계 결정 (LOCKED)

### D1. 결정론 orchestrator (LLM 미개입)
시나리오(`driver_overrides`)는 입력으로 주어지고, 발주량은 `apply_policy`로 결정론 산출, commit은 `GatePolicy`. S5 원칙("LLM read-only, 쓰기는 게이트된 결정론 seam")을 그대로 계승한다. → **LLM 도구 surface에 노출하지 않는다**(쓰기 경로). LLM이 "어떤 시나리오를 commit할지" 자율 선택하는 것은 frontier로 분리.

### D2. 재사용만 (신규 모델링 0)
`what_if_driver`(S6) + `apply_policy`(decision) + `WritebackStore`/`GatePolicy`/`OrderProposal`(S5)를 조립한다. 새 예측·정책 로직 없음.

### D3. 결정론·leakage 계승
타임스탬프는 caller 주입(`now`), `train_cutoff`도 caller 주입(S6 leakage 규칙 계승 — fit은 cutoff 이전만). LGBM seed 고정 → before/after predict 결정론. 게이트 단일 레버: store는 `require_approval=True`, 승인은 `GatePolicy`로 제어(S5와 동일).

### D4. 발주량 산출 = apply_policy(after_demand)
committed 발주량 = `apply_policy(item, wif.after_demand)[0]`. 비교용 base_order = `apply_policy(item, wif.before_demand)[0]`. what_if_driver의 risk 델타는 "같은 base_order 하에서 demand가 before→after로 이동할 때"의 do-nothing 위험을 보여주고(lineage), 실제 commit은 after_demand에 맞춘 조정 발주량이다.

### D5. what_if_driver `base_order` optional 확장 (순환 제거, fit 1회)
현 what_if_driver는 `base_order`를 필수 입력으로 받는데(risk 비교용), scenario_commit은 base_order를 `apply_policy(before_demand)`로 만들고 싶고 before_demand는 what_if_driver가 계산한다 → 순환. 해결: **what_if_driver의 `base_order`를 `float | None = None`로 확장**하고, None이면 내부에서 `apply_policy(item_id, before_demand)[0]`로 자동 산출한다(before_demand는 이미 계산됨). 이러면 fit 1회 유지, S6 standalone 호출도 그대로(하위호환), scenario_commit은 `base_order=None`으로 호출한 뒤 `wif.before_demand`/`wif.after_demand`에 `apply_policy`를 다시 적용(예측 아님, fit 없음)해 base/scenario order를 얻는다.

## 아키텍처

```
run_scenario_commit(dataset, store_id, item_id, period, driver_overrides,
                    writeback, gate, *, now, train_cutoff,
                    policy=PolicyParams(), risk=RiskParams()):
  1. wif = what_if_driver(daily, calendar, weather, store_id, item_id, period,
                          driver_overrides, base_order=None,       # None → 내부 apply_policy(before) (D5)
                          train_cutoff=train_cutoff, risk=risk)     # S6 재예측, fit 1회
  2. base_order     = apply_policy(item_id, wif.before_demand)[0]  # 현 발주량 (예측 아님, fit 없음)
  3. scenario_order = apply_policy(item_id, wif.after_demand)[0]   # 시나리오 조정 발주량
  4. proposal = OrderProposal(item_id, scenario_order, rationale)   # rationale = 시나리오 서술
  5. rec = writeback.propose_order(...) → gate(proposal) → approve/reject   # S5 게이트
  → ScenarioCommitResult(whatif=wif, base_order=base_order, committed=rec)
```

순환 제거는 D5 참조(what_if_driver base_order optional). base/scenario order는 wif가 반환한 before/after demand에 `apply_policy`를 다시 적용해 얻으므로 추가 fit·예측이 없다.

## 컴포넌트 (loop.py에 추가)

| 컴포넌트 | 책임 |
|---|---|
| `ScenarioCommitResult` (frozen) | `whatif: WhatIfDriverResult`, `base_order: float`, `committed: OrderRecord` |
| `run_scenario_commit(dataset, store_id, item_id, period, driver_overrides, writeback, gate, *, now, train_cutoff, policy=PolicyParams(), risk=RiskParams()) -> ScenarioCommitResult` | 위 1~5 오케스트레이션 |
| CLI `scenario-commit` (cli.py) | end-to-end 실행. 옵션 `--store --item --period "start,end" --drivers "is_rain=1,is_snow=1" --gate auto\|human --source --now --out`. `_select_gate_policy`(S5) 재사용. train_cutoff=period[0]. 결과(before/after demand, base/scenario order, 게이트 결과) 출력 + `--out` parquet |

신규 파일 없음. loop.py는 이미 scenario를 import하지 않으므로 `from . import scenario` 추가(단방향; scenario는 loop을 import 안 함 — 순환 없음).

## 데이터 흐름 (예시)

```
$ uv run bakery scenario-commit --store 광교 --item P012 --period 2026-07-06,2026-07-12 \
    --drivers "is_rain=1" --gate human
1. before_demand=45 → base_order=54 (apply_policy)
2. what_if_driver(is_rain=1): after_demand=42 (비 오면 -3)
3. scenario_order = apply_policy(42) = 50
4. proposal(P012, 50, "scenario is_rain=1: demand 45→42, order 54→50")
5. gate(human=approve_as_proposed) → APPROVED, writeback 확정
→ 출력: before/after demand, base/scenario order, 게이트 결과, out_of_support
```

## 에러 / 엣지

- **require_approval=False** store → ValueError (게이트 단일 레버; S5와 동일).
- **알 수 없는 driver 키** → what_if_driver가 ValueError (S6 검증 재사용).
- **빈 period rows / cutoff 이후 데이터 없음** → what_if_driver가 ValueError (S6).
- **게이트 거부** → committed 레코드 status=REJECTED.
- **out_of_support=True** (외삽) → 차단 안 함, 결과에 whatif.out_of_support로 노출(경고).
- **CLI drivers 파싱**: `"is_rain=1,is_snow=1"` → `{"is_rain":1.0,"is_snow":1.0}`. 형식 오류 → 명확한 CLI 에러.

## 테스트

- **결정론 (stub 모델)**: `scenario._fit_demand_model` monkeypatch → before/after 고정 → base_order/scenario_order/proposal/게이트/writeback 상태전이 검증.
- **게이트 3종**: auto_approve / approve_as_proposed / human_correct 각각.
- **require_approval=False 가드** → ValueError.
- **거부 경로** → REJECTED committed 레코드.
- **rationale 포맷** 검증.
- **CLI**: 명령 등록 + drivers 파싱 헬퍼 단위 테스트 (키-프리). 실행 smoke는 실 LGBM fit이라 heavy → 등록/파싱 테스트로 커버, 수동 smoke는 선택.
- 기존 전체 테스트 불변.

## 범위 밖 (TODO)

- LLM 자율 시나리오 선택(어떤 what-if를 commit할지 에이전트가 결정) — frontier, 분리.
- fit 크로스콜 캐싱 — D5로 호출당 fit 1회는 확보. 그 이상(세션 캐시)은 범위 밖.
- 다품목 배치 시나리오 commit — 이번엔 단일 (store,item). 다품목은 후속.

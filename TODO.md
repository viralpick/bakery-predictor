# TODO

## v7 PoC

- [x] **상류 레버 `what_if_driver`** (S6, PR#11) — 드라이버 가상변경 → 실 LightGBM(v2) 재예측 → risk/cost link 전파. 읽기 전용. 설계 docs/superpowers/specs/2026-06-30-whatif-driver-design.md.

- [x] **Scenario→commit closed-loop** (S7, PR#13) — `run_scenario_commit`: 시나리오 재예측 → `apply_policy` 조정 발주 → 게이트 → writeback. CLI `scenario-commit`. 설계 docs/superpowers/specs/2026-07-01-scenario-commit-design.md.

## v7 후속 (Stretch / non-blocking)

- [ ] what_if_driver: unresolved-lag predict가 silent 0.0 → 데모 시 `before_demand==0` 경고/센티넬 (non-blocking).
- [ ] LLM 자율 시나리오 선택 (어떤 what-if를 commit할지 에이전트가 결정) — frontier.
- [ ] 다품목 배치 scenario-commit / CLI `policy`(게이트)↔`PolicyParams` 용어 정리(`gate` 리네임) / cmd 간 `_parse_period` 중복 추출.

## 후속 (실데이터/운영 의존)

- [ ] W0 게이트 = 수요이전(흡수) 검증 (Stage2 진입, 운영데이터 필요).
- [ ] 아티제 발주 ① 포맷 통일 + ③ 사람보정값을 "발주vs판매" feature 통합 (실데이터).
- [x] q_order_top Q셋 재설계 (rank→explain 체인) — 관측 매진시각 기반 rank_stockout_earliness, 채점 item_id+qty.

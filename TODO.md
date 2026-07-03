# TODO

## v7 PoC

- [x] **상류 레버 `what_if_driver`** (S6, PR#11) — 드라이버 가상변경 → 실 LightGBM(v2) 재예측 → risk/cost link 전파. 읽기 전용. 설계 docs/superpowers/specs/2026-06-30-whatif-driver-design.md.

- [x] **Scenario→commit closed-loop** (S7, PR#13) — `run_scenario_commit`: 시나리오 재예측 → `apply_policy` 조정 발주 → 게이트 → writeback. CLI `scenario-commit`. 설계 docs/superpowers/specs/2026-07-01-scenario-commit-design.md.

## v7 후속 (Stretch / non-blocking)

- [x] what_if_driver: unresolved-lag predict가 silent 0.0 → 데모 시 `before_demand==0` 경고 (`_lever_warning`).
- [ ] LLM 자율 시나리오 선택 (어떤 what-if를 commit할지 에이전트가 결정) — frontier.
- [x] CLI `policy`(게이트)↔`PolicyParams` 용어 정리(`gate` 리네임) + cmd 간 `_parse_period`/`_write_and_label` 중복 추출.
- [x] 다품목 배치 scenario-commit — `what_if_driver_batch`(fit 공유) + `run_scenario_commit_batch` + CLI `scenario-commit-batch` (`--items` or 전체).

## 후속 (실데이터/운영 의존)

- [x] W0 게이트 = 수요이전(흡수) 검증 — 카테고리 총량보존 β/TOST 직접검정. 광교 bread/pastry absorb, walk-away 0개 → 통과. 결과 docs/w0_demand_absorption_result.md. (다매장 store_daily 경로 후속)
- [ ] 아티제 발주 ① 포맷 통일 + ③ 사람보정값을 "발주vs판매" feature 통합 (실데이터).
- [x] 흡수검증 다매장 확장 — scripts/absorption_4stores.py, 4매장 walk-away 0/20건. 광교·메세나 absorb, 삼성·광화문 inconclusive(잔차 confound 큼, walk-away 아님). docs/w0_demand_absorption_result.md §다매장.
- [ ] 흡수검증 후속 정리(degenerate control 카운트 노출 / placebo `--placebo` flag 커밋 / cmd panel 이중빌드 제거) — non-blocking.
- [x] q_order_top Q셋 재설계 (rank→explain 체인) — 관측 매진시각 기반 rank_stockout_earliness, 채점 item_id+qty.

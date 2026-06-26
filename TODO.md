# TODO

## v7 PoC — 기간 내 필수

- [ ] **상류 레버 `what_if_driver`** (closed-loop 상류) — 날씨/캘린더 드라이버 가상변경 → forecast 재실행 → OntologyLink 전파 (팔란티어 Scenario 동형).
  - ⚠️ **PoC 기간 내 필수** (Stretch 아님). closed-loop 하류(S5)와 짝.
  - 선행: forecast wiring (현재 `demand_point`은 potential_demand proxy, forecast 미연결).
  - 함수: `what_if_driver` + (Scenario→commit closed-loop 시 `commit_order` 연계).
  - 설계 참조: docs/superpowers/specs/2026-06-25-closed-loop-design.md §범위 밖.

## 후속 (실데이터/운영 의존)

- [ ] W0 게이트 = 수요이전(흡수) 검증 (Stage2 진입, 운영데이터 필요).
- [ ] 아티제 발주 ① 포맷 통일 + ③ 사람보정값을 "발주vs판매" feature 통합 (실데이터).
- [ ] q_order_top Q셋 재설계 (rank→explain 체인).

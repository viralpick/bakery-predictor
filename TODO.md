# TODO

## v7 PoC

- [x] **상류 레버 `what_if_driver`** (S6, PR#11) — 드라이버 가상변경 → 실 LightGBM(v2) 재예측 → risk/cost link 전파. 읽기 전용. 설계 docs/superpowers/specs/2026-06-30-whatif-driver-design.md.

## v7 후속 (Stretch / non-blocking)

- [ ] **Scenario→commit closed-loop** — what_if_driver 결과를 채택 → S5 `commit_order`로 발주 확정 (상류+하류 결합). Stretch.
- [ ] what_if_driver: unresolved-lag predict가 silent 0.0 → 데모 시 `before_demand==0` 경고/센티넬 (non-blocking).

## 후속 (실데이터/운영 의존)

- [ ] W0 게이트 = 수요이전(흡수) 검증 (Stage2 진입, 운영데이터 필요).
- [ ] 아티제 발주 ① 포맷 통일 + ③ 사람보정값을 "발주vs판매" feature 통합 (실데이터).
- [ ] q_order_top Q셋 재설계 (rank→explain 체인).

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

## 방법론 후속 (외부 벤치마크 리서치, 2026-07)

출처: 노션 "방법론 추가 검토 방안 — 외부 사례·연구 벤치마크". 뚜레쥬르(CJ) + FreshRetailNet/stockout-timing/newsvendor/FM/Calendric 학술 조사.

- [x] 명절 lead-up 피처 추가 — `days_to_seollal`/`days_to_chuseok`를 base `calendar_features.py`에 추가(v0~v3+stockout), lookup은 `data/calendar.py` 단일 출처로 v4와 공유. Calendric 논문 감사 결과 발견한 유일 구멍.
- [ ] **classification 패러다임 병렬 체계** — 수요를 연속값 회귀가 아니라 구간(bin) 분류로 예측(Huber & Stuckenschmidt 2020: bakery daily에서 classification > regression). 별도 모델 트랙 — bin 정의 / 평가지표(WAPE↔분류지표 정합) / decision layer 연결 설계 필요. 브레인스토밍부터. **non-blocking, 별도 세션.**
- [x] 수능일 feature — **검토 결과 미추가**. 광교 일별 수요(2021~25, 수능 5회) dow-통제 잔차비 ±7일 내 0.93~1.03, 당일 5년 [0.92,1.10,1.05,0.84,0.94]로 방향 불일치 → 감지 가능한 일관 효과 없음. 오피스+주거 카페라 수능 선물수요와 무관. 노이즈 회피 위해 피처화 안 함.
- [ ] WPE(편향 방향) 지표 리포트 병기 — potential_demand 가치=하향편향 제거. FreshRetailNet 근거(복원 WAPE 소폭↓지만 과소추정 −6.7%→~0%). 우선순위 높음, 비용 낮음.
- [ ] Decoupling Score(ρ_DS) 진단 — 복원 수요가 품절률과 상관 남았는지(=검열편향 잔량). 단 품목 아닌 **카테고리 합**에만 적용(품목 latent demand는 흡수·검열 이중편향으로 식별불가). leakage 테스트의 통계짝.
- [ ] (중기) FreshRetailNet 공개 데이터로 우리 방법 재현/벤치 + Chronos-2 LoRA 레퍼런스(WAPE 23.99%) — FM 트리거(다매장·cold-start) 시.

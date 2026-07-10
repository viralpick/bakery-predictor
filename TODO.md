# TODO

## v7 PoC

- [x] **상류 레버 `what_if_driver`** (S6, PR#11) — 드라이버 가상변경 → 실 LightGBM(v2) 재예측 → risk/cost link 전파. 읽기 전용. 설계 docs/superpowers/specs/2026-06-30-whatif-driver-design.md.

- [x] **Scenario→commit closed-loop** (S7, PR#13) — `run_scenario_commit`: 시나리오 재예측 → `apply_policy` 조정 발주 → 게이트 → writeback. CLI `scenario-commit`. 설계 docs/superpowers/specs/2026-07-01-scenario-commit-design.md.

## v7 후속 (Stretch / non-blocking)

- [x] what_if_driver: unresolved-lag predict가 silent 0.0 → 데모 시 `before_demand==0` 경고 (`_lever_warning`).
- [ ] LLM 자율 시나리오 선택 (어떤 what-if를 commit할지 에이전트가 결정) — frontier.
- [x] CLI `policy`(게이트)↔`PolicyParams` 용어 정리(`gate` 리네임) + cmd 간 `_parse_period`/`_write_and_label` 중복 추출.
- [x] 다품목 배치 scenario-commit — `what_if_driver_batch`(fit 공유) + `run_scenario_commit_batch` + CLI `scenario-commit-batch` (`--items` or 전체).

## 발주 calibration 후속 (PR#30 conformal 이후, 2026-07-09 합의 순서 3→4→1→2)

- [x] **#3 `potential_demand` 전역 감사·제거** (PR#31) — 소스별 규칙(real→adjusted_demand, synthetic→potential_demand)을 공유 헬퍼 `_resolve_demand_col`/`_resolve_demand_proxy`로 통일. 6개 real 소비처 전환(backtest/predict-next-week/alpha-sweep/business-report/ontology v7/v6-predict). **모델 레벨 `_default_target` 불변** — CLI/ontology 레이어 y_col 명시로 국소화. 데이터 레이어/schema는 deprecation 마커만(synthetic 생성기·arrival 헬퍼·필드 유지). `--closing-alpha` 옵션. 435 passed. defer(Minor): alpha-sweep 전 variant override(v2도구라 무해)/predict·v6 v2/v3 synthetic smoke만(real e2e 데이터부재).
- [x] **#4 stockout_classifier 재측정** — 고친 라벨(60.4%)에서 재측정. 재학습은 자동(런타임 fit, 데이터 고쳐진 시점). base_rate 92%→~0.48–0.59 균형 회복(degenerate 해소=재정의 성공), but AUC 평균~0.63 약판별·성탄주 0.45. 판정: 아티팩트 청산 완료, 분류기 자체는 약함, **현 PoC 비주류**(v2/v3·v6·ontology 미사용, predict-next-week v1 legacy만 소비)→추가 투자 없음. docs/stockout_classifier_retrain_result.md.
- [x] **#1 category 경로 conformal 적용 — 미실행(설계 결정, 2026-07-10)**. 검토 결과 **conformal은 item 경로 전용**으로 확정. 근거: (a) conformal은 item quantile 모델이 구조적 under-dispersed(q 0.85→0.99로 15배 좁혀도 초과율 0.679→0.421까지만)라 도입한 fix인데, (b) 카테고리 총합은 매끈한 aggregate라 LGBM production_q가 직접 예측하는 게 맞고 conformal을 덧씌우면 "보정의 보정"으로 지저분함. 카테고리 총합 under-calibration(초과율 **0.346**@q0.85, nominal 0.15 — 원 TODO의 "0.458"은 배분 **후** item-day 값이고 conformal이 직접 손대는 총합값은 0.346)은 loose end로 남기되, 카테고리 경로가 PR#27서 item-level 못 이긴 non-primary 경로라 추가 투자 안 함. 코드 변경 없음. 결정 기록 = docs/order_conformal_calibration_result.md §후속 #1.
- [x] **#2 conformal 잔차 진단 — 완료 + #2 종료(PoC 충분, 2026-07-10)**. 진단 스크립트 `scripts/diagnose_conformal_residual.py`, 결과 `docs/conformal_residual_diagnosis_result.md`. 헤드라인(s=0.74,n_folds=8) 재현 잔차 +0.046. **결론**: (1) 메모리의 드리프트 가설은 주범 아님 — cal/test score **median 동일**(+0.595 vs +0.593, location 이동 없음), 단 coverage 지배하는 q74가 cal 0.99→test 1.16으로 실재 상승(부차적, 더 최근/가중 cal 창은 저우선 미검증 레버). (2) **지배적 원인=volume 이질성** — pooled scale-정규화 Q_s가 저volume slow-mover under-cover(초과율 0.63, 필요 Q_s 4.65) ↔ 중/고volume over-cover(0.13, 필요 0.71~0.80). 평균-정규화가 간헐수요 tail 못 맞춤. (3) 요일은 2차(평일 −0.01 정확, 주말 +0.17; volume과 교차 미확인). **운영 측정**: 저volume=전체 shortfall의 79%(5879u), 중/고=과잉발주(overage 22.5K≫shortfall 7.5K) → mis-allocation. **종료 근거**: 전체 잔차 작고 운영 주력(평일+중/고) calibrate, 드리프트 반증. 보강 후보=volume-Mondrian(측정상 Pareto 여지 있으나 stale tier·소표본 리스크, PoC 필수 아님)은 미실행. 코드 변경 없음.

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
- [x] WPE(편향 방향) 지표 리포트 병기 — evaluation/metrics.py에서 구현. potential_demand 가치=하향편향 제거. FreshRetailNet 근거(복원 WAPE 소폭↓지만 과소추정 −6.7%→~0%).
- [x] Decoupling Score(ρ_DS) 진단 — evaluation/diagnostics.py (category-level)에서 구현. 복원 수요가 품절률과 상관 남았는지(=검열편향 잔량). 단 품목 아닌 **카테고리 합**에만 적용(품목 latent demand는 흡수·검열 이중편향으로 식별불가). leakage 테스트의 통계짝.
- [x] 전향적 KPI 비교 harness — evaluation/prospective.py (build_arrival_profile / simulate_soldout / simulate_item_day_kpis / compare_policies / reconstruct_baseline_order) + CLI prospective-eval (synthetic e2e). --source real은 실데이터 컬럼매핑 대기(NotImplementedError). 플랜 docs/superpowers/plans/2026-07-05-prospective-kpi-harness.md.
- [ ] (중기) FreshRetailNet 공개 데이터로 우리 방법 재현/벤치 + Chronos-2 LoRA 레퍼런스(WAPE 23.99%) — FM 트리거(다매장·cold-start) 시.

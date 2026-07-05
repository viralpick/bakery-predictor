# 회고 harness 실데이터 wiring + ρ_DS + minors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** `prospective-eval --source real`을 광교 과거(2021~25) 실데이터로 가동 — 우리 발주(production-quantile) vs 아티제 실제 생산량을 폐기/매진시각/매진률로 비교하고, ρ_DS(카테고리)·WPE 진단을 붙인다.

**Architecture:** 이미 머지된 `evaluation/prospective.py` harness의 `--source real` stub을 실제 loader로 대체. 데이터는 전부 실측 존재 — `bonavi_daily.parquet`(potential_demand·is_stockout·stockout_time·capacity·category_id·open_hours 이미 precompute) + `재고정보` 시트(생산량=baseline_order, 폐기량=actual waste). our_order는 LightGBM production-quantile(q0.85) backtest 예측.

**Tech Stack:** Python, pandas, LightGBM, typer, pytest, uv.

## Global Constraints
- Time leakage 금지: our_order 예측은 rolling/expanding backtest(시간순)로만. split 이후 feature.
- 품절 censored: potential_demand(복원)를 수요 입력으로. 매진시각 sanity는 품절정보 실측.
- exact-value 테스트 단언. 함수 ≤30줄.
- 실데이터 진입점은 loader — 하드코딩 경로는 기존 관례(`data/internal/...`) 따름.
- baseline_order = 재고정보.생산량(실측), actual_waste = 재고정보.폐기량(실측). our_order = production-quantile q0.85 (사용자 결정).

---

### Task A: 재고정보 loader (생산량·폐기량)
**Files:** Create `src/bakery/ingest/inventory.py` (or extend loader); Test `tests/test_inventory_loader.py`
**Interfaces — Produces:** `load_inventory(xlsx_path, store_id) -> pd.DataFrame` — columns `date, item_id, production_qty, waste_qty` (from 재고정보: 날짜→date, 품목코드→item_id, 생산량→production_qty, 폐기량→waste_qty), filtered to store via 점포코드(store_mapping).
- [ ] Step1: 실패 테스트 — 소형 합성 재고정보 DataFrame → load 결과 컬럼·값 정확 검증(exact).
- [ ] Step2: run → fail.
- [ ] Step3: 구현 — read_excel(sheet_name="재고정보"), rename, 점포코드 필터(store_mapping.yaml의 gwangyo 코드), dtype 정리. (점포코드 매핑은 `ingest/store_mapping.py` 재사용 — 구현자가 확인.)
- [ ] Step4: run → pass.
- [ ] Step5: commit `feat: 재고정보 loader (생산량/폐기량 실측)`.

### Task B: 실데이터 조립 `_load_prospective_inputs("real")`
**Files:** Modify `src/bakery/cli.py`; Test `tests/test_prospective_cli.py` (append)
**Interfaces — Consumes:** `load_inventory`(A), `bonavi_daily.parquet`. **Produces:** rows frame with `item_id, date, category_id, potential_demand, base_order(=production_qty), sold_units, is_stockout` + receipts(hour/qty) + unit_prices, for `build_arrival_profile`/`simulate_item_day_kpis`.
- [ ] Step1: 실패 테스트 — 소형 real-shaped fixture(bonavi_daily 서브셋 + inventory)로 조립 → base_order=production_qty, potential_demand 존재, category_id 채워짐 검증.
- [ ] Step2: fail. Step3: 구현 — bonavi_daily 로드(store/카테고리 필터) + load_inventory join(on date,item_id) → base_order. receipts는 bonavi_receipts.parquet(hour/qty). our_order는 Task C가 채움(여기선 컬럼 자리만). Step4: pass. Step5: commit `feat: prospective-eval real 조립 (bonavi_daily + 재고정보)`.

### Task C: our_order = production-quantile backtest
**Files:** Modify `src/bakery/cli.py` (real 경로에서 모델 예측); Test: 통합 스모크(작은 기간).
**Interfaces — Consumes:** LightGBM quantile(feature_set v2, production_quantile=0.85), 기존 backtest/predict 인프라. **Produces:** rows에 `our_order` = 시간순 rolling 예측 q0.85 per item-day.
- [ ] Step1: 실패/스모크 테스트 — 소기간 실데이터(또는 fixture)에서 our_order 컬럼 생성·비음수·행수 일치.
- [ ] Step2: fail. Step3: 구현 — expanding-window backtest로 q0.85 예측 생성(누수 없이), rows에 병합. 기간이 크면 최근 N주로 제한하고 `log`로 명시. Step4: pass + `uv run bakery prospective-eval --source real --store gwangyo` 실행. Step5: commit `feat: our_order production-quantile 예측 wiring`.

### Task D: ρ_DS 카테고리 레벨 wiring (real 경로)
**Files:** Modify `src/bakery/cli.py`; Test `tests/test_prospective_cli.py`
**Interfaces — Consumes:** `decoupling_score`. **Produces:** real 경로에서 카테고리별 (potential_demand, stockout_rate) 셀 → `decoupling_score` 출력(카테고리별 + 가중 종합).
- [ ] Step1: 실패 테스트 — category_id 있는 fixture에서 카테고리별 ρ_DS 산출(값 검증). item 레벨 계산 없음(제약).
- [ ] Step2: fail. Step3: 구현 — `_decoupling_by_category(rows)` 헬퍼: category_id별 stockout_rate=mean(is_stockout), demand=potential_demand 집계 → decoupling_score. CLI real 경로에서만 출력. Step4: pass. Step5: commit `feat: ρ_DS 카테고리 레벨 wiring (real 경로)`.

### Task E: item 4 minors — WPE in backtest + shape 가드
**Files:** Modify `src/bakery/evaluation/backtest.py`(WPE 병기), `metrics.py`/`diagnostics.py`(shape 가드); Tests 해당.
**Interfaces:** backtest fold 요약에 `wpe` 포함. `wpe`/`decoupling_score`에 shape mismatch 시 명시적 ValueError.
- [ ] Step1: 실패 테스트 — (a) backtest 요약 dict/df에 wpe 존재, (b) wpe/decoupling에 길이 불일치 입력 → ValueError.
- [ ] Step2: fail. Step3: 구현 — backtest aggregate에 wpe 추가(summarize 이미 wpe 포함 → 표면화), wpe에 `y_true/y_pred` 길이 체크, decoupling_score에 weights 길이 체크. Step4: pass. Step5: commit `feat: backtest WPE 병기 + 진단 shape 가드`.

### Task F: 회고 실행 + 결과 문서
**Files:** Create `docs/retro_harness_result.md`; run full suite.
- [ ] Step1: `uv run pytest -q` 전체 통과.
- [ ] Step2: `uv run bakery prospective-eval --source real --store gwangyo` 실행 → KPI 표(우리 q0.85 vs 아티제 생산량 + Δ) + 카테고리 ρ_DS + WPE 캡처.
- [ ] Step3: 결과 문서 작성 — 수치 + 해석(폐기/매진시각/매진률 Δ, 실측 폐기량 대비 sanity) + 한계(회고=아티제 실제 발주가 생산량과 동일 가정). Step4: commit `docs: 회고 harness 실행 결과`.

## Self-Review
- 실데이터 4매핑 해소: potential_demand(bonavi_daily✓)/waste(재고정보.폐기량✓)/base_order(재고정보.생산량✓)/our_order(Task C q0.85✓).
- Leakage: Task C expanding-window 시간순. exact 단언. ρ_DS 카테고리 한정(item 금지) 준수.
- 리스크: Task C backtest компute 규모 — 기간 제한 + log 명시. store_mapping 점포코드=구현자 확인.

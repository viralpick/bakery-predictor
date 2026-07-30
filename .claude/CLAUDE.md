# bakery-predictor

베이커리(아띠제) 매장 판매량/품절 수요예측 PoC. 배경 = `@spec.md`, PoC 범위 = `@docs/poc_scope_v6.md`.

---

## ★ 작업 진입 규칙 (backbone-first) — 새 작업 시작 시 가장 먼저 읽는다

이 repo에는 **재사용 가능한 실험 backbone**이 있다. 새 모델링·측정·가설검증을 **`scripts/`에 새 스크립트로 시작하지 않는다.** 반드시 아래 표면 중 하나를 **확장**한다.

> **왜**: 예전에 실험이 `scripts/` 85개로 흩어져 "뭘 돌려야 맞지"가 매번 반복됐고, **백테스트 엔진이 둘 있는데 하나(카테고리 총량 = 실제 발행 헤드라인)가 `scripts/`에 숨어 있었다.** 그래서 CLI가 item-level 보조 엔진인 줄도 모르고 비교 수치를 잘못 인용하는 사고가 났다. backbone은 이 혼란을 구조적으로 제거하기 위한 것이다. 스크립트를 새로 만들면 그 상태로 되돌아간다.

### 라우팅 표 (판단 대신 조회)

| 요청 유형 | 표면 | 첫 행동 (필수 산출물) |
|---|---|---|
| 예측 성능 / 모델·정책 비교 / 백테스트 / 새 forecaster·레이어 | **`bakery harness-run <yaml>`** | ① `experiments/*.yaml` 추가(파라미터만 바뀌면 여기서 끝) ② 새 예측기면 `harness/forecasters.py` 어댑터 + `harness/registry.py` 등록 ③ `tests/harness/` 테스트 |
| 입력 데이터 분석(EDA) / 가설 검증 | **`bakery analysis-run <yaml>`** | ① 계산을 `src/bakery/analysis/*.py` 순수함수로 ② `analysis/lab/handlers/*.py` 핸들러 ③ `analysis/lab/registry.py` 키 등록 ④ `experiments/analysis_*.yaml` on/off ⑤ `tests/analysis_lab/` 테스트 |
| 데이터 편입 / 스키마 / 새 파일 경로 | `build-data` · `refresh-external` · `check-integrity` | 새 데이터셋 경로는 **`src/bakery/data/paths.py` registry에만** 등록. 코드에 raw 경로 하드코딩 금지 |
| 온톨로지 / 액션·설명 레이어 | `src/bakery/{forecast,ontology}/` | 예측은 `forecast.forward` seam 호출(재구현 금지), 설명은 `ontology/explain.py` |
| 일회성 프로파일링·탐색 | `scripts/` **허용** | 단 (a) 일회성임을 파일 상단에 명시 (b) 결론이 재사용되면 **즉시 프리미티브+registry로 승격**하고 스크립트는 얇은 wrapper로 남긴다 |

**escape hatch는 위 마지막 행뿐이다.** "이번만 스크립트로"는 그 행에 해당할 때만 성립하고, 재사용되는 순간 승격 의무가 붙는다.

### 확장 시 불변 규칙

- **재구현 금지, 호출만** — 계산 로직은 `src/bakery/{models,features,analysis,evaluation,forecast}`의 프리미티브에 두고 harness/lab은 오케스트레이션만 한다.
- **엔진 동등성** — `harness/backtest_core.py`를 건드리면 카테고리 총량 경로가 **rtol=1e-9로 기존과 정확히 일치**해야 한다(`tests/harness/test_backtest_core_equivalence.py`, 52-fold). 이건 hard gate다.
- **YAML 1파일 = 1실험.** 실험 파라미터를 코드에 하드코딩하지 않는다.
- **off 항목도 리포트에 표기** — 껐다는 사실을 숨기면 리포트가 거짓말이 된다.

---

## 현재 위치

**진짜 PoC 실측 단계** (v0~v5는 착수 전 데이터 검증 단계였다). 산출물 = **점추정 + 품절/매진 위험 수치**(구간예측은 폐기). 검증 대상 = **광교 단독**(타 3매장은 보조 데이터).

검증 방식 = **4주 구축 + 4주 전향적(prospective) 실측**으로 기존 아띠제 발주 시스템과 운영 KPI(폐기비용↓ / 매진 median time↑ / 매진률↓) 비교. **전향 실측은 아직 미실시**(2026 실시간 피드 필요, 현 데이터 vintage는 2021~2025-12) — 지금까지의 모든 수치는 회고 backtest 기준이다.

### backbone 로드맵 (7단계)

| # | 단계 | 상태 |
|---|---|---|
| 1 | harness 스파인 추출 + distributional 배선 | ✅ PR#53·#54 |
| 2 | 실험 리포트 표면 (`harness/report.py`, 자기포함 HTML) | ✅ PR#55 |
| 3 | 데이터 파운데이션 재설계 (`data/{raw,interim,processed}` + `paths.py`) | ✅ PR#56 |
| 4 | 데이터 무결성 검사 (`check-integrity` / `check-conflict`) | ✅ PR#57 |
| 5 | 온톨로지 예측 배선 + 2층 설명 레이어 | ✅ PR#58·#59 |
| 6 | 데이터분석 + 가설검증 레이어 (`analysis-run`) | ✅ 완료 — `@docs/phase6_analysis_layer.md` |
| 7 | 아띠제 신규 데이터 편입 마무리 (4매장 multistore + 헤드라인 재측정) | ✅ PR#60·#61 |

Phase 6 스펙 = `@docs/superpowers/specs/2026-07-28-phase6-analysis-hypothesis-layer.md`, 플랜(18 태스크) = `@docs/superpowers/plans/2026-07-28-phase6-analysis-hypothesis-layer.md`.

### canonical 스택 (헤드라인 = 고객에게 보고하는 수치)

**`category_total` + `event_prior`** — 카테고리(빵) **총량** 예측 + 특수일 레벨-앵커 후처리 블렌드.

- `lightgbm_v0~v3`(item-level)은 **비교용 보조**다. 헤드라인이 아니다.
- `category_v4`(품목 비율 배분)는 총량 **아래** 배분 레이어이지 총량 예측의 대안이 아니다.
- 헤드라인 파라미터: `alpha=0.8`, `production_q=0.85`, `window_days=730`, `n_folds=52`, `horizon_days=7`.

---

## 절대 규칙

1. **측정 기준 헌장 (2026-07-15 확정) — 모든 측정·비교의 단일 기준**
   - **bulk(대량예약)는 판매·생산 둘 다 제외.** canonical 프레임(`bonavi_daily`/`multistore_daily`/`bonavi_receipts`)은 이미 제외됨 — 재적용 금지.
   - 수요 = `sold`(bulk 제외), **`adjusted_demand` = 정상 + 0.8 × 마감**. 카테고리는 `adjusted_demand_unit`.
   - **`potential_demand` 사용 금지** — `stockout_time` 다중이벤트 버그로 부풀려진 오염 소스. 컬럼이 남아 있어도 읽지 않는다.
   - 매진 2관점: ① 전체매진(폐기0 or Σ발주<Σadjusted, critical) ② SKU품절(폐기0 & 마감0 비율).
   - 지표: **메인 = WAPE**(MAPE 단독 금지 — 희소 품목에서 폭발). 폐기율 = 1차 KPI, item 매진 회피 = 2차.
   - ❓ **재검토 지시 3항목(2026-07-28 architect)** — 헌장 본체는 유지하되 아래는 열린 상태다. 확정으로 취급하지 않는다:
     (a) **bulk 재확인** — 7월 신규 데이터로 대량예약 재검토. 단서 = 대한제분 AX 산출 "생산필요량" vs 실제 "발주 요청량" 갭이 데이터에 보인다(현 `bulk.py` T1/T2 임계값은 잠정).
     (b) **마감할인 실수요 정의** — `정상 + 0.8×마감`은 "예측 정확도↑ → 폐기↓ → 마감할인을 안 하는 방향으로 운영 전략이 수정된다"는 가정 위에 있다. 그 가정이 아니면 α 튜닝이 아니라 **타깃 정의 자체**를 다시 잡아야 한다(아띠제 운영 방향 확인 필요).
     (c) **SKU 품절 정의** — 현재는 시간 무관 "그날 품절 발생 비율". 마감할인 시간대 품절 제외안 / 품절 예상시각 median·mean 병행안을 함께 분석할 것.
2. **Time leakage 금지** — lag/rolling feature는 split 이후 또는 명시적 cutoff 이전 데이터로만. 예측 시점 이후의 sales/weather/지하철 실측을 feature로 쓰지 않는다. `event_prior`도 `date < test_start` history로만 fit한다. `test_split_leakage.py` / `test_features_leakage.py` 통과 필수.
3. **Random split 금지** — train/val/test는 시간순. backtest는 rolling/expanding window, 단일 holdout 금지.
4. **품절 데이터는 censored** — 품절일 판매량은 실수요가 아니다. `is_stockout`/`is_stockout_defined`/`stockout_time`을 보존하고 무리하게 결측 처리하지 않는다. 판매량 모델과 품절 위험 모델은 분리한다.
5. **Synthetic ↔ Real 경계 명시** — `data/synthetic.py`는 PoC 한정. 실데이터 진입점은 `data/loader.py`·`data/bonavi_loader.py`이며 동일 schema(`data/schema.py`)를 반환한다.
6. **수치는 출처와 함께** — 헤드라인 수치를 인용할 때 어느 파이프라인·어느 vintage인지 같이 적는다. 옛 수치와 어긋나면 먼저 **측정 축**(모델/단위/타깃/소스/vintage)을 의심한다. 예: 광교 총량 WAPE 8.03 → **7.72는 신규 vintage 재설정이지 "더 정확해짐"이 아니다.**

---

## 실행

### 헤드라인 (backbone)

```bash
uv run bakery harness-run experiments/gwangyo_default.yaml   # ★ canonical 실험: 52-fold 백테스트 + 자기포함 HTML 리포트
uv run bakery harness-run experiments/gwangyo_compare.yaml   #   forecaster 다중 비교(category_total vs distributional_total)
uv run bakery analysis-run experiments/analysis_gwangyo.yaml  # 입력 데이터 분석 5종 + 가설 14종 (@docs/phase6_analysis_layer.md)
```

### 데이터 파이프라인

```bash
uv sync                                     # 의존성
uv run bakery build-data --diagnose         # raw→processed 재빌드 + 동등성 rtol=1e-9 진단
uv run bakery build-multistore              # 4매장 multistore_daily.parquet (광교 canonical 불변)
uv run bakery refresh-external --source all # 외부 8종 갱신 + 커버리지 보존 가드
uv run bakery check-integrity               # 무결성 게이트: 타깃누락/시트스왑=fail, 나머지=drift CSV (@docs/data_integrity.md)
uv run bakery check-conflict                # 옛/새 마스터 값 충돌 진단(vintage)
uv run bakery ingest-calendar / ingest-weather / ingest-forecast   # 실 API 백필 (.env 필요)
uv run pytest                               # 테스트 (addopts에 -q 있음 → -q 추가 금지, 카운트는 --color=no)
```

### 보조 · 레거시 CLI (헤드라인 아님 — 인용 시 주의)

```bash
uv run bakery backtest --source real --variants v0,v1,v2   # ⚠️ item-level LGBM 보조 엔진. 헤드라인 아님
uv run bakery predict-next-week --source real ...          #    운영형 forward 예측(item-level 산출)
uv run bakery v6-predict / business-report / stockout-risk #    결정 레이어·리포트 (백본 편입 전)
```

### 발주 KPI (백본 편입 완료 — PR#74~#77)

```bash
uv run bakery harness-run experiments/gwangyo_kpi.yaml   # 비용+매진+아띠제 절감률 → reports/gwangyo_kpi/kpi.csv
```

정의의 **단일 출처** = `src/bakery/evaluation/order_kpi.py`. 5축 확정: **A/B basis 병기**(A=아띠제 실측 `QT_OUT` / B=발주−실수요) · 원가율 **품목별**(`1511`=0.40 / `1513`=0.60) · 품절 손실은 **전체매진만**(k=0) · SKU 품절율 = **날별 비율의 평균** · 매진시각 = **전체 median + 날별 median 평균 병기**.

- ⚠️ **인용할 축은 `ΔvsB`(공정)** — `vs_actual_sim_pct`. `ΔvsA`(`vs_actual_pct`)는 A가 censored라 절감을 과소평가하는 **하한**이다. 옛 헤드라인 −37~45%도 `ΔvsB` 축이었다.
- ⚠️ `surplus_rate`(총량 층위 비율)와 KPI 비용은 **다른 축이다** — 같은 표에서 비교 금지.
- ⚠️ `kpi: true` 는 `order_level: item` 필수(`SpecError`로 강제). 둘 다 기본 off라 헤드라인 실험 속도는 보존된다.
- 정산 기록(옛↔새 축별 분해) = `@docs/kpi_plane_reconciliation.md`. `scripts/unified_policy_kpi.py` 는 **5정책 비교(nk/conformal) 전용 레거시**이며 헤드라인 아님.

---

## 디렉토리

**백본 (확장 지점)**
- `src/bakery/harness/` — 예측 실험 평면: `config.py`(ExperimentSpec) / `backtest_core.py`(★단일 엔진) / `forecasters.py` / `registry.py` / `event_priors.py` / `runner.py` / `report.py`
- `src/bakery/analysis/lab/` — 분석·가설 평면: `result.py` / `spec.py` / `inputs.py` / `registry.py` / `runner.py` / `report.py` / `handlers/` 9모듈(등록 19종). 게이트 3종 = `preds_required`(preds 의존 4종) / `needs_single_store`(광교 전용 소스 5종) / `multistore_required`
- `experiments/` — YAML 1파일 = 1실험 (`*.yaml` 예측, `analysis_*.yaml` 분석)
- `src/bakery/data/paths.py` — 데이터셋 경로 registry(18종). **데이터 접근은 여기를 통해서만**

**프리미티브**
- `src/bakery/data/` — schema / loader / bonavi_loader(+v2) / pipeline / integrity / coverage / bulk
- `src/bakery/features/` — date / lag / rolling / calendar / weather / cannibalization / stockout_history
- `src/bakery/models/` — seasonal_naive / moving_average / lightgbm_regressor / category_total / distributional_total / event_prior / stockout_classifier
- `src/bakery/analysis/` — 가설검증 프리미티브 17종(absorption / substitution / discount / seasonal / popularity / waste / basket …)
- `src/bakery/evaluation/` — split / metrics / backtest
- `src/bakery/forecast/` — forward 예측 공유 seam (CLI·온톨로지 공용)
- `src/bakery/ontology/` — AOS 도구 + 2층 설명(`explain.py`)
- `src/bakery/decision/` — 발주 결정·위험·lineage 레이어

**그 외**
- `data/{raw,interim,processed}/` — 3층 분리(byte-preserving)
- `tests/` — leakage 회귀 + `tests/harness/`(엔진 동등성) + `tests/analysis_lab/`
- `scripts/` — 레거시 85개. **신규 작성 대상이 아니다**(위 라우팅 표 참조)
- `reports/` — 산출물 (gitignored)
- `docs/superpowers/{specs,plans}/` — 단계별 설계 스펙·구현 플랜(작업 착수 시 해당 단계 스펙 먼저 읽는다)

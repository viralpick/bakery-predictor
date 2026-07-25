# 데이터 아티팩트 인벤토리 · 분류표 (Task 1, 2026-07-25)

> 데이터 파운데이션 재설계(`docs/superpowers/plans/2026-07-25-data-foundation-redesign.md`) Task 1 산출물.
> **읽기 전용 조사** — 이 작업에서 파일 이동/코드 수정은 하지 않았다. Task 2(`paths.py` registry)·Task 3(물리 이동)의 근거 표.

## 방법론

1. `grep -rnE "data/(internal|external)[^"')]*\.(parquet|xlsx|xls)" src/ scripts/ tests/ experiments/` + 보조 grep으로 경로 리터럴 전수 열거.
2. `data/internal`, `data/internal/v2`, `data/external` 전 파일에 대해 `ls -la` 확인 후, 파일명 리터럴 grep으로 소비처(consumers)를 src/scripts/tests 축으로 분리하고, 빌더(어느 CLI 커맨드/스크립트가 생성하는가)를 코드 추적으로 확정.
3. 제네릭 단어(예: `items`, `stores`)와 실제 파일 경로 리터럴을 혼동하지 않도록, 매칭된 라인을 개별 확인해 **false positive를 제거**했다 (아래 "방법론 주의" 참조).

### 방법론 주의 — false positive 제거

`items.parquet`/`stores.parquet`/`stockout.parquet` 파일명 자체가 아니라 `'items'`/`'stores'`/`'stockout'` 같은 흔한 단어로 느슨하게 grep하면 src/ 여러 곳(dict key, JSON schema, docstring)에서 오탐이 뜬다. 예:
- `src/bakery/ingest/competitor_api.py:69` `body.get("items")` — API 응답 JSON key, 무관.
- `src/bakery/ingest/store_mapping.py:79` `raw.get("stores", raw)` — yaml 최상위 key, 무관.
- `tests/test_store_daily_redefine.py:56` `"stockout" in c.lower()` — 컬럼명 필터, 무관.

각 항목을 라인 단위로 열어 실제 파일 경로 리터럴(`Path("data/internal/v2/...")` 또는 `V2 / "..."`)인지 확인한 뒤 카운트했다. **결론: `data/internal/v2/{inventory,items,stores,hours,stockout,discount_codes,sales_p1,sales_p2}.parquet`는 src/ 어디에서도 실제로 읽지 않는다(0 consumers) — scripts 전용.** 단, `waste_alpha_4stores.parquet`(v2/ 안에 있음에도)는 **src/bakery/cli.py에서 실제로 읽는다** (아래 §3 참조) — "src는 v2/를 안 쓴다"는 기존 가정의 유일한 예외.

## Step 1 — 경로 리터럴 전수 열거 (요약)

`grep -rnE "data/(internal|external)[^"')]*\.(parquet|xlsx|xls)" src/ scripts/ tests/ experiments/` → 77줄.

| 디렉토리 | 파일 수 (경로 리터럴 포함 파일) | 비고 |
|---|---|---|
| `src/` | 8개 파일 (`cli.py`, `analysis/discount.py`, `features/category_aggregate.py`, `data/bonavi_loader_v2.py`, `data/bonavi_loader.py`, `data/loader.py`, `ingest/living_population_csv.py`) | 전부 `data/internal`(v2 제외 1건) 또는 `data/external` 최상위 파일만 참조 |
| `scripts/` | 30개 파일 | `data/internal/v2/*` 참조는 scripts에만 존재 |
| `tests/` | 2개 파일 (`test_inventory_loader.py`, `test_forecast_pipeline.py`) | 둘 다 raw xlsx/forecast parquet 언급, v2 없음 |
| `experiments/*.yaml` | 0 | 실험 yaml은 CLI 기본 경로에 의존, 데이터 경로 리터럴 없음 |

**핵심 상수 정의처**: `src/bakery/config.py`의 `EXTERNAL_DATA_DIR = PROJECT_ROOT/"data"/"external"`가 유일한 중앙 상수다. **`INTERNAL_DATA_DIR` 상수는 존재하지 않는다** — internal 경로는 8개 파일에 문자열 리터럴로 중복 산재(`bonavi_daily.parquet` 7곳, `bonavi_receipts.parquet` 7곳, `sales_lines_clean.parquet` 2곳 — docstring/주석 제외 실제 `Path(...)` 구성만 집계; brief 추정치 9/7/2와 receipts·clean은 정확히 일치, daily는 7로 실측(brief 9는 주석 포함 추정으로 보임)). 이것이 Task 2 `paths.py` registry의 존재 이유.

## Step 2 — v2/ 빌더 확정 (핵심 발견)

**`data/internal/v2/*.parquet` 8종(`inventory`/`items`/`stores`/`hours`/`stockout`/`discount_codes`/`sales_p1`/`sales_p2`) + 병합본 `sales.parquet`의 1차 빌더는 `scripts/convert_xlsx_20260526.py`.** `보나비 데이터_20260526.xlsx`의 8개 시트를 영문 코드 헤더로 각각 parquet 저장 후, `판매정보`+`판매정보2`를 concat해 `sales.parquet`을 만든다. 전부 scripts 전용 소비 — src/tests 참조 0건 (확인됨, 위 false-positive 주의 참조).

**⚠️ 중요 불일치**: 현재 디스크의 `v2/sales.parquet`(72MB, mtime 7/23 23:46)은 `convert_xlsx_20260526.py`가 실제로 만드는 파일과 다르다.
- `sales_p1.parquet`(15MB, mtime 5/26) + `sales_p2.parquet`(6.4MB, mtime 5/26)을 합치면 약 21MB — 이는 `sales.parquet.pre-new-bak`(22MB, mtime 7/23 23:46, 스크립트가 남긴 백업으로 추정)의 크기와 정확히 일치한다.
- 반면 현재 `sales.parquet`(72MB)은 그보다 3배 이상 크다. p1/p2 자체는 5/26 이후 갱신되지 않았으므로, `convert_xlsx_20260526.py`를 그대로 재실행해도 72MB 버전은 재현되지 않고 22MB(구버전, `.pre-new-bak`와 동일)로 돌아간다.
- repo 전체(src/scripts) grep으로 이 72MB 확장본을 만드는 커밋된 스크립트를 찾지 못했다 — `2026-07-23-new-data-integration-design.md` 스펙 문서의 "166 canonical(+14% 품목)" 재빌드 작업과 시점이 일치하므로, **커밋되지 않은 애드혹(대화형) 작업으로 갱신된 것으로 추정**. `bonavi_daily_2026h1_covariate.parquet`(§3.4)와 동일한 패턴.
- **결론: `v2/sales.parquet`은 현재 상태 기준으로는 `rebuild-deterministic`이 아니라 `move-only`로 분류한다** (재실행하면 다른 파일이 나옴 — Task 3에서 이 파일을 옮길 때 "재빌드로 대체 가능"이라고 가정하면 안 됨).
- Task 2/3 담당자에게: 이 파일을 이동하기 전에 **결정적 재생성 스크립트를 새로 작성하거나(0721.xlsx 라인레벨 데이터 병합 로직 문서화), 현재 72MB 파일을 그대로 물리 이동(byte-preserving) 대상으로 취급**해야 한다. 소스가 불명확한 채로 셋 중 하나를 고르는 판단은 이 discovery 범위를 벗어나 architect 확인이 필요.

`v2/` 2차 파생 파일(`daily_4stores`, `daily_normal_vs_bulk`, `item_active_stats`, `sales_with_bulk_flag`, `waste_alpha_4stores`)의 빌더는 각각 `scripts/eda01_multistore_sales.py`, `scripts/bulk_detector.py`(2종), `scripts/bulk_detector.py`, `scripts/eda04_waste_alpha_calibration.py` — 전부 위 1차 v2 테이블(특히 위에서 불명확해진 `sales.parquet`)을 입력으로 삼으므로, **`sales.parquet`의 재현성 문제가 이들에게도 전이된다**.

## Step 3 — 분류표

**class 정의** (brief 기준): `rebuild-deterministic`(코드로 결정적 재생성 확인) / `move-only`(외부 API 유래 또는 재생성 근거 불명 orphan, 삭제 금지) / `cruft`(소비처 0 & 대체 가능, 삭제 후보) / `raw-source`(불변 원본).

### `data/internal/` (루트)

| 파일 | layer | class | builder | consumers (src/scripts/tests) | new_path |
|---|---|---|---|---|---|
| `보나비 데이터_20260520.xlsx` | raw | raw-source | (원본, 사람이 받음) | src: 5곳(`cli.py`×2 기본값, `bonavi_loader.py`, `discount.py`, `features/category_aggregate.py`) / scripts: 3곳 / tests: 1곳(`test_inventory_loader.py`) | `data/raw/internal/보나비 데이터_20260520.xlsx` |
| `보나비 데이터_20260526.xlsx` | raw | raw-source | (원본) | src: 5곳(`cli.py`×3, `bonavi_loader.py`, `bonavi_loader_v2.py`) / scripts: 4곳 / tests: 1곳 | `data/raw/internal/보나비 데이터_20260526.xlsx` |
| `보나비 판매 데이터_20260721.xlsx` | raw | raw-source | (원본) | src: 2곳(`cli.py`, `bonavi_loader_v2.py`) / scripts: 2곳 | `data/raw/internal/보나비 판매 데이터_20260721.xlsx` |
| `수원광교점 - 브레드 진열 시간(보안 해제 완료).xls` | raw | raw-source | (원본) | 0건 (아직 파이프라인 미연동 — 최근 수령분, 메모리 `project_new_data_20260721`와 일치) | `data/raw/internal/수원광교점 - 브레드 진열 시간(보안 해제 완료).xls` |
| `bonavi_daily.parquet` | processed | **rebuild-deterministic** | `format-bonavi`(v1, `bonavi_loader.build`) 또는 `format-bonavi-v2`(v2, `bonavi_loader_v2.build_v2`, rtol 1e-9 diff-검증됨 — 기존 메모리 진단과 일치) | src: 7곳(리터럴 `Path(...)` 기준) / scripts: 6곳+ / tests: 0 | `data/processed/internal/bonavi_daily.parquet` |
| `bonavi_daily.parquet.pre-v2-bak` | processed | **cruft** | (백업, `format-bonavi-v2` 실행 시 자동 생성 추정) | 0건 | 삭제 (백업 필요 시 별도 `data/_archive/`) |
| `bonavi_daily_2026h1_covariate.parquet` | processed | **move-only** (ambiguous, §3.4 상세) | 커밋된 스크립트 없음 — `2026-07-23-new-data-integration-design.md`에 "별도 생성 (7,107행/65품목/48,597개, sales-only, 라벨없음)"으로만 문서화 | 0건 (src/scripts/tests 전부 0) | `data/interim/bonavi_daily_2026h1_covariate.parquet` (재현 스크립트 없어 processed 승격 보류) |
| `bonavi_receipts.parquet` | processed | **rebuild-deterministic** | `format-bonavi`/`format-bonavi-v2` (`bonavi_loader.build`/`bonavi_loader_v2.build_v2`) | src: 7곳 / scripts: 5곳+ | `data/processed/internal/bonavi_receipts.parquet` |
| `bonavi_receipts.parquet.pre-v2-bak` | processed | **cruft** | (백업) | 0건 | 삭제 |
| `sales_lines_clean.parquet` | interim | **rebuild-deterministic** | `format-bonavi-v2` → `bonavi_loader_v2.convert_sales_to_parquet`(0721.xlsx 판매정보2 시트 헤더 스왑 교정, 메모리 `project_new_data_ingestion_pitfall`과 일치) | src: 2곳 / scripts: 0 | `data/interim/sales_lines_clean.parquet` |
| `store_mapping.yaml` | config (데이터 아님) | — | (수동 작성) | **0건 — 현재 코드 경로에서 실제로 로드되지 않음** (아래 §3.5) | `config/store_mapping.yaml` 또는 `src/bakery/config/` — `data/` 트리 밖 |
| `.DS_Store` | — | cruft | macOS 자동 생성 | 0건 | 삭제, `.gitignore` 확인만 |
| `v2/` | — | (하위 디렉토리, 아래 표) | | | |

### `data/internal/v2/`

| 파일 | layer | class | builder | consumers | new_path |
|---|---|---|---|---|---|
| `sales_p1.parquet` | interim | **cruft** | `convert_xlsx_20260526.py` (판매정보 시트) | scripts: 1(자기 자신, 병합용 중간산출) | 삭제 가능(재실행 시 재생성) — 유지 시 `data/interim/v2/sales_p1.parquet` |
| `sales_p2.parquet` | interim | **cruft** | `convert_xlsx_20260526.py` (판매정보2 시트) | scripts: 1(동일) | 삭제 가능 — 유지 시 `data/interim/v2/sales_p2.parquet` |
| `sales.parquet` | interim/processed 경계 | **move-only** ⚠️ (§Step2 참조, 현재 파일은 빌더 재실행으로 재현 안 됨) | 문서상 `convert_xlsx_20260526.py`이지만 **현재 디스크 파일은 그 스크립트의 산출물이 아님(72MB vs p1+p2=21MB)** | scripts: 6곳 이상(`store_daily.py`, `bulk_detector.py`, `all4_stores_backtest.py`, `v4_new_data_backtest.py`, `build_dashboard.py` 등) / src: 0 | `data/interim/v2/sales.parquet` (byte-preserving 이동만, 재빌드 시도 금지) |
| `sales.parquet.pre-new-bak` | interim | **cruft** | (구버전 백업, 위 불일치 발견의 물증) | 0건 | 삭제 전 보존 권장(유일하게 "재현 가능한 예전 상태"의 증거) |
| `inventory.parquet` | interim | **rebuild-deterministic** | `convert_xlsx_20260526.py` (재고정보 시트) | scripts: 9곳+ (`bulk_detector`, `store_daily`, `eda02/04`, `savings_analysis` 등) / src: 0 | `data/interim/v2/inventory.parquet` |
| `items.parquet` | interim | **rebuild-deterministic** | `convert_xlsx_20260526.py` (품목정보 시트) | scripts: 8곳+ / src: 0 (오탐 제거됨, §방법론 주의) | `data/interim/v2/items.parquet` |
| `stores.parquet` | interim | **rebuild-deterministic** | `convert_xlsx_20260526.py` (점포정보 시트) | scripts: 7곳+ / src: 0 (오탐 제거됨) | `data/interim/v2/stores.parquet` |
| `hours.parquet` | interim | **rebuild-deterministic** | `convert_xlsx_20260526.py` (영업시간 시트) | scripts: 2곳(`eda02_inventory_hours_stockout.py` 등) / src: 0 | `data/interim/v2/hours.parquet` |
| `stockout.parquet` | interim | **rebuild-deterministic** | `convert_xlsx_20260526.py` (품절정보 시트) | scripts: 5곳 / src: 0 / tests: 0 (오탐 제거됨) | `data/interim/v2/stockout.parquet` |
| `discount_codes.parquet` | interim | **rebuild-deterministic** | `convert_xlsx_20260526.py` (할인코드 시트) | scripts: 3곳(`eda04_waste_alpha_calibration.py`, `verify_other_discounts.py`) / src: 0 | `data/interim/v2/discount_codes.parquet` |
| `daily_4stores.parquet` | processed | **rebuild-deterministic**(단, 입력 `sales.parquet`이 move-only이므로 실제로는 그 안정성에 종속) | `scripts/eda01_multistore_sales.py` | scripts: 4곳(`plot_rain_by_dow`, `verify_stockout_revenue_4stores`, `diag_assumptions_multistore` 등) / src: 0 | `data/processed/internal/v2/daily_4stores.parquet` |
| `daily_normal_vs_bulk.parquet` | processed | **rebuild-deterministic**(同上 종속) | `scripts/bulk_detector.py` | scripts: 3곳(`bulk_detector`, `backtest_normal_target`, `build_dashboard`) / src: 0 | `data/processed/internal/v2/daily_normal_vs_bulk.parquet` |
| `item_active_stats.parquet` | processed | **cruft** | `scripts/bulk_detector.py` | scripts: 1곳(빌더 자기 자신만 — 외부 소비처 0) | 삭제 가능(재실행 시 재생성, 지금은 아무도 안 읽음) |
| `sales_with_bulk_flag.parquet` | processed | **cruft (경계 사례 — 완전 0건 아님, 아래 참조)** | `scripts/bulk_detector.py` | scripts: **3곳** — `absorption_4stores.py`, `all4_stores_backtest.py`, `v4_new_data_backtest.py` | 유지하되 저우선(아래 참조) — `data/processed/internal/v2/sales_with_bulk_flag.parquet` |
| `waste_alpha_4stores.parquet` | processed | **rebuild-deterministic**(단, `sales.parquet` 종속 리스크 상속) ⚠️ **src 소비 있음** | `scripts/eda04_waste_alpha_calibration.py` (v2/discount_codes+sales+inventory 입력) | **src: 2곳(`cli.py` `CLOSING_DEMAND_WASTE_PARQUET`, 실제 `pd.read_parquet` 2회)** / scripts: 2곳(`diag_assumptions_multistore.py`, `eda05_waste_breakdown.py`) | `data/processed/internal/waste_alpha_4stores.parquet` — **paths.py `_DATASETS`에 반드시 등록 필요 (Task 2 필수 항목, v2/ 유일 src-consumed 파일)** |

### `data/external/`

| 파일 | layer | class | builder | consumers | new_path |
|---|---|---|---|---|---|
| `calendar_raw.parquet` | processed | **move-only** | `ingest-calendar` (천문연 특일정보 API) | src: `data/loader.py:_load_real_dataset`(`external_dir / "calendar_raw.parquet"`) + `features/category_aggregate.py` / scripts: 8곳 | `data/processed/external/calendar_raw.parquet` |
| `weather_observed.parquet` | processed | **move-only** | `ingest-weather` (기상청 ASOS) | src: `data/loader.py`(동일 패턴) + `features/category_aggregate.py` / scripts: 5곳 | `data/processed/external/weather_observed.parquet` |
| `competitor_raw.parquet` | processed | **move-only** | `ingest-competitor`(소상공인진흥공단, cli.py:1359) | src: `data/loader.py`(존재 시 로드, 없으면 synthetic fallback) + `features/category_aggregate.py` | `data/processed/external/competitor_raw.parquet` |
| `consumption.parquet` | processed | **move-only** | `ingest-consumption`(서울 상권분석-소비, cli.py:1275) | src: `data/loader.py:153`(`external_dir / "consumption.parquet"`, 존재 시 로드) — 초기 리터럴-문자열 grep은 `data_dir /` 조합이라 놓쳤음, 코드 추적으로 확인 | `data/processed/external/consumption.parquet` |
| `population.parquet` | processed | **move-only** | `ingest-population`(행안부 admmSexdAgePpltn, cli.py:1259) | src: `data/loader.py:147`(동일 패턴) | `data/processed/external/population.parquet` |
| `living_population.parquet` | processed | **move-only** | `ingest-living-population`(서울 SPOP_LOCAL_RESD_DONG, cli.py:1236-1249) | src: `data/loader.py:139` + `cli.py:131`(`bundle.living_population.to_parquet`) | `data/processed/external/living_population.parquet` |
| `forecast_short_term.parquet` | processed | **move-only** | `ingest-forecast`(기상청 단기예보, cli.py:1373) | src: `cli.py:505` 인접 상수 | `data/processed/external/forecast_short_term.parquet` |
| `forecast_short_term_daily.parquet` | processed | **move-only** | `ingest-forecast` | src: `cli.py:505` (`EXTERNAL_DATA_DIR /`) | `data/processed/external/forecast_short_term_daily.parquet` |
| `forecast_mid_term_daily.parquet` | processed | **move-only** | `ingest-forecast` | src: `cli.py:506` | `data/processed/external/forecast_mid_term_daily.parquet` |
| `living_pop_zips/` (디렉토리, LOCAL_PEOPLE_DONG_*.zip 다수) | raw | **raw-source** | (서울 열린데이터광장에서 수동 다운로드) | src: 2곳(`cli.py`, `ingest/living_population_csv.py`) | `data/raw/external/living_pop_zips/` |
| `.DS_Store` | — | cruft | macOS | 0 | 삭제 |

## 애매한 파일 판정 근거 (brief 지정 6종)

### 3.1 `waste_alpha_4stores.parquet`
**판정: `rebuild-deterministic`, 단 `_DATASETS` 등록 필수 최우선 항목.** `src/bakery/cli.py:1616`에 `CLOSING_DEMAND_WASTE_PARQUET`로 정의되고 `:1630`, `:1713`에서 실제 `pd.read_parquet`로 읽는다 — v2/ 안에 있지만 **src가 v2/를 안 쓴다는 일반 가정의 유일한 예외**. 빌더는 `scripts/eda04_waste_alpha_calibration.py`이며 입력은 v2/`discount_codes`+`sales`+`inventory`(모두 rebuild-deterministic이나 `sales.parquet` 자체는 위 §Step2 불일치로 move-only) — 따라서 "결정적"이라는 라벨에 조건부 단서를 붙였다. Task 2는 이 파일을 `_DATASETS`에 반드시 추가해야 하며 (현재 초안 dict에는 없음), Task 3 물리 이동 시 `src/bakery/cli.py:1616`의 하드코딩 경로도 함께 재배선해야 한다.

### 3.2 `bonavi_daily_2026h1_covariate.parquet`
**판정: `move-only`(orphan, 삭제 금지·재빌드 스크립트 부재).** 파일명 리터럴로 src/scripts/tests 전체 grep 시 0건. "covariate" 키워드로도 무관한 docstring 1건(`substitution_did.py`)만 걸릴 뿐 실제 연관 없음. `2026-07-23-new-data-integration-design.md:119`에 "2026 H1 covariate: ... 별도 생성. 학습 타깃 아님"으로만 문서화되어 있고, 이를 만든 커밋된 스크립트가 없다 — 애드혹 생성물로 추정. 2026 상반기 데이터는 로드맵 7단계(신규 데이터, 아직 미착수) 대비 준비자료로 보이므로 삭제 후보(cruft)로 보지 않고 **재현 불가 orphan(move-only)**로 분류, 유지 권장.

### 3.3 `store_mapping.yaml`
**데이터 아님, config.** 브리핑 지시대로 별도 표기: "어디에 있어야 하는지"만 명시. **grep 근거상 현재 실제로 로드되지 않는다** — `load_store_mapping()`의 모든 호출부(`cli.py` 3곳, `synthetic.py`, `population_api.py`, `weather_api.py`, `forecast_api.py`, `competitor_api.py`, `loader.py`)가 `path=None`(또는 `mapping_path=None` 기본값)으로 호출하며, `None`이면 함수 내부의 `DEFAULT_STATIONS` 하드코딩 dict로 폴백한다(`store_mapping.py:76-77`). `data/internal/store_mapping.yaml`을 실제 경로로 넘기는 호출부는 코드베이스 어디에도 없다. **이 파일은 현재 dead weight다** — 삭제해도 동작에 영향 없거나, 반대로 진짜 real-data override 용도라면 실제로 배선해야 한다(이건 Task 1 범위를 벗어난 버그/설계 이슈이므로 architect 확인 필요, 여기서는 사실만 보고). 위치 제안: 데이터가 아니므로 `data/` 트리 밖 — `config/store_mapping.yaml` 또는 패키지 내 `src/bakery/config/store_mapping.yaml`.

### 3.4 `daily_normal_vs_bulk.parquet`
**판정: `rebuild-deterministic`(조건부, §Step2 sales.parquet 종속).** `scripts/bulk_detector.py`가 v2/`inventory`+`sales`+`items`에서 생성. scripts 소비처 3곳(`bulk_detector.py`, `backtest_normal_target.py`, `build_dashboard.py`) 확인, src/tests 0건. cruft 아님 — 실사용 중.

### 3.5 `item_active_stats.parquet`
**판정: `cruft`.** `scripts/bulk_detector.py`가 생성하지만, 생성 스크립트 자신 외에 아무도 읽지 않는다(grep 확인, 외부 소비처 0). 삭제해도 재실행 시 재생성됨.

### 3.6 `sales_with_bulk_flag.parquet`
**판정: `cruft` 후보이나 brief의 "소비처 0" 전제와 실측이 다르다 — 경계 사례로 보고.** `scripts/store_daily.py:50` 주석에 "구 sales_with_bulk_flag.parquet(whole-receipt) 대체, 패키지 CLI 경로와 단일 출처 통일"이라고 명시되어 새 canonical 경로(`bakery.data.bulk.flag_bulk_lines`)로 대체된 것으로 보이지만, **실측 grep 결과 여전히 3개 스크립트(`absorption_4stores.py`, `all4_stores_backtest.py`, `v4_new_data_backtest.py`)가 이 파일을 직접 읽는다** — 소비처가 0이 아니다. brief는 "필요 시 cruft"로 제안했지만, 근거를 확인하니 완전한 0-소비처 조건을 만족하지 않으므로 **즉시 삭제 대상으로 단정하지 않는다.** Task 2/3에서 저 3개 스크립트를 `store_daily.py` 경로로 마이그레이션한 뒤에야 진짜 cruft가 된다 — 지금은 "superseded-but-referenced"로 라벨링.

## Task 2/3에 전달할 리스크 요약

1. **`v2/sales.parquet` 재현 불가** — 현재 파일은 커밋된 스크립트로 재생성되지 않는다(§Step2). byte-preserving 물리 이동만 하고, 재빌드 시도 금지. **`daily_4stores`/`daily_normal_vs_bulk`/`waste_alpha_4stores`의 `rebuild-deterministic` 라벨은 이 `sales.parquet`이 byte-preserving으로 보존된다는 전제 하에서만 유효하다** — Task 3가 `sales.parquet`을 "Excel에서 재생성 가능하니 스킵 가능"으로 오독하면 이 3개 하류 파일도 함께 재현 불가 상태에 빠진다.
2. **`waste_alpha_4stores.parquet`는 src가 소비하는 유일한 v2/ 파일** — `_DATASETS` registry에 반드시 추가, `cli.py:1616` 하드코딩 경로 재배선 필요.
3. **`store_mapping.yaml`이 현재 dead weight** — 아무 코드도 이 파일을 실제로 로드하지 않음(전부 하드코딩 폴백). 이동 전에 실제로 쓰이게 할지, 폐기할지 architect 결정 필요.
4. **`sales_with_bulk_flag.parquet`는 완전한 cruft가 아님** — 3개 스크립트가 아직 참조. 스크립트 마이그레이션 후에만 삭제 가능.
5. `data/etc/` 디렉토리(아티제 실무자 미팅 자료, OCR 텍스트 등)는 brief 범위(`data/internal`, `data/internal/v2`, `data/external`) 밖이라 이번 표에서 제외했다 — Task 2/3에서 필요하면 별도 검토.

**Current goal**: Task 1 discovery 완료 — 전체 데이터 아티팩트 분류표 확정, Task 2(paths.py)·Task 3(물리 이동)의 근거 마련.
**Last decisions**: `v2/sales.parquet`=move-only(재현 불가 확인), `waste_alpha_4stores.parquet`=registry 필수 등록 대상, `store_mapping.yaml`=현재 미사용 확인, `sales_with_bulk_flag.parquet`=cruft 보류(3 consumer 실측).
**Open risks**: (a) `v2/sales.parquet` 72MB 확장본의 정확한 생성 로직 미문서화 — 유실 시 재현 불가, (b) `store_mapping.yaml` dead weight가 의도된 것인지 버그인지 미확인, (c) `bonavi_daily_2026h1_covariate.parquet` 용도/향후 통합 계획 미확인.
**Next first step**: Task 2 — `src/bakery/data/paths.py` registry 구현 시 이 표의 `new_path` 컬럼을 그대로 `_DATASETS`에 반영하고, 특히 `waste_alpha_4stores`를 누락 없이 추가.

# 데이터 파운데이션 재설계 (raw / interim / processed)

**날짜**: 2026-07-25
**상태**: 설계 승인됨 (사용자 확인 "이렇게 진행")
**로드맵**: Harness Backbone 마스터 로드맵 **3단계 — 데이터 프리프로세싱** (`project_harness_backbone`). 후속스펙 A(전처리 일반화)를 흡수. 4단계(무결성 검사 셋업)는 본 스펙의 커버리지 리포트를 씨앗으로 삼되 별도.
**참조**: `project_new_data_20260721`, `project_new_data_ingestion_pitfall`, `project_soldout_rate_dilution`, `project_measurement_charter`

---

## 1. 문제 정의

아띠제에서 받은 원본은 모두 Excel이고 `data/internal/`에 원본(`.xlsx`/`.xls`)과 파생(`.parquet`)이 **한 폴더에 섞여** 있다. `data/external/`도 raw zip과 파생 parquet이 혼재한다. 그 결과:

- **원본↔사용데이터 경계 없음** — 어느 게 불변 원본이고 어느 게 재생성 가능한 파생인지 디렉토리로 구분 안 됨. 실수로 파생을 원본처럼 다루거나 그 반대의 위험.
- **경로 하드코딩 산재** — `data/internal` 참조 74곳, `data/external` 26곳, `.parquet` 리터럴 231곳. 경로 상수는 `config.py`의 `EXTERNAL_DATA_DIR` 하나뿐. 파일이 어디 있는지가 **단일 출처가 없음** → 이동/변경이 곧 지뢰밭.
- **소스 간 커버리지 불일치(cross-source rot)** — Excel들은 같은 표의 vintage가 아니라 **상호보완 소스**다: `0520`=생산/폐기(inventory)·closing, `0526`=마스터/할인코드/hours, `0721`=라인레벨 판매(2021~2026-06), `진열시간.xls`=광교 브레드 진열. 파이프라인 경로마다 **다른 소스를 읽어** rot이 생겼다 (아래 §5에 규명된 불일치 목록).
- **외부 데이터 갱신 미체계** — 날짜가 지나면 신규 관측/예보가 생기는데, per-source ingest 명령을 사람이 일일이 챙겨야 하고 커버리지/최신성 확인 수단이 없다.

**목표**: 원본↔파생 위치 분리 + 경로 중앙화 + 재현 가능한 단일진입 파이프라인 + 모든 소스 크로스체크 커버리지 리포트 + 외부 갱신 CLI. **파운데이션이라 클리어하고 정교하게** — 순수 재배치는 수치를 바꾸지 않도록 동등성으로 보증한다.

## 2. 핵심 설계 원칙

1. **두 종류 변경을 분리하고 게이트를 다르게 건다.**
   - **순수 재배치**(파일 이동·경로 중앙화·단일진입 파이프라인이 *같은* processed parquet 생성): 모델 출력이 **바뀌면 안 됨**. Harness Phase 1과 동일한 동등성 규율로 보호 — 새 파이프라인으로 재빌드한 processed == 현재 processed, 수치 `rtol=1e-9` 일치.
   - **의도적 로직 수정**(downstream 수치 바뀜): 매진률 재스코프(§6)만 본 스펙에 동승(이미 승인). 각자 별도 커밋·별도 재측정 게이트.
2. **재구현 금지** — 기존 loader/build 로직(`bonavi_loader`, `bonavi_loader_v2`, `build_*`)을 **호출만** 한다. 파이프라인은 얇은 오케스트레이션.
3. **탐지 우선, 화해는 별도 결정** — 크로스체크는 소스 간 갭/충돌을 **표면화만** 한다. 수치를 바꾸는 화해(reconcile)/리베이스는 리포트가 근거를 제공한 뒤 개별 게이트로 결정한다 (closing 소스 통일이 대표 사례 — 본 스펙에서 **탐지만**).
4. **경로는 단일 출처** — 파일 위치를 named registry 한 곳에 모아, 100곳+ 흩어진 리터럴을 accessor로 대체한다.
5. **원본은 불변** — `raw/`는 read-only. 파이프라인은 raw를 읽어 interim/processed를 생성하되 raw를 절대 쓰지 않는다.

## 3. 디렉토리 레이아웃

```
data/
  raw/            # 불변 원본 (gitignored, read-only)
    internal/     보나비 데이터_20260520.xlsx / _20260526.xlsx
                  보나비 판매 데이터_20260721.xlsx / 수원광교점 진열시간.xls
    external/     living_pop_zips/ + api_cache/ (원 응답 캐시, 있으면)
  interim/        # 소스별 클린 (시트스왑 fix·semantic 컬럼매핑·dtype 정규화)
    sales_lines_clean.parquet
    ...           + _manifest.json (소스 vintage·행수·커버리지 스탬프)
  processed/      # canonical 사용데이터 (모델/하네스 진입점)
    internal/     sales/ items/ inventory/ stockout/
                  category_daily/ adjusted_demand/ store_daily/
    external/     weather/ calendar/ forecast/ consumption/
                  competitor/ living_population/ population/
```

- 기존 `data/internal/v2/`, `.pre-*-bak`, `sales_p1/p2`, `sales_with_bulk_flag` 등은 **소비처 0 확인 후** processed로 이동 또는 삭제 (cruft 정리는 grep으로 consumer 검증 후).
- `진열시간.xls`(DRM 해제·OLE2)는 raw/internal에 두고, interim 클린은 만들되 processed 편입은 Item 정적속성 배선(로드맵 5단계)까지 보류 — 본 스펙은 위치 정리 + 클린까지.
- `data/etc/`(baseline 로직·품목분류 문서)는 데이터가 아닌 참고자료 → 이동 없이 유지.

## 4. 경로 중앙화 (파이프라인 골격의 핵심)

- 신규 `src/bakery/data/paths.py`:
  - 레이어 루트 상수 `RAW_DIR / INTERIM_DIR / PROCESSED_DIR` (환경변수/인자 override 가능).
  - **named dataset registry**: `dataset("sales")` → `PROCESSED_DIR/internal/sales/...parquet`. 이름→경로 단일 매핑. provenance/vintage 태그 접근자 포함.
- **canonical `src/` + `tests/` 만 named accessor로 마이그레이션**(테스트 게이트). 일회성 `scripts/`(~55곳)는 **하위호환 심링크 shim**(옛 경로 → 신 위치, gitignore된 로컬 전용)으로 하드코딩 리터럴을 안 깨지게 유지 — dead 코드에 광범위 편집 강요 회피(사용자 결정).
- `config.py`의 `EXTERNAL_DATA_DIR`는 `paths.py`로 흡수(하위호환 alias 유지).
- **src가 실제 소비하는 데이터 파일은 ~10개**(bonavi_daily/receipts=재빌드-결정적, sales_lines_clean=interim, 외부 7종=move-only). `data/internal/v2/` 테이블(inventory/items/stores/stockout 등)은 **src 소비처 0**(scripts 전용) → cruft/orphan 분류 대상.

## 5. 단일진입 파이프라인 `bakery build-data`

- raw → interim → processed 스테이지를 순서대로 실행. 기존 loader/build 함수를 **호출**하는 오케스트레이터(`src/bakery/data/pipeline.py`).
- 스테이지: (1) raw Excel 파싱 → interim 클린 parquet (2) interim → processed canonical 테이블 (3) 크로스체크 리포트(§7) 생성.
- **동등성 = 게이트가 아니라 진단(advisor 교정)**: 재배치 자체는 **byte-preserving**(파일 이동은 내용 불변 → 수치 불가침)이라 재빌드로 보증할 필요가 없다. 재배치 안전망은 (1) grep 전수 열거 (2) 테스트 통과 (3) byte-identity 확인. `build-data`는 **별도 신규 능력**으로, 현재 processed는 `.pre-*-bak`이 증명하듯 수개월 in-place 변형으로 누적된 것이라 clean rebuild가 disk와 다를 수 있다.
  - **내부 결정적 테이블**(bonavi_daily/receipts): 진단 결과 `build_v2`(clean→daily) 재생성이 on-disk와 **rtol=1e-9 완전 일치 확인됨**(2026-07-25) → 이들엔 **진짜 동등성 게이트** 적용 가능.
  - **누적/orphan·외부 API 테이블**: 재생성 불가 → **move-only**(이동 후 존재 확인만). rebuild가 disk와 어긋나면 그건 파이프라인 버그 아니면 아티팩트 드리프트 → **§6 커버리지 리포트로 흘려보냄**(탐지), 재배치를 막지 않음.
- 재개/캐시는 최소한(스테이지 완료 마커). 임의 스테이지 재개 정교화는 본 스펙 범위 밖.

### 규명된 크로스소스 불일치 (탐지 대상, 화해는 별도 게이트)
`project_new_data_ingestion_pitfall` Open risks에서 이미 확인된 것들 — 리포트가 이걸 표면화해야 한다:
- **closing 소스 불일치**: `build_item_adjusted_demand`는 클린 parquet(`CD_USERDEF1`) 사용, `build_category_daily`는 여전히 옛 `0520`(`load_sales_with_discount`)에서 closing 읽음. 헤드라인 KPI(폐기 −38%·WAPE 7.8%)가 옛 closing에 얹혀 있음. **탐지만 — 통일 시 리베이스라 별도 결정.**
- **필드별 시간 커버리지 갭**: 생산/폐기/매진 라벨 = 2021~2025만, 2026 H1(+6개월)은 sales-only covariate.
- **품목 마스터 미매칭 67.9%**(광교, 음료+비베이커리+무명) → 갱신마스터 필요 (음료 트래픽 T5 blocked 원인).
- **카테고리 정의 불일치**: 예측력 파이프라인 3-cat(bread/pastry/sandwich) vs canonical 166=5-cat.

## 6. 크로스체크 / 무결성 커버리지 리포트 (사용자 핵심 요구)

- 모든 Excel(0520/0526/0721/진열시간)을 프로파일 → **커버리지 매트릭스**: `source × store × category × date-range × field`. 한쪽에 있고 다른 쪽엔 없는 것(missing/conflict)을 셀 단위로 표면화.
- **codex-data-cruncher로 위임** (451MB+138MB, 대화 밖 처리). ⚠️ **dispatch에 반드시 명시** (지난번 시트2 스왑 아티팩트 재발 방지):
  - **per-sheet English placeholder(row1) 헤더 매핑** — 위치·전역 헤더 금지.
  - **값 기준 판별** — `SALES_FG` 0/1 vs `SALES_TIME` 14자리로 시트2 컬럼 스왑 정정.
  - **앵커 검증** — 반품 1.88%(107,543), 광교 same-item 총량 510,585, 기존 canonical 전월 diff 0.
- 산출물 = **지속(durable) 리포트** (`reports/data_coverage/` HTML/parquet). 로드맵 4단계 무결성 프레임워크의 씨앗.
- **탐지만**: §5의 불일치들을 리포트에 적기만 하고 자동 화해/리베이스 안 함.

## 7. 외부 데이터 갱신 `bakery refresh-external`

- 기존 8종 ingest 어댑터(`ingest/*_api.py`)를 오케스트레이션. `ingest/`만 터치 → 내부 loader와 독립, **병렬 진행 가능**.
- **관측/실측형**(weather_observed·calendar·consumption·competitor·living_population·population): idempotent backfill — 있는 구간 skip, 오늘까지 새 날짜만 append.
- **예보형**(forecast_short/mid_term): 매번 덮어쓰기(forward-looking).
- 실행 후 **freshness/커버리지 요약**(소스별 최신 날짜·결측 구간) 출력.
- 스케줄링(cron)은 본 스펙 범위 밖 — 전향검증 단계에서 추가.

## 8. 매진률 진입점 fix (동승, 이미 승인)

- 문제(`project_soldout_rate_dilution`): 광교 item-day `is_stockout` = **0.151**은 완제품(생산기록 QT_MADE/QT_OUT 없는 `etc` 968/1150품목)을 분모에 넣어 검열 희석. 진짜 = 생산품목 기준 **0.605**.
- fix: `bonavi_loader.assign_stockout_fields` / `scripts/store_daily.build_store_daily`에서 완제품(inventory 미커버)을 **분모에서 명시적 제외 또는 censored NaN 처리**(0으로 깔지 않음 — 측정헌장 2번). median t(18시)는 유효하므로 rate만 교정.
- `test_store_daily_redefine` canary 해소(기대값을 0.151로 고착시키지 않고 재정의 로직으로 해소).
- **개별 게이트**: 수치를 바꾸므로 재배치 동등성 게이트와 **분리**한 별도 커밋·검증. 재측정 영향(store_daily 소비처)은 §9 열거 대상.

## 9. 시퀀싱 & 게이트

1. **경로 참조 전수 열거** (구현 전 필수, no-yolo 규칙): `data/internal`·`data/external`·`.parquet` 리터럴·loader 경로 상수를 `src/ scripts/ tests/ experiments/*.yaml` 전체 grep, **테스트 fixture 포함**. 어느 parquet이 실제 소비되는지(cruft 판별)도 이 패스에서 확정.
2. **재배치 먼저** (동등성 게이트) — 디렉토리 이동 + 경로 중앙화 + `build-data`. processed 수치 `rtol=1e-9` 불변 확인. 553+ 테스트 통과.
3. **매진률 fix** (개별 게이트) — §8, 별도 커밋. store_daily 소비처 재측정.
4. **refresh-external** (병렬) — §7, ingest만 터치.
5. **크로스체크 리포트** — §6, 재배치 후 언제든. closing 통일 등 화해 결정은 리포트 근거로 별도 세션.

## 10. 성공 기준

- [ ] `data/{raw,interim,processed}/` 레이아웃으로 원본↔파생 물리 분리 완료(이동 파일 byte-identity 확인).
- [ ] `src/bakery/data/paths.py` named registry로 **`src/`+`tests/`** 데이터 경로 리터럴 중앙화, 이 두 곳 하드코딩 잔여 0. `scripts/`는 하위호환 심링크 shim으로 유지(마이그레이션 대상 아님).
- [ ] `bakery build-data`가 내부 결정적 테이블(bonavi_daily/receipts)을 재생성하고 **rtol=1e-9 동등성 게이트** 통과. 외부/orphan은 move-only로 존재 확인. rebuild-vs-disk diff는 커버리지 리포트로 표면화.
- [ ] 전체 테스트 스위트(553+ / leakage 포함) 통과.
- [ ] `bakery refresh-external`이 8종 소스를 idempotent 갱신하고 freshness 요약 출력.
- [ ] 커버리지 매트릭스 리포트가 §5의 규명된 불일치 4종을 모두 셀 단위로 표면화(codex 산출물 앵커 검증 통과).
- [ ] 매진률 진입점 fix로 광교 생산품목 매진률 0.605 산출, `test_store_daily_redefine` canary 해소.

## 11. 범위 밖 (명시적 제외)

- **closing 소스 통일 / 헤드라인 KPI 리베이스** — 탐지만. 리포트 근거로 별도 결정.
- **2026 H1(0721 신규 6개월) canonical 편입** — 로드맵 7단계. 본 스펙은 현 vintage(2025-12 컷오프) 보존.
- **진열시간 processed 편입 / Item 정적속성 배선** — 로드맵 5단계. 위치 정리 + interim 클린까지만.
- **외부 갱신 스케줄링(cron)** — 전향검증 단계.
- **음료 트래픽(T5) / 갱신마스터** — 별도.
- **무결성 검사 프레임워크(반복 검증)** — 로드맵 4단계. 본 스펙은 커버리지 리포트(1회성 산출)까지.

## 12. 리스크

- **경로 마이그레이션 규모**(100곳+): mechanical하지만 광범위. 동등성 게이트 + 전체 테스트가 안전망. 태스크로 잘게 분해.
- **동등성 게이트가 잡아내는 것은 "현재 processed와 동일"뿐** — 현재 processed에 이미 있는 오류(예: category closing 옛0520)는 동등성으로 안 잡힘. 그건 §6 리포트가 담당(탐지). 두 안전망의 역할 분리를 혼동하지 말 것.
- **codex-data-cruncher 재발 위험** — 시트2 스왑 아티팩트 전례. dispatch 명세(§6) 엄수 + 앵커 검증 필수, 산출 그대로 신뢰 금지.
- **cruft 삭제** — `.pre-*-bak`/`sales_p1/p2` 등 소비처 0 확인 전 삭제 금지(no-yolo 4번).

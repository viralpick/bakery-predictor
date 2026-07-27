# 신규 데이터 편입 마무리 (Phase 7) — 실행 스펙

작성일 2026-07-27. 마스터 로드맵 7단계. 선행 스펙 = `2026-07-23-new-data-integration-design.md`(편입 배관·타깃 재정의·분석 팬아웃).

## 배경 — 이 스펙이 다루는 것 / 안 다루는 것

로드맵 7단계는 "아띠제 대규모 신규데이터 편입"이지만, **편입 자체는 이미 대부분 완료**돼 있다(2026-07-23 작업 + 파운데이션 재설계 PR#56). 착수 시점 코드 실측으로 확인:

- `pipeline.build_internal` → `bonavi_loader_v2.build_v2` → 신규 0721(`sales_xlsx`)로 canonical `bonavi_daily.parquet`를 직접 재생성. 편입 배관 완성.
- 현 canonical = 광교 단독·166품목(신규 타깃 정의)·2021~2025-12. adjusted_demand도 신규 클린 parquet(CD_USERDEF1)로 재배선(선행 스펙 Q1).
- 고객 지표 경로(`harness-run` = category_total+event_prior)의 `build_category_daily`가 이 166 canonical을 읽음.

따라서 이 스펙의 실질은 **"편입 마무리"** = ① 4매장 확장 ② 광교 헤드라인 KPI 재측정 ③ 컨플릭트 없는 정리 ④ 열린 결정 문서화.

**범위 밖(불변)**: 전향(prospective) 4주 실측 검증 — 컷오프 2026-06-30 이후 실시간 피드 필요. 2026 H1 covariate는 학습 타깃 아님. 원가율 미보유(폐기비용 판매가 계열 유지).

## 착수 전 검증 결과 (2026-07-27, 코드 실측)

사용자 조건: "우리가 세밀하게 잡아놓은 처리가 신규 canonical에 다 반영돼 있고, 데이터 컨플릭트가 없어야 4매장 canonical 생성." → 실측으로 충족 확인:

**① 처리 반영 (build_v2 빌드 경로):**

| 처리 | 반영 위치 | 확인 |
|---|---|---|
| 벌크 제외 | `load_sales_v2` `flag_bulk_lines` | ✅ |
| 반품 제거 | `load_returns_v2` → `aggregate_daily(returns=)` | ✅ |
| 생산량 연계 | `ingest.inventory.load_inventory`(재고정보 시트) → `aggregate_daily` | ✅ |
| stockout 재정의 | `assign_stockout_fields`(물리 leftover made>0 & waste<=0 + companion mask `is_stockout_defined`) | ✅ |
| 품목별/타깃 | `load_items_v2`(당일폐기Y − salad) | ✅ |
| 마감할인 α=0.8 **적용** | downstream `build_category_daily(alpha=0.8)` (canonical daily 아닌 target 생성 시점 — 정상 아키텍처) | ✅ |
| 마감할인 closing **소스** | `build_category_daily`가 `load_sales_with_discount()`/`load_closing_returns()`를 **default args**로 호출 → `DEFAULT_XLSX = legacy_xlsx_0520` = **옛 0520 파일** | ⚠️ **갭** |

⚠️ **착수 전 발견한 배선 갭 (advisor 지적으로 규명)**: α=0.8이 *적용*되는 것과, α가 곱해지는 *closing 데이터가 신규 소스*인 것은 다른 주장이다. 선행 스펙 Q1의 재배선은 **item-level** `build_item_adjusted_demand`만 신규 클린 parquet(CD_USERDEF1)로 옮겼고, **harness 헤드라인 타깃 `adjusted_demand_unit`(category-level)의 `build_category_daily`는 여전히 옛 0520 파일에서 closing을 읽는다**. 결과: 겹치는 139품목·구간은 근사 보존, 신규 27품목·확장구간은 `closing_qty=0`으로 빠져 `adjusted_demand=sold_units`로 과대. → 이 상태로 헤드라인을 재측정하면 틀린 고객 KPI를 보고하게 됨(`feedback_customer_kpi_measurement`의 "closing 클린vs옛0520 불일치" landmine). **remediation 필요** = 아래 작업 0. drop-in 가능 확인: `load_sales_with_discount_v2`/`load_closing_returns_v2`(`discount.py`)가 동일 스키마·라벨로 이미 존재(Q1 산출), 인자 `parquet_path=sales_lines_clean`·`store_code`.

**② 데이터 컨플릭트 (게이트 3종):**
- `bakery build-data --diagnose`: 재빌드 동등성 `max_diff=0`.
- `bakery check-conflict`: 옛/새 마스터 값 충돌 `conflicting=0`.
- `bakery check-integrity`: fail 0. drift 2건 — 할인코드 357 미정규화(기존 알려진 것)·비타깃 품목 1649개(전부 `is_target_scope=False` = 음료/완제품, 타깃 아님). **하드 충돌 없음.**

**③ 4매장 데이터 커버리지:**
- 판매 라인레벨(클린 parquet): 광교47(165만행)·삼성9(180만)·메세나29(117만)·광화문485(111만).
- 생산량/폐기량(재고정보 시트): 광교47(99K)·메세나29(91K)·삼성9(90K)·광화문485(63K).
- `STORE_CODE_MAPPING`: 4매장 확정(gw01/ss01/gh01/mp01).

→ **4매장 편입은 데이터 측면에서 클린.** build_v2가 광교만 뽑는 걸 4매장 루프로 확장하면 됨.

## 절대 규칙 준수 (프로젝트 CLAUDE.md)

- Time leakage 금지: 편입은 데이터 레이어. 재측정은 기존 leakage-safe harness 경로 그대로. leakage 테스트 전부 통과 유지.
- 품절 censored: `assign_stockout_fields` companion mask 로직 불변.
- Random split 금지: 재측정은 harness expanding window(52 folds).
- Synthetic↔Real 경계: 다매장 산출물도 `data/schema.py` `DAILY_COLUMNS` 준수.
- MAPE 단독 금지: 재측정 메인 = WAPE(총량·카테고리).

## 아키텍처 결정

### D1. 다매장 저장 = 별도 `multistore_daily.parquet` (광교 canonical 불변)

`build_category_daily`·`harness-run`은 store 필터 인자가 없어 canonical `bonavi_daily.parquet`를 통째로 읽는다. 4매장을 이 파일에 합치면 광교 헤드라인 총량이 4매장 합으로 오염된다.

**결정**: 기존 `bonavi_daily.parquet`(광교 단독)은 **byte-identical 불변** 유지(build-data equivalence `max_diff=0` 게이트 계속 통과). 4매장은 **별도 `data/processed/internal/multistore_daily.parquet`** 로 생성.

- 광교 헤드라인 오염을 구조적으로 불가능하게 함.
- 최소 침습: 기존 소비처(harness/category_aggregate) 무변경.
- multistore parquet의 광교(store_gw01) 파트는 기존 canonical과 **정합**해야 함(대조: 공통 품목 sold_units/is_stockout `max_diff=0`).

### D2. 광교 헤드라인 KPI 재측정 = 고객 지표 경로 고정

재측정은 반드시 `harness-run`(category_total + event_prior + adjusted_demand)로 한다. **item-level daily WAPE(선행 스펙 Q2의 0.22대)는 고객 지표가 아니므로 재활용하지 않는다** (`feedback_customer_kpi_measurement` "3연속 교정").

- **선행 게이트(필수)**: 작업 0(category-level closing 소스 재배선)이 완료되기 전에는 어떤 헤드라인 delta도 보고하지 않는다. 옛 0520 closing 위 재측정 = 틀린 KPI.
- 메인 delta: 광교 **총량 WAPE**(옛 headline naive 8.19 vs 우리 8.03 기준) + **폐기율(1차 KPI)** + **매진(2차)**.
- 정직한 프레이밍: 이 delta는 **146→166 품목/타깃정의 변화 + closing 소스 교체**분이지 2026 검증이 아님(날짜 여전히 2021~2025-12). 리포트에 경계 명시.

### D3. 타 3매장 = 참조 예측만 (타깃 아님)

PoC 검증대상 = 광교 단독(CLAUDE.md). 삼성·메세나·광화문은 multistore_daily 위에서 예측을 돌리되 **헤드라인·성공기준·최종 비교엔 쓰지 않는다**. pooling/매장간 분석의 보조 자산.

## 작업 단위

0. **[게이팅] category-level closing 소스 재배선** — `build_category_daily`의 `load_sales_with_discount()`/`load_closing_returns()` default(옛 0520) 호출을 신규 클린 parquet 경로(`load_sales_with_discount_v2`/`load_closing_returns_v2`)로 교체. 동일 스키마라 drop-in이나 헤드라인 타깃 값이 바뀌므로: 기존 테스트 기대값 재보정, 마감할인 반영 품목 수(item-level Q1은 130→157)·구간 정합 확인. **작업 3(헤드라인 재측정) 전 필수 완료.** 이 작업 없이는 delta 보고 금지.

1. **4매장 canonical 빌더** — `build_v2`(또는 얇은 상위 `build_multistore`)를 4매장 루프로 확장. 각 매장 `load_inventory`+`build_v2` → concat → `multistore_daily.parquet`. `paths.py` registry 등록. `build-data`/`pipeline` 배선(광교 bonavi_daily 경로는 불변).
2. **정합성 대조** — multistore의 광교 파트 vs 기존 canonical `max_diff=0`. 타 3매장 sanity(타깃 수·매진률·날짜범위)를 리포트로 표면화. leakage 테스트 전부 통과.
3. **광교 헤드라인 재측정** — `harness-run experiments/gwangyo_default.yaml` → 총량 WAPE + 폐기/매진, 옛 146 기준 delta 리포트(경계 명시). 고객 지표 경로 고정.
4. **타 3매장 참조 예측** — multistore_daily 위 예측 산출(별도 config 또는 store 인자). 참조용 표.
5. **정리 + 열린 결정 문서화** — drift 2건 상태·게이트 통과 durable 기록. 열린 결정(원가율 미보유·음료마스터 T5 blocked·진열시간 계획/실측·전향 실시간피드) 문서화.

## 성공 기준

- `multistore_daily.parquet` 생성, 4매장 포함, `DAILY_COLUMNS` 준수.
- 광교 canonical `bonavi_daily.parquet` byte-identical 불변(build-data `max_diff=0` 유지).
- multistore 광교 파트 = 기존 canonical 정합(`max_diff=0`).
- 게이트 3종(build-data/check-integrity/check-conflict) 통과 유지.
- **category-level closing 소스가 신규 클린 parquet(CD_USERDEF1)로 재배선됨** — 헤드라인 재측정 전 완료(작업 0).
- 광교 헤드라인 KPI(총량 WAPE·폐기·매진) 새 166 기준 + 신규 closing 소스로 재측정 + 옛 146 delta 리포트, 지표 경계 명시.
- 기존 leakage 테스트 + 전체 스위트 통과.

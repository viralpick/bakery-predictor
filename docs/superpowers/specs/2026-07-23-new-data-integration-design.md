# 신규 데이터 2종 편입 — 실행 스펙

> 작성 2026-07-23 · 대상: `data/internal/`에 신규 수령한 엑셀 2종
> 근거 문서: Notion "신규 데이터 2종 활용 검토" (3a602490cbe0801588c2d92ccfb24bad), 메모리 `project_new_data_20260721`
> 실행 형태: **단계 게이트 + 서브에이전트 팬아웃** / 이번 세션 목표 = **게이트 B(Foundation)까지**

## 대상 파일 (2026-07-23 존재/크기 확인)

- `data/internal/보나비 판매 데이터_20260721.xlsx` — **451MB** (라인레벨 POS, 6시트, 프로파일링 대상)
- `data/internal/수원광교점 - 브레드 진열 시간(보안 해제 완료).xls` — **37k** (광교 브레드 37품목 진열시각)

> ⚠️ 위 파일들의 **내용은 아직 이번 세션에서 재파싱하지 않았다.** 570만행·4매장 등 기존 팩트는 이전 세션이 Notion/메모리에 남긴 것 → Phase 0에서 직접 재검증한다.

## 예측 타깃 스코프 (사용자 정정 2026-07-23)

예측 대상은 기존과 동일하게 **당일생산/당일폐기 베이커리(빵)뿐**. 신규 파일의 음료·장기유통·완제품 등 전품목은 예측 타깃에서 **분리**하고 covariate/인사이트로만 사용. 분리 키 = 품목정보 **`당일폐기여부`**(Y=타깃 / N=분리).

## 절대 규칙 준수 (프로젝트 CLAUDE.md)

- Time leakage 금지: 판매시간·매진시각·당일 트래픽은 **라벨/평가 전용**, feature로 미래값 금지. 진열시간은 "계획 상수" 확인 시에만 static feature.
- 품절 데이터 censored 보존, 판매/품절 모델 분리.
- Random split 금지 (시간순 유지).
- Synthetic↔Real 경계: `data/loader.py`가 실데이터 진입점, `data/schema.py` schema 유지.
- 메인 지표 WAPE.
- 검증은 **광교 단독**, 타매장은 보조 (헌장).

---

## 아키텍처: 4단계 파이프라인

```
Phase 0  프로파일링/검증        ── 게이트 A (내가 검증) ──┐
Phase 1  canonical 재빌드+정합성 ── 게이트 B (사용자 승인) ┘  ← 이번 세션 목표
Phase 2  분석축 팬아웃 (6트랙, 병렬 서브에이전트)
Phase 3  온톨로지/액션 편입 + 종합
```

### Phase 0 — 프로파일링 (codex-data-cruncher 위임, 451MB)

나머지 모든 스펙 파라미터를 확정하는 사실 수집. 산출:

1. 스키마/무결성 — 6시트 실제 행수·컬럼·dtype, 매장 4곳 행수, 기간 min/max, 반품(판매구분=1) 비율.
2. `당일폐기여부` 조인 커버리지 — 신규 distinct 품목코드 중 마스터(541) 매칭 비율, 미매칭(음료 등) 규모.
3. 음료/비타깃 식별 — 미커버 품목 카테고리/품목명 패턴 잠정 분류(음료 후보 집합).
4. 할인코드 사전 — 코드별 빈도·시간대, 마감할인 코드(0069/0077/0320) 실사용량.
5. **proxy 타당성 실측** — (a) 음료 없는 순수-빵 영수증 비율, (b) 영수증당 음료 건수 분포.
6. 광교 정합성 기준선 — 광교 베이커리 4종 총량(기존 `bonavi_receipts.parquet` 대조용 기준 숫자).

**게이트 A (내가 검증):** 조인 커버리지·proxy 타당성 직접 확인. proxy 약하면 Phase 2 트래픽 트랙(T5) 강등/보류.

### Phase 1 — canonical 재빌드 + 정합성 (게이트 B, 사용자 승인)

- 신규 0721 → `loader.py` schema로 재빌드. 기존 로직 재적용: 반품 제거, stockout 재정의, bulk 필터(T1/T2), α=0.8.
- **⚠️ 게이트 A 발견 (2026-07-23)**: 0721 파일 `판매정보2` 시트만 `판매구분`(SALES_FG)/`판매시간`(SALES_TIME) 컬럼 순서가 나머지와 뒤바뀜(벤더 quirk). 어댑터는 반드시 **시트별 자체 헤더(English placeholder 행)로 컬럼 매핑** — 위치/전역 헤더 금지. 컬럼 의미: CD_USERDEF1=할인코드, CD_USERDEF2=셋트(SS/ST), SALES_FG=판매구분(0/1), SALES_TIME=판매시간(YYYYMMDDHHMMSS). 상세=메모리 `project_new_data_ingestion_pitfall`.
- 타깃 분리: `당일폐기여부=Y` 베이커리만 타깃, 나머지 covariate 테이블 분리 저장.
- **정합성 대조 필수**: 재빌드 후 광교 베이커리 총량 vs 기존 parquet. 불일치 시 원인 규명 전 게이트 통과 금지.
- 기존 leakage 테스트 전부 통과.

### Phase 2 — 분석축 팬아웃 (승인 후, 서브에이전트 병렬 6트랙)

| # | 트랙 | 의존 | 산출 |
|---|---|---|---|
| T1 | 반품 제거 순수요 감사 | P1 | 기존 처리 반품 분리 여부 + 순수요 재정의 |
| T2 | 할인코드→마감 α 실측 | P1 | `project_closing_discount_alpha` α 진전 |
| T3 | 판매시간→매진시각 재계산 | P1 | 라벨 재구성 (광교 카테고리품절≈0이라 이득 제한 명시) |
| T4 | 진열시간→수요율 정규화 | P1+진열 | 판매량÷가용시간, Item 속성화 |
| T5 | 음료 트래픽 심화 | P0 게이트 통과 시 | attach 정규화·효과 재해석·footfall·pooling·이상치 귀인 |
| T6 | 매장간/타입별 인사이트 | P1 | pooling 가중·매장생산 vs 완제품 제약 분리 |

각 트랙 서브에이전트 산출을 **내가 검증** 후 채택.

### Phase 3 — 온톨로지/액션 편입 + 종합

- Item 속성(`display_time`, `item_type`), `StoreTraffic` 파생, `SoldOut`/`Markdown` 이벤트, **진열/생산 타이밍 액션** 추가.
- Notion 검토 문서 갱신 + 메모리 갱신.

---

## 문서화된 가정 / 블록 (아티제 확인 대기 — 진행은 가정으로)

- **진열시간 = 계획/표준값** 가정 → leakage-safe static 속성. 실측 판명 시 관측 라벨로 재분류.
- **갱신 품목 마스터 부재** → 음료는 휴리스틱 식별. 확정 마스터 수령 시 교체.
- **원가율 미보유** → 폐기비용 판매가 계열 유지, 원가 gap 미해결 명시 (`project_data_gaps` 1순위).
- **전향 4주 비교** → 컷오프 2026-06-30로 불가, 실시간 피드 필요 (이번 범위 밖).
- **proxy 타당성** → Phase 0 게이트 A에서 실측 판정. 약하면 T5 축소.

## Phase 1 실행 결과 (2026-07-23)

- 신규 모듈 `src/bakery/data/bonavi_loader_v2.py`: `convert_sales_to_parquet`(per-sheet 값판별 스왑 교정) + `load_items_v2`(0526 한글 마스터, 타깃=Y−salad) + `load_sales_v2`/`load_returns_v2`/`load_receipts_v2` + `build_v2`. 집계·stockout·potential_demand는 `bonavi_loader` 재사용.
- 클린 캐시 `data/internal/sales_lines_clean.parquet`(77M) 생성. converter 원본 xlsx에서 재현 검증 완료.
- **정합성 대조 (v2 daily vs 기존 canonical, 광교 2021-2025)**: 공통 139품목 sold_units **(item,date) 100.00% 정확 일치, |diff|=0**, is_stockout rate 0.605=0.605. → ingestion·집계·반품·stockout 재정의 정확 재현.
- 타깃 재정의 차이(의도): v2 166품목(라벨구간) = 공통 139 + 신규포함 27(옛 분류기 누락 Y 베이커리) − old-only 7(당일폐기=N 크리스마스특별/머핀, 335개, 올바른 제외). 2026 신제품 21개는 라벨구간 밖.
- 테스트: `test_map_category_new_items`(17), `test_bonavi_loader_v2`(3, swap 유닛 + 타깃 정의).
- **⚠️ 미배선(게이트 B 대기)**: `build_v2`는 임시 경로로만 검증. 기존 canonical(`bonavi_daily.parquet`)·loader·CLI에 **아직 배선 안 함** — 승인 후 진행.

## Phase 2 팬아웃 결과 (2026-07-23, 각 트랙 메인스레드 검증 후 채택)

리포트: `<scratchpad>/phase2_t2_report.md`(마감α), `phase2_t4_report.md`+`phase2_t4_rate_table.csv`(진열시간), `phase2_t6_report.md`(매장간, 상관값 SUPERSEDED 주석).

**T1 — 반품 순수요 감사 (검증필: 메인)**: v2 net-out 정확 적용. 광교 타깃 정상 540,610 − 소매반품 9,868(1.83%, 161품목/7,648건) = canonical 530,748. 순수요 정제 정상.

**T2 — 할인코드 → 마감할인 (검증필: 독립 재계산)**:
- [검증사실] 광교 마감할인 코드 0069(20%)·0077(30%) **둘 다 20시대 집중**(평균 20.3/20.2시, 20시+ 98.5%/90.8%). 마감할인 일 비중 평균 15.5%·중앙 14.0%, 발생일 99.9%(상시). ★**정정**: 오염 캐시 시절 "0077=오후2시=마감아님"은 아티팩트 — 실제 오후2시대는 PAYCO(0121, 평균 13.5시)였고 혼동. 
- [추론+미해결] α 0.8+ **방향** 지지(마감 후 정가판매 대부분 소멸). 단 서브가 플래그한 "직전 2시간 정가 1.4배" 모순 미해소 → **0.8 확정 아님**. `project_closing_discount_alpha`의 저-α(미끼상품 잠식) 우려를 반증한 게 아니라 시간위치만 규명.

**T4 — 진열시간 → 수요율 정규화 (검증: rate table 교차확인, pre-display는 서브산출)**:
- [검증사실] 진열파일 37품목(매장생산26+완제품11), 코드 직접보유(이름매칭 불필요), daily 매칭 31/37(미매칭6=2026신제품/무판매). rate table item_id가 실제 광교 코드와 일치.
- [서브산출] pre-display 판매 0.27% → 진열시각=계획상수 leakage-safe 가정 지지. 완제품 전부 개점전 진열이라 rate 정규화 순위변동은 09:30~10:10 소수품목만(늦은진열 3종 평균 rank +3.0). ⚠️ uniform-traffic confound로 상승 ~11% 과대(서브 자기플래그). display_time=Item 정적속성 후보(`source=planned(assumed)`).

**T6 — 매장간·타입별 (검증: 독립 재계산으로 상관 정정)**:
- [검증사실] 4매장 pastry 최대(66~72%), 광교 bread 25.7%·주말 34.5%(오피스 14~24%). 광교 월별 카테고리 비율 std: bread 4.43·pastry 3.90(변동큼) vs cake/sandwich/salad ≤1.2(안정). ★상관 정정: 광교~메세나 r=**0.627**, 오피스 2매장 **−0.41(역상관)**. → 메세나만 카테고리별 선별 pooling 후보, 오피스 부적합(역상관). 매장생산/완제품 구분=마스터에 컬럼 없음(skip).

**T5 — 음료 트래픽: BLOCKED** — 음료 품목 무명(마스터 미커버) → 식별 불가. 갱신 품목 마스터 수령 시 unblock.

## ⚠️ 알려진 배선 갭 / 비교 주의 (durable 기록)

1. **정합성 "100%"의 정확한 범위**: sold_units·is_stockout만 (item,date) 완전일치. **adjusted_demand(헌장 TARGET)는 sold_units의 순수함수가 아님** — closing_qty를 `load_sales_with_discount(보나비 데이터_20260520.xlsx)` = **옛 파일**에서 읽음. 겹치는 139품목·구간은 근사보존, 신규 27품목·확장구간은 closing_qty=0으로 빠짐(adjusted_demand=sold_units로 과대). **결정 필요**: adjusted_demand 할인소스를 신규 클린 parquet(CD_USERDEF1)로 재배선. `potential_demand`는 substitution 2차패스 생략으로 옛것과 상이하나 real 경로 미사용(#3 감사, 폐기 확정컬럼).
2. **KPI 비교불가**: 기존 문서·메모리의 모든 KPI(naive WAPE 8.19, 폐기 −33~40%, 분포스택 full-window 등)는 **옛 146 canonical 기준**. 새 166 canonical(+14% 품목·믹스 변화)로 **백테스트 미재실행** → shift 크기 미상. "소폭 이동" 아님, **직접 비교 불가·재측정 필요**.
3. **2026 H1 covariate**: `data/internal/bonavi_daily_2026h1_covariate.parquet`(7,107행/65품목/48,597개, sales-only, 라벨없음) 별도 생성. 학습 타깃 아님.

## 사용자 결정 후속 (2026-07-23)

**Q1 — adjusted_demand 재배선 완료**: closing_qty 소스를 신규 클린 parquet(CD_USERDEF1)로 이동. 마감할인 반영 품목 130→157(신규 27품목 포함), 166 canonical과 소스 완전 정합. `discount.py` v2 로더 + `build_item_adjusted_demand` 클린 parquet 우선(옛 파일 CI 폴백). 536 통과. → 배선 갭 #1 해소.

**Q2 — 새 166 canonical KPI 재측정 (item-level daily WAPE, adjusted_demand, 4 folds real)**:
| 모델 | wape_all | wape_no_stockout | pct_under | pct_over |
|---|---|---|---|---|
| lightgbm_v1 | 0.2216 | 0.2478 | 21.4% | 33.8% |
| lightgbm(v0) | 0.2225 | 0.2597 | 21.9% | 36.1% |
| lightgbm_v2 | 0.2276 | 0.2567 | 23.9% | 34.2% |
| seasonal_naive | 0.2255 | 0.2943 | 13.2% | 43.9% |
| moving_average | 0.2498 | 0.3215 | 19.4% | 46.1% |
- ⚠️ **이 지표는 item-level daily WAPE로, 기존 헤드라인 "naive 8.19 vs 우리 8.03"(광교 총량 WAPE)과 다른 지표 — 직접 대체 아님.** 총량 WAPE·폐기(−33~40%)·매진 KPI는 별도 경로(business-report/order-sim)로 미재측정(후속).
- 정합 확인: item-level에서 LightGBM≈naive(과거 총량 결론과 일치), no-stockout·sandwich(v2 0.284 vs naive 0.339)는 v2 우위. pct_under 21-24%=과거 "v2 23%"와 일치. 리포트 `reports/new166/`.
- 배선 갭 #2(비교불가) 부분 해소: item-level baseline은 166 기준 확보. 총량·폐기 KPI 재측정은 후속 결정.

## 성공 기준 (게이트 B)

- Phase 0 산출로 타깃/비타깃 분리 규모와 proxy 타당성이 수치로 확정됨.
- canonical 재빌드본이 기존 광교 베이커리 총량과 정합(대조 리포트).
- 기존 leakage 테스트 전부 통과.
- 재빌드 산출물이 `data/schema.py` schema 준수.

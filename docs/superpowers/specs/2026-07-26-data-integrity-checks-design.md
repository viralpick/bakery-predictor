# 데이터 무결성 검사 (`bakery check-integrity`)

**날짜**: 2026-07-26
**상태**: 설계 승인 대기
**로드맵**: Harness Backbone 마스터 로드맵 **4단계 — 데이터 무결성 검사 셋업** (`project_harness_backbone`)
**참조**: `project_new_data_ingestion_pitfall`(sheet-2 스왑), `project_new_data_20260721`(미매칭 67.9%), `project_data_foundation_redesign`(3단계, coverage.py surface 층)

---

## 1. 목적 & 원칙

7단계 아티제 대규모 신규 데이터 ingest를 **de-risk**한다. 신규 데이터가 malformed(특히 sheet-2 컬럼 스왑류)이거나 마스터 갱신이 누락되면 **ingestion 경계에서 loud-fail**한다. 이 프로젝트 최고비용 실패(sheet-2 스왑 → "11개월 갭·정합성 −22%" 아티팩트, 전부 오진)를 잡는 게 존재 이유.

**핵심 원칙 (advisor 3-bucket + forward-invariant 규율):**
- 모든 후보 체크를 **실패 의미**로 분류한다:
  1. **Invariant** (실패 = 새 버그 → gate, exit non-zero)
  2. **Drift guard** (실패 = 알려진 상태가 바뀜 → 사람 확인, gate 아님)
  3. **Neither** (현실/이연 결정 → 커버리지 리포트에 유지, 스위트 아님)
- **4 conflict는 codify하지 않는다** (closing split·2026 label gap·미매칭·category 3v5 = 알려진·수용 상태 → coverage.py surface 층 유지).
- **coverage.py = surface 층**(리포트, 절대 안 실패). **integrity.py = checks 층**(loud-fail). 역할 분리.
- **기배포 2 메커니즘을 우산으로 통합**: `build-data --diagnose`(재빌드 결정성 rtol=1e-9)·refresh 커버리지 보존 가드. 재구현 아님 — 문서/CLI에서 "무결성 = 이 3개"로 통합 서술.

## 2. ★검사 대상 입력의 구분 (advisor 교정 — 스펙의 핵심)

각 체크가 **어느 입력에 도는지** 명시한다. 이걸 안 하면 "새 데이터 검사"라면서 현 파일에만 유효한 값으로 false-positive를 낸다.

| 입력 | 언제 | 어떤 체크 |
|---|---|---|
| **현 raw 재처리** (같은 파일) | build-data류 | **절대수 앵커** = vintage regression (예: 광교 same-item 510,585). 새 데이터엔 절대 안 검. |
| **새 drop** (미래 아티제) | check-integrity 본령 | **구조 invariant** + **비율 soft-range** + **FK/known-set regression**. |

## 3. 앵커/체크 분류 (advisor — forward invariant만 gate)

| 체크 | 종류 | 입력 | 기준 |
|---|---|---|---|
| `SALES_FG ∈ {'0','1'}` | **Invariant** | 새 drop | 값 판별. sheet-2 스왑이면 타임스탬프가 들어와 즉시 fail. (실측 확인: 현 clean = {'0','1'}) |
| `SALES_TIME` 14자리 숫자(YYYYMMDDHHMMSS) | **Invariant** | 새 drop | 값 판별. 스왑 시 0/1이 들어와 fail. |
| `source_sheet`별 컬럼 semantic 일치 | **Invariant** | 새 drop | 시트별 값 기준 판별(판매정보2 스왑 재발). |
| 17 컬럼 존재 + dtype 계약 | **Invariant** | 새 drop | 스키마 드리프트. (실측: DT_SALE/CD_ITEM string, AM_* int64 등) |
| (NO_POS, SLIP_NO, SLIP_LINE) 라인 유일 | **Invariant** | 새 drop | 중복 라인. |
| `DT_SALE` 품목×월 큰 갭 없음 | **Invariant** | 새 drop | 조용한 데이터 누락(품목×월 분해, sheet-2 교훈). |
| 반품비율 ≈ 1.88% | **Drift(soft-range 1~3%)** | 새 drop | 비율 invariant. 범위 벗어나면 사람 확인(gate 아님). (실측 0.0188) |
| 광교 same-item 총량 510,585 | **vintage regression** | 현 raw 재처리만 | 새 drop엔 검사 안 함(false-positive 방지). |

## 4. ★FK / 코드 누락·충돌 검사 (사용자 핵심 요구 — 아티제 역문의용)

새 데이터가 들어올 때 기존 마스터와 **매칭 안 되는 코드를 전부 잡아** 보고한다(누락 + 충돌 둘 다, 사용자 확정). **실측: 판매 품목 1867종 중 1649종(전매장/전품목 기준)이 마스터 미매칭.** 두 종류를 구분한다:
- **누락(missing)**: 새 데이터의 코드가 마스터에 아예 없음.
- **충돌(conflict)**: 같은 코드가 옛 마스터(0520) vs 새 마스터(0526)에서 **값이 다름**(품목명·할인율·카테고리 등). 실측 확인: 0520·0526 둘 다 품목정보/점포정보/품절정보/할인코드 시트 보유 → 두 vintage 대조 가능.

아래 fail/보고 분리는 누락·충돌 **양쪽에 동일 적용**한다.

### 4a. Fail 게이트 (well-defined invariant만)
- **품목 = known-target-set regression** (advisor blocker #1 해결):
  - "타깃 품목인데 마스터에 없음"은 **구현 불가**(타깃 여부=마스터 속성 CD_USERDEF4, orphan은 정의상 마스터에 없어 판별 순환). clean parquet에 폐기 flag 없음(17컬럼 확인).
  - 대신 **현재 알려진 타깃 집합**(현 processed의 광교 166 타깃 품목)이 새 마스터에서 **여전히 전부 resolve되는가**. 실패 = 타깃 품목이 재ingest 시 조용히 유실 = 진짜 리스크. → **fail**.
- **할인코드 = used-code regression** (비대칭, well-defined):
  - 할인코드는 판매의 CD_USERDEF1에서 "사용됨"이 **판매만으로 관측 가능**. → "판매에서 실제 사용된 할인코드가 마스터(discount_codes)에 없음" = **fail**. (품목과 달리 관측 가능하므로 구현 가능.)

### 4b. 충돌(conflict) 검사 — 같은 코드, 다른 값 (옛 마스터 vs 새 마스터)
- 4종 마스터의 **공통 코드**에 대해 옛(0520) vs 새(0526) 값 필드를 대조. 다르면 conflict.
  - 품목: NM_ITEM(품목명)·카테고리·규격. 할인: RT_DISC(할인율)·유효기간. 점포: 매장속성. 품절: 해당 스냅샷 성격상 대조 최소.
- **fail vs drift**: 타깃 품목·사용중 할인코드의 값이 바뀌면 **fail**(모델 라벨/α에 영향). 그 외 코드 값 변경은 **drift 보고**.
- 산출물: `reports/integrity/conflicting_codes.csv` — `code / kind / field / old_value / new_value / is_target_scope`.

### 4c. 보고 층 (drift + 아티제 문의 CSV — 항상 산출)
- **pass/fail과 무관하게 모든 미매칭(누락+충돌) 코드를 CSV export.** 4종 마스터 전부 크로스체크(사용자 확정): 품목(items.CD_ITEM)·할인(discount_codes.CD_DISC)·점포(stores.CD_PARTNER)·품절(stockout).
- `missing_codes.csv` 컬럼: `code / kind(item|discount|store|stockout) / sale_count / date_range / is_target_scope(Y=fail대상·N=drift) / source_sheet`.
  - 판매건수·기간을 붙여 아티제가 "이 코드 뭔지" 답하기 쉽게. is_target_scope로 우선순위.
- 산출물: `reports/integrity/{missing,conflicting}_codes.csv` (gitignored). 이게 **아티제 데이터 요청서 근거**([[project_new_data_20260721]] 갱신마스터 요청).
- **신규 orphan(타깃일 수도, 판별 불가)** = 자동 분류 불가 → CSV로 벤더 왕복이 유일 해결(게이트가 커버하는 척 안 함).

### 4c. Drift baseline 분모 고정 (advisor #3)
- 미매칭률 drift guard는 **분모를 명시**한다: **광교·전품목** 기준(메모리 67.9%와 정합) 하나로 고정. "여전히 ~68%인가"를 추적, 급증 시 사람 확인. (88.3%는 전매장 혼합 = 다른 모수, 혼용 금지.)

## 5. 아키텍처

- **신규 `src/bakery/data/integrity.py`** — 순수함수 체크. 각 체크 = `(df, ...) -> list[Violation]`. `Violation(check: str, severity: Literal["fail","drift"], detail: str, count: int)` 데이터클래스. 합성 fixture로 데이터 없이 pytest 가능.
- **`bakery check-integrity [--source sales_lines_clean|all] [--strict]`** — 실데이터 로드 → 체크 실행 → 콘솔 요약/위반 테이블 + `missing_codes.csv` 산출. **exit code = fail-severity 위반 있을 때만 non-zero**(drift는 0). build-data/refresh-external과 병렬 CLI.
- **pytest**: 순수함수는 합성 fixture로 정확값 테스트(데이터 없이). 실데이터 통합 테스트는 data-absent skip-guard.

## 6. 성공 기준

- [ ] `src/bakery/data/integrity.py` — §3·§4 체크가 순수함수로, `Violation` 반환. 합성 fixture pytest 정확값 통과.
- [ ] `bakery check-integrity`가 현 clean 데이터로 구동: 구조 invariant 전부 pass(현 데이터 정상), 반품비율 soft-range pass, known-target 166 resolve pass.
- [ ] sheet-2 스왑 **재현 fixture**(SALES_FG에 타임스탬프)로 값 판별 체크가 **fail**함을 테스트로 고정(회귀 방지 = 이 프로젝트 최고비용 실패).
- [ ] `missing_codes.csv` 생성: 품목·할인·점포·품절 4종 누락 코드 + 판매건수/기간/타깃여부. 실측(품목 ~1649 누락) 반영.
- [ ] `conflicting_codes.csv` 생성: 옛(0520) vs 새(0526) 마스터 공통코드 값 대조(품목명·할인율 등) + field/old/new/타깃여부.
- [ ] exit code: fail 위반(타깃/사용중 코드의 누락 or 충돌) 시 non-zero, drift만이면 0.
- [ ] `build-data --diagnose`·refresh 가드를 "무결성 3메커니즘"으로 문서 통합(우산).

## 7. 범위 밖

- 4 conflict codify(closing split·2026 gap·미매칭·category) → coverage.py surface 유지.
- interim→processed / processed 자체 검증 → 최소 dtype만(핵심은 raw→interim).
- 스케줄링/CI 자동화 → 데이터 gitignored라 수동 게이트(전향검증 단계).
- 신규 orphan 자동 타깃 분류 → 불가(벤더 왕복).
- 절대수 앵커를 새 drop에 적용 → 금지(vintage regression 전용).

## 8. 리스크

- **타깃 known-set의 출처**: 현 processed 광교 166 타깃이 새 마스터 재ingest 시 기준. 이 집합 자체가 오염되면 게이트가 잘못된 기준을 씀 → known-set은 현 canonical(검증된 bonavi_daily item_id)에서 도출, 하드코딩 금지.
- **soft-range threshold 튜닝**: 반품 1~3%는 현 1.88% 기준 추정. 아티제 새 데이터로 재보정 여지(주석 명시).
- **CSV가 커질 수 있음**(1649 품목): 아티제 문의용이라 OK, 단 타깃여부·판매건수로 정렬해 우선순위 상위부터.

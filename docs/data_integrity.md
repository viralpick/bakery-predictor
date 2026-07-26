# 데이터 무결성 (3 메커니즘 우산)

> 신규 데이터가 malformed이거나 마스터가 갱신 누락되면 **경계에서 잡는다.** 이 프로젝트 최고비용 실패(2026-07 sheet-2 컬럼 스왑 → "11개월 갭·정합성 −22%" 아티팩트, 전부 오진)가 존재 이유. 7단계 아티제 대규모 ingest de-risk가 목적.

무결성은 3개 메커니즘의 **우산**이다. 각자 다른 층에서 다른 실패를 잡는다 — 하나로 뭉치지 말 것.

## 1. `bakery build-data --diagnose` — 재빌드 결정성 (rtol=1e-9)

- **무엇**: raw(clean parquet) → processed(bonavi_daily/receipts)를 tempdir에 재생성 후 on-disk와 수치 대조.
- **언제**: canonical 재빌드가 disk와 어긋나는가(파이프라인 버그 or 누적 드리프트).
- **결과**: OK(max_diff=0) / DRIFT. **DRIFT는 비블록**(커버리지 리포트로 surface, 재배치 안 막음).
- 근거: `src/bakery/data/pipeline.py` `equivalence_diff`. 내부 결정적 테이블만(numeric).

## 2. `bakery refresh-external` — 커버리지 보존 가드 (히스토리 clobber 방지)

- **무엇**: 외부 8종 어댑터가 bounded window로 파일을 overwrite할 때, 스냅샷→호출→**커버리지 축소(행수/min-date) 감지 시 복원**.
- **언제**: 갱신이 기존 히스토리를 파괴하려 할 때(weather 하드코딩 start=2024가 2021~2025를 자를 뻔한 사례).
- **결과**: 축소면 복원+`applied=False`(--force override). forecast는 sliding window라 우회.
- 근거: `src/bakery/ingest/refresh.py` `refresh_source`/`_apply_coverage_guard`.

## 3. `bakery check-integrity` — ingestion 경계 invariant + FK 누락/충돌

- **무엇**: 신규 판매 데이터의 구조 invariant + 마스터 코드 매칭. `raw→interim` 경계.
- **언제**: 신규 데이터(특히 7단계 아티제 drop)를 편입하기 직전 게이트.
- **결과**: 타깃 품목 누락·컬럼 스왑 = **fail(exit 1)**. 나머지 미매칭·비율 이탈 = **drift(exit 0, CSV 보고)**.
- 근거: `src/bakery/data/integrity.py` (순수함수) + `check-integrity`(forward 게이트)/`check-conflict`(vintage 진단).

### 체크 분류 (실패 의미로 3분류 — advisor 규율)
| 체크 | 종류 | 기준 |
|---|---|---|
| SALES_FG ∈ {'0','1'} / SALES_TIME 14자리 / 시트 semantic | **fail** | 값 판별. **sheet-2 스왑 즉시 감지** |
| 라인 유일 (CD_PARTNER+DT_SALE+NO_POS+SLIP_NO+SLIP_LINE) | **fail** | 전체 grain 키(3-col은 매장/일자마다 리셋되어 유일X) |
| 스키마/dtype (float64 정규화 포함) | **fail** | 컬럼 드리프트 |
| 타깃 품목 known-set resolve (현 166) | **fail** | 타깃 유실 = 마스터 갱신 누락. ★orphan 분류 아님(현 타깃이 새 마스터서 resolve되나) |
| 사용 할인코드 resolve | **drift** | 코드체계 3/4자리 혼재로 정규화 미확정 → 오탐 방지. 아티제 문의 |
| 반품비율 soft-range 1~3% | **drift** | 비율 invariant(현 1.88%) |
| 날짜 연속성 | **drift** | 조용한 데이터 누락 |
| 코드 충돌(옛 vs 새 마스터 당일폐기 플래그) | **fail**(타깃) | `check-conflict` 별도(vintage 진단, one-shot) |
| 절대 총량(광교 510,585) | **vintage regression** | 현 raw 재처리에만. **새 drop엔 검사 안 함**(false-positive 방지) |

### 산출물 (아티제 역문의 근거)
- `reports/integrity/missing_codes.csv` — 마스터 미매칭 코드 전체(code/kind/is_target_scope). pass/fail 무관 항상 산출.
- `reports/integrity/conflicting_codes.csv` — 옛/새 마스터 값 충돌.
- is_target_scope=True부터 우선 문의(모델 영향).

## coverage.py vs integrity.py — 층 구분 (혼동 금지)
- **`coverage.py`(surface 층)**: 커버리지 매트릭스 렌더. **절대 안 실패**. 알려진·수용 상태(closing 소스 split·2026 label gap·미매칭·category 3v5) = detect-only 리포트.
- **`integrity.py`(checks 층)**: **loud-fail**. 신규 데이터가 malformed면 게이트.
- 4 conflict를 integrity check로 codify하지 **않는다**(알려진 상태 → coverage 유지).

## 실전 검증 실적 (도구가 만들어지는 중 발견)
1. **미등록 할인코드 `357`** (199건) — 판매엔 있으나 마스터 없음. drift로 보고 → 아티제 문의 대상. (정규화 규칙 3/4자리 혼재로 fail 아닌 drift.)
2. **line_uniqueness 키 결함** — 초기 3-col 키가 매장/일자마다 리셋되어 5.72M/5.73M 행 오탐. CD_PARTNER+DT_SALE 추가로 0 dupes 확정. 실데이터 스케일에서만 드러난 진짜 결함.

## 로드맵 위치
로드맵 4단계. 7단계(아티제 신규 데이터 canonical 편입) 시 `check-integrity`를 편입 직전 게이트로 실행. 편입 배관=3단계 `build-data`, 안전검사=이 4단계.

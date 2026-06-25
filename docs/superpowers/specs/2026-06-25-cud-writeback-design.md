# CUD Writeback 골격 (S4) — 설계

> v7 PoC LayerB Should 티어(`docs/poc_scope_v7.md` §6, §8.3 `commit_order`, §9, §12, §13).
> 작성일: 2026-06-25. 선행: ontology mock(S1) + functions(S2) + decision layer(8d13157) 머지됨.

---

## §0. 무엇이고 무엇이 아닌가

발주 추천을 **확정(writeback)**하는 데이터/함수 뼈대 — 사람 승인 게이트(토글) + 시점 무결성 메타데이터. **LLM 무관, 순수 로직 + 테스트.**

이 store는 **"우리 모델이 그 시점에 발주를 이렇게 예측했다"는 전향적 정확도 스냅샷**이다. **학습 데이터가 아니다.**

### 발주량 3종 (도메인 정리)
| # | 무엇 | 이 store가 담나 |
|---|---|---|
| 1 | 기존 아티제 자체 로직 발주 (모델 적용 전 baseline) | ✗ (별개 데이터, 아티제 포맷) |
| 2 | 우리 모델(LGBM+decision)이 뱉은 추천값 | ✓ `proposed_qty` |
| 3 | 2를 사람이 리스크 보정해 실제 발주한 값 | ✓ `approved_qty` |

학습의 "발주 vs 판매" feature는 **실제 일어난 발주(1+3)** 가 필요하고 아티제 포맷과 통일해야 하는 별개 작업 → **범위 밖**. 우리 store의 `approved_qty`(3)가 *미래에* 그 후보가 될 수 있다.

---

## §1. `valid_as_of`의 의미 (leakage 방어 아님)

발주량(2번)은 학습 input도 target도 아니므로(target=`potential_demand`=판매량), **"writeback → 학습 feature 누출" leakage는 현 구조에서 성립하지 않는다.** `docs/poc_scope_v7.md` §9의 "writeback = leakage 입구"는 발주량을 feature화하는 미래 시나리오 전제이거나 과잉 — 현재는 해당 없음.

`valid_as_of`의 진짜 의미는 **전향적 평가 무결성 + audit**:
- "이 추천이 *그 시점에* 확정됐다"는 사후-조작-불가 증거. 실제 판매가 나온 뒤 추천을 바꿔치기하지 못하게 막아, PoC의 **"4주 전향적 실측"**(우리 추천 vs 실제 판매로 폐기·매진 KPI 측정)이 정직함을 보장.
- 누가·언제·무엇을 확정했나의 audit (closed-loop 신뢰 기반, §13 버전스탬프).

조건부 leakage 훅: *만약* 나중에 발주량/공급제약을 feature로 쓰게 되면 그때 `confirmed_as_of(cutoff)`가 시점 방어로도 쓰일 수 있다(현재는 평가 무결성 용도).

---

## §2. 데이터 모델 — `OrderRecord` (frozen dataclass)

| 필드 | 타입 | 의미 |
|---|---|---|
| `record_id` | str | 고유 id |
| `store_id` | str | 매장 |
| `item_id` | str | 품목 |
| `date` | str (YYYY-MM-DD) | **발주 대상일** (이 발주가 적용되는 날) |
| `proposed_qty` | float | **2번** — 모델이 뱉은 추천값 |
| `approved_qty` | float \| None | **3번** — 사람이 보정·승인한 실제 값 (미승인 시 None) |
| `status` | str | `PENDING` \| `APPROVED` \| `REJECTED` |
| `approver` | str \| None | 승인자 id (자율 토글 시 `"autonomous"`) |
| `proposed_at` | str (ISO) | 추천 제안 시각 (호출자가 주입 — 결정성 위해) |
| `valid_as_of` | str \| None | **승인 확정 시각** (APPROVED일 때만; 전향적 평가 무결성 기준점) |

파생: `override = approved_qty − proposed_qty` (사람 보정량; APPROVED일 때만 의미) — property로 노출.

**시각 주입 원칙**: `proposed_at`/`valid_as_of`는 store가 `datetime.now()`를 부르지 않고 **호출자가 명시 전달**한다(테스트 결정성 + 절대규칙 정합). 기본 인자로 시각을 받는다.

---

## §3. `WritebackStore` (append-only, in-memory)

```
WritebackStore(require_approval: bool = True)
```

| 메서드 | 동작 |
|---|---|
| `propose_order(store_id, item_id, date, proposed_qty, *, proposed_at) -> OrderRecord` | PENDING 레코드 생성·추가. **토글 OFF면 즉시 APPROVED** (`approved_qty=proposed_qty`, `approver="autonomous"`, `valid_as_of=proposed_at`) |
| `approve(record_id, approver, *, approved_at, approved_qty=None) -> OrderRecord` | PENDING→APPROVED. `approved_qty` 생략 시 `proposed_qty` 사용(보정 없음), 주면 보정값(3번). `valid_as_of=approved_at` 기록. 자율 모드(require_approval=False)에서 propose된 레코드는 이미 APPROVED라, 아래 상태전이 가드에 의해 재승인 시 `ValueError` |
| `reject(record_id, approver) -> OrderRecord` | PENDING→REJECTED |
| `confirmed_as_of(cutoff) -> list[OrderRecord]` | `status==APPROVED and valid_as_of <= cutoff`인 레코드만 (전향적 스냅샷 재현). PENDING·미래 확정분·REJECTED 제외 |
| `to_frame() -> pd.DataFrame` | 전체 레코드 DataFrame (검사·리포트) |
| `to_parquet(path)` / `from_parquet(path)` (classmethod) | 얇은 직렬화 헬퍼 (영속은 선택; batch materialize 확장 지점) |

상태 전이 규칙(가드): 이미 APPROVED/REJECTED인 레코드 재승인/재거부 시 `ValueError`. 없는 `record_id` 시 `KeyError`.

### 토글 (require_approval)
- **True (기본, 안전)**: propose=PENDING → 사람 approve 필요. 사람 승인 게이트 시연.
- **False (자율 시연, frontier 방향)**: propose가 즉시 APPROVED. "안전 인터락 없음" — *시연*일 뿐 자동 발주 트리거는 없음. 게이트 유무를 토글로 대조.

---

## §4. 파일 구조

```
src/bakery/ontology/writeback.py   # OrderRecord + WritebackStore
tests/test_writeback.py            # 상태전이·토글·confirmed_as_of·override·직렬화
```

`src/bakery/ontology/__init__.py`에 `OrderRecord`, `WritebackStore` export.

---

## §5. 테스트 (키 불필요, 전부 결정론)

1. `propose_order` → PENDING, approved_qty None, valid_as_of None.
2. `approve`(보정 없음) → APPROVED, approved_qty==proposed_qty, override==0, valid_as_of==approved_at.
3. `approve`(보정 있음) → approved_qty=보정값, override==차이.
4. `reject` → REJECTED.
5. 이미 처리된 레코드 재승인 → ValueError; 없는 id → KeyError.
6. **토글 OFF** → propose 즉시 APPROVED, approver=="autonomous", valid_as_of==proposed_at.
7. **`confirmed_as_of(cutoff)`** — valid_as_of>cutoff 확정분·PENDING·REJECTED 제외, valid_as_of<=cutoff APPROVED만 (전향적 무결성 핵심 회귀 테스트).
8. `to_frame` 컬럼/행 수, `to_parquet`→`from_parquet` round-trip 동일성.

---

## §6. 범위 밖 (명시)

- OntologyFunction 등록 / grounding 에이전트 연결 (closed-loop 미니 시연) — S3와 별개, 후속.
- 실 DB / app-api 통합.
- 완전 자율 자동발주 (토글 OFF는 *시연*, 자동 트리거 없음).
- 1번(아티제) 포맷 통일 + 3번을 학습 "발주 vs 판매" feature로 통합 — 별개 데이터 작업.
- batch materialize loader 통합 (parquet 헬퍼만 두고 확장 지점으로 남김, §12).

---

## §7. Open Questions

| # | 질문 | 처리 |
|---|---|---|
| Q1 | `valid_as_of`/`proposed_at` 타입 — ISO str vs datetime | 결정성·직렬화 용이 위해 ISO str(YYYY-MM-DDTHH:MM:SS) 채택, cutoff 비교는 문자열 사전순(ISO라 안전) 또는 pd.Timestamp 파싱. 구현 시 확정 |
| Q2 | `record_id` 생성 — 호출자 주입 vs store 자동(순번) | store 자동 순번(`r{n}`)이 단순·결정론. 구현 시 |

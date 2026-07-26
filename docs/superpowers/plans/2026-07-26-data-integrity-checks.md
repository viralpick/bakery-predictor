# 데이터 무결성 검사 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bakery check-integrity` CLI로 신규 데이터가 malformed(sheet-swap류)이거나 마스터 코드가 누락·충돌하면 ingestion 경계에서 loud-fail하고, 미매칭 코드를 아티제 문의용 CSV로 산출한다.

**Architecture:** 신규 `src/bakery/data/integrity.py` = 순수함수 체크 모음(`(df,...) -> list[Violation]`). `bakery check-integrity` CLI가 실데이터 로드→체크 실행→콘솔 요약+CSV 산출+exit code. 검증 로직은 데이터 없이 합성 fixture로 pytest. coverage.py(surface 층)와 별개의 checks 층.

**Tech Stack:** Python 3.11, uv, pandas, pyarrow, typer(CLI), pytest.

## Global Constraints

- **forward-invariant 규율** — 새 drop에 도는 체크만 gate. 절대수 앵커(광교 510,585)는 vintage regression 전용(현 raw 재처리에만), 새 데이터엔 절대 검사 안 함. 비율(반품 1.88%)은 soft-range(1~3%)로 drift.
- **fail vs drift 분리** — 타깃 품목(현 bonavi_daily 166 item_id)의 누락/충돌 = `severity="fail"`(exit non-zero). 그 외 미매칭 = `severity="drift"`(exit 0, CSV 보고만).
- **⚠️할인코드 정규화 미확정 → used-discount는 fail 아닌 drift (advisor #1 실측 반영)** — 실측: `used - master` = `'357'` 1종(3자리, 199건). 그러나 마스터가 4자리 zero-pad인데 판매엔 3/4자리 혼재(`0121` vs `317`), zero-pad(4) 정규화하면 오히려 28종으로 늘어 깨짐(0317/0357 등 마스터에 없음). **코드 체계 혼재로 정규화 규칙 불명확 → used-discount 누락을 fail로 박으면 오탐.** used-discount 누락/충돌은 **drift + CSV 보고**로 두고, 정규화 규칙은 아티제 확인(문의 대상). fail 게이트는 **타깃 품목 known-set regression만** 확정. [[feedback_verified_vs_inferred]] 준수(검증 전 사실 단정 금지).
- **per-sheet English placeholder 매핑** — 옛/새 마스터 대조는 한글 헤더 아닌 **English 코드(CD_ITEM/CD_DISC/CD_PARTNER)** 기반. 0520=`POS메뉴명` vs 0526=`품목명`으로 한글 헤더가 달라 한글 대조는 오탐([[project_new_data_ingestion_pitfall]]).
- **타깃 known-set은 canonical에서 도출** — 하드코딩 금지. `pd.read_parquet(paths.dataset("bonavi_daily"))["item_id"].unique()` = 166종.
- **coverage.py 안 건드림** — surface 층 유지(절대 안 실패). integrity.py는 별도 checks 층.
- **재구현 금지** — 마스터 로드는 기존 `bonavi_loader_v2.load_items_v2` 등 재사용. 데이터 gitignored → pytest는 합성 fixture(순수함수) + 실데이터는 skip-guard.
- **테스트 규칙** — 정확값 `==`/`.count()`. sheet-2 스왑 재현 fixture로 fail 고정.
- **pytest** — `uv run pytest --color=no` (addopts에 -q 있음).
- **커밋 트레일러** — `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_015231oEXPA9S82TZf9Ue933`.
- **브랜치** — `feat/data-integrity-checks` (스펙 커밋됨: d1f8bfa, 201c487).

---

## Task 1: Violation 타입 + 구조 invariant 체크 (값 판별·dtype·no-dup)

목적: sheet-2 스왑을 잡는 핵심 invariant. 순수함수로 데이터 없이 테스트.

**Files:**
- Create: `src/bakery/data/integrity.py`
- Test: `tests/test_integrity_structural.py`

**Interfaces:**
- Produces:
  - `Violation(check: str, severity: str, detail: str, count: int)` — dataclass. severity ∈ {"fail","drift"}.
  - `check_sales_fg_domain(sales: pd.DataFrame) -> list[Violation]` — SALES_FG ∈ {'0','1'} 위반.
  - `check_sales_time_format(sales: pd.DataFrame) -> list[Violation]` — SALES_TIME 14자리 숫자 위반.
  - `check_line_uniqueness(sales: pd.DataFrame) -> list[Violation]` — (NO_POS, SLIP_NO, SLIP_LINE) 중복.
  - `check_schema(sales: pd.DataFrame, expected: dict[str,str]) -> list[Violation]` — 컬럼 존재+dtype.

- [ ] **Step 1: 실패 테스트 작성 (값 판별 — sheet-swap 재현)**

```python
# tests/test_integrity_structural.py
import pandas as pd
import pytest
from bakery.data import integrity


def _good_sales():
    return pd.DataFrame({
        "NO_POS": ["1", "1", "2"], "SLIP_NO": ["10", "11", "10"], "SLIP_LINE": ["1", "1", "1"],
        "SALES_FG": ["0", "1", "0"],
        "SALES_TIME": ["20260101120000", "20260101130000", "20260102090000"],
    })


def test_sales_fg_domain_clean_passes():
    assert integrity.check_sales_fg_domain(_good_sales()) == []


def test_sales_fg_domain_catches_sheet_swap():
    # sheet-2 스왑: SALES_FG 자리에 타임스탬프가 들어옴 → 즉시 fail
    swapped = _good_sales()
    swapped["SALES_FG"] = ["20260101120000", "20260101130000", "20260102090000"]
    v = integrity.check_sales_fg_domain(swapped)
    assert len(v) == 1
    assert v[0].severity == "fail"
    assert v[0].count == 3   # 3행 전부 도메인 위반


def test_sales_time_format_catches_swap():
    swapped = _good_sales()
    swapped["SALES_TIME"] = ["0", "1", "0"]   # 스왑 시 0/1이 들어옴
    v = integrity.check_sales_time_format(swapped)
    assert len(v) == 1 and v[0].severity == "fail" and v[0].count == 3


def test_line_uniqueness_catches_dup():
    dup = _good_sales()
    dup.loc[2, ["NO_POS", "SLIP_NO", "SLIP_LINE"]] = ["1", "10", "1"]  # row0 중복
    v = integrity.check_line_uniqueness(dup)
    assert len(v) == 1 and v[0].count == 2   # 중복 그룹 2행


def test_schema_catches_missing_col_and_dtype():
    df = _good_sales().astype({"SALES_FG": "int64"})  # dtype 위반 유도
    del df["SALES_TIME"]  # 컬럼 누락
    v = integrity.check_schema(df, {"SALES_FG": "string", "SALES_TIME": "string"})
    kinds = sorted(x.detail.split(":")[0] for x in v)
    assert kinds == ["dtype", "missing"]
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_integrity_structural.py -v --color=no`
Expected: FAIL — `bakery.data.integrity` 없음.

- [ ] **Step 3: integrity.py 구조 체크 구현**

```python
# src/bakery/data/integrity.py
"""데이터 무결성 체크 — 순수함수. 신규 데이터가 malformed이면 loud-fail.
coverage.py(surface 층, 안 실패)와 별개의 checks 층. detect-only 아님 — 게이트."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Violation:
    check: str
    severity: str   # "fail" | "drift"
    detail: str
    count: int


def check_sales_fg_domain(sales: pd.DataFrame) -> list[Violation]:
    """SALES_FG는 '0'(정상)/'1'(반품)만. sheet-2 스왑이면 타임스탬프가 들어와 위반."""
    bad = ~sales["SALES_FG"].astype(str).isin({"0", "1"})
    if not bad.any():
        return []
    return [Violation("sales_fg_domain", "fail",
                      f"SALES_FG not in {{0,1}} (sheet-swap 의심)", int(bad.sum()))]


def check_sales_time_format(sales: pd.DataFrame) -> list[Violation]:
    """SALES_TIME은 14자리 숫자(YYYYMMDDHHMMSS). 스왑이면 0/1이 들어와 위반."""
    s = sales["SALES_TIME"].astype(str)
    bad = ~s.str.fullmatch(r"\d{14}")
    if not bad.any():
        return []
    return [Violation("sales_time_format", "fail",
                      "SALES_TIME not 14-digit (sheet-swap 의심)", int(bad.sum()))]


def check_line_uniqueness(sales: pd.DataFrame) -> list[Violation]:
    """(NO_POS, SLIP_NO, SLIP_LINE) 라인 유일."""
    keys = ["NO_POS", "SLIP_NO", "SLIP_LINE"]
    dup = sales.duplicated(subset=keys, keep=False)
    if not dup.any():
        return []
    return [Violation("line_uniqueness", "fail",
                      "duplicate (NO_POS,SLIP_NO,SLIP_LINE)", int(dup.sum()))]


def check_schema(sales: pd.DataFrame, expected: dict[str, str]) -> list[Violation]:
    """컬럼 존재 + dtype 계약."""
    out: list[Violation] = []
    for col, dtype in expected.items():
        if col not in sales.columns:
            out.append(Violation("schema", "fail", f"missing: {col}", 1))
        elif str(sales[col].dtype) != dtype:
            out.append(Violation("schema", "fail",
                                  f"dtype: {col} is {sales[col].dtype}, want {dtype}", 1))
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_integrity_structural.py -v --color=no`
Expected: PASS (6 tests).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/integrity.py tests/test_integrity_structural.py
git commit -m "feat(data): 무결성 구조 invariant 체크 (sheet-swap 감지) (Task 1)"
```

---

## Task 2: 날짜 연속성 + 반품비율 soft-range 체크

목적: 조용한 데이터 누락(품목×월 갭)과 비율 drift.

**Files:**
- Modify: `src/bakery/data/integrity.py`
- Test: `tests/test_integrity_ratios.py`

**Interfaces:**
- Consumes: `Violation`.
- Produces:
  - `check_date_contiguity(sales: pd.DataFrame, max_gap_days: int = 45) -> list[Violation]` — DT_SALE 월 단위 큰 갭(severity="drift").
  - `check_return_ratio(sales: pd.DataFrame, lo: float = 0.01, hi: float = 0.03) -> list[Violation]` — SALES_FG=='1' 비율 range(severity="drift").

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_integrity_ratios.py
import pandas as pd
from bakery.data import integrity


def test_return_ratio_in_range_passes():
    # 반품 2% (100건 중 2) → range [1%,3%] 통과
    fg = ["1"] * 2 + ["0"] * 98
    df = pd.DataFrame({"SALES_FG": fg})
    assert integrity.check_return_ratio(df) == []


def test_return_ratio_out_of_range_drifts():
    fg = ["1"] * 10 + ["0"] * 90   # 10% → range 벗어남
    df = pd.DataFrame({"SALES_FG": fg})
    v = integrity.check_return_ratio(df)
    assert len(v) == 1 and v[0].severity == "drift"


def test_date_contiguity_catches_gap():
    # 2026-01, 그다음 2026-04 (3개월 갭) → drift
    dates = ["20260101"] * 3 + ["20260401"] * 3
    df = pd.DataFrame({"DT_SALE": dates})
    v = integrity.check_date_contiguity(df, max_gap_days=45)
    assert len(v) == 1 and v[0].severity == "drift"


def test_date_contiguity_contiguous_passes():
    dates = ["20260101", "20260115", "20260201", "20260215"]
    df = pd.DataFrame({"DT_SALE": dates})
    assert integrity.check_date_contiguity(df, max_gap_days=45) == []
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_integrity_ratios.py -v --color=no`
Expected: FAIL — 함수 없음.

- [ ] **Step 3: 구현**

```python
# src/bakery/data/integrity.py 에 추가
def check_return_ratio(sales: pd.DataFrame, lo: float = 0.01, hi: float = 0.03) -> list[Violation]:
    """반품비율(SALES_FG=='1')은 사업 비율이라 soft-range. 벗어나면 drift(gate 아님).
    현 clean 실측 1.88%. 범위는 아티제 새 데이터로 재보정 여지."""
    total = len(sales)
    if total == 0:
        return []
    ratio = float((sales["SALES_FG"].astype(str) == "1").mean())
    if lo <= ratio <= hi:
        return []
    return [Violation("return_ratio", "drift",
                      f"return ratio {ratio:.4f} outside [{lo},{hi}]", total)]


def check_date_contiguity(sales: pd.DataFrame, max_gap_days: int = 45) -> list[Violation]:
    """DT_SALE 정렬 후 연속 날짜 간 최대 갭이 max_gap_days 초과면 drift(조용한 누락)."""
    dates = pd.to_datetime(sales["DT_SALE"].astype(str), format="%Y%m%d", errors="coerce").dropna()
    if len(dates) < 2:
        return []
    uniq = pd.Series(sorted(dates.unique()))
    gaps = uniq.diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0
    if max_gap <= max_gap_days:
        return []
    return [Violation("date_contiguity", "drift",
                      f"max date gap {max_gap}d > {max_gap_days}d", max_gap)]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_integrity_ratios.py -v --color=no`
Expected: PASS (4 tests).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/integrity.py tests/test_integrity_ratios.py
git commit -m "feat(data): 날짜 연속성 + 반품비율 soft-range 체크 (Task 2)"
```

---

## Task 3: FK 코드 누락 검사 (타깃=fail / 전체=drift + CSV)

목적: 판매 코드가 마스터에 없는 것을 잡고, 타깃/사용중이면 fail·나머지는 drift로 CSV 보고.

**Files:**
- Modify: `src/bakery/data/integrity.py`
- Test: `tests/test_integrity_fk.py`

**Interfaces:**
- Consumes: `Violation`.
- Produces:
  - `find_missing_codes(sale_codes: set[str], master_codes: set[str], kind: str, target_codes: set[str], used_codes: set[str]) -> pd.DataFrame` — 누락 코드 표(code, kind, is_target_scope). 순수함수.
  - `check_target_items_resolve(sale_item_codes: set, master_item_codes: set, target_items: set) -> list[Violation]` — 타깃 품목이 마스터에서 resolve 안 되면 fail.
  - `check_used_discounts_resolve(used_disc: set, master_disc: set) -> list[Violation]` — 사용된 할인코드 누락 fail.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_integrity_fk.py
import pandas as pd
from bakery.data import integrity


def test_target_item_missing_from_master_fails():
    # 타깃 품목 A103이 마스터에 없음 → fail
    v = integrity.check_target_items_resolve(
        sale_item_codes={"A101", "A102", "A103"},
        master_item_codes={"A101", "A102"},   # A103 없음
        target_items={"A101", "A103"},         # A103은 타깃
    )
    assert len(v) == 1 and v[0].severity == "fail" and v[0].count == 1
    assert "A103" in v[0].detail


def test_non_target_orphan_does_not_fail():
    # A103 orphan이지만 타깃 아님 → fail 없음 (drift는 CSV에서 처리)
    v = integrity.check_target_items_resolve(
        sale_item_codes={"A101", "A103"}, master_item_codes={"A101"},
        target_items={"A101"},
    )
    assert v == []


def test_used_discount_missing_is_drift_not_fail():
    # ★코드 정규화 규칙 미확정(3/4자리 혼재)이라 fail 아닌 drift (advisor #1 실측)
    v = integrity.check_used_discounts_resolve(
        used_disc={"0069", "9999"}, master_disc={"0069"})  # 9999 사용됐는데 마스터 없음
    assert len(v) == 1 and v[0].severity == "drift" and "9999" in v[0].detail


def test_find_missing_codes_marks_target_scope():
    df = integrity.find_missing_codes(
        sale_codes={"A101", "A102", "A103"}, master_codes={"A101"},
        kind="item", target_codes={"A102"}, used_codes=set())
    # A102, A103 누락. A102는 타깃 → is_target_scope True
    miss = df.set_index("code")["is_target_scope"].to_dict()
    assert miss == {"A102": True, "A103": False}
    assert set(df["kind"]) == {"item"}
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_integrity_fk.py -v --color=no`
Expected: FAIL — 함수 없음.

- [ ] **Step 3: 구현**

```python
# src/bakery/data/integrity.py 에 추가
def check_target_items_resolve(sale_item_codes: set[str], master_item_codes: set[str],
                               target_items: set[str]) -> list[Violation]:
    """알려진 타깃 품목이 마스터에서 resolve 안 되면 fail(타깃 유실 = 마스터 갱신 누락).
    ★orphan을 타깃으로 분류하는 게 아니라, 이미 아는 타깃 집합의 회귀를 본다."""
    missing = (target_items & sale_item_codes) - master_item_codes
    if not missing:
        return []
    sample = ", ".join(sorted(missing)[:5])
    return [Violation("target_items_resolve", "fail",
                      f"target items missing from master: {sample}", len(missing))]


def check_used_discounts_resolve(used_disc: set[str], master_disc: set[str]) -> list[Violation]:
    """판매에서 사용된 할인코드가 마스터에 없음 → drift(fail 아님).
    ★코드 체계 3/4자리 혼재로 정규화 규칙 미확정(advisor #1 실측: '357' 1종, zero-pad하면
    28종으로 깨짐) → 오탐 방지 위해 drift로 보고, 정규화는 아티제 확인. CSV로 문의."""
    missing = used_disc - master_disc
    if not missing:
        return []
    sample = ", ".join(sorted(missing)[:5])
    return [Violation("used_discounts_resolve", "drift",
                      f"used discounts unresolved (정규화 미확정): {sample}", len(missing))]


def find_missing_codes(sale_codes: set[str], master_codes: set[str], kind: str,
                       target_codes: set[str], used_codes: set[str]) -> pd.DataFrame:
    """마스터에 없는 판매 코드 표. is_target_scope=타깃 품목 or 사용된 코드(fail 대상)."""
    missing = sorted(sale_codes - master_codes)
    scope = target_codes | used_codes
    return pd.DataFrame({
        "code": missing,
        "kind": kind,
        "is_target_scope": [c in scope for c in missing],
    })
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_integrity_fk.py -v --color=no`
Expected: PASS (4 tests).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/integrity.py tests/test_integrity_fk.py
git commit -m "feat(data): FK 코드 누락 검사 (타깃 known-set regression + CSV) (Task 3)"
```

---

## Task 4: 코드 충돌(conflict) 검사 (옛 마스터 vs 새 마스터, English-키 기반)

목적: 같은 코드인데 값이 바뀐 것을 대조로 잡는다. **★advisor #2: 대조 필드는 품목명(cosmetic)이 아니라 모델에 영향 주는 신호 우선** — 품목 `당일폐기여부(CD_USERDEF4)`(타깃 플래그 변경=라벨 오염)·카테고리, 할인 `RT_DISC(할인율)`·유효기간(마감 α 영향). 품목명(NM_ITEM)은 보조. `find_conflicting_codes`는 fields 인자로 일반화돼 있어 어느 필드든 받는다(순수함수 불변, 호출부에서 필드 지정).

**Files:**
- Modify: `src/bakery/data/integrity.py`
- Test: `tests/test_integrity_conflict.py`

**Interfaces:**
- Consumes: `Violation`.
- Produces:
  - `find_conflicting_codes(old_master: pd.DataFrame, new_master: pd.DataFrame, key: str, fields: list[str], kind: str, scope_codes: set[str]) -> pd.DataFrame` — 공통 코드 중 field 값이 다른 것(code, kind, field, old_value, new_value, is_target_scope). 순수함수.
  - `check_scope_conflicts(conflicts: pd.DataFrame) -> list[Violation]` — is_target_scope 충돌이 있으면 fail.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_integrity_conflict.py
import pandas as pd
from bakery.data import integrity


def _master(codes, names):
    return pd.DataFrame({"CD_ITEM": codes, "NM_ITEM": names})


def test_conflict_detects_changed_value():
    old = _master(["A1", "A2"], ["빵", "케이크"])
    new = _master(["A1", "A2"], ["빵", "케잌"])   # A2 이름 변경
    df = integrity.find_conflicting_codes(old, new, key="CD_ITEM", fields=["NM_ITEM"],
                                          kind="item", scope_codes={"A2"})
    assert list(df["code"]) == ["A2"]
    assert df.iloc[0]["old_value"] == "케이크" and df.iloc[0]["new_value"] == "케잌"
    assert bool(df.iloc[0]["is_target_scope"]) is True


def test_conflict_ignores_missing_codes():
    # 한쪽에만 있는 코드는 conflict 아님(누락은 Task 3 소관)
    old = _master(["A1"], ["빵"])
    new = _master(["A1", "A2"], ["빵", "쿠키"])
    df = integrity.find_conflicting_codes(old, new, key="CD_ITEM", fields=["NM_ITEM"],
                                          kind="item", scope_codes=set())
    assert df.empty


def test_scope_conflict_fails_else_drift():
    conflicts = pd.DataFrame({"code": ["A2", "A3"], "is_target_scope": [True, False]})
    v = integrity.check_scope_conflicts(conflicts)
    assert len(v) == 1 and v[0].severity == "fail" and v[0].count == 1  # A2만 fail
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_integrity_conflict.py -v --color=no`
Expected: FAIL — 함수 없음.

- [ ] **Step 3: 구현**

```python
# src/bakery/data/integrity.py 에 추가
def find_conflicting_codes(old_master: pd.DataFrame, new_master: pd.DataFrame, key: str,
                           fields: list[str], kind: str, scope_codes: set[str]) -> pd.DataFrame:
    """공통 코드(양쪽 존재) 중 field 값이 다른 것. English 코드(key) 기반 —
    한글 헤더는 vintage마다 다를 수 있어 신뢰 금지([[project_new_data_ingestion_pitfall]])."""
    o = old_master.set_index(key)
    n = new_master.set_index(key)
    common = o.index.intersection(n.index)
    rows = []
    for field in fields:
        if field not in o.columns or field not in n.columns:
            continue
        ov, nv = o.loc[common, field].astype(str), n.loc[common, field].astype(str)
        diff = common[(ov.values != nv.values)]
        for code in diff:
            rows.append({"code": code, "kind": kind, "field": field,
                         "old_value": str(o.loc[code, field]), "new_value": str(n.loc[code, field]),
                         "is_target_scope": code in scope_codes})
    return pd.DataFrame(rows, columns=["code", "kind", "field", "old_value", "new_value", "is_target_scope"])


def check_scope_conflicts(conflicts: pd.DataFrame) -> list[Violation]:
    """타깃/사용중 코드의 값이 바뀌면 fail(모델 라벨/α에 영향)."""
    if conflicts.empty or "is_target_scope" not in conflicts.columns:
        return []
    scoped = conflicts[conflicts["is_target_scope"].astype(bool)]
    if scoped.empty:
        return []
    sample = ", ".join(scoped["code"].astype(str).unique()[:5])
    return [Violation("scope_conflicts", "fail",
                      f"target/used codes changed value: {sample}", len(scoped))]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_integrity_conflict.py -v --color=no`
Expected: PASS (3 tests).

- [ ] **Step 5: 커밋**

```bash
git add src/bakery/data/integrity.py tests/test_integrity_conflict.py
git commit -m "feat(data): 코드 충돌 검사 (옛/새 마스터 대조, English-키) (Task 4)"
```

---

## Task 5: `bakery check-integrity` CLI (실데이터 구동 + CSV + exit code)

목적: 순수함수들을 실데이터로 오케스트레이션. 콘솔 요약 + missing/conflicting CSV + exit code.

**Files:**
- Modify: `src/bakery/cli.py` (신규 `@app.command("check-integrity")`)
- Modify: `src/bakery/data/integrity.py` (`run_all` 오케스트레이터 + CSV writer)
- Test: `tests/test_integrity_cli.py`

**Interfaces:**
- Consumes: 모든 체크 함수, `paths`, `bonavi_loader_v2.load_items_v2`.
- Produces:
  - `run_all(sales, master_item_codes, target_items, used_discounts, master_disc, schema=None) -> tuple[list[Violation], pd.DataFrame]` — (violations, missing_df). forward 게이트.
  - `run_conflict_diagnostic(old_item_master, new_item_master, fields, scope_codes) -> tuple[list[Violation], pd.DataFrame]` — (violations, conflicting_df). vintage 진단.
  - `has_fail(violations) -> bool`, `write_missing(df, out_dir)`, `write_conflicting(df, out_dir)`.
  - CLI: `check-integrity`(forward 게이트) + `check-conflict`(vintage one-shot 진단).

- [ ] **Step 1: 실패 테스트 작성 (run_all 오케스트레이션 + exit 규칙)**

```python
# tests/test_integrity_cli.py
import pandas as pd
from bakery.data import integrity


def test_run_all_aggregates_and_exit_severity():
    # 합성: 정상 sales + 타깃 누락 1건 → fail 존재
    sales = pd.DataFrame({
        "NO_POS": ["1"], "SLIP_NO": ["1"], "SLIP_LINE": ["1"],
        "SALES_FG": ["0"], "SALES_TIME": ["20260101120000"],
        "DT_SALE": ["20260101"], "CD_ITEM": ["A101"], "CD_USERDEF1": [""],
    })
    violations, missing = integrity.run_all(
        sales=sales,
        master_item_codes={"A999"},           # A101 없음
        target_items={"A101"}, used_discounts=set(), master_disc={"0069"},
        schema={},
    )
    # 타깃 A101 누락 → fail
    assert any(v.severity == "fail" and v.check == "target_items_resolve" for v in violations)
    assert integrity.has_fail(violations) is True
    # A101이 missing_df에 target_scope=True로
    assert missing.set_index("code").loc["A101", "is_target_scope"]


def test_run_conflict_diagnostic_target_flag_flip_fails():
    # 타깃 플래그(CD_USERDEF4) Y→N 변경 = 라벨 오염 → fail
    old = pd.DataFrame({"CD_ITEM": ["A101"], "CD_USERDEF4": ["Y"]})
    new = pd.DataFrame({"CD_ITEM": ["A101"], "CD_USERDEF4": ["N"]})
    v, conflicting = integrity.run_conflict_diagnostic(
        old, new, fields=["CD_USERDEF4"], scope_codes={"A101"})
    assert integrity.has_fail(v) is True
    assert conflicting.iloc[0]["old_value"] == "Y" and conflicting.iloc[0]["new_value"] == "N"


def test_has_fail_false_when_only_drift():
    v = [integrity.Violation("return_ratio", "drift", "x", 1)]
    assert integrity.has_fail(v) is False
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_integrity_cli.py -v --color=no`
Expected: FAIL — `run_all`/`has_fail` 없음.

- [ ] **Step 3: run_all + has_fail + write_reports 구현**

```python
# src/bakery/data/integrity.py 에 추가
from pathlib import Path

_SALES_SCHEMA = {  # clean parquet 계약 (실측: DT_SALE/CD_ITEM string, SALES_FG string ...)
    "CD_ITEM": "string", "SALES_FG": "string", "SALES_TIME": "string", "DT_SALE": "string",
}


def has_fail(violations: list[Violation]) -> bool:
    return any(v.severity == "fail" for v in violations)


def run_all(sales, master_item_codes, target_items, used_discounts, master_disc,
            schema=None) -> tuple:
    """구조/FK 체크(새 drop에 매번 도는 forward 게이트) → (violations, missing_df).
    ★conflict(옛vs새 마스터)는 vintage 대조라 run_all에 없음 — run_conflict_diagnostic 분리
    (advisor #3: 하드코딩 0520vs0526은 one-shot, hot-path xlsx 재파싱 금지)."""
    schema = schema if schema is not None else _SALES_SCHEMA
    sale_items = set(sales["CD_ITEM"].dropna().astype(str))
    v: list[Violation] = []
    v += check_sales_fg_domain(sales)
    v += check_sales_time_format(sales)
    v += check_line_uniqueness(sales)
    v += check_schema(sales, schema)
    v += check_return_ratio(sales)
    v += check_date_contiguity(sales)
    v += check_target_items_resolve(sale_items, master_item_codes, target_items)  # fail 게이트
    v += check_used_discounts_resolve(used_discounts, master_disc)                # drift
    missing = find_missing_codes(sale_items, master_item_codes, "item",
                                 target_items, used_discounts)
    return v, missing


def run_conflict_diagnostic(old_item_master, new_item_master, fields, scope_codes) -> tuple:
    """옛 vs 새 마스터 값 충돌 대조 (one-shot 진단, 새 drop 편입 시점에만).
    → (violations, conflicting_df). CD_USERDEF4(타깃플래그)·카테고리 등 모델영향 필드 우선."""
    conflicting = find_conflicting_codes(old_item_master, new_item_master, "CD_ITEM",
                                         fields, "item", scope_codes)
    return check_scope_conflicts(conflicting), conflicting


def write_missing(missing_df, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(out_dir / "missing_codes.csv", index=False)


def write_conflicting(conflicting_df, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conflicting_df.to_csv(out_dir / "conflicting_codes.csv", index=False)
```

- [ ] **Step 4: run_all 테스트 통과 확인**

Run: `uv run pytest tests/test_integrity_cli.py -v --color=no`
Expected: PASS (2 tests).

- [ ] **Step 5: CLI 커맨드 배선**

```python
# src/bakery/cli.py 에 추가 (from .data import integrity, paths; from .data import bonavi_loader_v2 as v2)
@app.command("check-integrity")
def cmd_check_integrity(source: str = "sales_lines_clean", strict: bool = False) -> None:
    """신규 데이터 forward 무결성 게이트. 타깃 품목 누락=fail(exit non-zero),
    나머지 미매칭=drift(CSV 보고). reports/integrity/missing_codes.csv 산출.
    ★conflict(옛vs새 마스터)는 별도 `check-conflict`(vintage 진단)."""
    import sys
    import pandas as pd
    from .data import integrity, paths
    from .data import bonavi_loader_v2 as v2

    sales = pd.read_parquet(paths.dataset(source))
    daily = pd.read_parquet(paths.dataset("bonavi_daily"))
    target_items = set(daily["item_id"].astype(str))           # canonical 166 타깃
    items = v2.load_items_v2()
    master_item_codes = set(items["item_id"].astype(str))
    used_disc = set(sales["CD_USERDEF1"].replace("", pd.NA).dropna().astype(str))
    disc = pd.read_parquet(paths.INTERIM_DIR / "v2" / "discount_codes.parquet")
    master_disc = set(disc["CD_DISC"].astype(str))

    violations, missing = integrity.run_all(
        sales=sales, master_item_codes=master_item_codes, target_items=target_items,
        used_discounts=used_disc, master_disc=master_disc,
    )
    out_dir = REPORTS_DIR / "integrity"
    integrity.write_missing(missing, out_dir)
    for vi in violations:
        color = "red" if vi.severity == "fail" else "yellow"
        console.print(f"[{color}]{vi.severity.upper()}[/] {vi.check}: {vi.detail} (n={vi.count})")
    console.print(f"[dim]missing={len(missing)} → {out_dir}/missing_codes.csv[/]")
    if integrity.has_fail(violations):
        console.print("[red]무결성 실패 — 타깃 품목 누락/컬럼 스왑[/]")
        sys.exit(1)
    console.print("[green]무결성 통과 (fail 없음, drift는 CSV 참조)[/]")


@app.command("check-conflict")
def cmd_check_conflict() -> None:
    """옛(0520) vs 새(0526) 마스터 값 충돌 one-shot 진단. 타깃 플래그·할인율 변경=fail.
    ★vintage 대조라 편입 시점에만 수동 실행(hot-path 아님). 위치기준 헤더 rename은
    코드열 검증 후 사용(sheet-2 교훈)."""
    import pandas as pd
    from .data import integrity, paths

    daily = pd.read_parquet(paths.dataset("bonavi_daily"))
    target_items = set(daily["item_id"].astype(str))

    def _load_items(key):
        df = pd.read_excel(paths.dataset(key), sheet_name="품목정보", dtype=str)
        # 위치기준 rename (품목정보에 English placeholder 행 없음) — 0열=코드, 6열=당일폐기여부.
        # ★가드: 0열이 코드 패턴(숫자 문자열)인지 assert (위치 신뢰 금지, sheet-2 교훈)
        first = df.iloc[:, 0].dropna().astype(str)
        assert first.str.fullmatch(r"\d+").mean() > 0.9, f"{key} 0열이 코드 아님 — 헤더 위치 확인"
        return df.rename(columns={df.columns[0]: "CD_ITEM", df.columns[6]: "CD_USERDEF4"})

    old_im, new_im = _load_items("legacy_xlsx_0520"), _load_items("master_xlsx")
    violations, conflicting = integrity.run_conflict_diagnostic(
        old_im[["CD_ITEM", "CD_USERDEF4"]], new_im[["CD_ITEM", "CD_USERDEF4"]],
        fields=["CD_USERDEF4"], scope_codes=target_items)   # 당일폐기 플래그 변경 우선
    out_dir = REPORTS_DIR / "integrity"
    integrity.write_conflicting(conflicting, out_dir)
    for vi in violations:
        console.print(f"[{'red' if vi.severity=='fail' else 'yellow'}]{vi.severity.upper()}[/] {vi.check}: {vi.detail}")
    console.print(f"[dim]conflicting={len(conflicting)} → {out_dir}/conflicting_codes.csv[/]")
```

- [ ] **Step 6: CLI smoke (실데이터)**

Run:
```bash
uv run bakery check-integrity 2>&1 | tail -15; echo "exit=$?"
uv run bakery check-conflict 2>&1 | tail -8
ls -la reports/integrity/
head -5 reports/integrity/missing_codes.csv
```
Expected:
- `check-integrity`: 구조 invariant 통과(현 데이터 정상), **타깃 166 전부 resolve → fail 없음 → exit 0**(실측 확인). used-discount `'357'` 등은 **drift**로 출력(fail 아님). missing_codes.csv에 ~1649 품목 미매칭 + 타깃여부(전부 False = 현 canonical과 정합).
- `check-conflict`: 0520 vs 0526 당일폐기 플래그 대조. 타깃 품목 플래그 변경 있으면 fail, 없으면 통과. conflicting_codes.csv 산출.
- 현 데이터가 exit 0이 정상(도구가 건전 기준선 확인). fail 나오면 실제 이슈이므로 self-heal 아닌 조사 후 report 기록.

- [ ] **Step 7: import clean + 커밋**

Run: `uv run python -c "import bakery.cli"`
```bash
git add src/bakery/cli.py src/bakery/data/integrity.py tests/test_integrity_cli.py
git commit -m "feat(cli): check-integrity(forward 게이트)+check-conflict(vintage 진단) (Task 5)"
```

---

## Task 6: 우산 문서 통합 (무결성 3메커니즘)

목적: check-integrity + build-data --diagnose + refresh 커버리지 가드를 "무결성 = 이 3개"로 통합 서술. 재구현 아님.

**Files:**
- Modify: `.claude/CLAUDE.md` (실행 섹션에 check-integrity 추가)
- Create: `docs/data_integrity.md` (3메커니즘 우산 요약)

**Interfaces:** 없음(문서만).

- [ ] **Step 1: docs/data_integrity.md 작성**

3메커니즘 표: (1) `build-data --diagnose`=재빌드 결정성 rtol=1e-9 (2) `refresh-external` 커버리지 보존 가드=히스토리 clobber 방지 (3) `check-integrity`=ingestion 경계 invariant+FK 누락/충돌. 각 무엇을 언제 잡는지. coverage.py=surface(안 실패) vs integrity=gate 구분 명시.

- [ ] **Step 2: CLAUDE.md 실행 섹션에 한 줄 추가**

`uv run bakery check-integrity` — 신규 데이터 편입 전 무결성 게이트(7단계 아티제 ingest de-risk).

- [ ] **Step 3: 커밋**

```bash
git add docs/data_integrity.md .claude/CLAUDE.md
git commit -m "docs: 무결성 3메커니즘 우산 통합 (Task 6)"
```

---

## Self-Review (작성자 체크)

**1. Spec coverage:**
- §1 목적/원칙 → 전체 ✓ / §2 입력 구분(vintage 전용) → Global Constraints + Task 4 conflict 분리 ✓ / §3 구조 invariant → Task 1,2 ✓ / §4a fail게이트(타깃 known-set) → Task 3 ✓ / §4b conflict(CD_USERDEF4·RT_DISC) → Task 4,5 ✓ / §4c CSV → Task 3,5 ✓ / §5 CLI+순수함수 → Task 5 ✓ / 우산 통합 → Task 6 ✓.
- ★blocker 해결(타깃=known-set regression, orphan 분류 안 함) → Task 3 `check_target_items_resolve` ✓.
- ★절대수 앵커는 gate 안 함(vintage 전용) → 어느 체크도 510,585 안 씀 ✓.

**2. advisor 3교정 반영:**
- **#1 할인코드 정규화 미확정** → `check_used_discounts_resolve` fail→**drift** 강등(실측 '357' 1종, zero-pad하면 28종 깨짐, 코드체계 3/4자리 혼재). 아티제 문의. 현 데이터 exit 0.
- **#2 conflict 필드** → NM_ITEM(cosmetic) 대신 **CD_USERDEF4(당일폐기 플래그)** 우선(라벨 오염). RT_DISC(할인율)는 할인 마스터 대조 시(후속). `find_conflicting_codes` fields 일반화라 순수함수 불변.
- **#3 conflict = one-shot 진단** → `run_all`에서 분리→`run_conflict_diagnostic` + `check-conflict` 별도 커맨드(vintage, hot-path xlsx 재파싱 제거). 위치기준 rename에 **코드열 assert 가드**(위치 신뢰 금지, sheet-2 교훈).

**3. Type consistency:** `Violation(check,severity,detail,count)` · `find_missing_codes`/`find_conflicting_codes`/`check_*_resolve`/`run_all`(→2-tuple)/`run_conflict_diagnostic`/`has_fail`/`write_missing`/`write_conflicting` — 정의(초기 Task)와 사용(Task 5) 시그니처 일치.

**주의 (구현 시):**
- Task 5 `check-conflict`의 위치기준 rename(0열=CD_ITEM, 6열=CD_USERDEF4)은 **코드열 assert 가드 필수**(0열이 `\d+` 패턴 >90%). 실패 시 STOP(헤더 위치 변동 = sheet-2류 재발).
- Task 5 실데이터 smoke: 타깃 166 전부 resolve(실측 0 미resolve 확인) → exit 0이 정상.
- coverage.py 안 건드림(surface 층 유지).

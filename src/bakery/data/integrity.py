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
    """SALES_FG는 '0'(정상)/'1'(반품)만. sheet-2 스왑이면 타임스탬프가 들어와 위반.

    float64로 로드되면 0.0/1.0처럼 ".0"이 붙어 문자열 비교가 깨지므로 정규화한다.
    """
    norm = sales["SALES_FG"].astype(str).str.replace(r"\.0$", "", regex=True)
    bad = ~norm.isin({"0", "1"})
    if not bad.any():
        return []
    return [Violation("sales_fg_domain", "fail",
                      "SALES_FG not in {0,1} (sheet-swap 의심)", int(bad.sum()))]


def check_sales_time_format(sales: pd.DataFrame) -> list[Violation]:
    """SALES_TIME은 14자리 숫자(YYYYMMDDHHMMSS). 스왑이면 0/1이 들어와 위반.

    float64로 로드되면 "20260101120000.0"처럼 ".0"이 붙으므로 정규화 후 검사한다.
    """
    norm = sales["SALES_TIME"].astype(str).str.replace(r"\.0$", "", regex=True)
    bad = ~norm.str.fullmatch(r"\d{14}")
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

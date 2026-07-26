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

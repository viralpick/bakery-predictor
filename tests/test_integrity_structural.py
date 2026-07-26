import pandas as pd

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

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


def test_sales_fg_domain_accepts_float_coded():
    # float64로 로드되면 0.0/1.0처럼 ".0"이 붙음 — 정상 데이터로 취급해야 함
    floaty = _good_sales()
    floaty["SALES_FG"] = [0.0, 1.0, 0.0]
    assert integrity.check_sales_fg_domain(floaty) == []


def test_sales_time_format_accepts_float_timestamp():
    # float64로 로드되면 "20260101120000.0"처럼 ".0"이 붙음 — 정상 데이터로 취급해야 함
    floaty = _good_sales()
    floaty["SALES_TIME"] = ["20260101120000.0", "20260101130000.0", "20260102090000.0"]
    assert integrity.check_sales_time_format(floaty) == []


def test_line_uniqueness_catches_dup():
    dup = _good_sales()
    dup.loc[2, ["NO_POS", "SLIP_NO", "SLIP_LINE"]] = ["1", "10", "1"]  # row0 중복
    v = integrity.check_line_uniqueness(dup)
    assert len(v) == 1 and v[0].count == 2   # 중복 그룹 2행


def test_line_uniqueness_full_grain_distinguishes_store_and_date():
    """NO_POS/SLIP_NO는 매장×일자별로 리셋되므로 같은 (NO_POS,SLIP_NO,SLIP_LINE)이
    다른 매장/다른 날짜에 나오는 건 정상(위반 아님) — CD_PARTNER+DT_SALE까지 키에
    있어야 진짜 중복만 잡는다(Task 5 real-data 회귀 방지)."""
    same_line_different_store_and_date = pd.DataFrame({
        "CD_PARTNER": ["store_a", "store_b"], "DT_SALE": ["20260101", "20260102"],
        "NO_POS": ["1", "1"], "SLIP_NO": ["10", "10"], "SLIP_LINE": ["1", "1"],
    })
    assert integrity.check_line_uniqueness(same_line_different_store_and_date) == []

    true_full_key_dup = pd.DataFrame({
        "CD_PARTNER": ["store_a", "store_a"], "DT_SALE": ["20260101", "20260101"],
        "NO_POS": ["1", "1"], "SLIP_NO": ["10", "10"], "SLIP_LINE": ["1", "1"],
    })
    v = integrity.check_line_uniqueness(true_full_key_dup)
    assert len(v) == 1 and v[0].severity == "fail" and v[0].count == 2


def test_schema_catches_missing_col_and_dtype():
    df = _good_sales().astype({"SALES_FG": "int64"})  # dtype 위반 유도
    del df["SALES_TIME"]  # 컬럼 누락
    v = integrity.check_schema(df, {"SALES_FG": "string", "SALES_TIME": "string"})
    kinds = sorted(x.detail.split(":")[0] for x in v)
    assert kinds == ["dtype", "missing"]

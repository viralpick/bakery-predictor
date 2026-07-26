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


def test_return_ratio_float_coded_normalizes():
    # float64 SALES_FG (1.0/0.0)도 정규화 후 2% → range 통과 (형제 체크와 일관)
    fg = [1.0] * 2 + [0.0] * 98
    df = pd.DataFrame({"SALES_FG": fg})
    assert integrity.check_return_ratio(df) == []

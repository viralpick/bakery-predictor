import pandas as pd
import pytest

from bakery.analysis.month_dow import (
    MONTH_DOW_VALUE_COLUMNS,
    adjust_effect_table,
    month_dow_matrix,
)

DOW_LABELS = ["월", "화", "수", "목", "금", "토", "일"]


def _series():
    """2025-01-06(월) ~ 01-12(일) 1주 + 02-03(월) 1일."""
    dates = list(pd.date_range("2025-01-06", periods=7, freq="D")) + [pd.Timestamp("2025-02-03")]
    return pd.DataFrame({
        "date": dates,
        "sold_total_unit": [100, 110, 120, 130, 140, 200, 190, 300],
        "sold_closing": [10, 10, 10, 10, 10, 20, 20, 30],
        "adjusted_demand_unit": [92, 102, 112, 122, 132, 184, 174, 276],
    })


def test_value_columns_declared():
    assert MONTH_DOW_VALUE_COLUMNS == ("sold_total_unit", "adjusted_demand_unit",
                                       "sold_closing")


def test_month_dow_matrix_shape_and_values():
    matrix = month_dow_matrix(_series(), "sold_total_unit")
    assert matrix.columns.tolist() == DOW_LABELS
    assert matrix.index.tolist() == [1, 2]
    assert matrix.loc[1, "월"] == 100.0
    assert matrix.loc[1, "일"] == 190.0
    assert matrix.loc[2, "월"] == 300.0
    assert pd.isna(matrix.loc[2, "화"])          # 2월 화요일 관측 없음


def test_adjust_effect_table_exact():
    table = adjust_effect_table(_series())
    monday_jan = table[(table["month"] == 1) & (table["dow"] == 0)].iloc[0]
    assert monday_jan["raw_mean"] == 100.0
    assert monday_jan["adjusted_mean"] == 92.0
    assert monday_jan["closing_mean"] == 10.0
    assert monday_jan["delta"] == -8.0
    assert monday_jan["delta_pct"] == pytest.approx(-8.0)


def test_adjust_effect_table_covers_every_observed_cell():
    table = adjust_effect_table(_series())
    assert len(table) == 8          # 1월 7요일 + 2월 월요일


def test_script_delegates_to_primitive():
    import sys
    sys.path.insert(0, "scripts")
    import verify_month_dow_adjust

    assert verify_month_dow_adjust.month_dow_matrix is month_dow_matrix

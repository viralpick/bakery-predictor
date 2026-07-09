import pandas as pd
import pytest

from bakery.features.scale import compute_item_scale


def _daily():
    return pd.DataFrame({
        "item_id": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01",
                                 "2021-01-01", "2021-01-02"]),
        "adjusted_demand": [10.0, 20.0, 999.0, 0.0, 0.0],
    })


def test_mean_over_pre_cutoff_only():
    # cutoff 2021-01-15 → A: mean(10,20)=15 (999 제외); B: mean(0,0)=0 → floor 1.0
    scale = compute_item_scale(_daily(), before_date=pd.Timestamp("2021-01-15"))
    assert scale["A"] == pytest.approx(15.0)
    assert scale["B"] == pytest.approx(1.0)  # floor


def test_leakage_rows_on_or_after_cutoff_excluded():
    # cutoff 2021-01-02 (strict <) → A: only 2021-01-01=10.0
    scale = compute_item_scale(_daily(), before_date=pd.Timestamp("2021-01-02"))
    assert scale["A"] == pytest.approx(10.0)


def test_floor_applies_to_custom_value():
    scale = compute_item_scale(_daily(), before_date=pd.Timestamp("2021-01-15"), floor=5.0)
    assert scale["B"] == pytest.approx(5.0)


def test_all_nan_pre_cutoff_values_fall_back_to_floor_not_nan():
    # item C의 pre-cutoff adjusted_demand가 전부 NaN → mean()도 NaN.
    # Python max(nan, floor)는 nan을 그대로 반환하는 함정이 있어 floor로 명시 처리해야 함.
    daily = pd.DataFrame({
        "item_id": ["C", "C"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "adjusted_demand": [float("nan"), float("nan")],
    })
    scale = compute_item_scale(daily, before_date=pd.Timestamp("2021-01-15"), floor=1.0)
    assert scale["C"] == pytest.approx(1.0)

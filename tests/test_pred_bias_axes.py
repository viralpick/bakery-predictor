import pandas as pd

from bakery.analysis.pred_bias import (
    EXTREME_THRESHOLDS,
    SUMMER_MONTHS,
    WEEKEND_DOW,
    WINTER_MONTHS,
    bias_by_axis,
    robust_z,
    segment_contrast,
    stockout_rate_percent,
    wpe_percent,
)


def _preds():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-11", "2025-01-12"]),
        "actual": [100.0, 100.0, 200.0, 200.0],
        "expected": [90.0, 110.0, 180.0, 220.0],
        "production": [95.0, 120.0, 190.0, 250.0],
    })


def test_constants():
    assert SUMMER_MONTHS == (6, 7, 8, 9)
    assert WINTER_MONTHS == (12, 1, 2)
    assert WEEKEND_DOW == (5, 6)
    assert EXTREME_THRESHOLDS == {"heatwave_max_ta": 33.0, "coldwave_min_ta": -10.0,
                                  "heavy_rain_mm": 30.0}


def test_wpe_percent_exact():
    # Σexpected 600, Σactual 600 → 0%
    assert wpe_percent(_preds()) == 0.0
    biased = _preds().assign(expected=[80.0, 80.0, 160.0, 160.0])
    # (480 − 600)/600 × 100 = −20%
    assert wpe_percent(biased) == -20.0


def test_stockout_rate_percent_exact():
    # actual > production: 100>95 (True), 100>120 (False), 200>190 (True), 200>250 (False)
    assert stockout_rate_percent(_preds()) == 50.0


def test_bias_by_axis_groups_and_computes():
    # fixture 날짜: 01-06(월=0), 01-07(화=1), 01-11(토=5), 01-12(일=6) — 각 1건
    preds = _preds().assign(dow=lambda d: d["date"].dt.dayofweek)
    table = bias_by_axis(preds, "dow").set_index("dow")
    assert set(table.index) == {0, 1, 5, 6}
    assert table["n"].tolist() == [1, 1, 1, 1]
    # 월요일: expected 90, actual 100 → (90−100)/100×100 = −10%
    assert table.loc[0, "wpe"] == -10.0
    # 화요일: expected 110 > actual 100 → +10%
    assert table.loc[1, "wpe"] == 10.0
    # 매진률: 월 actual 100 > production 95 → 100%
    assert table.loc[0, "stockout_rate"] == 100.0
    assert table.loc[1, "stockout_rate"] == 0.0


def test_robust_z_is_zero_at_median():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = robust_z(values)
    assert z[2] == 0.0                            # median=3
    assert z[4] > 0.0


def test_robust_z_handles_zero_mad():
    z = robust_z(pd.Series([2.0, 2.0, 2.0]))
    assert z.tolist() == [0.0, 0.0, 0.0]


def test_segment_contrast_ci_is_deterministic():
    preds = _preds()
    mask = preds["date"].dt.dayofweek.isin(WEEKEND_DOW)
    first = segment_contrast(preds, mask, n_boot=50, seed=42)
    second = segment_contrast(preds, mask, n_boot=50, seed=42)
    assert first["wpe_diff"] == second["wpe_diff"]
    assert first["ci"].tolist() == second["ci"].tolist()
    assert first["n_segment"] == 2
    assert first["n_rest"] == 2

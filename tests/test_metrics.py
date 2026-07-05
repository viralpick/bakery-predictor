import numpy as np
import pandas as pd
import pytest

from bakery.evaluation.metrics import (
    coverage,
    coverage_by_group,
    grouped_wape,
    interval_width,
    mae,
    mase,
    pinball_loss,
    rmse,
    summarize,
    wape,
    wpe,
)


def test_wape_perfect_prediction_is_zero():
    y = np.array([10, 20, 30], dtype=float)
    assert wape(y, y) == 0.0


def test_wape_constant_underprediction():
    # predict half: residual sum = 30, denom = 60 → wape = 0.5
    y = np.array([10, 20, 30], dtype=float)
    yhat = y / 2
    assert wape(y, yhat) == 0.5


def test_wape_handles_zero_target_with_nan():
    # all-zero target → undefined; we return nan rather than crashing
    y = np.zeros(5)
    yhat = np.array([1, 0, 0, 0, 0], dtype=float)
    assert np.isnan(wape(y, yhat))


def test_mae_rmse_basic():
    y = np.array([0, 0, 0], dtype=float)
    yhat = np.array([1, 1, 1], dtype=float)
    assert mae(y, yhat) == 1.0
    assert rmse(y, yhat) == 1.0


def test_shape_mismatch_raises():
    try:
        mae(np.zeros(3), np.zeros(4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_grouped_wape_returns_one_row_per_group():
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "y": [10.0, 20.0, 5.0, 5.0],
            "yhat": [10.0, 10.0, 0.0, 10.0],
        }
    )
    result = grouped_wape(df, by=["g"], y_col="y", yhat_col="yhat")
    a = result.loc[result["g"] == "a", "wape"].iloc[0]
    b = result.loc[result["g"] == "b", "wape"].iloc[0]
    assert a == 10 / 30
    assert b == 10 / 10


def test_summarize_keys():
    out = summarize(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert set(out) == {"wape", "wpe", "mae", "rmse"}


def test_coverage_all_inside():
    actual = np.array([5.0, 10.0, 15.0])
    lower = np.array([0.0, 5.0, 10.0])
    upper = np.array([10.0, 15.0, 20.0])
    assert coverage(actual, lower, upper) == 1.0


def test_coverage_boundary_counts_as_inside():
    # closed interval: actual exactly on lower or upper bound is covered
    actual = np.array([0.0, 20.0])
    lower = np.array([0.0, 10.0])
    upper = np.array([10.0, 20.0])
    assert coverage(actual, lower, upper) == 1.0


def test_coverage_partial():
    actual = np.array([5.0, 25.0, 15.0, -1.0])
    lower = np.array([0.0, 10.0, 10.0, 0.0])
    upper = np.array([10.0, 20.0, 20.0, 10.0])
    # row0 inside, row1 above, row2 inside, row3 below → 2/4
    assert coverage(actual, lower, upper) == 0.5


def test_coverage_by_group_returns_per_group_rate():
    actual = np.array([5.0, 25.0, 15.0, 21.0])
    lower = np.array([0.0, 10.0, 10.0, 10.0])
    upper = np.array([10.0, 20.0, 20.0, 20.0])
    group = np.array(["mon", "mon", "tue", "tue"])
    out = coverage_by_group(actual, lower, upper, group)
    assert out["mon"] == 0.5  # 5 inside, 25 above
    assert out["tue"] == 0.5  # 15 inside, 21 above


def test_interval_width_mean():
    lower = np.array([0.0, 5.0, 10.0])
    upper = np.array([10.0, 15.0, 30.0])
    # widths 10, 10, 20 → mean 40/3
    assert interval_width(lower, upper) == 40.0 / 3.0


def test_pinball_loss_underprediction_weighted_by_q():
    # pred below actual → weighted by q
    actual = np.array([10.0])
    pred = np.array([0.0])
    assert pinball_loss(actual, pred, q=0.9) == pytest.approx(9.0)


def test_pinball_loss_overprediction_weighted_by_one_minus_q():
    # pred above actual → weighted by (1-q)
    actual = np.array([0.0])
    pred = np.array([10.0])
    assert pinball_loss(actual, pred, q=0.9) == pytest.approx(1.0)


def test_pinball_loss_perfect_is_zero():
    actual = np.array([5.0, 10.0])
    assert pinball_loss(actual, actual, q=0.5) == 0.0


def test_mase_equals_one_when_model_matches_naive_error():
    # seasonal-naive (season=1) on train: |2-1|+|3-2| → MAE 1.0
    train = np.array([1.0, 2.0, 3.0])
    actual = np.array([10.0, 20.0])
    pred = np.array([11.0, 19.0])  # MAE 1.0
    assert mase(actual, pred, train, season=1) == 1.0


def test_mase_halves_when_model_error_is_half_naive():
    train = np.array([1.0, 2.0, 3.0])  # naive MAE 1.0
    actual = np.array([10.0, 20.0])
    pred = np.array([10.5, 19.5])  # MAE 0.5
    assert mase(actual, pred, train, season=1) == 0.5


def test_wpe_sign_and_value():
    y = np.array([10.0, 10.0, 10.0, 10.0])
    over = np.array([12.0, 12.0, 12.0, 12.0])   # 과대예측 → +
    under = np.array([8.0, 8.0, 8.0, 8.0])      # 과소예측 → −
    exact = np.array([10.0, 10.0, 10.0, 10.0])
    assert wpe(y, over) == 0.2      # (48-40)/40
    assert wpe(y, under) == -0.2    # (32-40)/40
    assert wpe(y, exact) == 0.0


def test_summarize_includes_wpe():
    y = np.array([10.0, 20.0])
    yhat = np.array([11.0, 19.0])
    out = summarize(y, yhat)
    assert "wpe" in out
    assert out["wpe"] == 0.0        # (+1 -1)/30

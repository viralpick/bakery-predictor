import numpy as np
import pandas as pd

from bakery.evaluation.metrics import grouped_wape, mae, rmse, summarize, wape


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
    assert set(out) == {"wape", "mae", "rmse"}

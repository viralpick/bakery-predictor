import numpy as np
import pytest
from bakery.evaluation.metrics import quantile_exceedance_rate


def test_exceedance_rate_counts_strict_exceed():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([15.0, 15.0, 35.0, 35.0])  # exceed at idx1, idx3
    assert quantile_exceedance_rate(y_true, y_pred) == 0.5


def test_exceedance_all_covered_is_zero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    assert quantile_exceedance_rate(y_true, y_pred) == 0.0


def test_exceedance_shape_mismatch_raises():
    with pytest.raises(ValueError):
        quantile_exceedance_rate(np.array([1.0, 2.0]), np.array([1.0]))

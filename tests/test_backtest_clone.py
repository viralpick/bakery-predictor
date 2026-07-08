from bakery.evaluation.backtest import _clone
from bakery.models.lightgbm_regressor import GlobalLGBM


def test_clone_preserves_non_default_y_col():
    original = GlobalLGBM(feature_set="v2", y_col="adjusted_demand")
    clone = _clone(original)
    assert clone.y_col == "adjusted_demand"


def test_clone_preserves_feature_set_and_default_y_col():
    original = GlobalLGBM(feature_set="v2")  # default y_col = potential_demand
    clone = _clone(original)
    assert clone.feature_set == "v2"
    assert clone.y_col == "potential_demand"

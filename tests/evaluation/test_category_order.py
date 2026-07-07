import numpy as np
import pandas as pd
from bakery.cli import _category_total_fold_predictions


def _synth_category_features(n_days: int) -> pd.DataFrame:
    """합성 카테고리-일별 프레임: date + target + 숫자 feature 2개. lag/holiday 없이
    fit_category_total의 select_feature_cols가 쓸 수 있는 최소 형태."""
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    rng = np.arange(n_days, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "adjusted_demand_unit": 100.0 + 10.0 * np.sin(rng / 7.0) + rng * 0.1,
        "dow": dates.dayofweek.astype(float),
        "trend": rng,
    })


def test_fold_predictions_shape_and_folds():
    feats = _synth_category_features(365 + 2 * 30)
    out = _category_total_fold_predictions(
        feats, production_quantile=0.85, horizon_days=30, n_folds=2,
    )
    # 계약: fold별 horizon_days개 test date, fold 라벨 {0,1}, 발주 비음수.
    assert set(out["fold"].unique()) == {0, 1}
    assert len(out) == 2 * 30
    assert (out["total_order"] >= 0).all()


def test_fold_predictions_raises_when_insufficient_days():
    feats = _synth_category_features(100)
    try:
        _category_total_fold_predictions(feats, production_quantile=0.85, horizon_days=30, n_folds=2)
        assert False, "expected ValueError"
    except ValueError:
        pass

import numpy as np
import pandas as pd
from bakery.cli import _category_total_fold_predictions
from bakery.models.item_proportion import distribute_total


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


def test_distribute_total_preserves_category_sum():
    # history: 두 품목, cutoff 이전 판매로 비율 형성
    hist = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-20", "2024-01-20"]),
        "item_id": ["a", "b", "a", "b"],
        "category_id": ["bread", "bread", "bread", "bread"],
        "sold_units": [10, 30, 10, 30],
        "is_stockout": [False, False, False, False],
        "stockout_time": [pd.NaT] * 4,
    })
    totals = pd.Series({pd.Timestamp("2024-02-01"): 100.0})
    res = distribute_total(hist, totals)
    # 배분 보존: 그 날 품목 발주 합 == 카테고리 총합.
    day_sum = res.quantities.groupby("date")["qty"].sum().iloc[0]
    assert round(day_sum, 6) == 100.0

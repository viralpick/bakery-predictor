import pandas as pd
import pytest

from bakery.cli import _apply_conformal_to_folds


def _preds():
    # 2 folds × 2 items. fold 0 = cal, fold 1 = test (frac 0.5, 2 folds → n_cal=1).
    return pd.DataFrame({
        "item_id": ["A", "B", "A", "B"],
        "date": pd.to_datetime(["2021-03-01", "2021-03-01", "2021-04-01", "2021-04-01"]),
        "fold": [0, 0, 1, 1],
        "adjusted_demand": [12.0, 6.0, 10.0, 5.0],
        "yhat": [10.0, 5.0, 10.0, 5.0],
    })


def test_only_test_folds_returned_and_schema():
    scale = {"A": 4.0, "B": 2.0}
    out = _apply_conformal_to_folds(_preds(), scale, service_level=0.5, cal_fold_frac=0.5)
    assert list(out.columns) == ["item_id", "date", "fold", "our_order"]
    assert set(out["fold"].unique()) == {1}          # cal fold 0 제외
    assert len(out) == 2


def test_our_order_equals_base_plus_qs_times_scale():
    # cal fold0 scores: A (12-10)/4=0.5, B (6-5)/2=0.5 → q_s(s=0.5, higher)=0.5
    # test fold1: A 10+0.5*4=12, B 5+0.5*2=6
    scale = {"A": 4.0, "B": 2.0}
    out = _apply_conformal_to_folds(_preds(), scale, service_level=0.5, cal_fold_frac=0.5)
    got = dict(zip(out["item_id"], out["our_order"]))
    assert got["A"] == pytest.approx(12.0)
    assert got["B"] == pytest.approx(6.0)


def test_missing_item_scale_defaults_to_floor_one():
    preds = _preds().assign(item_id=["A", "C", "A", "C"])  # C not in scale dict
    scale = {"A": 4.0}
    out = _apply_conformal_to_folds(preds, scale, service_level=0.5, cal_fold_frac=0.5)
    # cal scores: A 0.5, C (6-5)/1=1.0 → q_s higher@0.5 of [0.5,1.0]=1.0
    # test C: 5 + 1.0*1.0 = 6.0
    got = dict(zip(out["item_id"], out["our_order"]))
    assert got["C"] == pytest.approx(6.0)


def test_cal_test_split_is_chronological_not_by_fold_index():
    # fold 0 = LATER (2021-05-01), fold 1 = EARLIER (2021-02-01):
    # generate_time_splits gives fold=0 the most-recent window, so cal/test
    # must split on DATE, not raw fold int. cal = earlier (fold 1), test = fold 0.
    preds = pd.DataFrame({
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2021-05-01", "2021-02-01"]),
        "fold": [0, 1],
        "adjusted_demand": [10.0, 14.0],
        "yhat": [10.0, 10.0],
    })
    scale = {"A": 4.0}
    out = _apply_conformal_to_folds(preds, scale, service_level=0.5, cal_fold_frac=0.5)
    assert set(out["fold"].unique()) == {0}          # fold 0 (later) = test
    # cal = fold 1 (earlier): score (14-10)/4 = 1.0 → q_s(higher@0.5)=1.0
    # test fold 0: 10 + 1.0*4 = 14
    assert out["our_order"].iloc[0] == pytest.approx(14.0)

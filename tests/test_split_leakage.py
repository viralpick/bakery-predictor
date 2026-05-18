import pandas as pd
import pytest

from bakery.evaluation.split import apply_split, assert_no_leakage, generate_time_splits


def _toy_df(n_days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for store in ["s1", "s2"]:
            for item in ["i1", "i2"]:
                rows.append({"date": d, "store_id": store, "item_id": item, "sold_units": 10})
    return pd.DataFrame(rows)


def test_splits_are_chronological_and_disjoint():
    df = _toy_df()
    windows = generate_time_splits(df["date"], n_splits=3, val_horizon_days=7, step_days=7)
    assert len(windows) == 3
    for w in windows:
        assert w.train_end < w.val_start
        assert w.val_start <= w.val_end


def test_apply_split_passes_leakage_check():
    df = _toy_df()
    windows = generate_time_splits(df["date"], n_splits=2, val_horizon_days=7, step_days=7)
    for w in windows:
        train, val = apply_split(df, w)
        assert train["date"].max() < val["date"].min()
        assert not train.empty
        assert not val.empty


def test_assert_no_leakage_catches_overlap():
    overlap_train = pd.DataFrame({"date": pd.to_datetime(["2024-06-01", "2024-06-08"])})
    overlap_val = pd.DataFrame({"date": pd.to_datetime(["2024-06-08", "2024-06-09"])})
    with pytest.raises(AssertionError, match="time leakage"):
        assert_no_leakage(overlap_train, overlap_val)


def test_expanding_train_grows_across_folds():
    df = _toy_df()
    windows = generate_time_splits(df["date"], n_splits=3, mode="expanding")
    train_sizes = [(w.train_end - w.train_start).days for w in windows]
    assert train_sizes == sorted(train_sizes)


def test_rolling_train_size_stable():
    df = _toy_df()
    windows = generate_time_splits(df["date"], n_splits=3, mode="rolling", min_train_days=60)
    spans = {(w.train_end - w.train_start).days for w in windows}
    assert spans == {59}


def test_insufficient_history_raises():
    df = _toy_df(n_days=30)
    with pytest.raises(ValueError, match="unique days"):
        generate_time_splits(df["date"], n_splits=2, val_horizon_days=7, min_train_days=90)

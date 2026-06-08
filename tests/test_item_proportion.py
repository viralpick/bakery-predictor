import numpy as np
import pandas as pd
import pytest

from bakery.models.item_proportion import distribute_interval


def _toy_history(n_days: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    means = {"i1": 30, "i2": 10, "i3": 5}
    rows = []
    for d in dates:
        for item, mu in means.items():
            rows.append(
                {
                    "date": d,
                    "item_id": item,
                    "category_id": "c1",
                    "sold_units": int(rng.poisson(mu)),
                    "is_stockout": False,
                    "stockout_time": pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def _interval(td) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": [td], "lower": [80.0], "anchor": [100.0], "upper": [130.0]}
    )


def test_item_bounds_sum_to_category_bounds():
    hist = _toy_history()
    td = hist["date"].max() + pd.Timedelta(days=1)
    out = distribute_interval(hist, _interval(td))
    # proportions sum to 1 → item bounds must reconstruct the category bounds
    assert out["item_lower"].sum() == pytest.approx(80.0)
    assert out["item_anchor"].sum() == pytest.approx(100.0)
    assert out["item_upper"].sum() == pytest.approx(130.0)


def test_item_interval_ordering_preserved():
    hist = _toy_history()
    td = hist["date"].max() + pd.Timedelta(days=1)
    out = distribute_interval(hist, _interval(td))
    assert (out["item_lower"] <= out["item_anchor"]).all()
    assert (out["item_anchor"] <= out["item_upper"]).all()


def test_item_width_scales_with_proportion():
    hist = _toy_history()
    td = hist["date"].max() + pd.Timedelta(days=1)
    out = distribute_interval(hist, _interval(td))
    cat_width = 130.0 - 80.0
    item_width = out["item_upper"] - out["item_lower"]
    assert np.allclose(item_width.to_numpy(), (out["proportion"] * cat_width).to_numpy())


def test_higher_selling_item_gets_larger_share():
    hist = _toy_history()
    td = hist["date"].max() + pd.Timedelta(days=1)
    out = distribute_interval(hist, _interval(td)).set_index("item_id")
    assert out.loc["i1", "proportion"] > out.loc["i2", "proportion"] > out.loc["i3", "proportion"]

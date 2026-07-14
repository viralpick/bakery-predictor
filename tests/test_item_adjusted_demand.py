import pandas as pd
import pytest

from bakery.features.category_aggregate import build_item_adjusted_demand


def _daily(sold):
    return pd.DataFrame({
        "item_id": ["A", "B"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        "sold_units": sold,
    })


def _closing(qty_by_item):
    return pd.DataFrame({
        "item_id": list(qty_by_item.keys()),
        "date": pd.to_datetime(["2021-01-01"] * len(qty_by_item)),
        "qty": list(qty_by_item.values()),
    })


def test_alpha_half_discounts_closing():
    # A: sold=10, closing=4 → adjusted = 10 - 4*(1-0.5) = 8
    daily = _daily([10, 20])
    closing = _closing({"A": 4})  # B has no closing
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=0.5)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(8.0)
    assert got["B"] == pytest.approx(20.0)  # no closing → adjusted == sold


def test_alpha_one_equals_sold():
    daily = _daily([10, 20])
    closing = _closing({"A": 4, "B": 5})
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=1.0)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(10.0)  # adjusted == sold
    assert got["B"] == pytest.approx(20.0)


def test_alpha_zero_equals_normal():
    # adjusted = sold - closing (all closing removed)
    daily = _daily([10, 20])
    closing = _closing({"A": 4, "B": 5})
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=0.0)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(6.0)
    assert got["B"] == pytest.approx(15.0)


def test_multiple_closing_rows_same_item_day_are_summed():
    # A: sold=20, two closing rows qty=3와 qty=2 → groupby.sum으로 합산(5),
    # adjusted = 20 - 5*(1-0.5) = 17.5
    daily = _daily([20, 20])
    closing = pd.DataFrame({
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        "qty": [3, 2],
    })
    out = build_item_adjusted_demand(daily, discount_rows=closing, alpha=0.5)
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(17.5)


def test_input_not_mutated():
    daily = _daily([10, 20])
    build_item_adjusted_demand(daily, discount_rows=_closing({"A": 4}), alpha=0.5)
    assert "adjusted_demand" not in daily.columns


def _closing_returns(ret_by_item):
    return pd.DataFrame({
        "item_id": list(ret_by_item.keys()),
        "date": pd.to_datetime(["2021-01-01"] * len(ret_by_item)),
        "ret_qty": list(ret_by_item.values()),
    })


def test_closing_returns_netted_from_closing_qty():
    # A: sold=10(net), closing gross=4, 마감반품=1 → closing_net=3
    # adjusted = 10 − 3*(1−0.5) = 8.5  (반품 미반영 시 8.0 → seam)
    daily = _daily([10, 20])
    out = build_item_adjusted_demand(
        daily, discount_rows=_closing({"A": 4}),
        closing_returns=_closing_returns({"A": 1}), alpha=0.5,
    )
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(8.5)
    assert got["B"] == pytest.approx(20.0)


def test_closing_returns_clip_at_zero():
    # 마감반품(5) > closing(2) → closing_net=0, adjusted == sold
    daily = _daily([10, 20])
    out = build_item_adjusted_demand(
        daily, discount_rows=_closing({"A": 2}),
        closing_returns=_closing_returns({"A": 5}), alpha=0.5,
    )
    got = dict(zip(out["item_id"], out["adjusted_demand"]))
    assert got["A"] == pytest.approx(10.0)

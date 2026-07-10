import pandas as pd
from bakery.cli import _resolve_demand_col


def _daily():
    return pd.DataFrame({
        "store_id": ["S", "S"],
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "sold_units": [10, 20],
        "potential_demand": [10.0, 20.0],
    })


def test_synthetic_returns_potential_and_same_frame():
    df = _daily()
    out, col = _resolve_demand_col(df, "synthetic", 0.5)
    assert col == "potential_demand"
    assert out is df  # synthetic은 프레임 비변형


def test_real_attaches_adjusted_demand():
    df = _daily()
    discount = pd.DataFrame({
        "item_id": ["A"],
        "date": pd.to_datetime(["2026-01-02"]),
        "qty": [4],
    })
    out, col = _resolve_demand_col(df, "real", 0.5, discount_rows=discount)
    assert col == "adjusted_demand"
    row = out.set_index("date").loc["2026-01-02"]
    # adjusted = sold - closing*(1-alpha) = 20 - 4*0.5 = 18
    assert float(row["adjusted_demand"]) == 18.0
    # closing 매칭 없는 날은 adjusted == sold
    assert float(out.set_index("date").loc["2026-01-01"]["adjusted_demand"]) == 10.0

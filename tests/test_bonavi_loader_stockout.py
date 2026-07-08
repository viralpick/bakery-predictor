import numpy as np
import pandas as pd
from bakery.data.bonavi_loader import aggregate_daily, assign_stockout_fields


def test_assign_stockout_fields_truth_table():
    df = pd.DataFrame({
        "production_qty": [10.0, 5.0, 8.0, 3.0, np.nan],
        "waste_qty":      [0.0,  2.0, -1.0, 4.0, np.nan],
        "last_sale_ts": pd.to_datetime([
            "2024-01-01 21:30", "2024-01-01 20:00", "2024-01-01 15:00",
            "2024-01-01 19:00", "2024-01-01 12:00",
        ]),
    })
    out = assign_stockout_fields(df)
    # made>0 & waste<=0 → True: row0(10,0), row2(8,-1). row1(waste2>0)=F, row3(waste4)=F, row4(nan)=F
    assert out["is_stockout"].tolist() == [True, False, True, False, False]
    assert out.loc[0, "stockout_time"] == pd.Timestamp("2024-01-01 21:30")
    assert pd.isna(out.loc[1, "stockout_time"])
    assert out.loc[2, "stockout_time"] == pd.Timestamp("2024-01-01 15:00")
    assert pd.isna(out.loc[4, "stockout_time"])


def test_assign_stockout_fields_missing_inventory_is_false():
    # 재고정보 결측(NaN made/waste) → is_stockout False, stockout_time NaT (판매 있어도)
    df = pd.DataFrame({
        "production_qty": [np.nan],
        "waste_qty": [np.nan],
        "last_sale_ts": pd.to_datetime(["2024-01-01 21:00"]),
    })
    out = assign_stockout_fields(df)
    assert out["is_stockout"].tolist() == [False]
    assert pd.isna(out.loc[0, "stockout_time"])


def test_aggregate_daily_wires_inventory_and_last_sale_into_stockout():
    """End-to-end wiring test: aggregate_daily merges inventory (production/waste)
    and last_sale (last_sale_ts) into assign_stockout_fields correctly. Guards
    against a future column rename silently making everything is_stockout=False
    with green CI (only the pure helper was tested before)."""
    store_id = "1000000047"
    date = pd.Timestamp("2024-01-01")

    sales = pd.DataFrame({
        "store_id": [store_id, store_id, store_id],
        "item_id": ["A", "B", "C"],
        "date": [date, date, date],
        "qty": [5, 3, 2],
    })
    items = pd.DataFrame({
        "item_id": ["A", "B", "C"],
        "category_id": ["bread", "bread", "bread"],
    })
    # C is intentionally absent from inventory — models an item with no
    # inventory record for the day.
    inventory = pd.DataFrame({
        "date": [date, date],
        "item_id": ["A", "B"],
        "production_qty": [10.0, 5.0],
        "waste_qty": [0.0, 2.0],
    })
    last_sale = pd.DataFrame({
        "date": [date, date, date],
        "item_id": ["A", "B", "C"],
        "last_sale_ts": [
            pd.Timestamp("2024-01-01 21:30"),
            pd.Timestamp("2024-01-01 20:00"),
            pd.Timestamp("2024-01-01 18:00"),
        ],
    })

    daily = aggregate_daily(sales, items, inventory, last_sale, measured_profiles=None)
    by_item = daily.set_index("item_id")

    # A: production>0 & waste<=0 → true stockout, stockout_time = last_sale_ts
    assert by_item.loc["A", "is_stockout"] == True  # noqa: E712
    assert by_item.loc["A", "stockout_time"] == pd.Timestamp("2024-01-01 21:30")

    # B: waste>0 → not a stockout, stockout_time NaT
    assert by_item.loc["B", "is_stockout"] == False  # noqa: E712
    assert pd.isna(by_item.loc["B", "stockout_time"])

    # C: missing from inventory → NaN production/waste → not a stockout
    assert by_item.loc["C", "is_stockout"] == False  # noqa: E712
    assert pd.isna(by_item.loc["C", "stockout_time"])

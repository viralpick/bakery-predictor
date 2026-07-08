import numpy as np
import pandas as pd
from bakery.data.bonavi_loader import assign_stockout_fields


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

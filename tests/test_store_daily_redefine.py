import numpy as np
import pandas as pd
import pytest

from bakery.data.bonavi_loader import assign_stockout_fields


def test_assign_stockout_fields_redefinition_exact():
    # 재정의: is_stockout = (production_qty>0 & waste_qty<=0), stockout_time=last_sale_ts(완판 시)
    df = pd.DataFrame({
        "production_qty": [10, 10, 0, 10, 10],
        "waste_qty":      [0, 3, 0, -2, np.nan],   # 완판 / 잔여 / 미생산 / 음수(반품, 완판) / 결측
        "last_sale_ts": pd.to_datetime([
            "2024-01-01 20:00", "2024-01-01 21:00", "2024-01-01 19:00",
            "2024-01-01 18:30", "2024-01-01 17:00"]),
    })
    out = assign_stockout_fields(df)
    assert list(out["is_stockout"]) == [True, False, False, True, False]
    # 완판 행만 stockout_time 채워짐
    assert out["stockout_time"].iloc[0] == pd.Timestamp("2024-01-01 20:00")
    assert pd.isna(out["stockout_time"].iloc[1])
    assert pd.isna(out["stockout_time"].iloc[2])
    assert out["stockout_time"].iloc[3] == pd.Timestamp("2024-01-01 18:30")
    assert pd.isna(out["stockout_time"].iloc[4])


def test_build_store_daily_uses_redefinition():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from store_daily import build_store_daily
    d = build_store_daily("1000000047", "store_gw01", exclude_bulk=True)
    # 재정의 후 광교 item-day is_stockout 비율은 옛 92%가 아니라 ~60%대여야 함
    rate = d["is_stockout"].mean()
    assert 0.50 < rate < 0.70, f"expected redefined ~0.60, got {rate:.3f}"
    # 반환 컬럼 스키마 불변
    assert set(["date","item_id","sold_units","store_id","category_id","stockout_time","is_stockout"]).issubset(d.columns)
    # 완판(is_stockout) 행은 stockout_time 있고, 아닌 행은 NaT
    so = d[d["is_stockout"]]; nso = d[~d["is_stockout"]]
    assert so["stockout_time"].notna().all()
    assert nso["stockout_time"].isna().all()


def test_stockout_cols_excluded_from_training_features():
    # 재정의된 is_stockout/stockout_time이 학습 feature로 새지 않음을 고정
    # (품절 컬럼은 LEAK_COLS로 제외되므로 store_daily 재정의는 HTML/발주 예측에 영향 없음)
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from store_daily import build_store_daily, build_store_closing_rows
    from bakery.features.category_aggregate import build_category_daily, build_features
    from bakery.models.category_total import select_feature_cols
    daily = build_store_daily("1000000047", "store_gw01", exclude_bulk=True)
    cd = build_category_daily(daily_raw=daily,
                              discount_rows=build_store_closing_rows("1000000047"), alpha=0.8)
    feat = build_features(cd, target_col="adjusted_demand_unit")
    cols = select_feature_cols(feat, "adjusted_demand_unit")
    leaked = [c for c in cols if "stockout" in c.lower()]
    assert leaked == [], f"stockout cols leaked into features: {leaked}"

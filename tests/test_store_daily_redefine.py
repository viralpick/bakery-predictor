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


def test_assign_stockout_fields_adds_defined_mask():
    """is_stockout_defined = 생산기록 존재(inventory 커버). 완제품(made 결측)은 False.
    is_stockout 자체는 boolean 불변(소비처 무사고)."""
    df = pd.DataFrame({
        "production_qty": [10, 0, np.nan, 10],   # 생산O / 미생산(0) / 완제품(결측) / 생산O
        "waste_qty":      [0, 0, np.nan, 3],
        "last_sale_ts": pd.to_datetime(
            ["2024-01-01 18:00", "2024-01-01 19:00", "2024-01-01 20:00", "2024-01-01 21:00"]),
    })
    out = assign_stockout_fields(df)
    # is_stockout: boolean 그대로 (완제품·미생산=False, dtype bool)
    assert out["is_stockout"].dtype == bool
    assert list(out["is_stockout"]) == [True, False, False, False]
    # companion: 생산기록 있는 행만 True (완제품[idx2]만 False)
    assert out["is_stockout_defined"].dtype == bool
    assert list(out["is_stockout_defined"]) == [True, True, False, True]


def test_build_store_daily_uses_redefinition():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from store_daily import build_store_daily
    d = build_store_daily("1000000047", "store_gw01", exclude_bulk=True)
    # ★생산품목(inventory 커버) 기준 매진률 = 0.605 (완제품 검열 희석 제외)
    produced = d[d["is_stockout_defined"]]
    rate = produced["is_stockout"].mean()
    # round(rate, 2)==0.61 (0.6051 rounds up at 2dp) — check at 3dp to match the ~0.605 design value
    assert round(rate, 3) == 0.605, f"expected produced-scope ~0.605, got {rate:.3f}"
    # 완제품이 실제로 분모에 섞여 있었음을 고정 (전체는 여전히 희석되어 낮음)
    assert d["is_stockout"].mean() < 0.30   # 전체(완제품 포함)는 ~0.151
    # 스키마: 신규 companion 컬럼 존재, is_stockout은 여전히 bool
    assert d["is_stockout"].dtype == bool
    assert "is_stockout_defined" in d.columns
    # median 매진시각은 유효(희석은 rate만)
    so = d[d["is_stockout"]]
    assert so["stockout_time"].dt.hour.median() == 18


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

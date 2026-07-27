"""build_category_daily의 closing 소스가 신규 클린 parquet(CD_USERDEF1)여야 한다."""
from unittest.mock import patch
import pandas as pd
from bakery.features import category_aggregate as ca


def test_build_category_daily_uses_v2_closing_when_clean_parquet_exists():
    """클린 parquet 존재 시 build_category_daily가 v2 closing 로더를 호출한다."""
    with patch("bakery.analysis.discount.CLEAN_PARQUET_DEFAULT") as mock_path, \
         patch("bakery.analysis.discount.load_sales_with_discount_v2") as mock_v2_sales, \
         patch("bakery.analysis.discount.load_closing_returns_v2") as mock_v2_ret, \
         patch("bakery.analysis.discount.load_sales_with_discount") as mock_old_sales:
        mock_path.exists.return_value = True
        # v2 sales가 빈 closing_discount를 반환하도록
        empty = pd.DataFrame({"item_id": [], "date": [], "qty": []})
        mock_v2_sales.return_value.closing_discount.return_value = empty
        mock_v2_ret.return_value = pd.DataFrame({"item_id": [], "date": [], "ret_qty": []})
        # 최소 daily_raw 주입(파일 IO 회피)
        daily_raw = pd.DataFrame({
            "store_id": ["store_gw01"], "item_id": ["x1"], "category_id": ["bread"],
            "date": pd.to_datetime(["2024-01-01"]), "sold_units": [10],
            "is_stockout": [False], "stockout_time": [pd.NaT],
            "open_hours": [12], "capacity": [20], "potential_demand": [10.0],
        })
        ca.build_category_daily(daily_raw=daily_raw)
        mock_v2_sales.assert_called_once()   # 신규 소스 사용
        mock_old_sales.assert_not_called()   # 옛 0520 미사용

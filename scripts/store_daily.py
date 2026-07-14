"""매장별 item-level daily / closing / 카테고리매핑 빌더 (공유 모듈).

v4_new_data_backtest 의 광교 canonical 경로를 store_cd 로 매개화한 것.
plotting(interval_backtest_4stores) / 분석(substitution_4stores) 양쪽에서 재사용.
matplotlib 등 무거운 의존 없음 — 순수 데이터 빌더만.
"""
from __future__ import annotations

import pandas as pd

from bakery.data.bonavi_loader import _aggregate_returns, assign_stockout_fields
from bakery.data.bulk import flag_bulk_lines
from v4_new_data_backtest import CLOSING_CODES, V2, map_category

STORE_MAP = {
    "1000000047": ("광교", "store_gw01"),
    "1000000009": ("삼성타운", "store_ss01"),
    "1000000029": ("메세나폴리스", "store_mp01"),
    "1000000485": ("광화문", "store_gh01"),
}


def item_category_map() -> dict[str, str]:
    items = pd.read_parquet(V2 / "items.parquet")
    items["item_id"] = items["CD_ITEM"].astype(str)
    items["category_id"] = items["NM_ITEM"].apply(map_category)
    return items.set_index("item_id")["category_id"].to_dict()


def build_store_daily(store_cd: str, store_id: str, exclude_bulk: bool = True) -> pd.DataFrame:
    """build_new_data_daily 의 store_cd 매개화 버전 — 전 카테고리 item-level daily."""
    sales = pd.read_parquet(V2 / "sales.parquet")
    sales = sales[sales["CD_PARTNER"].astype(str) == store_cd]
    # 반품(SALES_FG=1) net-out용 — 단품(SS)만, (item,date)별 소매반품 집계.
    # 대량취소 제외는 _aggregate_returns(단매장 bonavi_loader와 단일 출처 공유).
    _ret_raw = sales[(sales["SALES_FG"].astype(str) == "1")
                     & (sales["CD_USERDEF2"].astype(str) == "SS")]
    returns = _aggregate_returns(pd.DataFrame({
        "item_id": _ret_raw["CD_ITEM"].astype(str),
        "date": pd.to_datetime(_ret_raw["DT_SALE"].astype(str)),
        "qty": pd.to_numeric(_ret_raw["QT_SALE"], errors="coerce").fillna(0),
    }))
    sales = sales[sales["SALES_FG"].astype(str) == "0"]
    sales = sales[sales["CD_USERDEF2"].astype(str) == "SS"]
    sales["date"] = pd.to_datetime(sales["DT_SALE"].astype(str))
    sales["QT_SALE"] = pd.to_numeric(sales["QT_SALE"], errors="coerce").fillna(0)

    if exclude_bulk:
        # 신규 bulk 검출 (bakery.data.bulk.flag_bulk_lines) — line-level 제거, 매장×품목 통계.
        # 구 sales_with_bulk_flag.parquet(whole-receipt) 대체, 패키지 CLI 경로와 단일 출처 통일.
        lines = pd.DataFrame(
            {
                "receipt_id": sales["CD_PARTNER"].astype(str) + "_" + sales["DT_SALE"].astype(str)
                + "_" + sales["NO_POS"].astype(str) + "_" + sales["SLIP_NO"].astype(str),
                "item_id": sales["CD_ITEM"].astype(str),
                "date": sales["date"],
                "qty": sales["QT_SALE"],
            },
            index=sales.index,
        )
        sales = sales[~flag_bulk_lines(lines).to_numpy()]

    daily = sales.groupby(["date", "CD_ITEM"])["QT_SALE"].sum().reset_index()
    daily = daily.rename(columns={"CD_ITEM": "item_id", "QT_SALE": "sold_units"})
    daily["item_id"] = daily["item_id"].astype(str)
    daily["store_id"] = store_id

    # 반품 net-out (SALES_FG=1) — sold_units = max(sold − 소매반품, 0). 단매장 파이프라인과 동일.
    if not returns.empty:
        daily = daily.merge(returns, on=["item_id", "date"], how="left")
        daily["sold_units"] = (daily["sold_units"] - daily["ret_qty"].fillna(0)).clip(lower=0)
        daily = daily.drop(columns=["ret_qty"])

    cat_map = item_category_map()
    daily["category_id"] = daily["item_id"].map(cat_map).fillna("etc")

    # === 재정의 stockout: 폐기0 & 완판 (단매장 assign_stockout_fields 공식 공유) ===
    # ② 생산/폐기 (bulk 무관 물리수량) → 재정의 입력 컬럼명으로
    inv = pd.read_parquet(V2 / "inventory.parquet")
    inv = inv[inv["CD_PARTNER"].astype(str) == store_cd].copy()
    inv["date"] = pd.to_datetime(inv["DT_SALE"].astype(str))
    inv["item_id"] = inv["CD_ITEM"].astype(str)
    inv = inv.rename(columns={"QT_MADE": "production_qty", "QT_OUT": "waste_qty"})
    inv = inv[["date", "item_id", "production_qty", "waste_qty"]]

    # ③ 마지막 실판매 시각 — bulk 제외본 sales에서 (sold_units와 동일 필터)
    ls = sales.copy()
    ls["date"] = pd.to_datetime(ls["DT_SALE"].astype(str))
    ls["item_id"] = ls["CD_ITEM"].astype(str)
    ls["last_sale_ts"] = pd.to_datetime(ls["SALES_TIME"].astype(str),
                                        format="%Y%m%d%H%M%S", errors="coerce")
    ls = ls.groupby(["date", "item_id"], as_index=False)["last_sale_ts"].max()

    # ④ merge → 재정의 공식
    daily = daily.merge(inv, on=["date", "item_id"], how="left")
    daily = daily.merge(ls, on=["date", "item_id"], how="left")
    daily = assign_stockout_fields(daily)
    daily = daily.drop(columns=["production_qty", "waste_qty", "last_sale_ts"])
    return daily


def build_store_closing_rows(store_cd: str) -> pd.DataFrame:
    """build_closing_rows 의 store_cd 매개화 버전."""
    sales = pd.read_parquet(V2 / "sales.parquet")
    sales = sales[sales["CD_PARTNER"].astype(str) == store_cd]
    sales["CD_USERDEF1"] = sales["CD_USERDEF1"].astype(str)
    # 마감할인 반품(SALES_FG=1 + CLOSING_CODES) net-out용 — sold_units/단매장과 기준 통일.
    _ret = sales[(sales["SALES_FG"].astype(str) == "1")
                 & (sales["CD_USERDEF1"].isin(CLOSING_CODES))]
    closing_returns = _aggregate_returns(pd.DataFrame({
        "item_id": _ret["CD_ITEM"].astype(str),
        "date": pd.to_datetime(_ret["DT_SALE"].astype(str)),
        "qty": pd.to_numeric(_ret["QT_SALE"], errors="coerce").fillna(0),
    }))
    sales = sales[sales["SALES_FG"].astype(str) == "0"]
    sales = sales[sales["CD_USERDEF1"].isin(CLOSING_CODES)]
    sales["date"] = pd.to_datetime(sales["DT_SALE"].astype(str))
    sales["QT_SALE"] = pd.to_numeric(sales["QT_SALE"], errors="coerce").fillna(0)
    sales["AM_PAYMENT"] = pd.to_numeric(sales["AM_PAYMENT"], errors="coerce").fillna(0)
    sales["AM_DC"] = pd.to_numeric(sales["AM_DC"], errors="coerce").fillna(0)
    out = (
        sales.groupby(["date", "CD_ITEM"]).agg(
            qty=("QT_SALE", "sum"),
            closing_revenue=("AM_PAYMENT", "sum"),
            discount_amt=("AM_DC", "sum"),
        ).reset_index().rename(columns={"CD_ITEM": "item_id"})
    )
    out["item_id"] = out["item_id"].astype(str)
    # 마감 반품 net-out — qty 기준 통일 (revenue/discount_amt는 cost 분석용이라 gross 유지).
    if not closing_returns.empty:
        out = out.merge(closing_returns, on=["date", "item_id"], how="left")
        out["qty"] = (out["qty"] - out["ret_qty"].fillna(0)).clip(lower=0)
        out = out.drop(columns=["ret_qty"])
    return out

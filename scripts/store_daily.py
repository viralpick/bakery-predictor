"""매장별 item-level daily / closing / 카테고리매핑 빌더 (공유 모듈).

v4_new_data_backtest 의 광교 canonical 경로를 store_cd 로 매개화한 것.
plotting(interval_backtest_4stores) / 분석(substitution_4stores) 양쪽에서 재사용.
matplotlib 등 무거운 의존 없음 — 순수 데이터 빌더만.
"""
from __future__ import annotations

import pandas as pd

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

    cat_map = item_category_map()
    daily["category_id"] = daily["item_id"].map(cat_map).fillna("etc")

    so = pd.read_parquet(V2 / "stockout.parquet")
    so = so[so["CD_PARTNER"].astype(str) == store_cd].copy()
    so["date"] = pd.to_datetime(so["DT_SALE"].astype(str))
    so["item_id"] = so["CD_ITEM"].astype(str)
    so["SOLD_TIME"] = pd.to_numeric(so["SOLD_TIME"], errors="coerce")
    so["hh"] = so["SOLD_TIME"].astype("Int64") // 100
    so["mm"] = so["SOLD_TIME"].astype("Int64") % 100
    so["stockout_time"] = pd.to_datetime(
        so["date"].astype(str) + " " + so["hh"].astype(str) + ":" + so["mm"].astype(str),
        errors="coerce",
    )
    so_first = (
        so.sort_values("stockout_time")
        .groupby(["date", "item_id"])["stockout_time"].first().reset_index()
    )
    daily = daily.merge(so_first, on=["date", "item_id"], how="left")
    daily["is_stockout"] = daily["stockout_time"].notna()
    return daily


def build_store_closing_rows(store_cd: str) -> pd.DataFrame:
    """build_closing_rows 의 store_cd 매개화 버전."""
    sales = pd.read_parquet(V2 / "sales.parquet")
    sales = sales[sales["CD_PARTNER"].astype(str) == store_cd]
    sales = sales[sales["SALES_FG"].astype(str) == "0"]
    sales["CD_USERDEF1"] = sales["CD_USERDEF1"].astype(str)
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
    return out

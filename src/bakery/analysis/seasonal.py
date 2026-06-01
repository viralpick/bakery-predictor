"""Seasonal/premium item exclusion.

광교 데이터에 마감 세일 정책이 다른 16개 품목 존재:
  - 시즌 한정 (밤 시리즈 가을~겨울, 크리스마스 위싱 리스/파네토네)
  - 프리미엄 라인 (AOP, 밀레앙 — 풀가격만)
  - 단종/장기 보관 N 품목 (당일 폐기 안 함)

이들은 마감 세일 0건이거나 풀가격 정책 → discount/popularity/waste 분석에서
**일반 마감 대상 품목과 섞이면 노이즈**. 별도로 분리해서 분석.

제외 매출 비중은 0.8% (4,062 / 510,585 unit) → 일반 분석 결과 영향 최소.

When 다매장 데이터 도착 시: store별로 시즌 품목 다를 수 있어 store_id 기반
필터링 확장 필요.
"""

from __future__ import annotations

import pandas as pd


# 광교 (store_gw01) 풀가격 정책 / 시즌 한정 16개
# 각 항목: (item_id, name, reason)
EXCLUDED_ITEMS_GWANGYO: dict[str, tuple[str, str]] = {
    # 밤 시즌 라인업 (가을-겨울 한정, 마감 세일 0)
    "151100003265": ("AOP 버터 맘모스 빵",       "premium_no_closing"),
    "151100003266": ("우리 밤 식빵",             "seasonal_chestnut"),
    "151100003342": ("밀레앙 퀸아망",            "premium_milleens"),
    "151100002542": ("보늬밤 식빵_half",         "seasonal_chestnut"),
    "151100003343": ("밀레앙 팽 페르뒤",         "premium_milleens"),
    "151100002541": ("보늬밤 식빵",              "seasonal_chestnut"),
    "151100002581": ("(I)밤 크림브레드",         "seasonal_chestnut"),
    "151100002543": ("보늬 밤 크림브레드",       "seasonal_chestnut"),
    # 크리스마스 시즌 (장기 보관 N + 시즌)
    "151100002210": ("위싱 리스 바브카 (홀)",    "seasonal_christmas"),
    "151100002226": ("위싱 리스 바브카 조각",    "seasonal_christmas"),
    "151100002209": ("스노우 파네토네",          "seasonal_christmas"),
    # 단종 / N 머핀류
    "151100002163": ("바닐라 빈 크림 몽블랑",    "discard_N_general"),
    "152100001437": ("C발로나초코크럼블머핀",    "discard_N_PB_muffin"),
    "152100001436": ("C블루베리크럼블머핀",      "discard_N_PB_muffin"),
    "152100001434": ("C우리밤찰빵",              "discard_N_PB"),
    # 거의 안 팔린 한정판
    "151100002864": ("데니쉬 큐브 식빵",         "negligible_no_closing"),
}


def is_excluded(item_id: str, store_id: str = "store_gw01") -> bool:
    """Return True if item is in store's seasonal/premium exclusion set."""
    if store_id == "store_gw01":
        return str(item_id) in EXCLUDED_ITEMS_GWANGYO
    return False


def filter_seasonal(
    df: pd.DataFrame,
    *,
    store_id: str = "store_gw01",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Drop excluded rows from any df with item_id."""
    if store_id != "store_gw01":
        return df
    mask = ~df[item_col].astype(str).isin(EXCLUDED_ITEMS_GWANGYO)
    return df[mask]


def excluded_summary() -> pd.DataFrame:
    """Inspection helper — returns the exclusion table as a DataFrame."""
    rows = [
        {"item_id": iid, "name": name, "reason": reason}
        for iid, (name, reason) in EXCLUDED_ITEMS_GWANGYO.items()
    ]
    return pd.DataFrame(rows)

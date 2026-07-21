"""매진→매출 영향 검증 재실행 — stockout 재정의(PR#28) 반영.

옛 build_stockout_daily 는 하루 '첫 품절이벤트'(SOLD_TIME min)로 매진 시각을 잡아
early_share 를 부풀렸다(로더 버그와 동일). 이를 교정 정의로 교체해 L1~L4 재실행:
  - 매진 품목 = 폐기0 (QT_MADE>0 & QT_OUT<=0)  [inventory]
  - 매진 시각 = 그 item-day 마지막 실판매(SALES_TIME 최대)  [_last_sale_ts]
나머지 레이어/판정은 원 스크립트 재사용(monkeypatch).

실행: PYTHONPATH=scripts uv run python scripts/verify_stockout_revenue_fixed.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import verify_stockout_revenue_4stores as V
from absorption_4stores import _last_sale_ts

V2 = Path("data/internal/v2")
STORE_MAP = {
    "1000000047": "광교",
    "1000000009": "삼성타운",
    "1000000029": "메세나폴리스",
    "1000000485": "광화문",
}


def build_stockout_daily_fixed() -> pd.DataFrame:
    inv = pd.read_parquet(V2 / "inventory.parquet")
    inv["store"] = inv["CD_PARTNER"].astype(str).map(STORE_MAP)
    inv["date"] = pd.to_datetime(inv["DT_SALE"].astype(str))
    inv["item_id"] = inv["CD_ITEM"].astype(str)
    made = pd.to_numeric(inv["QT_MADE"], errors="coerce")
    waste = pd.to_numeric(inv["QT_OUT"], errors="coerce")
    inv["is_so"] = ((made > 0) & (waste <= 0)).fillna(False)
    so = inv[inv["is_so"] & inv["store"].notna()][["store", "date", "item_id"]]

    frames = []
    for cd, name in STORE_MAP.items():
        lt = _last_sale_ts(cd, exclude_bulk=True)
        lt["store"] = name
        frames.append(lt)
    lts = pd.concat(frames, ignore_index=True)

    m = so.merge(lts, on=["store", "date", "item_id"], how="left").dropna(subset=["last_sale_ts"])
    ts = pd.to_datetime(m["last_sale_ts"])
    m["smin"] = ts.dt.hour * 60 + ts.dt.minute
    m["is_early"] = m["smin"] < V.EARLY_HOUR * 60
    daily = m.groupby(["store", "date"]).agg(
        n_stockout=("item_id", "nunique"),
        n_early_stockout=("is_early", "sum"),
        median_stockout_min=("smin", "median"),
    ).reset_index()
    daily["early_share"] = daily["n_early_stockout"] / daily["n_stockout"]
    return daily


if __name__ == "__main__":
    V.build_stockout_daily = build_stockout_daily_fixed  # monkeypatch
    print(">>> 매진 재정의(폐기0 & 마지막 실판매) 기반 재실행 (EARLY_HOUR=15:00)\n")
    V.main()

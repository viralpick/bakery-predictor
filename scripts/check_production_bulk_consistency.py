"""Step 0 정합성 게이트 — 생산 bulk 제외 배선 전 잔차 진단.

헌장 §1: 생산 = 판매 + 폐기. bulk를 생산에서 빼면 이 항등식이 raw에서 성립해야 한다.
잔차 = (QT_MADE − bulk_qty) − (sold_units + waste_qty) 를 item-day별로 본다.

움직이는 부분 3개(advisor):
  (a) bulk 소스 오차 — receipts(hour dropna population) vs 정확값(load_sales raw before/after)
  (b) 반품 net-out — sold_units엔 반품 차감됨, QT_MADE엔 없음
  (c) raw slack — 기록오류

이 스크립트는 소스·위치를 결정하기 위한 탐색용. 결과 요약만 출력.
실행: uv run python scripts/check_production_bulk_consistency.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.data.bonavi_loader import load_sales
from bakery.data.bulk import flag_bulk_lines
from bakery.features.category_aggregate import TARGET_CATEGORIES
from bakery.ingest.inventory import load_inventory

STORE_ID = "store_gw01"
STORE_CODE = "1000000047"
SALES_XLSX = "data/internal/보나비 데이터_20260520.xlsx"
INV_XLSX = "data/internal/보나비 데이터_20260526.xlsx"
DAILY_PARQUET = "data/internal/bonavi_daily.parquet"
RECEIPTS_PARQUET = "data/internal/bonavi_receipts.parquet"


def _bulk_qty_from_receipts() -> pd.DataFrame:
    """receipts(is_bulk) → item-day별 bulk 판매량 (부정확 소스: hour dropna population)."""
    rec = pd.read_parquet(RECEIPTS_PARQUET)
    rec = rec[rec["is_bulk"].astype(bool)]
    rec["item_id"] = rec["item_id"].astype(str)
    rec["date"] = pd.to_datetime(rec["date"]).dt.normalize()
    g = rec.groupby(["item_id", "date"])["qty"].sum().reset_index()
    return g.rename(columns={"qty": "bulk_qty_receipts"})


def _bulk_qty_exact() -> pd.DataFrame:
    """load_sales 필터 직전 raw에서 정확한 bulk 판매량 재구성 (정확 소스)."""
    raw = pd.read_excel(SALES_XLSX, "판매정보")
    set_col = next(c for c in raw.columns if c.startswith("셋트상품구분"))
    sale_col = next(c for c in raw.columns if c.startswith("판매구분"))
    pos_col = next(c for c in raw.columns if c.startswith("POS번호"))
    receipt_col = next(c for c in raw.columns if c.startswith("영수증번호"))
    raw = raw[raw["점포코드"].astype(str) == STORE_CODE]
    raw = raw[raw[set_col] == "SS"]
    raw = raw[raw[sale_col].astype(str) == "0"].copy()
    raw["date"] = pd.to_datetime(raw["판매일자"].astype(str), format="%Y%m%d").dt.normalize()
    lines = pd.DataFrame(
        {
            "receipt_id": raw["판매일자"].astype(str) + "_"
            + raw[pos_col].astype(str) + "_" + raw[receipt_col].astype(str),
            "item_id": raw["품목코드"].astype(str),
            "date": raw["date"],
            "qty": pd.to_numeric(raw["판매수량"], errors="coerce").fillna(0),
        },
        index=raw.index,
    )
    is_bulk = flag_bulk_lines(lines).to_numpy()
    b = lines[is_bulk]
    g = b.groupby(["item_id", "date"])["qty"].sum().reset_index()
    return g.rename(columns={"qty": "bulk_qty_exact"})


def _describe(name: str, resid: pd.Series) -> None:
    q = resid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"\n[{name}] n={len(resid)}")
    print(f"  mean={resid.mean():+.2f}  median={resid.median():+.1f}  std={resid.std():.2f}")
    print(f"  |resid|=0 비율={float((resid == 0).mean()):.3f}  "
          f"|resid|<=2 비율={float((resid.abs() <= 2).mean()):.3f}")
    print(f"  min={resid.min():.0f}  max={resid.max():.0f}")
    print(f"  분위 1/5/25/50/75/95/99: "
          f"{q.iloc[0]:.0f}/{q.iloc[1]:.0f}/{q.iloc[2]:.0f}/{q.iloc[3]:.0f}/"
          f"{q.iloc[4]:.0f}/{q.iloc[5]:.0f}/{q.iloc[6]:.0f}")


def main() -> None:
    # sold_units (bulk 제외 + 반품 net-out) — bonavi_daily
    daily = pd.read_parquet(DAILY_PARQUET)
    daily = daily[daily["store_id"] == STORE_ID]
    daily = daily[daily["category_id"].isin(TARGET_CATEGORIES)]
    daily["item_id"] = daily["item_id"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    base = daily[["item_id", "date", "sold_units", "category_id"]].copy()

    # inventory (QT_MADE, QT_OUT)
    inv = load_inventory(INV_XLSX, STORE_ID)
    inv["item_id"] = inv["item_id"].astype(str)
    inv["date"] = pd.to_datetime(inv["date"], format="%Y%m%d").dt.normalize()
    inv = inv[["item_id", "date", "production_qty", "waste_qty"]]

    # inner join = 평가셋과 동일 population (_assemble_real_rows도 inner)
    m = base.merge(inv, on=["item_id", "date"], how="inner")
    print(f"공통 population(daily ∩ inventory): {len(m)} item-days "
          f"({m['item_id'].nunique()} items, {m['date'].nunique()} days)")

    m = m.merge(_bulk_qty_receipts_wrap(), on=["item_id", "date"], how="left")
    m = m.merge(_bulk_qty_exact(), on=["item_id", "date"], how="left")
    m["bulk_qty_receipts"] = m["bulk_qty_receipts"].fillna(0.0)
    m["bulk_qty_exact"] = m["bulk_qty_exact"].fillna(0.0)

    made = m["production_qty"].astype(float)
    sold = m["sold_units"].astype(float)
    waste = m["waste_qty"].clip(lower=0).astype(float)  # 음수 폐기는 clip(로더 정책과 일치)

    # 잔차 3종
    _describe("bulk 미제외: QT_MADE − (sold+waste)", made - (sold + waste))
    _describe("bulk 제외(receipts): (QT_MADE−bulk_r) − (sold+waste)",
              (made - m["bulk_qty_receipts"]) - (sold + waste))
    _describe("bulk 제외(exact): (QT_MADE−bulk_e) − (sold+waste)",
              (made - m["bulk_qty_exact"]) - (sold + waste))

    # bulk 소스 비교
    n_bulk_r = int((m["bulk_qty_receipts"] > 0).sum())
    n_bulk_e = int((m["bulk_qty_exact"] > 0).sum())
    diff = (m["bulk_qty_receipts"] - m["bulk_qty_exact"])
    print(f"\n[bulk 소스 비교] receipts>0={n_bulk_r} 일, exact>0={n_bulk_e} 일")
    print(f"  bulk_r 총합={m['bulk_qty_receipts'].sum():.0f}  bulk_e 총합={m['bulk_qty_exact'].sum():.0f}")
    print(f"  (receipts−exact) 불일치 item-day={int((diff != 0).sum())}, "
          f"mean diff={diff.mean():+.2f}")

    # 음수 잔차(생산<판매+폐기) = raw slack 방향 확인
    r_excl = (made - m["bulk_qty_exact"]) - (sold + waste)
    print(f"\n[방향성] bulk제외(exact) 잔차: 양수(생산>판매+폐기)={int((r_excl>0).sum())}, "
          f"0={int((r_excl==0).sum())}, 음수={int((r_excl<0).sum())}")


def _bulk_qty_receipts_wrap() -> pd.DataFrame:
    return _bulk_qty_from_receipts()


if __name__ == "__main__":
    main()

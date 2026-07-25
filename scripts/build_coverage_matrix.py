"""크로스체크 커버리지 매트릭스 리포트 생성 (Task 10).

기존에 검증된 parquet 테이블(sales_lines_clean + v2 interim)만 읽는다.
451MB/138MB 원본 Excel은 열지 않는다 — sales_lines_clean.parquet이 이미
시트2 스왑 정정 + 앵커 검증(반품 1.88%·광교 same-item 510,585) 완료된
프로파일링 산출물이기 때문(재프로파일 불필요).

산출: reports/data_coverage/coverage.html, reports/data_coverage/cells.parquet
탐지만 한다 — 갭/충돌을 교정하지 않는다.
"""
from __future__ import annotations

import pandas as pd

from bakery.data import coverage, paths

# CD_PARTNER -> 매장명 (scripts/store_daily.py STORE_MAP과 동일 관례)
STORE_MAP: dict[str, str] = {
    "1000000047": "광교",
    "1000000009": "삼성타운",
    "1000000029": "메세나폴리스",
    "1000000485": "광화문",
}

# display_time_xls는 37KB(451MB/138MB 원본과 무관)라 직접 read_excel로 확인:
# 37개 품목 x 계획 진열시간, 날짜축 없음(광교 단독). 재확인: pd.read_excel(..., engine="xlrd").shape == (37, 7)
DISPLAY_XLS_ITEM_COUNT = 37

KNOWN_CONFLICTS: list[str] = [
    "closing 소스 불일치: build_category_daily=옛0520 vs build_item_adjusted_demand=clean parquet",
    "2026 H1 라벨 갭: 생산/폐기/매진 라벨은 2021-2025만, 2026 상반기는 sales-only",
    "품목 미매칭 67.9%: 광교 판매품목 다수가 마스터에 없음(음료/비베이커리)",
    "카테고리 정의 불일치: 예측력 3-cat(bread/pastry/sandwich) vs canonical 5-cat",
]


def _to_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d").dt.strftime("%Y-%m")


def _sales_field_cells(sales: pd.DataFrame, field: str, mask: pd.Series) -> pd.DataFrame:
    """0721_sales 소스에서 store x month별 present/rows 집계 (field=sold/closing 공용)."""
    subset = sales.loc[mask, ["store", "month"]]
    counts = subset.groupby(["store", "month"], observed=True).size()
    full_index = pd.MultiIndex.from_product(
        [sales["store"].unique(), sorted(sales["month"].unique())],
        names=["store", "month"],
    )
    counts = counts.reindex(full_index, fill_value=0)
    out = counts.reset_index(name="rows")
    out["source"] = "0721_sales"
    out["field"] = field
    out["present"] = out["rows"] > 0
    return out[["source", "store", "field", "month", "present", "rows"]]


def _inventory_field_cells(
    inventory: pd.DataFrame, all_months: list[str], source: str, field: str, qty_col: str
) -> pd.DataFrame:
    """v2 inventory(생산/폐기)에서 store x month별 present/rows 집계.

    present = 해당 store-month에 qty_col > 0 인 행이 하나라도 있는지.
    2021-2025만 존재 → 2026 달은 자동으로 결측(present=False)으로 표면화된다.
    """
    subset = inventory[inventory[qty_col] > 0]
    counts = subset.groupby(["store", "month"], observed=True).size()
    full_index = pd.MultiIndex.from_product(
        [sorted(STORE_MAP.values()), all_months], names=["store", "month"]
    )
    counts = counts.reindex(full_index, fill_value=0)
    out = counts.reset_index(name="rows")
    out["source"] = source
    out["field"] = field
    out["present"] = out["rows"] > 0
    return out[["source", "store", "field", "month", "present", "rows"]]


def _stockout_cells(stockout: pd.DataFrame, all_months: list[str]) -> pd.DataFrame:
    counts = stockout.groupby(["store", "month"], observed=True).size()
    full_index = pd.MultiIndex.from_product(
        [sorted(STORE_MAP.values()), all_months], names=["store", "month"]
    )
    counts = counts.reindex(full_index, fill_value=0)
    out = counts.reset_index(name="rows")
    out["source"] = "0526_inventory"
    out["field"] = "stockout"
    out["present"] = out["rows"] > 0
    return out[["source", "store", "field", "month", "present", "rows"]]


def _master_match_cells(sales: pd.DataFrame, item_ids: set[str]) -> pd.DataFrame:
    """0526_master(items.parquet) 대비 판매품목 매칭 커버리지 — store x month."""
    matched = sales.assign(matched=sales["CD_ITEM"].isin(item_ids))
    counts = matched[matched["matched"]].groupby(["store", "month"], observed=True).size()
    full_index = pd.MultiIndex.from_product(
        [sales["store"].unique(), sorted(sales["month"].unique())],
        names=["store", "month"],
    )
    counts = counts.reindex(full_index, fill_value=0)
    out = counts.reset_index(name="rows")
    out["source"] = "0526_master"
    out["field"] = "sold"
    out["present"] = out["rows"] > 0
    return out[["source", "store", "field", "month", "present", "rows"]]


def _display_cell(item_count: int) -> pd.DataFrame:
    """display_xls: 품목-레벨 계획값(날짜축 없음, 37개 광교 한정) — 단일 셀로 표면화."""
    return pd.DataFrame({
        "source": ["display_xls"],
        "store": ["광교"],
        "field": ["display"],
        "month": ["static"],
        "present": [item_count > 0],
        "rows": [item_count],
    })


def build_cells() -> pd.DataFrame:
    sales = pd.read_parquet(
        paths.dataset("sales_lines_clean"),
        columns=["CD_PARTNER", "DT_SALE", "CD_ITEM", "CD_USERDEF1"],
    )
    sales = sales[sales["CD_PARTNER"].isin(STORE_MAP)].copy()
    sales["store"] = sales["CD_PARTNER"].map(STORE_MAP)
    sales["month"] = _to_month(sales["DT_SALE"])

    inventory = pd.read_parquet(paths.INTERIM_DIR / "v2" / "inventory.parquet")
    inventory = inventory[inventory["CD_PARTNER"].isin(STORE_MAP)].copy()
    inventory["store"] = inventory["CD_PARTNER"].map(STORE_MAP)
    inventory["month"] = _to_month(inventory["DT_SALE"])

    stockout = pd.read_parquet(paths.INTERIM_DIR / "v2" / "stockout.parquet")
    stockout = stockout[stockout["CD_PARTNER"].isin(STORE_MAP)].copy()
    stockout["store"] = stockout["CD_PARTNER"].map(STORE_MAP)
    stockout["month"] = _to_month(stockout["DT_SALE"])

    items = pd.read_parquet(paths.INTERIM_DIR / "v2" / "items.parquet", columns=["CD_ITEM"])
    item_ids = set(items["CD_ITEM"])

    # sales의 전체 월 축(2021-01~2026-06)을 모든 필드가 공유 — inventory/stockout은
    # 2025-12까지만 존재하므로 2026 상반기가 "missing" 셀로 그대로 표면화된다.
    all_months = sorted(sales["month"].unique())

    cells = pd.concat(
        [
            _sales_field_cells(sales, "sold", sales["CD_ITEM"].notna()),
            _sales_field_cells(sales, "closing", sales["CD_USERDEF1"].notna()),
            _inventory_field_cells(inventory, all_months, "0526_inventory", "production", "QT_MADE"),
            _inventory_field_cells(inventory, all_months, "0526_inventory", "waste", "QT_OUT"),
            _stockout_cells(stockout, all_months),
            _master_match_cells(sales, item_ids),
            _display_cell(DISPLAY_XLS_ITEM_COUNT),
        ],
        ignore_index=True,
    )
    return cells


def main() -> None:
    cells = build_cells()

    out_dir = paths.PROJECT_ROOT / "reports" / "data_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out_dir / "cells.parquet", index=False)

    html = coverage.render_coverage_matrix(cells, conflicts=KNOWN_CONFLICTS)
    (out_dir / "coverage.html").write_text(html, encoding="utf-8")

    print(f"cells: {len(cells)} rows")
    print(cells.groupby(["source", "field"], observed=True)["present"].agg(["sum", "count"]))
    missing_count = int((~cells["present"]).sum())
    print(f"missing cells: {missing_count} / {len(cells)}")
    for c in KNOWN_CONFLICTS:
        assert c in html, f"conflict not surfaced in html: {c}"
    print(f"conflicts surfaced: {len(KNOWN_CONFLICTS)}")
    print(f"written: {out_dir / 'coverage.html'}")


if __name__ == "__main__":
    main()

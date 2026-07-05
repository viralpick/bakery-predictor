"""Load inventory data (생산량, 폐기량) from 재고정보 sheet.

Maps 날짜, 점포코드, 품목코드, 생산량, 폐기량 to
date, item_id, production_qty, waste_qty and filters to a single store.
"""

from pathlib import Path

import pandas as pd


# Mapping from store_id (e.g., "store_gw01") to 점포코드 (e.g., "1000000047")
STORE_CODE_MAPPING = {
    "store_gw01": "1000000047",  # 광교 (아티제 아브뉴프랑광교점)
    "store_ss01": "1000000009",  # 삼성타운
    "store_gh01": "1000000485",  # 광화문
    "store_mp01": "1000000029",  # 메세나폴리스
}


def _normalize_inventory(raw: pd.DataFrame, store_code: str) -> pd.DataFrame:
    """Normalize and filter inventory DataFrame to a single store.

    Args:
        raw: DataFrame with columns 날짜, 점포코드, 품목코드, 생산량, 폐기량
        store_code: 점포코드 (e.g., "1000000047" for gwangyo)

    Returns:
        DataFrame with columns: date, item_id, production_qty, waste_qty
    """
    # Filter to store
    df = raw[raw["점포코드"] == store_code].copy()

    # Map columns
    df = df[["날짜", "품목코드", "생산량", "폐기량"]]
    df.columns = ["date", "item_id", "production_qty", "waste_qty"]

    # Convert types
    df["item_id"] = df["item_id"].astype(str)
    df["production_qty"] = pd.to_numeric(df["production_qty"], errors="coerce").fillna(0).astype("int64")
    df["waste_qty"] = pd.to_numeric(df["waste_qty"], errors="coerce").fillna(0).astype("int64")

    return df.reset_index(drop=True)


def load_inventory(xlsx_path: str, store_id: str) -> pd.DataFrame:
    """Load inventory data from 재고정보 sheet, filtered to one store.

    Args:
        xlsx_path: Path to bonavi data Excel file
        store_id: Store identifier (e.g., "store_gw01")

    Returns:
        DataFrame with columns: date, item_id, production_qty, waste_qty

    Raises:
        ValueError: If store_id is not recognized
        FileNotFoundError: If xlsx_path does not exist
    """
    if store_id not in STORE_CODE_MAPPING:
        raise ValueError(f"Unknown store_id: {store_id}. Must be one of {list(STORE_CODE_MAPPING.keys())}")

    if not Path(xlsx_path).exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    store_code = STORE_CODE_MAPPING[store_id]
    raw = pd.read_excel(xlsx_path, sheet_name="재고정보")

    return _normalize_inventory(raw, store_code=store_code)

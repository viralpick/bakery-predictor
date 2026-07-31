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

    # ★파일별 컬럼명이 다르다 — master는 "생산량", 추가 데이터(20260721)는 "총생산량"이고
    #   후자에는 제시량(AX 산출)·추가량이 더 있다. 둘 다 받는다.
    made_col = "총생산량" if "총생산량" in df.columns else "생산량"
    if made_col not in df.columns:
        raise ValueError(
            f"재고정보에 생산량 컬럼이 없다(기대: 생산량 또는 총생산량). 실제: {list(df.columns)}"
        )
    keep = {"날짜": "date", "품목코드": "item_id", made_col: "production_qty", "폐기량": "waste_qty"}
    # 신규 컬럼은 있을 때만 싣는다(없는 파일에서 조용히 0을 만들지 않는다).
    for src, dst in (("제시량", "suggested_qty"), ("추가량", "added_qty")):
        if src in df.columns:
            keep[src] = dst
    df = df[list(keep)].rename(columns=keep)

    # Convert types
    df["item_id"] = df["item_id"].astype(str)
    for col in [c for c in df.columns if c.endswith("_qty")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    return df.reset_index(drop=True)


def handle_negative_waste(
    inv: pd.DataFrame, *, policy: str = "clip"
) -> tuple[pd.DataFrame, dict]:
    """재고정보 폐기량 음수(반품/보정 추정) 처리 + 리포트.

    광교 실데이터에 음수 ~3.3%(min −31) 관측. actual-waste sanity(Task 2) 전에
    반드시 통과시킨다. 현재 policy는 clip-at-0만 지원(음수를 반품으로 보고 폐기 0 처리).
    """
    if policy != "clip":
        raise ValueError(f"unsupported policy: {policy!r} (only 'clip')")
    w = pd.to_numeric(inv["waste_qty"], errors="coerce")
    report = {
        "policy": policy,
        "n_negative": int((w < 0).sum()),
        "n_total": int(len(w)),
        "min_value": float(w.min()) if len(w) else 0.0,
    }
    out = inv.copy()
    out["waste_qty"] = w.clip(lower=0)
    return out, report


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

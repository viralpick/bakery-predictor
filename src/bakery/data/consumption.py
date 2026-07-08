"""Admin-dong quarterly consumption (서울 상권분석서비스 소비-행정동).

OA-22166 publishes estimated quarterly spend per admin dong, broken down
by category (음식 / 소매 / 의료 / 교육 / ...). For PoC we keep the two
columns relevant to bakery demand: total spend, and food/retail spend
combined.

Real loader filter must produce columns matching `CONSUMPTION_COLUMNS`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..ingest.store_mapping import StationMapping

CONSUMPTION_COLUMNS: dict[str, str] = {
    "admin_dong_code": "string",
    "quarter": "string",          # e.g. "2024Q1"
    "total_spend": "float64",     # KRW
    "food_retail_spend": "float64",
}

_SYNTH_QUARTERLY: dict[str, dict[str, float]] = {
    "11680565": {"total": 1.6e10, "food_retail": 4.2e9},
    "11440660": {"total": 3.1e10, "food_retail": 1.4e10},
    "11560540": {"total": 2.4e10, "food_retail": 7.5e9},
}
_DEFAULT_TOTAL = 1.5e10
_DEFAULT_FOOD_RETAIL = 5e9


def build_synthetic_consumption(
    *,
    mapping: dict[str, StationMapping],
    quarter_start: str = "2024Q1",
    quarter_end: str = "2025Q4",
    quarterly_growth: float = 0.012,
) -> pd.DataFrame:
    dong_codes = sorted({s["admin_dong_code"] for s in mapping.values()})
    quarters = pd.period_range(quarter_start, quarter_end, freq="Q")
    rows: list[dict] = []
    for dong in dong_codes:
        baseline = _SYNTH_QUARTERLY.get(dong, {"total": _DEFAULT_TOTAL, "food_retail": _DEFAULT_FOOD_RETAIL})
        for i, q in enumerate(quarters):
            growth = (1 + quarterly_growth) ** i
            rows.append(
                {
                    "admin_dong_code": dong,
                    "quarter": str(q),
                    "total_spend": float(baseline["total"] * growth),
                    "food_retail_spend": float(baseline["food_retail"] * growth),
                }
            )
    return _coerce_dtypes(pd.DataFrame(rows))


def load_consumption_from_local(
    parquet_path: Path | str, *, admin_dong_codes: list[str]
) -> pd.DataFrame:
    raw = pd.read_parquet(parquet_path)
    raw["admin_dong_code"] = raw["admin_dong_code"].astype("string")
    return _coerce_dtypes(raw[raw["admin_dong_code"].isin(admin_dong_codes)].copy())


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["admin_dong_code"] = df["admin_dong_code"].astype("string")
    df["quarter"] = df["quarter"].astype("string")
    df["total_spend"] = df["total_spend"].astype("float64")
    df["food_retail_spend"] = df["food_retail_spend"].astype("float64")
    return df[list(CONSUMPTION_COLUMNS.keys())].reset_index(drop=True)


def validate_consumption(df: pd.DataFrame) -> None:
    missing = set(CONSUMPTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"consumption frame missing columns: {sorted(missing)}")
    if (df["total_spend"] < 0).any() or (df["food_retail_spend"] < 0).any():
        raise ValueError("consumption frame has negative spend")
    if (df["food_retail_spend"] > df["total_spend"]).any():
        raise ValueError("consumption: food_retail_spend exceeds total_spend in some rows")

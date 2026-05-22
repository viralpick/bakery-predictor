"""Static store features from quarterly consumption — log-scaled.

Spec §2.6 names two variables; we expose both as log-scale to keep LightGBM
splits sensible when one dong's consumption is 5× another's:

  consumption_total_log         log of mean quarterly total spend
  consumption_food_retail_log   log of mean quarterly food+retail spend
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest.store_mapping import StationMapping

CONSUMPTION_FEATURE_COLUMNS: list[str] = [
    "consumption_total_log",
    "consumption_food_retail_log",
]

_DEFAULTS = {
    "consumption_total_log": float(np.log(1.5e10)),
    "consumption_food_retail_log": float(np.log(5e9)),
}


def compute_store_consumption_features(
    consumption_df: pd.DataFrame,
    mapping: dict[str, StationMapping],
) -> pd.DataFrame:
    rows: list[dict] = []
    for store_id, entry in mapping.items():
        sub = consumption_df[consumption_df["admin_dong_code"] == entry["admin_dong_code"]]
        if sub.empty:
            rows.append({"store_id": store_id, **_DEFAULTS})
            continue
        total = float(sub["total_spend"].mean())
        food_retail = float(sub["food_retail_spend"].mean())
        rows.append(
            {
                "store_id": store_id,
                "consumption_total_log": float(np.log(max(total, 1.0))),
                "consumption_food_retail_log": float(np.log(max(food_retail, 1.0))),
            }
        )
    return pd.DataFrame(rows).astype({"store_id": "string"})


def add_consumption_features(df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    missing = set(CONSUMPTION_FEATURE_COLUMNS) - set(static_features.columns)
    if missing:
        raise ValueError(
            f"static_features missing columns: {sorted(missing)}. "
            "Call compute_store_consumption_features() first."
        )
    if "store_id" not in df.columns:
        raise ValueError("df missing 'store_id' — required for per-store consumption merge")
    return df.merge(
        static_features[["store_id", *CONSUMPTION_FEATURE_COLUMNS]],
        on="store_id", how="left",
    )

"""Static store-level features derived from admin-dong living population.

Three baseline features per store, all forecast-safe (computed from past
windows and constant within a store):

  living_pop_daily_avg     daily mean over the trailing window (raw scale)
  living_pop_lunch_share   lunch-hour (11~14) mean / daily mean — office hub if high
  living_pop_weekend_ratio weekend mean / weekday mean — destination if >1, office if <1

`compute_store_living_features(living_pop_df, mapping)` joins via
`admin_dong_code`. If a store's dong is missing, falls back to Seoul-wide
defaults so the v3 model never sees NaN for these columns.
"""

from __future__ import annotations

import pandas as pd

from ..ingest.store_mapping import StationMapping

LIVING_POP_FEATURE_COLUMNS: list[str] = [
    "living_pop_daily_avg",
    "living_pop_lunch_share",
    "living_pop_weekend_ratio",
]

_DEFAULT_DAILY_AVG = 18_000.0
_DEFAULT_LUNCH_SHARE = 1.10
_DEFAULT_WEEKEND_RATIO = 1.00

LUNCH_HOURS = range(11, 14)


def compute_store_living_features(
    living_pop_df: pd.DataFrame,
    mapping: dict[str, StationMapping],
) -> pd.DataFrame:
    """One row per store_id with the three living-population baseline features."""
    rows: list[dict] = []
    for store_id, entry in mapping.items():
        dong = entry["admin_dong_code"]
        sub = living_pop_df[living_pop_df["admin_dong_code"] == dong]
        if sub.empty:
            rows.append(
                {
                    "store_id": store_id,
                    "living_pop_daily_avg": _DEFAULT_DAILY_AVG,
                    "living_pop_lunch_share": _DEFAULT_LUNCH_SHARE,
                    "living_pop_weekend_ratio": _DEFAULT_WEEKEND_RATIO,
                }
            )
            continue
        rows.append(_compute_one_store(store_id, sub))
    return pd.DataFrame(rows).astype({"store_id": "string"})


def _compute_one_store(store_id: str, sub: pd.DataFrame) -> dict:
    daily = sub.groupby("date", as_index=False)["total_pop"].mean()
    daily_avg = float(daily["total_pop"].mean()) if not daily.empty else _DEFAULT_DAILY_AVG
    lunch_rows = sub[sub["hour"].isin(LUNCH_HOURS)]
    lunch_avg = float(lunch_rows["total_pop"].mean()) if not lunch_rows.empty else daily_avg
    lunch_share = lunch_avg / daily_avg if daily_avg > 0 else _DEFAULT_LUNCH_SHARE
    sub_with_dow = sub.assign(_dow=pd.to_datetime(sub["date"]).dt.dayofweek)
    weekday_pop = sub_with_dow[sub_with_dow["_dow"] < 5]["total_pop"].mean()
    weekend_pop = sub_with_dow[sub_with_dow["_dow"] >= 5]["total_pop"].mean()
    if pd.isna(weekday_pop) or weekday_pop == 0:
        weekend_ratio = _DEFAULT_WEEKEND_RATIO
    else:
        weekend_ratio = float(weekend_pop / weekday_pop) if pd.notna(weekend_pop) else _DEFAULT_WEEKEND_RATIO
    return {
        "store_id": store_id,
        "living_pop_daily_avg": float(daily_avg),
        "living_pop_lunch_share": float(lunch_share),
        "living_pop_weekend_ratio": float(weekend_ratio),
    }


def add_living_pop_features(df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    missing = set(LIVING_POP_FEATURE_COLUMNS) - set(static_features.columns)
    if missing:
        raise ValueError(
            f"static_features missing columns: {sorted(missing)}. "
            "Call compute_store_living_features() first."
        )
    if "store_id" not in df.columns:
        raise ValueError("df missing 'store_id' — required for per-store living-pop merge")
    return df.merge(
        static_features[["store_id", *LIVING_POP_FEATURE_COLUMNS]],
        on="store_id", how="left",
    )

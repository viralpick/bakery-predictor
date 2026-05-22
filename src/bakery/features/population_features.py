"""Age/sex composition features per spec §2.5.

Four static per-store shares (snapshot averaged across whatever months are
in `population_df` to smooth fluctuation):

  pop_share_0_9         가족/영유아 비중 → cake·sweets 친화
  pop_share_20_39       청년층 비중 → sandwich·pastry 친화
  pop_share_30_49_female 가족 구매층 비중 → 정기 베이커리 수요 proxy
  pop_share_60_plus     고령층 비중 → bread 친화

All forecast-safe (static per store).
"""

from __future__ import annotations

import pandas as pd

from ..ingest.store_mapping import StationMapping

POPULATION_FEATURE_COLUMNS: list[str] = [
    "pop_share_0_9",
    "pop_share_20_39",
    "pop_share_30_49_female",
    "pop_share_60_plus",
]

_DEFAULTS = {
    "pop_share_0_9": 0.07,
    "pop_share_20_39": 0.30,
    "pop_share_30_49_female": 0.18,
    "pop_share_60_plus": 0.20,
}


def compute_store_population_features(
    population_df: pd.DataFrame,
    mapping: dict[str, StationMapping],
) -> pd.DataFrame:
    """One row per store_id with the four age/sex composition shares."""
    rows: list[dict] = []
    for store_id, entry in mapping.items():
        dong = entry["admin_dong_code"]
        sub = population_df[population_df["admin_dong_code"] == dong]
        if sub.empty:
            rows.append({"store_id": store_id, **_DEFAULTS})
            continue
        avg = sub.groupby(["age_bin", "sex"], as_index=False)["population"].mean()
        total = float(avg["population"].sum())
        if total <= 0:
            rows.append({"store_id": store_id, **_DEFAULTS})
            continue
        rows.append(
            {
                "store_id": store_id,
                "pop_share_0_9": _share(avg, total, age_bins=["0_9"]),
                "pop_share_20_39": _share(avg, total, age_bins=["20_29", "30_39"]),
                "pop_share_30_49_female": _share(
                    avg, total, age_bins=["30_39", "40_49"], sex="F"
                ),
                "pop_share_60_plus": _share(avg, total, age_bins=["60_plus"]),
            }
        )
    return pd.DataFrame(rows).astype({"store_id": "string"})


def _share(avg: pd.DataFrame, total: float, *, age_bins: list[str], sex: str | None = None) -> float:
    mask = avg["age_bin"].isin(age_bins)
    if sex is not None:
        mask &= avg["sex"] == sex
    return float(avg.loc[mask, "population"].sum() / total)


def add_population_features(df: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    missing = set(POPULATION_FEATURE_COLUMNS) - set(static_features.columns)
    if missing:
        raise ValueError(
            f"static_features missing columns: {sorted(missing)}. "
            "Call compute_store_population_features() first."
        )
    if "store_id" not in df.columns:
        raise ValueError("df missing 'store_id' — required for per-store population merge")
    return df.merge(
        static_features[["store_id", *POPULATION_FEATURE_COLUMNS]],
        on="store_id", how="left",
    )

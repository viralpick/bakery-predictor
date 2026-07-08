"""Administrative-dong resident population by age/sex (monthly snapshot).

Source: 행정안전부 "행정동별 성·연령별 주민등록 인구수" (data.go.kr 15108072).
The dataset publishes monthly snapshots of 등록 인구 per admin dong, broken
out by 1-year age cohorts × sex. For baking-predictor we collapse to the
broad bins spec §2.5 names and keep one row per (dong, ym, age_bin, sex).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.store_mapping import StationMapping

POPULATION_COLUMNS: dict[str, str] = {
    "admin_dong_code": "string",
    "ym": "string",           # YYYYMM snapshot
    "age_bin": "string",      # "0_9" / "10_19" / "20_29" / "30_39" / "40_49" / "50_59" / "60_plus"
    "sex": "string",          # "M" / "F"
    "population": "int32",
}

AGE_BINS = ("0_9", "10_19", "20_29", "30_39", "40_49", "50_59", "60_plus")

# Per-dong stylized age × sex profile. Numbers are 1-row-per-store rough
# baselines; total ≈ resident population, splits reflect Seoul ward shapes.
_SYNTH_PROFILE: dict[str, dict[str, dict[str, float]]] = {
    "11680565": {  # 청담동 — older affluent
        "M": {"0_9": 0.04, "10_19": 0.05, "20_29": 0.10, "30_39": 0.12,
              "40_49": 0.18, "50_59": 0.18, "60_plus": 0.33},
        "F": {"0_9": 0.04, "10_19": 0.04, "20_29": 0.11, "30_39": 0.13,
              "40_49": 0.18, "50_59": 0.18, "60_plus": 0.32},
    },
    "11440660": {  # 서교동 — young transit
        "M": {"0_9": 0.03, "10_19": 0.05, "20_29": 0.32, "30_39": 0.22,
              "40_49": 0.14, "50_59": 0.12, "60_plus": 0.12},
        "F": {"0_9": 0.03, "10_19": 0.05, "20_29": 0.34, "30_39": 0.22,
              "40_49": 0.13, "50_59": 0.12, "60_plus": 0.11},
    },
    "11560540": {  # 여의동 — office hub, lighter residents
        "M": {"0_9": 0.05, "10_19": 0.06, "20_29": 0.16, "30_39": 0.20,
              "40_49": 0.22, "50_59": 0.16, "60_plus": 0.15},
        "F": {"0_9": 0.05, "10_19": 0.06, "20_29": 0.17, "30_39": 0.20,
              "40_49": 0.22, "50_59": 0.16, "60_plus": 0.14},
    },
}
_DEFAULT_TOTAL_M = 11_000
_DEFAULT_TOTAL_F = 11_500
_DEFAULT_PROFILE = {
    "M": {"0_9": 0.06, "10_19": 0.08, "20_29": 0.16, "30_39": 0.18,
          "40_49": 0.18, "50_59": 0.16, "60_plus": 0.18},
    "F": {"0_9": 0.06, "10_19": 0.08, "20_29": 0.16, "30_39": 0.18,
          "40_49": 0.18, "50_59": 0.16, "60_plus": 0.18},
}
_DONG_TOTALS: dict[str, tuple[int, int]] = {
    "11680565": (5_400, 5_800),
    "11440660": (12_500, 13_200),
    "11560540": (4_800, 5_200),
}


def build_synthetic_population(
    *,
    mapping: dict[str, StationMapping],
    ym_start: str = "2024-01",
    ym_end: str = "2025-12",
    monthly_growth: float = 0.0008,
) -> pd.DataFrame:
    """Hand-crafted monthly population snapshots, one row per (dong, ym, bin, sex)."""
    dong_codes = sorted({s["admin_dong_code"] for s in mapping.values()})
    months = pd.period_range(ym_start, ym_end, freq="M")
    rows: list[dict] = []
    for dong in dong_codes:
        totals = _DONG_TOTALS.get(dong, (_DEFAULT_TOTAL_M, _DEFAULT_TOTAL_F))
        profile = _SYNTH_PROFILE.get(dong, _DEFAULT_PROFILE)
        for i, month in enumerate(months):
            growth = (1 + monthly_growth) ** i
            for sex in ("M", "F"):
                total = totals[0 if sex == "M" else 1] * growth
                splits = profile[sex]
                for age_bin in AGE_BINS:
                    rows.append(
                        {
                            "admin_dong_code": dong,
                            "ym": str(month),
                            "age_bin": age_bin,
                            "sex": sex,
                            "population": int(round(total * splits[age_bin])),
                        }
                    )
    return _coerce_dtypes(pd.DataFrame(rows))


def load_population_from_local(
    parquet_path: Path | str,
    *,
    admin_dong_codes: list[str],
) -> pd.DataFrame:
    raw = pd.read_parquet(parquet_path)
    raw["admin_dong_code"] = raw["admin_dong_code"].astype("string")
    return _coerce_dtypes(raw[raw["admin_dong_code"].isin(admin_dong_codes)].copy())


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["admin_dong_code"] = df["admin_dong_code"].astype("string")
    df["ym"] = df["ym"].astype("string")
    df["age_bin"] = df["age_bin"].astype("string")
    df["sex"] = df["sex"].astype("string")
    df["population"] = df["population"].astype("int32")
    return df[list(POPULATION_COLUMNS.keys())].reset_index(drop=True)


def validate_population(df: pd.DataFrame) -> None:
    missing = set(POPULATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"population frame missing columns: {sorted(missing)}")
    if (df["population"] < 0).any():
        raise ValueError("population frame has negative counts")
    bad_bin = set(df["age_bin"].unique()) - set(AGE_BINS)
    if bad_bin:
        raise ValueError(f"unknown age_bin values: {sorted(bad_bin)}")

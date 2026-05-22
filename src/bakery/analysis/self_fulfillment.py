"""Self-fulfilling stockout pattern analysis.

If v0 (sold_units target) is run in production:
  품절 → low sold → model learns "low demand" → low production → 품절 반복.

We detect that pattern in historical sales by checking, per (store, item, dow):
  - how consistent the stockout time is week-to-week
  - the coefficient of variation of sold_units within (item, dow) — low CV +
    high stockout rate ⇒ store hit a sales ceiling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tuning knobs — chosen for bakery-scale daily data; surface as args if needed.
HIGH_STOCKOUT_RATE = 0.5    # 50% 이상 일자에서 품절이면 후보
LOW_CV_CEILING = 0.4         # CV ≤ 0.4 → 매주 비슷한 매출 (천장 의심)


def per_item_dow_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    """For each (store_id, item_id, dow), summarize sold_units + stockout dynamics."""
    df = daily.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["stockout_hour"] = df["stockout_time"].dt.hour
    grouped = df.groupby(["store_id", "item_id", "dow"], observed=True)
    out = grouped.agg(
        sold_mean=("sold_units", "mean"),
        sold_std=("sold_units", "std"),
        sold_total=("sold_units", "sum"),
        stockout_rate=("is_stockout", "mean"),
        stockout_hour_mean=("stockout_hour", "mean"),
        stockout_hour_std=("stockout_hour", "std"),
        n_weeks=("date", "count"),
    ).reset_index()
    out["sold_cv"] = (out["sold_std"] / out["sold_mean"].replace(0, np.nan)).fillna(0)
    out["self_fulfilling_score"] = (
        (out["stockout_rate"] >= HIGH_STOCKOUT_RATE).astype(int)
        * (out["sold_cv"] <= LOW_CV_CEILING).astype(int)
    )
    return out


def top_self_fulfilling_items(daily: pd.DataFrame, *, n: int = 15) -> pd.DataFrame:
    """Items most likely capped by self-fulfilling stockout.

    Ranked by total `sold_total` among (store, item, dow) groups that flag
    self_fulfilling_score == 1 — biggest absolute revenue impact first.
    """
    pat = per_item_dow_pattern(daily)
    flagged = pat[pat["self_fulfilling_score"] == 1].copy()
    item_summary = (
        flagged.groupby(["store_id", "item_id"], observed=True)
        .agg(
            sold_total=("sold_total", "sum"),
            avg_stockout_rate=("stockout_rate", "mean"),
            avg_sold_cv=("sold_cv", "mean"),
            avg_stockout_hour=("stockout_hour_mean", "mean"),
            covered_dows=("dow", "nunique"),
        )
        .reset_index()
        .sort_values("sold_total", ascending=False)
    )
    return item_summary.head(n).reset_index(drop=True)


def stockout_hour_distribution(
    daily: pd.DataFrame, item_ids: list[str] | None = None
) -> pd.DataFrame:
    """Per (item_id, dow) average stockout hour + std + week count.

    Low std (e.g. ≤ 1.5h) on a popular item means "매주 같은 시각 품절" — the
    clearest visual signature of a sales ceiling.
    """
    df = daily[daily["is_stockout"]].copy()
    if item_ids:
        df = df[df["item_id"].isin(item_ids)]
    df["dow"] = df["date"].dt.dayofweek
    df["stockout_hour"] = df["stockout_time"].dt.hour + df["stockout_time"].dt.minute / 60
    out = (
        df.groupby(["store_id", "item_id", "dow"], observed=True)
        .agg(
            stockout_hour_mean=("stockout_hour", "mean"),
            stockout_hour_std=("stockout_hour", "std"),
            n_weeks=("date", "nunique"),
        )
        .reset_index()
    )
    out["stockout_hour_std"] = out["stockout_hour_std"].fillna(0)
    return out


def estimated_lost_demand(
    daily: pd.DataFrame, *, hour_weights: dict[int, float] | None = None
) -> pd.DataFrame:
    """Per (store, item, date) estimate of how many extra units would have sold
    if no stockout — same formula as features/potential_demand.py but kept
    self-contained here so it can run on raw daily without needing the full
    bakery package.

    Default: linear hour weight (open=7~22h evenly distributed). Override via
    `hour_weights` if you have a calibrated profile per store.
    """
    if hour_weights is None:
        hour_weights = {h: 1.0 for h in range(7, 22)}
    cum = _cumulative_weight_lookup(hour_weights)
    df = daily.copy()
    has_stockout = df["is_stockout"] & df["stockout_time"].notna()
    df["stockout_hour"] = df["stockout_time"].dt.hour + df["stockout_time"].dt.minute / 60
    df["cum_weight"] = df["stockout_hour"].map(
        lambda h: cum.get(int(h), 1.0) if pd.notna(h) else 1.0
    )
    df.loc[df["cum_weight"] < 0.15, "cum_weight"] = 0.15  # floor, same as features/potential_demand.py
    df["potential_demand"] = np.where(
        has_stockout,
        np.minimum(df["sold_units"] / df["cum_weight"], df["sold_units"] * 3.0),
        df["sold_units"],
    )
    df["lost_units"] = (df["potential_demand"] - df["sold_units"]).clip(lower=0)
    return df[["store_id", "item_id", "date", "sold_units", "stockout_time",
               "potential_demand", "lost_units"]]


def _cumulative_weight_lookup(hour_weights: dict[int, float]) -> dict[int, float]:
    """For each open hour h, fraction of daily demand that has accumulated by h."""
    hours = sorted(hour_weights)
    total = sum(hour_weights.values())
    cum: dict[int, float] = {}
    running = 0.0
    for h in hours:
        running += hour_weights[h]
        cum[h] = running / total if total else 0.0
    return cum

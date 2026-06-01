"""Stage 2: Item Proportion Model.

Stage 1 produces a single category-total production number.
Stage 2 distributes it across items using:
  - base   : recent N-day sold share (popularity baseline)
  - trend  : ±20% based on recent_90d / prior_90d
  - early stockout: +10% if item is in bottom-quartile of stockout hour
  - high closing : -10% if item is in top-quartile of closing rate
  - new product  : 1.2× for items active < new_product_window days
  - normalize so Σ proportions = 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Tuning constants
BASE_WINDOW_DAYS       = 28      # 최근 28일 점유율
TREND_RECENT_DAYS      = 90      # 최근 90일
TREND_PRIOR_DAYS       = 90      # 직전 90일

# 분위수 기반 연속 boost — 인기도/과잉도에 비례
STOCKOUT_MAX_BOOST     = 0.20    # 가장 인기품 → +20%, 가장 늦게 매진 → 0%
CLOSING_MAX_REDUCE     = 0.20    # 가장 과잉 → -20%, 가장 안 과잉 → 0%
TREND_MAX_BOOST        = 0.20    # 추세 강도에 비례 ±20%
TREND_SIGNIFICANCE     = 0.15    # |trend_pct| < 15% 면 무시 (noise)

NEW_PRODUCT_WINDOW     = 90      # 진입 < 90d = 신제품
NEW_PRODUCT_BOOST      = 0.20    # 1.2× (단순 binary)


@dataclass
class ItemProportionResult:
    proportions: pd.DataFrame   # date, item_id, proportion, base, trend, stockout, closing, new
    quantities:  pd.DataFrame   # date, item_id, qty (= total × proportion)


def _per_item_signals(
    history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute per-item signals from history up to (not including) cutoff_date.

    history: daily DataFrame [date, item_id, category_id, sold_units,
                              is_stockout, stockout_time, closing_qty (optional)]
    """
    base_start  = cutoff_date - pd.Timedelta(days=BASE_WINDOW_DAYS)
    recent_start = cutoff_date - pd.Timedelta(days=TREND_RECENT_DAYS)
    prior_start  = cutoff_date - pd.Timedelta(days=TREND_RECENT_DAYS + TREND_PRIOR_DAYS)

    base   = history[(history["date"] >= base_start)  & (history["date"] < cutoff_date)]
    recent = history[(history["date"] >= recent_start) & (history["date"] < cutoff_date)]
    prior  = history[(history["date"] >= prior_start) & (history["date"] < recent_start)]

    # Base proportion (raw sold share in recent N days)
    base_sold = base.groupby("item_id")["sold_units"].sum().rename("base_sold")
    cat_map   = history.drop_duplicates("item_id").set_index("item_id")["category_id"]

    # Trend
    r_avg = recent.groupby("item_id")["sold_units"].mean().rename("recent_avg")
    p_avg = prior.groupby("item_id")["sold_units"].mean().rename("prior_avg")
    trend = pd.concat([r_avg, p_avg], axis=1).fillna(0)
    trend["trend_pct"] = np.where(
        trend["prior_avg"] > 0.5,
        (trend["recent_avg"] - trend["prior_avg"]) / trend["prior_avg"], 0.0,
    )

    # Stockout hour
    so = history[history["is_stockout"] & (history["date"] < cutoff_date)].copy()
    so["so_hour"] = pd.to_datetime(so["stockout_time"]).dt.hour
    avg_so = so.groupby("item_id")["so_hour"].mean().rename("avg_stockout_h")

    # Closing rate
    if "closing_qty" in history.columns:
        cd_agg = base.groupby("item_id").agg(
            total_sold=("sold_units", "sum"),
            closing_qty=("closing_qty", "sum"),
        )
        cd_agg["closing_rate"] = cd_agg["closing_qty"] / cd_agg["total_sold"].replace(0, np.nan)
        cd_agg["closing_rate"] = cd_agg["closing_rate"].fillna(0)
        closing_rate = cd_agg["closing_rate"]
    else:
        closing_rate = pd.Series(0.0, index=base_sold.index, name="closing_rate")

    # First sold date (for new product)
    first_seen = history.groupby("item_id")["date"].min().rename("first_sold")

    df = pd.concat([base_sold, trend["trend_pct"], avg_so, closing_rate, first_seen], axis=1)
    df["category_id"] = df.index.map(cat_map)
    df["days_since_first"] = (cutoff_date - df["first_sold"]).dt.days
    return df.dropna(subset=["category_id"])


def _classify_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category rank percentile (연속 분위수).

    stockout_rank_pct : avg_stockout_h의 percentile (0=가장 이른 매진, 1=가장 늦)
    closing_rank_pct  : closing_rate의 percentile (0=가장 적음, 1=가장 많음)
    """
    out_chunks = []
    for cat, group in df.groupby("category_id"):
        g = group.copy()
        # stockout_h: 작을수록 인기 → rank ascending (작은 값 = rank 0)
        valid_h = g["avg_stockout_h"]
        g["stockout_rank_pct"] = valid_h.rank(pct=True, ascending=True).fillna(1.0)
        # closing_rate: 클수록 과잉 → rank ascending (큰 값 = rank 1)
        g["closing_rank_pct"] = g["closing_rate"].rank(pct=True, ascending=True).fillna(0.0)
        out_chunks.append(g)
    return pd.concat(out_chunks)


def compute_proportions(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """For a single target_date, compute item proportions from history < target_date."""
    sig = _per_item_signals(history, target_date)
    sig = _classify_signals(sig)

    # 분위수 기반 연속 boost (인기도/과잉도 강도에 비례)

    # adj_trend: |trend_pct| > 15%일 때만 활성, 강도에 비례 ±20% 까지
    trend_strength = (sig["trend_pct"].abs() - TREND_SIGNIFICANCE).clip(lower=0) / TREND_SIGNIFICANCE
    trend_strength = trend_strength.clip(upper=1.0)  # 0~1
    trend_direction = np.sign(sig["trend_pct"])      # +1 / 0 / -1
    sig["adj_trend"] = 1.0 + TREND_MAX_BOOST * trend_strength * trend_direction

    # adj_stockout: stockout_rank_pct 작을수록 (인기품) +boost
    # rank=0 → +max, rank=1 → 0
    sig["adj_stockout"] = 1.0 + STOCKOUT_MAX_BOOST * (1 - sig["stockout_rank_pct"])

    # adj_closing: closing_rank_pct 클수록 (과잉) -reduce
    # rank=1 → -max, rank=0 → 0
    sig["adj_closing"] = 1.0 - CLOSING_MAX_REDUCE * sig["closing_rank_pct"]

    # adj_new: 신제품 binary (4주 진단으로 별도 처리)
    sig["adj_new"] = np.where(sig["days_since_first"] < NEW_PRODUCT_WINDOW, 1 + NEW_PRODUCT_BOOST, 1.0)

    # Raw weighted base
    sig["raw_weight"] = sig["base_sold"] * sig["adj_trend"] * sig["adj_stockout"] * sig["adj_closing"] * sig["adj_new"]
    # Filter out zero-base items
    sig = sig[sig["base_sold"] > 0].copy()

    # Normalize globally (Σ proportion = 1)
    total_weight = sig["raw_weight"].sum()
    if total_weight == 0:
        sig["proportion"] = 0.0
    else:
        sig["proportion"] = sig["raw_weight"] / total_weight

    out = sig.reset_index()
    out["target_date"] = target_date
    return out[[
        "target_date", "item_id", "category_id", "proportion",
        "base_sold", "trend_pct", "avg_stockout_h", "closing_rate",
        "days_since_first", "adj_trend", "adj_stockout", "adj_closing", "adj_new",
    ]]


def distribute_total(
    history: pd.DataFrame,
    total_by_date: pd.Series,
) -> ItemProportionResult:
    """For each date in total_by_date, distribute the total across items."""
    all_props = []
    all_qty = []
    for d, total in total_by_date.items():
        d = pd.Timestamp(d)
        props = compute_proportions(history, d)
        props["qty"] = props["proportion"] * total
        all_props.append(props)
        all_qty.append(props[["target_date", "item_id", "qty"]].rename(columns={"target_date": "date"}))
    return ItemProportionResult(
        proportions=pd.concat(all_props, ignore_index=True),
        quantities=pd.concat(all_qty, ignore_index=True),
    )

"""Per-item popularity signals + production quantity recommendation.

Combines 4 signals from daily sales + closing-discount data:
  1. avg_stockout_hour  — earlier = more popular
  2. stockout_freq      — higher = under-supply
  3. closing_discount_per_unit — higher = over-supply
  4. trend (recent 90d vs prior 90d) — direction shift

Output: per-item recommendation in {strong_up, up, hold, down, strong_down}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Thresholds — tuned for 광교 5y data.
# 광교는 거의 모든 품목이 매일 품절되는 매장 (baseline 부재) →
# stockout_freq는 변별력 없음. avg_stockout_h(상대적 분위수)와 closing_rate
# 분위수로 분류.
MIN_DAYS_SOLD       = 60
RECENT_WINDOW_DAYS  = 90
TREND_UP_THRESHOLD  = 0.15
TREND_DOWN_THRESHOLD = -0.15

# 분위수 기준 (per-category 또는 전체)
STOCKOUT_H_EARLY_PCT  = 0.25  # 하위 25% 시간 = 일찍 품절 = 인기
STOCKOUT_H_LATE_PCT   = 0.75  # 상위 25% 시간 = 늦게 품절 = 한계
CLOSING_RATE_LOW_PCT  = 0.25  # 하위 25% 마감비율 = 거의 안 남음 = 인기
CLOSING_RATE_HIGH_PCT = 0.75  # 상위 25% 마감비율 = 잘 안 팔림 = 과잉


@dataclass
class PopularityResult:
    signals: pd.DataFrame   # one row per item
    recommendations: pd.DataFrame  # ordered, with rationale


def _compute_trend(daily: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Return per-item trend: recent vs prior mean daily sold."""
    recent_start = today - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    prior_start  = today - pd.Timedelta(days=2 * RECENT_WINDOW_DAYS)

    recent = daily[(daily["date"] > recent_start) & (daily["date"] <= today)]
    prior  = daily[(daily["date"] > prior_start)  & (daily["date"] <= recent_start)]

    r = recent.groupby("item_id")["sold_units"].mean().rename("recent_avg")
    p = prior.groupby("item_id")["sold_units"].mean().rename("prior_avg")
    df = pd.concat([r, p], axis=1).fillna(0)
    df["trend_pct"] = np.where(
        df["prior_avg"] > 0.5,
        (df["recent_avg"] - df["prior_avg"]) / df["prior_avg"],
        0.0,
    )
    return df.reset_index()


def compute_popularity_signals(
    daily: pd.DataFrame,
    closing_discount: pd.DataFrame,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """daily: bonavi_daily.parquet schema.
    closing_discount: rows from DiscountSales.closing_discount() — needs item_id, date, qty.
    """
    today = today or pd.Timestamp(daily["date"].max())

    # Stockout hour from stockout_time
    sd = daily[daily["is_stockout"]].copy()
    sd["stockout_h"] = pd.to_datetime(sd["stockout_time"]).dt.hour
    so_stats = sd.groupby("item_id").agg(
        stockout_days   = ("date", "nunique"),
        avg_stockout_h  = ("stockout_h", "mean"),
        median_stockout_h = ("stockout_h", "median"),
    )

    # All-time per-item daily stats
    base = daily.groupby("item_id").agg(
        category_id     = ("category_id", "first"),
        days_sold       = ("date", "nunique"),
        total_sold      = ("sold_units", "sum"),
        avg_daily_sold  = ("sold_units", "mean"),
        stockout_freq_all = ("is_stockout", "mean"),
    )

    # Closing-discount aggregated per item
    cd = closing_discount.copy()
    cd_agg = cd.groupby("item_id").agg(
        closing_qty = ("qty", "sum"),
        closing_days = ("date", "nunique"),
    )

    # Merge
    out = base.join(so_stats, how="left").join(cd_agg, how="left")
    out["closing_qty"]  = out["closing_qty"].fillna(0)
    out["closing_days"] = out["closing_days"].fillna(0)
    out["stockout_days"] = out["stockout_days"].fillna(0)
    out["closing_rate_per_sold"] = out["closing_qty"] / out["total_sold"].replace(0, np.nan)
    out["closing_rate_per_sold"] = out["closing_rate_per_sold"].fillna(0)

    # Trend
    trend = _compute_trend(daily, today).set_index("item_id")
    out = out.join(trend[["recent_avg", "prior_avg", "trend_pct"]], how="left").fillna({"trend_pct": 0})

    out = out[out["days_sold"] >= MIN_DAYS_SOLD].reset_index()
    return out


def _classify_quantile(row: pd.Series) -> tuple[str, str]:
    """Quantile-based classification (광교 baseline-부재 매장에 맞춤).

    score = early_stockout(2) + low_closing(2) + trend(±1) + late_stockout(-2) + high_closing(-2)
    """
    score = 0
    parts: list[str] = []

    if row["early_stockout"]:
        score += 2
        parts.append(f"일찍 품절({row['avg_stockout_h']:.1f}시, 하위25%)")
    if row["late_stockout"]:
        score -= 2
        parts.append(f"늦게 품절({row['avg_stockout_h']:.1f}시, 상위25%)")

    if row["low_closing"]:
        score += 2
        parts.append(f"마감 할인 적음({row['closing_rate_per_sold']*100:.0f}%)")
    if row["high_closing"]:
        score -= 2
        parts.append(f"마감 할인 많음({row['closing_rate_per_sold']*100:.0f}%, 과잉)")

    if row["trend_pct"] >= TREND_UP_THRESHOLD:
        score += 1
        parts.append(f"추세 {row['trend_pct']*100:+.0f}%")
    elif row["trend_pct"] <= TREND_DOWN_THRESHOLD:
        score -= 1
        parts.append(f"추세 {row['trend_pct']*100:+.0f}%")

    if score >= 3:
        rec = "strong_up"
    elif score >= 1:
        rec = "up"
    elif score <= -3:
        rec = "strong_down"
    elif score <= -1:
        rec = "down"
    else:
        rec = "hold"
        if not parts:
            parts.append("안정 — 신호 없음")

    return rec, "; ".join(parts)


def recommend_quantities(signals: pd.DataFrame) -> pd.DataFrame:
    """Per-category quantile-based binning."""
    out = signals.copy()

    # per-category 분위수 (광교는 cat별 특성 다름: bread vs pastry vs cake)
    flags = []
    for cat, group in out.groupby("category_id"):
        # closing_rate: 전 품목 대상 분위수
        cr_low  = group["closing_rate_per_sold"].quantile(CLOSING_RATE_LOW_PCT)
        cr_high = group["closing_rate_per_sold"].quantile(CLOSING_RATE_HIGH_PCT)

        # stockout_h: 품절 발생 품목에 한해
        valid_h = group["avg_stockout_h"].dropna()
        if len(valid_h) >= 4:
            so_early = valid_h.quantile(STOCKOUT_H_EARLY_PCT)
            so_late  = valid_h.quantile(STOCKOUT_H_LATE_PCT)
        else:
            so_early = float("-inf")
            so_late  = float("inf")

        g = group.copy()
        g["early_stockout"] = g["avg_stockout_h"] <= so_early
        g["late_stockout"]  = (g["avg_stockout_h"] >= so_late) & g["avg_stockout_h"].notna()
        g["low_closing"]    = g["closing_rate_per_sold"] <= cr_low
        g["high_closing"]   = g["closing_rate_per_sold"] >= cr_high
        flags.append(g)

    out = pd.concat(flags, ignore_index=True)

    recs = out.apply(_classify_quantile, axis=1, result_type="expand")
    out["recommendation"] = recs[0]
    out["rationale"] = recs[1]

    rec_order = {"strong_up": 0, "up": 1, "hold": 2, "down": 3, "strong_down": 4}
    out["_order"] = out["recommendation"].map(rec_order)
    out = out.sort_values(["_order", "total_sold"], ascending=[True, False]).drop(columns="_order")
    return out

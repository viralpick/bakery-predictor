"""Stage 3: New Product Tracker.

신제품 식별 → 4주 메트릭 추적 → 정규/보류/fade-out 의사결정.

modeling_v4.md §2 Stage 3 spec:
- 감지: first_sold_date < NEW_PRODUCT_WINDOW (90d) ago
- 추적: 4주 누적 평균 sold / stockout 시각 / 마감 할인 비율 / revenue/capacity
- 의사결정:
  - promote   : 평균 sold ≥ category median × 0.5 + closing_rate < 0.20
  - fade_out  : 평균 sold < category median × 0.2 + closing_rate ≥ 0.40
  - hold      : 위 조건 미달
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Spec 상수 (modeling_v4.md §2 Stage 3)
NEW_PRODUCT_WINDOW_DAYS = 90    # first_sold < 90d → 신제품
TRACKING_WINDOW_DAYS    = 28    # 4주 (= 28일)
PROMOTE_SOLD_RATIO      = 0.5   # category median × 0.5 이상
PROMOTE_CLOSING_MAX     = 0.20  # 마감 할인 비율 < 20%
FADE_SOLD_RATIO         = 0.2   # category median × 0.2 미만
FADE_CLOSING_MIN        = 0.40  # 마감 할인 비율 ≥ 40%


@dataclass
class NewProductMetric:
    item_id: str
    name: str
    category_id: str
    first_sold: pd.Timestamp
    days_since_first: int
    n_tracking_days: int
    avg_daily_sold: float
    avg_stockout_h: float
    closing_rate: float          # closing_qty / total_sold during tracking
    revenue_per_capacity: float  # paid / capacity (활용도)
    decision: str                # "promote" / "hold" / "fade_out"
    rationale: str


def detect_new_products(
    daily: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = NEW_PRODUCT_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return items whose first_sold_date is within `window_days` of `as_of`.

    daily: must have ['date', 'item_id', 'category_id']
    Returns: item_id, category_id, first_sold, days_since_first
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    history = d[d["date"] <= as_of]

    first_seen = history.groupby("item_id").agg(
        first_sold = ("date", "min"),
        category_id = ("category_id", "first"),
    ).reset_index()
    first_seen["days_since_first"] = (as_of - first_seen["first_sold"]).dt.days
    return first_seen[first_seen["days_since_first"] < window_days].sort_values("days_since_first")


def compute_metrics(
    daily: pd.DataFrame,
    closing_per_item: pd.DataFrame,
    item_id: str,
    as_of: pd.Timestamp,
    window_days: int = TRACKING_WINDOW_DAYS,
) -> dict | None:
    """Track 4주 (or fewer if newer) metrics for a single item.

    closing_per_item: DataFrame [date, item_id, closing_qty]
    Returns dict with avg_daily_sold, avg_stockout_h, closing_rate, revenue_per_capacity.
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    track_start = as_of - pd.Timedelta(days=window_days)
    item_data = d[(d["item_id"] == item_id) & (d["date"] > track_start) & (d["date"] <= as_of)].copy()

    if len(item_data) == 0:
        return None

    item_data["stockout_h"] = pd.to_datetime(item_data["stockout_time"]).dt.hour

    # Closing rate
    cd = closing_per_item[
        (closing_per_item["item_id"] == item_id) &
        (closing_per_item["date"] > track_start) &
        (closing_per_item["date"] <= as_of)
    ]
    closing_qty = cd["closing_qty"].sum() if len(cd) else 0
    total_sold  = item_data["sold_units"].sum()
    closing_rate = closing_qty / total_sold if total_sold > 0 else 0

    # Revenue per capacity (assume capacity from daily)
    if "capacity" in item_data.columns:
        capacity_sum = item_data["capacity"].sum()
        revenue_per_cap = total_sold / capacity_sum if capacity_sum > 0 else 0
    else:
        revenue_per_cap = np.nan

    return {
        "item_id": item_id,
        "n_tracking_days": len(item_data),
        "avg_daily_sold":  item_data["sold_units"].mean(),
        "avg_stockout_h":  item_data["stockout_h"].mean(),
        "closing_rate":    closing_rate,
        "revenue_per_capacity": revenue_per_cap,
        "total_sold":      total_sold,
        "stockout_freq":   item_data["is_stockout"].mean(),
    }


def decide(
    metrics: dict,
    category_median_sold: float,
) -> tuple[str, str]:
    """Apply 정규/보류/fade-out logic.

    Returns: (decision, rationale)
    """
    avg = metrics["avg_daily_sold"]
    cr = metrics["closing_rate"]

    promote_sold_threshold = category_median_sold * PROMOTE_SOLD_RATIO
    fade_sold_threshold    = category_median_sold * FADE_SOLD_RATIO

    if avg >= promote_sold_threshold and cr < PROMOTE_CLOSING_MAX:
        return ("promote", f"avg_sold {avg:.1f} ≥ {promote_sold_threshold:.1f} + 마감 {cr*100:.0f}% < 20%")
    if avg < fade_sold_threshold and cr >= FADE_CLOSING_MIN:
        return ("fade_out", f"avg_sold {avg:.1f} < {fade_sold_threshold:.1f} + 마감 {cr*100:.0f}% ≥ 40%")
    return ("hold", f"avg_sold {avg:.1f}, 마감 {cr*100:.0f}% — 추가 관찰")


def track_all_new_products(
    daily: pd.DataFrame,
    closing_per_item: pd.DataFrame,
    name_map: pd.Series,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Detect + track + decide for all current new products."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    new_items = detect_new_products(d, as_of)

    # Category median sold (over all history, all items)
    item_avg = d[d["date"] <= as_of].groupby(["category_id", "item_id"])["sold_units"].mean().reset_index()
    cat_median = item_avg.groupby("category_id")["sold_units"].median().to_dict()

    rows = []
    for _, ni in new_items.iterrows():
        metrics = compute_metrics(d, closing_per_item, ni["item_id"], as_of)
        if metrics is None:
            continue
        cat_med = cat_median.get(ni["category_id"], 1.0)
        decision, rationale = decide(metrics, cat_med)
        rows.append({
            "item_id": ni["item_id"],
            "name": name_map.get(ni["item_id"], "?") if isinstance(name_map, pd.Series) else "?",
            "category_id": ni["category_id"],
            "first_sold": ni["first_sold"],
            "days_since_first": ni["days_since_first"],
            "category_median_sold": cat_med,
            **{k: v for k, v in metrics.items() if k != "item_id"},
            "decision": decision,
            "rationale": rationale,
        })
    return pd.DataFrame(rows)

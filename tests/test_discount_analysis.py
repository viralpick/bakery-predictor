"""Smoke tests for discount/popularity/waste analysis modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.analysis.discount import (
    DiscountSales, GWANGYO_CODE_LABELS, classify_code,
    discount_summary, label_summary,
)
from bakery.analysis.popularity import (
    compute_popularity_signals, recommend_quantities,
)
from bakery.analysis.waste import (
    CapacityMinusSoldEstimator, ClosingDiscountLoss,
    business_impact_summary,
)


def _toy_discount_rows() -> pd.DataFrame:
    """Synthetic line-items: 3 days × 3 items × discount mix."""
    rows = []
    for day_n, d in enumerate(pd.date_range("2024-01-01", periods=3)):
        for hour in [10, 15, 20, 21]:
            # full price at 10/15, closing at 20/21
            for item in ["bread1", "pastry1", "pastry2"]:
                if hour >= 20:
                    rows.append({
                        "receipt_id": f"r{day_n}_{hour}_{item}_c", "date": d,
                        "hour": hour, "minute": 0, "item_id": item, "qty": 2,
                        "unit_price": 4000.0, "paid": 5600.0,
                        "discount_amt": 2400.0, "discount_code": "0077",
                        "label": "closing", "is_set": False,
                    })
                else:
                    rows.append({
                        "receipt_id": f"r{day_n}_{hour}_{item}", "date": d,
                        "hour": hour, "minute": 0, "item_id": item, "qty": 1,
                        "unit_price": 4000.0, "paid": 4000.0,
                        "discount_amt": 0.0, "discount_code": "",
                        "label": "none", "is_set": False,
                    })
    return pd.DataFrame(rows)


def _toy_daily() -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2024-01-01", periods=60):
        for item, cat in [("bread1", "bread"), ("pastry1", "pastry"), ("pastry2", "pastry")]:
            rows.append({
                "store_id": "s1", "item_id": item, "category_id": cat,
                "date": d, "sold_units": 10, "is_stockout": d.day % 3 == 0,
                "stockout_time": d + pd.Timedelta(hours=14) if d.day % 3 == 0 else pd.NaT,
                "open_hours": 13, "capacity": 15, "potential_demand": 12.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# discount.py
# ---------------------------------------------------------------------------

def test_all_30_gwangyo_codes_classified():
    """All 30 광교 codes must have a label (no 'other' for known codes)."""
    for code in GWANGYO_CODE_LABELS:
        assert classify_code(code) != "other", f"missing classification: {code}"


def test_classify_known_codes():
    assert classify_code("0077") == "closing"
    assert classify_code("0069") == "closing"
    assert classify_code("0121") == "payment"
    assert classify_code("0081") == "staff"
    assert classify_code("unknown") == "other"


def test_discount_summary_aggregates():
    rows = _toy_discount_rows()
    ds = DiscountSales(rows=rows)
    summ = discount_summary(ds)
    # Only closing discounts produce > 0 discount_amt in the toy data
    assert len(summ) == 1
    assert summ.iloc[0]["discount_code"] == "0077"
    assert summ.iloc[0]["label"] == "closing"
    assert summ.iloc[0]["share_at_pm8"] == 1.0


def test_label_summary_share_at_pm8_for_closing():
    rows = _toy_discount_rows()
    ds = DiscountSales(rows=rows)
    lsum = label_summary(ds)
    closing = lsum[lsum["label"] == "closing"].iloc[0]
    # All closing rows in toy data are at hour ≥ 20
    assert closing["share_at_pm8"] == 1.0


# ---------------------------------------------------------------------------
# popularity.py
# ---------------------------------------------------------------------------

def test_popularity_signals_non_empty():
    daily = _toy_daily()
    cd_rows = _toy_discount_rows()
    cd_rows = cd_rows[cd_rows["label"] == "closing"]
    signals = compute_popularity_signals(daily, cd_rows)
    assert len(signals) == 3
    assert {"avg_stockout_h", "closing_rate_per_sold", "trend_pct"}.issubset(signals.columns)


def test_recommendations_assigned():
    daily = _toy_daily()
    cd_rows = _toy_discount_rows()
    cd_rows = cd_rows[cd_rows["label"] == "closing"]
    signals = compute_popularity_signals(daily, cd_rows)
    recs = recommend_quantities(signals)
    valid = {"strong_up", "up", "hold", "down", "strong_down"}
    assert set(recs["recommendation"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# waste.py
# ---------------------------------------------------------------------------

def test_capacity_minus_sold_estimator():
    daily = _toy_daily()
    cd_rows = _toy_discount_rows()
    cd_rows = cd_rows[cd_rows["label"] == "closing"]
    est = CapacityMinusSoldEstimator(daily=daily, closing_discount=cd_rows)
    df = est.per_day_item()
    assert (df["estimated_waste_qty"] >= 0).all()
    # capacity=15, sold=10, closing varies → est_waste ≤ 5
    assert df["estimated_waste_qty"].max() <= 5


def test_closing_discount_loss_aggregates():
    rows = _toy_discount_rows()
    cd_rows = rows[rows["label"] == "closing"]
    cdl = ClosingDiscountLoss(discounts=cd_rows)
    daily_loss = cdl.daily()
    assert len(daily_loss) == 3  # 3 days
    assert (daily_loss["revenue_loss_won"] > 0).all()


def test_business_impact_summary_keys():
    daily = _toy_daily()
    cd_rows = _toy_discount_rows()
    cd_rows = cd_rows[cd_rows["label"] == "closing"]
    cdl = ClosingDiscountLoss(discounts=cd_rows)
    est = CapacityMinusSoldEstimator(daily=daily, closing_discount=cd_rows)
    impact = business_impact_summary(cdl, est, daily)
    required = {
        "annual_closing_qty", "annual_revenue_loss_won",
        "annual_paid_recovered_won" if False else "total_paid_recovered_won",
        "annual_cost_saved_if_zero_closing_discount_won",
    }
    assert required.issubset(impact.keys())
    assert impact["years"] > 0

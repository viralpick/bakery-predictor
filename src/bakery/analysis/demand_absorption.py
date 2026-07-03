"""카테고리 총량 수요이전 흡수 검증 (W0 게이트).

leave-one-out 총량보존 계수 β: 카테고리 내 품목 조기품절(품절강도 T)이
같은 카테고리 총 sold(Y)를 떨어뜨리는가. β≈0 = 흡수(총량 보존), β<0 = walk-away.
confound(고수요일=품절많은날)는 OtherCatSold(그날 전반 traffic) + cat_baseline
(c의 최근 4주 동일요일 평균, lag)로 이중 통제. 타깃은 raw sold_units(순환 회피).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CLOSE_HOUR = 22
BASELINE_WEEKS = 4
PANEL_COLUMNS = [
    "store_id", "category_id", "date", "cat_sold", "stockout_hours",
    "other_cat_sold", "cat_baseline", "dow", "month", "trend",
]


def _stockout_hours(sub: pd.DataFrame, close_hour: int) -> float:
    """Category-day stockout intensity = Σ max(close_hour − stockout time-of-day, 0)."""
    so = pd.to_datetime(sub["stockout_time"])
    tod = so.dt.hour + so.dt.minute / 60.0
    return float((close_hour - tod).clip(lower=0.0).fillna(0.0).sum())


def _category_day_frame(daily: pd.DataFrame, close_hour: int) -> pd.DataFrame:
    """Aggregate item×day → (store, category, date) with cat_sold + stockout_hours."""
    grp = daily.groupby(["store_id", "category_id", "date"], observed=True)
    agg = grp.agg(cat_sold=("sold_units", "sum")).reset_index()
    hours = (grp.apply(lambda s: _stockout_hours(s, close_hour), include_groups=False)
             .rename("stockout_hours").reset_index())
    return agg.merge(hours, on=["store_id", "category_id", "date"])


def _add_other_cat_sold(cat_day: pd.DataFrame) -> pd.DataFrame:
    """OtherCatSold = same store-day total sold across all OTHER categories."""
    store_day = (cat_day.groupby(["store_id", "date"], observed=True)["cat_sold"]
                 .sum().rename("store_total").reset_index())
    out = cat_day.merge(store_day, on=["store_id", "date"])
    out["other_cat_sold"] = out["store_total"] - out["cat_sold"]
    return out.drop(columns=["store_total"])


def _add_leakage_safe_baseline(cat_day: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """cat_baseline = mean of same (store,category,dow) cat_sold over the prior
    `weeks` occurrences, strictly before the row's date (no leakage)."""
    df = cat_day.sort_values("date").copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    def _roll(g: pd.DataFrame) -> pd.Series:
        return g["cat_sold"].shift(1).rolling(weeks, min_periods=weeks).mean()
    df["cat_baseline"] = (df.groupby(["store_id", "category_id", "dow"], observed=True,
                                     group_keys=False).apply(_roll, include_groups=False))
    return df


def build_absorption_panel(daily: pd.DataFrame, *, close_hour: int = DEFAULT_CLOSE_HOUR,
                           baseline_weeks: int = BASELINE_WEEKS) -> pd.DataFrame:
    """Build the (store, category, date) regression panel. Rows without a full
    baseline window are dropped. Target/controls are all raw sold_units."""
    cat_day = _category_day_frame(daily, close_hour)
    cat_day = _add_other_cat_sold(cat_day)
    cat_day = _add_leakage_safe_baseline(cat_day, baseline_weeks)
    cat_day = cat_day.dropna(subset=["cat_baseline"]).copy()
    dt = pd.to_datetime(cat_day["date"])
    cat_day["month"] = dt.dt.month
    cat_day["trend"] = (dt - dt.min()).dt.days.astype(float)
    return cat_day[PANEL_COLUMNS].reset_index(drop=True)

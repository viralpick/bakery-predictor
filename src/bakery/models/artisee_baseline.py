"""고객사(아띠제) 현행 발주 baseline 재구현.

제시량 = 적용수량(3주 주중/주말 평균) × S/O 증산배수 × 요일 스케일링 → 반올림.
⚠️ predict()가 반환하는 값은 sold_units 예측이 아니라 **발주 제시량(order qty)**이다.
전향 KPI 비교의 competitor. 설계: docs/superpowers/specs/2026-07-15-artisee-baseline-design.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAY_MAX_DOW = 4  # 월(0)~금(4) = weekday


def dow_group(dates: pd.Series) -> pd.Series:
    dts = pd.to_datetime(dates)
    dow = dts.dt.dayofweek
    return pd.Series(np.where(dow <= WEEKDAY_MAX_DOW, "weekday", "weekend"), index=dts.index)


def _recent(daily: pd.DataFrame, weeks: int) -> pd.DataFrame:
    cutoff = daily["date"].max() - pd.Timedelta(weeks=weeks)
    return daily[daily["date"] > cutoff]


def applied_quantity(daily: pd.DataFrame, *, weeks: int = 3,
                     spike_ratio: float = 1.3) -> pd.DataFrame:
    recent = _recent(daily, weeks)
    recent = recent[~recent["is_holiday"].astype(bool)].copy()
    recent["dow_group"] = dow_group(recent["date"])
    keys = ["store_id", "item_id", "dow_group"]
    med = recent.groupby(keys)["sold_units"].transform("median")
    capped = np.minimum(recent["sold_units"], med * spike_ratio)
    recent = recent.assign(_capped=capped)
    out = (recent.groupby(keys)["_capped"].mean()
           .rename("base_qty").reset_index())
    return out


def build_item_residual_curve(hourly: pd.DataFrame, *,
                              months: int = 3) -> dict[str, np.ndarray]:
    cutoff = hourly["date"].max() - pd.DateOffset(months=months)
    recent = hourly[hourly["date"] > cutoff]
    out: dict[str, np.ndarray] = {}
    for item_id, g in recent.groupby("item_id"):
        residuals: list[np.ndarray] = []
        for _, day in g.groupby("date"):
            hourly_qty = (day.groupby("hour")["qty"].sum()
                          .reindex(range(24), fill_value=0.0).to_numpy(dtype=float))
            total = hourly_qty.sum()
            if total <= 0:
                continue
            residuals.append(1.0 - np.cumsum(hourly_qty) / total)
        if residuals:
            out[str(item_id)] = np.vstack(residuals).mean(axis=0)
    return out


def _missed_pct(row, curves: dict[str, np.ndarray]) -> float:
    if not bool(row["is_stockout"]) or pd.isna(row["stockout_time"]):
        return 0.0
    curve = curves.get(str(row["item_id"]))
    if curve is None:
        return 0.0
    hour = int(pd.Timestamp(row["stockout_time"]).hour)
    return float(curve[min(hour, 23)])


def soldout_multiplier(daily: pd.DataFrame, curves: dict[str, np.ndarray], *,
                       weeks: int = 3) -> pd.DataFrame:
    recent = _recent(daily, weeks).copy()
    recent["dow_group"] = dow_group(recent["date"])
    recent["missed"] = recent.apply(lambda r: _missed_pct(r, curves), axis=1)
    out = (recent.groupby(["store_id", "item_id", "dow_group"])["missed"].mean()
           .rename("multiplier").reset_index())
    out["multiplier"] = 1.0 + out["multiplier"]
    return out

"""고객사(아띠제) 현행 발주 baseline 재구현.

제시량 = 적용수량(3주 주중/주말 평균) × S/O 증산배수 × 요일 스케일링 → 반올림.
⚠️ predict()가 반환하는 값은 sold_units 예측이 아니라 **발주 제시량(order qty)**이다.
전향 KPI 비교의 competitor. 설계: docs/superpowers/specs/2026-07-15-artisee-baseline-design.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAY_MAX_DOW = 4  # 월(0)~금(4) = weekday
WEEKEND_REPRESENTATIVE_DOW = 5  # 토 — 명절/휴장일(주말 취급)의 요일 스케일 대표값


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


def dow_scaling(daily: pd.DataFrame, *, weeks: int = 3) -> pd.DataFrame:
    recent = _recent(daily, weeks)
    recent = recent[~recent["is_holiday"].astype(bool)].copy()
    recent["dow"] = pd.to_datetime(recent["date"]).dt.dayofweek
    recent["dow_group"] = dow_group(recent["date"])
    keys = ["store_id", "item_id"]
    dow_mean = recent.groupby(keys + ["dow", "dow_group"])["sold_units"].mean()
    grp_mean = recent.groupby(keys + ["dow_group"])["sold_units"].mean()
    df = dow_mean.rename("dm").reset_index().merge(
        grp_mean.rename("gm").reset_index(), on=keys + ["dow_group"])
    df["weight"] = np.where(df["gm"] > 0, df["dm"] / df["gm"], 1.0)
    return df[keys + ["dow", "weight"]]


def round_order(raw: pd.Series, item_ids: pd.Series, *, rounding: str = "generic",
                multiple_map: dict[str, int] | None = None) -> pd.Series:
    """C4 반올림: generic 또는 N배수 floor."""
    if rounding == "generic" or not multiple_map:
        return raw.round()
    ns = item_ids.map(lambda i: multiple_map.get(str(i), 1)).astype(float)
    floored = np.floor(raw / ns) * ns
    return pd.Series(floored.to_numpy(), index=raw.index)


class ArtiseeBaseline:
    """제시량 = base × multiplier × weight → 반올림."""

    name = "artisee_baseline"

    def __init__(self, *, weeks: int = 3, curve_months: int = 3,
                 rounding: str = "generic", multiple_map: dict[str, int] | None = None):
        self.weeks = weeks
        self.curve_months = curve_months
        self.rounding = rounding
        self.multiple_map = multiple_map
        self._base = self._mult = self._dow = None

    def fit(self, daily: pd.DataFrame, hourly: pd.DataFrame) -> "ArtiseeBaseline":
        """학습."""
        curves = build_item_residual_curve(hourly, months=self.curve_months)
        self._base = applied_quantity(daily, weeks=self.weeks)
        self._mult = soldout_multiplier(daily, curves, weeks=self.weeks)
        self._dow = dow_scaling(daily, weeks=self.weeks)
        return self

    def predict(self, target: pd.DataFrame) -> pd.Series:
        """예측: target.index 정렬 제시량.

        M1: target에 `is_holiday`(bool) 컬럼이 있고 True인 행은 명절/건물휴장=주말
        취급(스펙) — dow_group=weekend, 요일 스케일 대표값=토(WEEKEND_REPRESENTATIVE_DOW).
        컬럼이 없으면 기존과 동일(하위호환).
        """
        if self._base is None:
            raise RuntimeError("call fit() before predict()")
        out = target.copy()
        out["dow"] = pd.to_datetime(out["date"]).dt.dayofweek
        out["dow_group"] = dow_group(out["date"])
        if "is_holiday" in out.columns:
            is_holiday = out["is_holiday"].fillna(False).astype(bool)
            out.loc[is_holiday, "dow_group"] = "weekend"
            out.loc[is_holiday, "dow"] = WEEKEND_REPRESENTATIVE_DOW
        keys = ["store_id", "item_id", "dow_group"]
        merged = (out.merge(self._base, on=keys, how="left")
                     .merge(self._mult, on=keys, how="left")
                     .merge(self._dow, on=["store_id", "item_id", "dow"], how="left"))
        merged["base_qty"] = merged["base_qty"].fillna(0.0)
        merged["multiplier"] = merged["multiplier"].fillna(1.0)
        merged["weight"] = merged["weight"].fillna(1.0)
        raw = merged["base_qty"] * merged["multiplier"] * merged["weight"]
        order = round_order(raw, merged["item_id"], rounding=self.rounding,
                            multiple_map=self.multiple_map)
        return pd.Series(order.to_numpy(), index=target.index, name="artisee_order")

"""전향적 KPI harness — 우리 발주 추천 vs 현행 발주를 동일 시뮬로 비교.

반사실: 두 발주량을 같은 복원수요 + 시간대 도착곡선에 넣어 폐기/매진시각/
매진률을 계산한다. 1차 지표는 Δ(우리−아티제)로, 시뮬 편향이 양쪽에 상쇄된다.
매진시각은 potential_demand의 도착곡선을 재사용해 '누적수요가 발주량에
도달하는 시각'을 역산한다. 폐기/lost 비용은 business_metrics 재사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.potential_demand import StoreHours, bakery_hour_profile
from .business_metrics import CostParams, simulate_profit


def simulate_soldout(
    order_qty: float,
    daily_demand: float,
    profile: np.ndarray,
    *,
    open_hour: int,
    close_hour: int,
) -> tuple[float | None, bool]:
    """발주량 하에서 매진시각(hour_float)과 매진여부. 미매진이면 (None, False)."""
    if daily_demand <= 0 or order_qty >= daily_demand:
        return (None, False)
    if order_qty <= 0:
        return (float(open_hour), True)
    target = order_qty / daily_demand          # 도달해야 할 누적 비중
    pre = 0.0
    for h in range(open_hour, close_hour):
        w_h = float(profile[h])
        if pre + w_h >= target:
            frac = (target - pre) / w_h if w_h > 0 else 0.0
            return (h + frac, True)
        pre += w_h
    return (float(close_hour), True)           # 수치오차 fallback


def build_arrival_profile(
    receipts: pd.DataFrame,
    *,
    group_cols: list[str],
    exclude_keys: set | None = None,
    exclude_cols: list[str] | None = None,
) -> dict[tuple, np.ndarray]:
    """그룹별 24-length 시간당 수량 벡터. bakery_hour_profile(measured=)용 raw."""
    df = receipts
    if exclude_keys:
        key_cols = exclude_cols or group_cols
        keys = list(zip(*[df[c].astype(str) for c in key_cols]))
        df = df[[k not in exclude_keys for k in keys]]
    out: dict[tuple, np.ndarray] = {}
    for gkey, g in df.groupby(group_cols):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        vec = np.zeros(24, dtype=float)
        by_hour = g.groupby("hour")["qty"].sum()
        for h, q in by_hour.items():
            vec[int(h)] = float(q)
        out[gkey] = vec
    return out


def reconstruct_baseline_order(
    df: pd.DataFrame,
    *,
    normal_col: str = "normal_units",
    closing_col: str = "closing_units",
    waste_col: str = "waste_units",
) -> pd.Series:
    """생산 = 정상판매 + 마감판매 + 폐기. 회고 검증의 현행 발주 proxy."""
    parts = [df[c].fillna(0.0).astype(float) for c in (normal_col, closing_col, waste_col)]
    return parts[0] + parts[1] + parts[2]


def simulate_item_day_kpis(
    rows: pd.DataFrame,
    profiles: dict[tuple, np.ndarray],
    *,
    order_col: str,
    store_hours: StoreHours,
    group_cols: list[str],
    params: CostParams | None = None,
    unit_prices=None,
) -> pd.DataFrame:
    """item-day별 폐기/lost 비용(business_metrics) + 매진시각/매진여부."""
    params = params or CostParams()
    # 폐기/lost: simulate_profit 재사용 (yhat=발주량, true=potential_demand)
    prof_in = rows.rename(columns={order_col: "yhat"}).copy()
    prof_in["sold_units"] = prof_in["potential_demand"]
    costed = simulate_profit(
        prof_in, unit_prices=unit_prices, params=params,
        yhat_col="yhat", sold_col="sold_units", potential_col="potential_demand",
    )
    # 매진시각: 그룹 프로필로 역산
    soldout_hours, stockouts = [], []
    for _, r in rows.iterrows():
        gkey = tuple(str(r[c]) for c in group_cols)
        raw = profiles.get(gkey)
        prof = bakery_hour_profile(
            store_hours.open_hour, store_hours.close_hour,
            measured=raw if raw is not None else None,
        )
        t, is_so = simulate_soldout(
            float(r[order_col]), float(r["potential_demand"]), prof,
            open_hour=store_hours.open_hour, close_hour=store_hours.close_hour,
        )
        soldout_hours.append(t if t is not None else np.nan)
        stockouts.append(is_so)
    out = rows.copy()
    out["waste_units"] = costed["waste_units"].to_numpy()
    out["lost_sale_units"] = costed["lost_sale_units"].to_numpy()
    out["waste_cost_krw"] = costed["waste_cost_krw"].to_numpy()
    out["lost_margin_krw"] = costed["lost_margin_krw"].to_numpy()
    out["soldout_hour"] = soldout_hours
    out["is_stockout"] = stockouts
    return out

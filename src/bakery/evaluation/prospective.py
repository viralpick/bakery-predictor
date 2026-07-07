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
        # 계약: 키는 항상 str-cast tuple. simulate_item_day_kpis의
        # tuple(str(r[c]) for c in group_cols) 조회와 dtype 상관없이 일치시켜
        # non-string group_cols(int store_id 등)에서 silent fallback을 막는다.
        out[tuple(str(x) for x in gkey)] = vec
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


def _summarize_policy(kpis: pd.DataFrame) -> dict[str, float]:
    """KPI 프레임을 정책별 요약(합계/평균)으로 변환."""
    so = kpis["is_stockout"].astype(bool)
    soldout_median = kpis.loc[so, "soldout_hour"].median() if so.any() else float("nan")
    return {
        "waste_cost_krw": float(kpis["waste_cost_krw"].sum()),
        "lost_margin_krw": float(kpis["lost_margin_krw"].sum()),
        "stockout_rate": float(so.mean()),
        "soldout_median_h": float(soldout_median),
    }


def compare_policies(our_kpis: pd.DataFrame, base_kpis: pd.DataFrame) -> pd.DataFrame:
    """우리·아티제 정책 KPI 요약 + Δ(우리−아티제) 1행."""
    our = _summarize_policy(our_kpis)
    base = _summarize_policy(base_kpis)
    delta = {k: our[k] - base[k] for k in our}
    return pd.DataFrame([
        {"policy": "our", **our},
        {"policy": "baseline", **base},
        {"policy": "delta", **delta},
    ])


def compare_policies_by_fold(
    our_kpis: pd.DataFrame, base_kpis: pd.DataFrame
) -> pd.DataFrame:
    """fold별 Δ(우리−baseline) KPI. 각 프레임은 fold 컬럼을 가져야 한다."""
    rows = []
    for fold in sorted(our_kpis["fold"].unique()):
        our = _summarize_policy(our_kpis[our_kpis["fold"] == fold])
        base = _summarize_policy(base_kpis[base_kpis["fold"] == fold])
        rows.append({"fold": int(fold), **{k: our[k] - base[k] for k in our}})
    return pd.DataFrame(rows)


def aggregate_fold_kpis(per_fold: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """fold별 Δ를 metric별 mean ± 95%CI(정규근사)로 집계. fold 수 적음 — caveat 문서화."""
    out = []
    for col in metric_cols:
        vals = per_fold[col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        n = int(len(vals))
        mean = float(np.mean(vals)) if n else float("nan")
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 else float("nan")
        half = 1.96 * sem if n > 1 else float("nan")
        out.append({"metric": col, "mean": mean, "std": std, "sem": sem, "n": n,
                    "ci95_low": mean - half, "ci95_high": mean + half})
    return pd.DataFrame(out)


def compare_actual_vs_simulated_waste(
    rows: pd.DataFrame, base_kpis: pd.DataFrame
) -> dict:
    """실측 폐기량(rows.waste_qty) 총합 vs 시뮬 폐기(base_kpis.waste_units) 총합 대조.

    baseline 발주=생산량이므로 시뮬 폐기(생산−복원수요)와 실측 폐기(생산−판매)는
    복원분만큼 구조적으로 다르다. ratio가 1에서 크게 벗어나면 시뮬/복원 가정 재점검 신호.
    """
    actual = float(pd.to_numeric(rows["waste_qty"], errors="coerce").fillna(0.0).sum())
    simulated = float(base_kpis["waste_units"].sum())
    ratio = simulated / actual if actual != 0 else float("nan")
    return {"actual_total": actual, "simulated_total": simulated,
            "ratio": ratio, "n_rows": int(len(rows))}


def characterize_baseline_proxy(rows: pd.DataFrame, waste_report: dict) -> dict:
    """'생산량 ≈ 발주' proxy를 깨는 요인 정량화(실발주 대조는 전향 단계).

    - stockout_share: 생산=판매+폐기 항등식이 복원분만큼 깨지는 item-day 비율
      (모집단: 평가 대상 `rows`)
    - negative_waste_share: 반품/보정으로 clip된 비율(Task 1 report)
      (모집단: `waste_report`가 집계된 소스 인벤토리 전체 — `rows`와 scope가 다름)
    """
    n = int(len(rows))
    stockout_share = float(rows["is_stockout"].astype(bool).mean()) if n else float("nan")
    n_total = waste_report["n_total"]
    neg_share = waste_report["n_negative"] / n_total if n_total else float("nan")
    return {
        "n_item_days": n,
        "stockout_share": stockout_share,
        "negative_waste_share": float(neg_share),
        "carryover_note": (
            "base_order=생산량 proxy. 당일폐기 N 품목 이월·당일 재생산은 미분리 — "
            "전향 실발주 피드 수령 시 select_base_order(source='order_feed')로 교체."
        ),
    }

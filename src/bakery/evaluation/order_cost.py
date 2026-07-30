"""발주 정책(forecaster) 비교용 비용 KPI 프리미티브.

WAPE(center 지표)나 harness의 자기참조적 surplus_rate로는 "어느 발주 정책이
비용상 유리한가"를 답할 수 없다. 이 모듈은 품목유형별 원가율 + 매진시각
시뮬을 결합해 폐기·품절 비용을 KRW로 환산한다.

기존 `business_metrics.CostParams`는 전역 단일 cost_rate(0.30)/margin_rate(0.50)를
쓴다. 이 모듈은 품목유형(매장자체생산/완제품)별로 원가율이 달라 별도로 다루며,
기존 `CostParams` 계약은 수정하지 않는다(기존 소비처 유지).

품목유형 판정 = 품목코드 접두어 4자리(2026-07-30 실측):
    1511 → 매장자체생산, 원가율 0.40
    1513 → 완제품,       원가율 0.60
`수원광교점- 브레드 품목.xlsx`(41품목 정답)과 교차검증해 오분류 0건
(tests/evaluation/test_order_cost.py). 미등록 접두어는 fails-loud(ValueError) —
조용히 기본값을 쓰지 않는다.

비용 산식:
    폐기 1개 비용 = cost_rate × price
    품절 1개 비용 = (1 − cost_rate) × price × absorption_k
`absorption_k`(기본 1.0=완전손실)는 명시적 파라미터다. 이 프로젝트는 카테고리
총량보존(walk-away 0/20, W0 수요이전 흡수 검증)을 실측했으므로 실제 k는 1보다
작을 가능성이 높다 — 호출부에서 스윕해서 확인할 것.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.potential_demand import StoreHours
from .prospective import simulate_item_day_kpis

# 품목코드 접두어(4자리) → 품목유형. 미등록 접두어는 classify_item_kind가 fails-loud.
ITEM_KIND_PREFIXES: dict[str, str] = {
    "1511": "store_produced",
    "1513": "finished_goods",
}
COST_RATE_BY_KIND: dict[str, float] = {
    "store_produced": 0.40,
    "finished_goods": 0.60,
}
DEFAULT_ABSORPTION_K: float = 1.0
DEFAULT_EARLY_STOCKOUT_HOUR: int = 20


def classify_item_kind(item_ids: pd.Series | list) -> pd.Series:
    """품목코드 접두어(4자리) 기반 품목유형 분류. 미등록 접두어는 ValueError."""
    ids = pd.Series(item_ids).astype(str)
    prefixes = ids.str[:4]
    is_known = prefixes.isin(ITEM_KIND_PREFIXES)
    if not is_known.all():
        unknown = sorted(set(prefixes[~is_known]))
        raise ValueError(
            f"미등록 품목코드 접두어: {unknown} — ITEM_KIND_PREFIXES에 등록 필요 "
            "(조용히 기본값을 쓰지 않음, fails-loud)."
        )
    return prefixes.map(ITEM_KIND_PREFIXES)


def cost_rate_for(item_ids: pd.Series | list) -> pd.Series:
    """품목별 원가율(품목유형 매핑)."""
    return classify_item_kind(item_ids).map(COST_RATE_BY_KIND)


def order_cost(
    rows: pd.DataFrame,
    *,
    order_col: str,
    demand_col: str,
    price_col: str,
    absorption_k: float = DEFAULT_ABSORPTION_K,
) -> pd.DataFrame:
    """행별 폐기/품절 비용을 품목유형별 원가율로 계산해 rows에 컬럼 추가.

    rows는 `item_id` 컬럼을 가져야 한다(품목유형 판정용).
    추가 컬럼: waste_units, short_units, waste_cost_krw, lost_margin_krw,
    total_cost_krw.
    """
    out = rows.copy()
    cost_rate = cost_rate_for(out["item_id"])
    order = out[order_col].astype(float)
    demand = out[demand_col].astype(float)
    price = out[price_col].astype(float)
    out["waste_units"] = (order - demand).clip(lower=0.0)
    out["short_units"] = (demand - order).clip(lower=0.0)
    out["waste_cost_krw"] = cost_rate * price * out["waste_units"]
    out["lost_margin_krw"] = (1.0 - cost_rate) * price * out["short_units"] * absorption_k
    out["total_cost_krw"] = out["waste_cost_krw"] + out["lost_margin_krw"]
    return out


def stockout_timing(
    rows: pd.DataFrame,
    profiles: dict[tuple, np.ndarray],
    *,
    order_col: str,
    demand_col: str,
    store_hours: StoreHours,
    group_cols: list[str],
    early_hour: int = DEFAULT_EARLY_STOCKOUT_HOUR,
) -> pd.DataFrame:
    """매진시각/매진여부/조기매진여부 — prospective.simulate_item_day_kpis 재사용.

    simulate_item_day_kpis가 함께 계산하는 waste/lost 비용 컬럼은 전역 단일
    CostParams 기반이라(품목유형별 원가율 미반영) 버리고, soldout_hour/
    is_stockout만 취한다. 비용은 order_cost()로 별도 계산한다.
    """
    kpis = simulate_item_day_kpis(
        rows, profiles, order_col=order_col, store_hours=store_hours,
        group_cols=group_cols, demand_col=demand_col,
    )
    out = rows.copy()
    out["soldout_hour"] = kpis["soldout_hour"].to_numpy()
    out["is_stockout"] = kpis["is_stockout"].to_numpy()
    out["is_early_stockout"] = out["is_stockout"] & (out["soldout_hour"] < early_hour)
    return out


def summarize_order_kpi(
    costed: pd.DataFrame, *, early_hour: int = DEFAULT_EARLY_STOCKOUT_HOUR
) -> dict:
    """비용/매진 KPI 요약.

    costed는 order_cost() + stockout_timing() 출력을 병합한 프레임(또는 동일
    컬럼을 갖는 합성 데이터)이어야 한다. 필요 컬럼: waste_cost_krw,
    lost_margin_krw, total_cost_krw, waste_units, short_units, is_stockout,
    soldout_hour, date.
    """
    is_so = costed["is_stockout"].astype(bool)
    on_so = costed.loc[is_so]
    is_early_all = is_so & (costed["soldout_hour"] < early_hour)
    daily_median = on_so.groupby("date")["soldout_hour"].median() if len(on_so) else pd.Series(dtype=float)
    return {
        "waste_cost_krw": float(costed["waste_cost_krw"].sum()),
        "lost_margin_krw": float(costed["lost_margin_krw"].sum()),
        "total_cost_krw": float(costed["total_cost_krw"].sum()),
        "waste_units": float(costed["waste_units"].sum()),
        "short_units": float(costed["short_units"].sum()),
        "shortfall_mean_on_stockout_days": float(on_so["short_units"].mean()) if len(on_so) else float("nan"),
        "shortfall_mean_all_days": float(costed["short_units"].mean()),
        "early_stockout_rate": float((on_so["soldout_hour"] < early_hour).mean()) if len(on_so) else float("nan"),
        "early_stockout_rate_all": float(is_early_all.mean()),
        "stockout_rate": float(is_so.mean()),
        "soldout_hour_median_mean": float(daily_median.mean()) if len(daily_median) else float("nan"),
    }

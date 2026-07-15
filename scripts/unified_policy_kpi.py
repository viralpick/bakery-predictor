"""통합 발주정책 KPI 비교 — ground truth = 아띠제 실제 판매/발주량.

모든 정책을 **동일 population(같은 item-day)·동일 수요모델(adjusted_demand)·동일 arrival
profile**로 시뮬해 폐기/매진을 비교한다. 기준(reference) = 아띠제 실제 생산량(production_qty).

정책:
- actual_production : 아띠제 실제 생산량(QT_MADE). 기준.
- artisee_reimpl    : 우리가 재구현한 아띠제 로직(3주평균×sold-out×요일).
- our_item_q0.85    : item-level v2 LGBM production-quantile.
- our_cat_quantile  : 총량 q0.85 → 비율 배분.
- our_cat_nk        : 총량 q0.85 + N버퍼 → 배분.
- our_cat_conformal : 총량 median + conformal margin(s) → 배분.

핵심: waste(1차 KPI, 낮을수록 좋음) 우선, stockout(2차) 참고. delta = policy − actual_production;
waste delta 음수 = 아띠제 실생산보다 덜 버림 = 좋음.

실행: uv run python scripts/unified_policy_kpi.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.cli import (
    _real_prospective_inputs, _stockout_item_days,
    _category_order_predictions, _artisee_baseline_order,
)
from bakery.features.potential_demand import StoreHours
from bakery.evaluation.prospective import build_arrival_profile, simulate_item_day_kpis

STORE = "store_gw01"
N_FOLDS = 8
VAL_WEEKS = 8
ALPHA = 0.8
Q = 0.85
DEMAND = "adjusted_demand"
KEYS = ["item_id", "date"]


def _kpi(rows: pd.DataFrame, profiles, order_col: str) -> dict:
    sh = StoreHours(STORE, 8, 22)
    k = simulate_item_day_kpis(
        rows, profiles, order_col=order_col, store_hours=sh,
        group_cols=["item_id"], demand_col=DEMAND,
    )
    so = k["is_stockout"].astype(bool)
    return {
        "waste_krw": float(k["waste_cost_krw"].sum()),
        "lost_margin_krw": float(k["lost_margin_krw"].sum()),
        "stockout_rate": float(so.mean()),
        "soldout_median_h": float(k.loc[so, "soldout_hour"].median()) if so.any() else float("nan"),
    }


def _join_order(rows: pd.DataFrame, preds: pd.DataFrame, col: str) -> pd.Series:
    """preds[item_id,date,our_order] → rows 키에 정렬된 order Series(누락=NaN)."""
    p = preds[KEYS + ["our_order"]].rename(columns={"our_order": col}).copy()
    p["item_id"] = p["item_id"].astype(str)
    p["date"] = pd.to_datetime(p["date"])
    m = rows[KEYS].merge(p, on=KEYS, how="left")
    return m[col].to_numpy()


def main() -> None:
    # 1) rows 조립 (base_order=actual production, our_order=item q0.85). item 경로가 평가창 정의.
    rows, receipts, _ = _real_prospective_inputs(
        STORE, production_quantile=Q, val_weeks=VAL_WEEKS, n_folds=N_FOLDS,
        order_level="item", alpha=ALPHA,
    )
    rows = rows.copy()
    rows["item_id"] = rows["item_id"].astype(str)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["order_actual_production"] = rows["base_order"].to_numpy()
    rows["order_our_item_q085"] = rows["our_order"].to_numpy()
    rows["order_artisee_reimpl"] = _artisee_baseline_order(STORE, rows).to_numpy()

    # 2) 카테고리 정책들(총량→배분)을 같은 키에 join
    cat_specs = {
        "order_our_cat_quantile": dict(margin_method="quantile"),
        "order_our_cat_nk": dict(margin_method="nk", nk_mult=1.0, nk_add=40.0),
    }
    for col, kw in cat_specs.items():
        preds = _category_order_predictions(
            STORE, production_quantile=Q, val_weeks=VAL_WEEKS, n_folds=N_FOLDS, alpha=ALPHA, **kw
        )
        rows[col] = _join_order(rows, preds, col)

    # 3) 결측(정책이 못 채운 item-day) 제거 → 전 정책 동일 population
    order_cols = ["order_actual_production", "order_our_item_q085", "order_artisee_reimpl",
                  "order_our_cat_quantile", "order_our_cat_nk"]
    before = len(rows)
    rows = rows.dropna(subset=order_cols).reset_index(drop=True)
    print(f"공통 population: {len(rows)} / {before} item-days (전 정책 발주 존재)")

    profiles = build_arrival_profile(
        receipts, group_cols=["item_id"],
        exclude_keys=_stockout_item_days(rows), exclude_cols=["item_id", "date"],
    )

    # 4) 정책별 KPI
    ref = _kpi(rows, profiles, "order_actual_production")
    print(f"\n[기준] actual_production: waste={ref['waste_krw']:,.0f} "
          f"stockout={ref['stockout_rate']:.3f} soldout_h={ref['soldout_median_h']:.2f}")
    result_rows = [dict(policy="actual_production", **ref, waste_delta_pct=0.0)]
    for col in order_cols[1:]:
        m = _kpi(rows, profiles, col)
        wd = (m["waste_krw"] - ref["waste_krw"]) / max(ref["waste_krw"], 1) * 100
        result_rows.append(dict(policy=col.replace("order_", ""), **m, waste_delta_pct=wd))
        print(f"{col.replace('order_',''):22s} waste={m['waste_krw']:>12,.0f} "
              f"({wd:+.1f}%)  stockout={m['stockout_rate']:.3f}  soldout_h={m['soldout_median_h']:.2f}")

    out = pd.DataFrame(result_rows)
    out.to_csv("reports/unified_policy_kpi.csv", index=False)
    print("\nwrote reports/unified_policy_kpi.csv")
    print("※ waste_delta_pct 음수 = 아띠제 실생산보다 덜 버림(좋음). stockout 낮을수록 좋음(2차).")


if __name__ == "__main__":
    main()

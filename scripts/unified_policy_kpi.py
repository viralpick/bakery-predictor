"""통합 발주정책 KPI — 측정 헌장(project_measurement_charter) 기준 재구현.

기준(reference) = 아띠제 실제 생산(=발주) QT_MADE(bulk 제외) / 실측 폐기 QT_OUT.
수요 잣대 = adjusted_demand(정상+0.8×마감). potential_demand 폐기.

두 기준을 **병기**(헌장 §6 투명성):
  A. Production 실측  — 폐기=실측 QT_OUT, 매진=실제 폐기0 기반. 아띠제 현행 실제 성적.
  B. 모델 시뮬       — 폐기=max(발주−adjusted,0), 매진=발주<adjusted 기반.
                       actual_production도 B로 한 번 더 시뮬 → A와의 간극 = censoring 크기.

관점(헌장 §4):
  ①(전체매진, CRITICAL) — 빵류 총량 소진.  A: 그날 총폐기 ≤ WASTE_TOL / B: Σ발주 < Σadjusted.
  ②(SKU 품절, 덜 위험)  — 각 날 품절 SKU 비율의 날별 평균.
                          A: 폐기0 SKU / B: 발주<adjusted SKU.

방향성(헌장 §6): adjusted는 실수요 하한 → 모델 폐기(절대)=상한 → **폐기 절감=하한**(보수적).
모델 매진율도 하한(실제 더 나쁠 수 있음). 폐기=1차 KPI, 매진=2차.

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
COST_RATE = 0.30      # business_metrics.CostParams.cost_rate와 일치 (폐기 단위비용 = 판매가×0.30)
PRICE_FALLBACK = 4000.0
WASTE_TOL = 5         # 헌장 §4관점①: 입력오류 감안 그날 총폐기 ≤5도 전체매진 취급


def _waste_krw(units: pd.Series, item_ids: pd.Series, unit_prices: dict) -> float:
    """폐기량(units) → 폐기비용 KRW. A/B 동일 비용식(판매가×cost_rate)으로 비교가능."""
    prices = item_ids.astype(str).map(lambda i: unit_prices.get(i, PRICE_FALLBACK))
    return float((units.clip(lower=0).to_numpy() * prices.to_numpy() * COST_RATE).sum())


def _kpi_actual(rows: pd.DataFrame, unit_prices: dict) -> dict:
    """헌장 §5A — Production 실측(QT_OUT 그대로). 아띠제 현행 실제 성적."""
    waste = pd.to_numeric(rows["waste_qty"], errors="coerce").fillna(0.0).clip(lower=0)
    cat_sellout = rows.assign(_w=waste.to_numpy()).groupby("date")["_w"].sum() <= WASTE_TOL  # 관점①
    # 관점②: 코드베이스 표준 is_stockout(made>0 & waste<=0) — 생산 0인 품목은 매진 아님(가드).
    sku_soldout = rows.groupby("date")["is_stockout"].apply(lambda s: float(s.astype(bool).mean()))
    return {
        "basis": "A_actual",
        "waste_krw": _waste_krw(waste, rows["item_id"], unit_prices),
        "waste_units": float(waste.sum()),
        "cat_sellout_day": float(cat_sellout.mean()),
        "cat_shortfall_on_sellout": float("nan"),
        "cat_undersupply_rate": float("nan"),
        "sku_soldout_rate": float(sku_soldout.mean()),
        "soldout_median_h": float("nan"),
    }


def _kpi_model(rows: pd.DataFrame, profiles, order_col: str, unit_prices: dict) -> dict:
    """헌장 §5B — 모델(발주 vs adjusted_demand 시뮬). 폐기=max(발주−adjusted,0).

    unit_prices를 넘겨 A(_kpi_actual)와 동일한 실제 품목단가 basis로 KRW 계산
    (미전달 시 simulate_profit이 flat 3000원 fallback → A/B 비교 무효)."""
    sh = StoreHours(STORE, 8, 22)
    k = simulate_item_day_kpis(
        rows, profiles, order_col=order_col, store_hours=sh,
        group_cols=["item_id"], demand_col=DEMAND, unit_prices=unit_prices,
    )
    order = pd.to_numeric(rows[order_col], errors="coerce").fillna(0.0).to_numpy()
    dem = pd.to_numeric(rows[DEMAND], errors="coerce").fillna(0.0).to_numpy()
    frame = pd.DataFrame({"date": rows["date"].to_numpy(), "o": order, "d": dem})
    g = frame.groupby("date")
    o_sum, d_sum = g["o"].sum(), g["d"].sum()
    cat_sellout = o_sum < d_sum                                        # 관점①
    shortfall = (d_sum - o_sum).clip(lower=0) / d_sum.clip(lower=1)
    sku_soldout = g.apply(lambda s: float((s["o"] < s["d"]).mean()))   # 관점②(날별 평균)
    so = k["is_stockout"].astype(bool)
    return {
        "basis": "B_sim",
        "waste_krw": float(k["waste_cost_krw"].sum()),
        "waste_units": float(k["waste_units"].sum()),
        "cat_sellout_day": float(cat_sellout.mean()),
        "cat_shortfall_on_sellout": float(shortfall[cat_sellout].mean()) if cat_sellout.any() else 0.0,
        "cat_undersupply_rate": float((d_sum - o_sum).clip(lower=0).sum() / d_sum.sum()),
        "sku_soldout_rate": float(sku_soldout.mean()),
        "soldout_median_h": float(k.loc[so, "soldout_hour"].median()) if so.any() else float("nan"),
    }


def _join_order(rows: pd.DataFrame, preds: pd.DataFrame, col: str) -> np.ndarray:
    """preds[item_id,date,our_order] → rows 키에 정렬된 order(누락=NaN)."""
    p = preds[KEYS + ["our_order"]].rename(columns={"our_order": col}).copy()
    p["item_id"] = p["item_id"].astype(str)
    p["date"] = pd.to_datetime(p["date"])
    m = rows[KEYS].merge(p, on=KEYS, how="left")
    return m[col].to_numpy()


def _assemble_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """공통 population(전 정책 발주 존재) + arrival profile 조립."""
    rows, receipts, unit_prices = _real_prospective_inputs(
        STORE, production_quantile=Q, val_weeks=VAL_WEEKS, n_folds=N_FOLDS,
        order_level="item", alpha=ALPHA,
    )
    rows = rows.copy()
    rows["item_id"] = rows["item_id"].astype(str)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["order_actual_production"] = rows["base_order"].to_numpy()  # QT_MADE(bulk 제외)
    rows["order_artisee_reimpl"] = _artisee_baseline_order(STORE, rows).to_numpy()

    cat_specs = {
        "order_our_cat_quantile": dict(margin_method="quantile"),
        "order_our_cat_nk15": dict(margin_method="nk", nk_mult=1.0, nk_add=15.0),
        "order_our_cat_nk30": dict(margin_method="nk", nk_mult=1.0, nk_add=30.0),
        "order_our_cat_conformal": dict(margin_method="conformal", service_level=0.85, cal_fold_frac=0.5),
    }
    for col, kw in cat_specs.items():
        preds = _category_order_predictions(
            STORE, production_quantile=Q, val_weeks=VAL_WEEKS, n_folds=N_FOLDS, alpha=ALPHA, **kw
        )
        rows[col] = _join_order(rows, preds, col)

    order_cols = ["order_actual_production", "order_artisee_reimpl", "order_our_cat_quantile",
                  "order_our_cat_nk15", "order_our_cat_nk30", "order_our_cat_conformal"]
    before = len(rows)
    rows = rows.dropna(subset=order_cols).reset_index(drop=True)
    print(f"공통 population: {len(rows)} / {before} item-days (conformal test창, 전 정책 발주 존재)")

    profiles = build_arrival_profile(
        receipts, group_cols=["item_id"],
        exclude_keys=_stockout_item_days(rows), exclude_cols=["item_id", "date"],
    )
    return rows, profiles, unit_prices, order_cols


def _fmt(name: str, m: dict, wd: float) -> str:
    sh = "" if np.isnan(m["cat_shortfall_on_sellout"]) else \
        f"(부족{m['cat_shortfall_on_sellout']*100:.1f}%/총{m['cat_undersupply_rate']*100:.1f}%)"
    return (f"{name:26s} [{m['basis']:8s}] 폐기={m['waste_krw']:>11,.0f}원({wd:+6.1f}%) "
            f"매진①={m['cat_sellout_day']:.3f}{sh} 매진②={m['sku_soldout_rate']:.3f}")


def _delta(w: float, ref_w: float) -> float:
    return (w - ref_w) / max(ref_w, 1) * 100


def main() -> None:
    rows, profiles, unit_prices, order_cols = _assemble_rows()

    # A: 아띠제 실측(현행, QT_OUT). B: 아띠제 발주(QT_MADE)를 시뮬 → A와의 간극 = censoring.
    ref_a = _kpi_actual(rows, unit_prices)                                       # 폐기 delta 기준(헌장: 현행=실측)
    ref_b = _kpi_model(rows, profiles, "order_actual_production", unit_prices)    # 동일basis 공정비교 기준
    result_rows = [
        dict(policy="actual_production", **ref_a,
             waste_delta_vs_actualA_pct=0.0, waste_delta_vs_simB_pct=float("nan")),
        dict(policy="actual_production_sim", **ref_b,
             waste_delta_vs_actualA_pct=_delta(ref_b["waste_krw"], ref_a["waste_krw"]),
             waste_delta_vs_simB_pct=0.0),
    ]
    for col in order_cols[1:]:                                                    # 모델 + artisee_reimpl (전부 B)
        m = _kpi_model(rows, profiles, col, unit_prices)
        result_rows.append(dict(
            policy=col.replace("order_", ""), **m,
            waste_delta_vs_actualA_pct=_delta(m["waste_krw"], ref_a["waste_krw"]),
            waste_delta_vs_simB_pct=_delta(m["waste_krw"], ref_b["waste_krw"]),
        ))

    # 출력 — advisor #2: 폐기(1차) / 매진(2차, 동일 B basis) / censoring(A vs B) 패널 분리.
    print("\n=== 폐기 (1차 KPI) — Δ는 현행 실측A / 현행시뮬B 둘 다 ===")
    print(f"{'policy':26s} {'basis':8s} {'폐기KRW':>12s}  ΔvsA(하한)  ΔvsB(공정)")
    for r in result_rows:
        da = "  ref" if r["policy"] == "actual_production" else f"{r['waste_delta_vs_actualA_pct']:+7.1f}%"
        db = "" if np.isnan(r["waste_delta_vs_simB_pct"]) else \
            ("  ref" if r["policy"] == "actual_production_sim" else f"{r['waste_delta_vs_simB_pct']:+7.1f}%")
        print(f"{r['policy']:26s} {r['basis']:8s} {r['waste_krw']:>12,.0f}  {da:>9s}  {db:>9s}")

    print("\n=== 매진 (2차 KPI) — 동일 B basis(발주<adjusted)로만 비교. actual A는 아래 censoring 패널 ===")
    print(f"{'policy':26s} {'매진①(전체)':>12s} {'매진②(SKU)':>11s}")
    for r in result_rows:
        if r["policy"] == "actual_production":
            continue  # A basis (폐기0) — B와 같은 칸에 두지 않음
        print(f"{r['policy']:26s} {r['cat_sellout_day']:>12.3f} {r['sku_soldout_rate']:>11.3f}")

    print("\n=== censoring 패널 — 현행 actual A(실측 폐기0) vs B(시뮬 발주<adjusted) ===")
    print(f"  매진①  A={ref_a['cat_sellout_day']:.3f}  B={ref_b['cat_sellout_day']:.3f}")
    print(f"  매진②  A={ref_a['sku_soldout_rate']:.3f}  B={ref_b['sku_soldout_rate']:.3f}"
          "   ← 실제 매진이 시뮬보다 훨씬 큼 = censoring")

    out = pd.DataFrame(result_rows)
    out.to_csv("reports/unified_policy_kpi.csv", index=False)
    print("\nwrote reports/unified_policy_kpi.csv")
    print("※ 폐기 음수%=현행보다 덜 버림(좋음). adjusted=실수요 하한 → 모델폐기=상한 → 절감=하한(보수적).")
    print("※ 매진 모델 basis=발주<adjusted(헌장 §5B, 단 aggregation은 날별평균 — pooled 아님).")


if __name__ == "__main__":
    main()

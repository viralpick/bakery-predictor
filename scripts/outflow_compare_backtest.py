"""F3 — Compare four outflow strategies for the censored-demand correction.

Hypothesis: the choice of outflow_ratio (per-item substitution shrink factor)
materially changes potential_demand on stockout days, which propagates to
v2/v3 forecasts. We re-attach potential_demand four ways on the same daily
frame, then run the standard 4-fold rolling backtest for each.

Four modes:
  - rd     : current production behavior — per-item RD outflow (mean 0.6)
  - mnl    : uniform OUTFLOW_CAP=0.7 from MNL/IIA limit
  - none   : outflow=0 (full censoring correction, no substitution credit)
  - weak   : uniform 0.3 (modest substitution credit)

Outputs:
  reports/outflow_compare_summary.csv  — WAPE/pct_under by mode × model
  reports/outflow_compare_folds.csv    — per-fold detail
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from bakery.analysis.mnl_substitution import fit_mnl_per_category
from bakery.analysis.substitution import compute_substitution_matrix
from bakery.cli import _build_forecasters, _enrich_if_needed, _load_dataset, _parse_variants
from bakery.data.bonavi_loader import (
    DEFAULT_STORE_CODE,
    XLSX_DEFAULT,
    load_sales,
    measure_hour_profile,
)
from bakery.evaluation.backtest import run_backtest
from bakery.evaluation.split import generate_time_splits
from bakery.features.potential_demand import StoreHours, attach_potential_demand


def main() -> None:
    # Load the full DailyDataset (real source) — keeps external sources hooked
    # in so `_enrich_if_needed` can attach competitor/calendar/weather/etc.
    base_ds = _load_dataset("real", None)
    daily = base_ds.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    receipts = pd.read_parquet("data/internal/bonavi_receipts.parquet")
    receipts["date"] = pd.to_datetime(receipts["date"])

    sales = load_sales(XLSX_DEFAULT, store_code=DEFAULT_STORE_CODE)
    measured_profiles = measure_hour_profile(sales)
    measured_profiles = {"store_gw01": next(iter(measured_profiles.values()))}

    rd = compute_substitution_matrix(daily, receipts, include_inter_category=True)
    mnl = fit_mnl_per_category(receipts, daily)

    weak = pd.Series(0.3, index=rd.outflow_ratio.index, name="outflow_ratio")
    outflow_modes = {
        "rd": rd.outflow_ratio,
        "mnl": mnl.outflow_ratio,
        "none": pd.Series(dtype=float),
        "weak": weak,
    }

    store_hours = [StoreHours(store_id="store_gw01", open_hour=7, close_hour=22)]
    windows = generate_time_splits(daily["date"], n_splits=4, val_horizon_days=7, step_days=7)
    variants = _parse_variants("v0,v1,v2")

    summary_rows = []
    fold_rows = []
    for mode, outflow in outflow_modes.items():
        print(f"\n=== mode={mode} ===")
        if outflow.empty:
            daily_mode = attach_potential_demand(daily, store_hours, measured_profiles=measured_profiles)
        else:
            daily_mode = attach_potential_demand(
                daily, store_hours,
                outflow_ratio=outflow,
                measured_profiles=measured_profiles,
            )
        # Stockout-day potential_demand stats — how much did this outflow change?
        sd = daily_mode[daily_mode["is_stockout"]]
        print(f"  stockout days: {len(sd)}, mean potential/sold ratio: "
              f"{(sd['potential_demand']/sd['sold_units'].replace(0,1)).mean():.3f}")
        ds = dataclasses.replace(base_ds, daily=daily_mode)
        enriched = _enrich_if_needed(ds, variants)
        forecasters = _build_forecasters(variants)
        fold_df, _ = run_backtest(enriched, forecasters, windows)
        fold_df["mode"] = mode
        fold_rows.append(fold_df)
        agg = fold_df.groupby("model")[
            ["wape_all", "wape_no_stockout", "pct_underpredict", "pct_overpredict"]
        ].mean()
        agg["mode"] = mode
        agg = agg.reset_index()
        summary_rows.append(agg)
        print(agg.to_string(index=False))

    summary = pd.concat(summary_rows, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    summary.to_csv("reports/outflow_compare_summary.csv", index=False)
    folds.to_csv("reports/outflow_compare_folds.csv", index=False)
    print("\nwrote reports/outflow_compare_*.csv")

    print("\n=== final summary (lower wape = better) ===")
    pivot = summary.pivot(index="model", columns="mode", values="wape_all")
    print(pivot.round(4))
    print("\n=== pct_underpredict by mode (lower = production safer) ===")
    pivot_u = summary.pivot(index="model", columns="mode", values="pct_underpredict")
    print(pivot_u.round(4))


if __name__ == "__main__":
    main()

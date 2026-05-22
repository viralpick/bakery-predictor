"""F3 quick — 2 outflow modes × v2 model only, 2 folds.

The full 4-mode × 3-model × 4-fold backtest was hitting a stdout-buffer hang
and taking >17 min. This stripped version covers the extreme contrast that
actually matters (rd vs none) to demonstrate the *qualitative* shape of the
plug-in effect, without the long tail of marginal variants.
"""

from __future__ import annotations

import dataclasses
import sys
import time

import pandas as pd

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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("loading...")
    base_ds = _load_dataset("real", None)
    daily = base_ds.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    receipts = pd.read_parquet("data/internal/bonavi_receipts.parquet")
    receipts["date"] = pd.to_datetime(receipts["date"])

    sales = load_sales(XLSX_DEFAULT, store_code=DEFAULT_STORE_CODE)
    measured_profiles = measure_hour_profile(sales)
    measured_profiles = {"store_gw01": next(iter(measured_profiles.values()))}

    log("computing RD outflow...")
    rd = compute_substitution_matrix(daily, receipts, include_inter_category=True)
    log(f"  RD outflow mean: {rd.outflow_ratio.mean():.3f}, median: {rd.outflow_ratio.median():.3f}")

    modes = {"rd": rd.outflow_ratio, "none": pd.Series(dtype=float)}
    store_hours = [StoreHours(store_id="store_gw01", open_hour=7, close_hour=22)]

    log("splitting...")
    windows = generate_time_splits(daily["date"], n_splits=2, val_horizon_days=7, step_days=7)
    log(f"  {len(windows)} folds")

    variants = _parse_variants("v0,v2")
    log(f"  variants: {variants}")

    summary_rows = []
    fold_rows = []
    for mode, outflow in modes.items():
        log(f"=== mode={mode} ===")
        t0 = time.time()
        if outflow.empty:
            daily_mode = attach_potential_demand(daily, store_hours, measured_profiles=measured_profiles)
        else:
            daily_mode = attach_potential_demand(
                daily, store_hours, outflow_ratio=outflow, measured_profiles=measured_profiles,
            )
        sd = daily_mode[daily_mode["is_stockout"]]
        log(f"  attached in {time.time()-t0:.1f}s, mean potential/sold ratio: "
            f"{(sd['potential_demand']/sd['sold_units'].replace(0,1)).mean():.3f}")

        t0 = time.time()
        ds = dataclasses.replace(base_ds, daily=daily_mode)
        enriched = _enrich_if_needed(ds, variants)
        log(f"  enriched in {time.time()-t0:.1f}s, rows={len(enriched)}")

        t0 = time.time()
        forecasters = _build_forecasters(variants)
        fold_df, _ = run_backtest(enriched, forecasters, windows)
        log(f"  backtest in {time.time()-t0:.1f}s")
        fold_df["mode"] = mode
        fold_rows.append(fold_df)
        agg = fold_df.groupby("model")[
            ["wape_all", "wape_no_stockout", "pct_underpredict", "pct_overpredict"]
        ].mean()
        agg["mode"] = mode
        agg = agg.reset_index()
        summary_rows.append(agg)
        log(agg.to_string(index=False))

    summary = pd.concat(summary_rows, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    summary.to_csv("reports/outflow_compare_summary.csv", index=False)
    folds.to_csv("reports/outflow_compare_folds.csv", index=False)
    log("wrote reports/outflow_compare_*.csv")

    log("=== WAPE pivot (lower = better) ===")
    pivot = summary.pivot(index="model", columns="mode", values="wape_all")
    log("\n" + pivot.round(4).to_string())
    log("=== pct_underpredict pivot (lower = production safer) ===")
    pivot_u = summary.pivot(index="model", columns="mode", values="pct_underpredict")
    log("\n" + pivot_u.round(4).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

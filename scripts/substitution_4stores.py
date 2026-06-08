"""4매장 substitution 4방법 비교 — 광교 단독 분석을 매장별로 일반화.

방법:
  1) RD (compute_substitution_matrix)      — 매진 cutoff 회귀불연속, outflow/sub_rate
  2) MNL (fit_mnl_per_category)            — 단일품목 영수증 conditional logit, s_share
  3) Nested logit (fit_nested_logit)       — nest별 λ_g (1=IIA/약한 대체, <1=강한 대체)
  4) DiD (compute_did_substitution)        — 시간대 hourly DiD, β_did(대체효과)

입력 빌드(매장별):
  daily    = interval_backtest_4stores.build_store_daily(store_cd, store_id, exclude_bulk=False)
             (전 카테고리 item-level: item_id/category_id/sold_units/is_stockout/stockout_time)
  receipts = sales.parquet → (receipt_id, date, item_id, hour, minute, timestamp)
  profiles = receipt 시간분포 length-24 (measure_hour_profile 동일 로직)

광교는 과거 bonavi_receipts.parquet(450,606행) 대신 sales.parquet에서 동일 규칙으로
재산출 → 4매장 내부 일관 비교. 절대값은 과거 광교 단독 run과 미세하게 다를 수 있음.

산출물(reports/substitution_4stores/):
  summary.csv                 — 매장 × 4방법 headline 지표 비교
  <store>_<method>.csv        — 매장별 raw 결과
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.analysis.mnl_substitution import fit_mnl_per_category
from bakery.analysis.nested_logit import fit_nested_logit
from bakery.analysis.substitution import compute_substitution_matrix
from bakery.analysis.substitution_did import compute_did_substitution
from bakery.data.bonavi_loader import measure_hour_profile
from store_daily import STORE_MAP, build_store_daily
from v4_new_data_backtest import V2

OUT_DIR = Path("reports/substitution_4stores")


def build_store_receipts(store_cd: str) -> pd.DataFrame:
    """sales.parquet → bonavi_receipts 스키마 (receipt_id, date, item_id, hour, minute, timestamp)."""
    sales = pd.read_parquet(
        V2 / "sales.parquet",
        columns=[
            "CD_PARTNER", "DT_SALE", "NO_POS", "SLIP_NO", "CD_ITEM",
            "SALES_FG", "CD_USERDEF2", "SALES_TIME",
        ],
    )
    sales = sales[sales["CD_PARTNER"].astype(str) == store_cd]
    sales = sales[sales["SALES_FG"].astype(str) == "0"]
    sales = sales[sales["CD_USERDEF2"].astype(str) == "SS"]
    sales["date"] = pd.to_datetime(sales["DT_SALE"].astype(str))
    sales["item_id"] = sales["CD_ITEM"].astype(str)
    st = sales["SALES_TIME"].astype(str).str.zfill(14)
    sales["hour"] = pd.to_numeric(st.str[8:10], errors="coerce")
    sales["minute"] = pd.to_numeric(st.str[10:12], errors="coerce")
    sales["timestamp"] = pd.to_datetime(st, format="%Y%m%d%H%M%S", errors="coerce")
    sales["receipt_id"] = (
        sales["CD_PARTNER"].astype(str) + "_" + sales["DT_SALE"].astype(str)
        + "_" + sales["NO_POS"].astype(str) + "_" + sales["SLIP_NO"].astype(str)
    )
    out = sales[["receipt_id", "date", "item_id", "hour", "minute", "timestamp"]]
    return out.dropna(subset=["hour"]).reset_index(drop=True)


def build_profiles(receipts: pd.DataFrame, store_id: str) -> dict[str, np.ndarray]:
    s = receipts[["hour"]].copy()
    s["store_id"] = store_id
    s["qty"] = 1.0
    return measure_hour_profile(s)


def _sig(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def summarize_store(name: str, rd, mnl, nl, did) -> dict:
    rd_within = rd.coefficients[rd.coefficients["same_category"]]
    did_sig_pos = (
        ((did.coefficients["p_value"] < 0.05) & (did.coefficients["beta_did"] > 0)).mean()
        if len(did.coefficients) else float("nan")
    )
    within = nl.substitution[nl.substitution["same_nest"]]
    cross = nl.substitution[~nl.substitution["same_nest"]]
    wc_ratio = (
        within["s_share"].mean() / cross["s_share"].mean()
        if len(cross) and cross["s_share"].mean() > 0 else float("nan")
    )
    return {
        "store": name,
        "RD_mean_sub_rate_within": _sig(rd_within["sub_rate"]),
        "RD_mean_outflow": _sig(rd.outflow_ratio),
        "MNL_mean_s_share": _sig(mnl.substitution["s_share"]),
        "MNL_mean_outflow": _sig(mnl.outflow_ratio),
        "Nested_mean_lambda": _sig(nl.lambdas),
        "Nested_within_over_cross": wc_ratio,
        "DiD_mean_beta": _sig(did.coefficients["beta_did"]),
        "DiD_pct_sig_pos": did_sig_pos,
        "DiD_mean_outflow": _sig(did.outflow_ratio),
    }


def run_store(store_cd: str) -> dict:
    name, store_id = STORE_MAP[store_cd]
    print(f"\n========== [{name}] ==========")
    daily = build_store_daily(store_cd, store_id, exclude_bulk=False)
    receipts = build_store_receipts(store_cd)
    profiles = build_profiles(receipts, store_id)
    print(f"  daily {len(daily):,}행 / receipts {len(receipts):,}행 / "
          f"고유영수증 {receipts['receipt_id'].nunique():,}")

    print("  [1/4] RD substitution matrix ...")
    rd = compute_substitution_matrix(
        daily, receipts, include_inter_category=False, hour_profiles=profiles
    )
    print("  [2/4] MNL per-category ...")
    mnl = fit_mnl_per_category(receipts, daily)
    print("  [3/4] Nested logit ...")
    nl = fit_nested_logit(receipts, daily)
    print("  [4/4] DiD ...")
    did = compute_did_substitution(daily, receipts, profiles)

    rd.coefficients.to_csv(OUT_DIR / f"{name}_rd.csv", index=False)
    mnl.substitution.to_csv(OUT_DIR / f"{name}_mnl.csv", index=False)
    nl.substitution.to_csv(OUT_DIR / f"{name}_nested.csv", index=False)
    nl.lambdas.to_csv(OUT_DIR / f"{name}_nested_lambdas.csv")
    did.coefficients.to_csv(OUT_DIR / f"{name}_did.csv", index=False)
    return summarize_store(name, rd, mnl, nl, did)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_store(cd) for cd in STORE_MAP]
    summary = pd.DataFrame(rows).set_index("store")
    summary.to_csv(OUT_DIR / "summary.csv")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n" + "=" * 80)
    print("4매장 substitution 비교 (headline)")
    print("=" * 80)
    print(summary.round(4).to_string())
    print(f"\nsaved {OUT_DIR}/summary.csv (+ 매장별 raw csv)")


if __name__ == "__main__":
    main()

"""G2 재검증: is_stockout/stockout_time 재정의가 Stage2 품목비율 인기 신호를
바꾸는가 (하위2 ②).

통제 비교: build_store_daily(옛 정의: 첫 순간품절 이벤트, is_stockout 92%) 산출물의
sold_units/closing 은 고정하고, absorption_4stores.apply_fixed_stockout 로 처치변수
(is_stockout/stockout_time)만 새 정의(폐기0=완판, 마지막 실판매)로 override 한 뒤,
item_proportion.compute_proportions(라이브 Stage2 배선)와 popularity 추천을 옛/새로
각각 돌려 avg_stockout_h→stockout_rank_pct→adj_stockout→proportion 차이를 잰다.

Y(sold_units)·closing 동일 → stockout_time 재정의 효과만 격리.

실행: PYTHONPATH=scripts uv run python scripts/revalidate_popularity_stockout.py
산출: reports/revalidate_g2/popularity_stockout_{store}.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bakery.models.item_proportion import STOCKOUT_MAX_BOOST, compute_proportions
from absorption_4stores import apply_fixed_stockout
from store_daily import STORE_MAP, build_store_closing_rows, build_store_daily

OUT_DIR = Path("reports/revalidate_g2")
CUTOFF = pd.Timestamp("2025-12-31")   # base=최근28일, stockout_h=cutoff 이전 전체
GENERAL_CATEGORIES = ("bread", "pastry")


def _with_closing(daily: pd.DataFrame, store_cd: str) -> pd.DataFrame:
    """history 에 closing_qty 부여 (item_proportion 옵션 입력, 옛/새 동일)."""
    closing = build_store_closing_rows(store_cd)[["date", "item_id", "qty"]]
    closing = closing.rename(columns={"qty": "closing_qty"})
    out = daily.merge(closing, on=["date", "item_id"], how="left")
    out["closing_qty"] = out["closing_qty"].fillna(0)
    return out


def _proportions(daily: pd.DataFrame) -> pd.DataFrame:
    props = compute_proportions(daily, CUTOFF)
    return props.set_index("item_id")


def compare_store(store_cd: str, store_id: str) -> pd.DataFrame:
    old_daily = _with_closing(build_store_daily(store_cd, store_id, exclude_bulk=True), store_cd)
    # apply_fixed_stockout 는 sold_units/closing_qty 를 보존하고 처치변수만 override.
    new_daily = apply_fixed_stockout(old_daily, store_cd, exclude_bulk=True)

    old_p = _proportions(old_daily)
    new_p = _proportions(new_daily)

    keep = ["category_id", "avg_stockout_h", "adj_stockout", "proportion"]
    cmp = old_p[keep].join(new_p[keep], lsuffix="_old", rsuffix="_new", how="outer")
    cmp["store_id"] = store_id
    cmp["prop_delta"] = cmp["proportion_new"] - cmp["proportion_old"]
    cmp["adj_delta"] = cmp["adj_stockout_new"] - cmp["adj_stockout_old"]
    return cmp.reset_index()


def _rank_pct(series: pd.Series) -> pd.Series:
    """item_proportion._classify_signals 와 동일: ascending, NaN→1.0."""
    return series.rank(pct=True, ascending=True).fillna(1.0)


def summarize(cmp: pd.DataFrame, store_id: str) -> dict:
    gen = cmp[cmp["category_id_new"].fillna(cmp["category_id_old"]).isin(GENERAL_CATEGORIES)].copy()
    # per-category rank_pct 재현 후 Spearman (옛 vs 새 순서 정렬 상관)
    rhos = []
    for _, g in gen.groupby(gen["category_id_new"].fillna(gen["category_id_old"])):
        old_r = _rank_pct(g["avg_stockout_h_old"])
        new_r = _rank_pct(g["avg_stockout_h_new"])
        if g["avg_stockout_h_old"].notna().sum() >= 4 and g["avg_stockout_h_new"].notna().sum() >= 4:
            rho, _ = spearmanr(old_r, new_r)
            rhos.append(rho)
    n_items = len(gen)
    nan_old = int(gen["avg_stockout_h_old"].isna().sum())
    nan_new = int(gen["avg_stockout_h_new"].isna().sum())
    return {
        "store_id": store_id,
        "n_general_items": n_items,
        "spearman_rank_pct": float(np.nanmean(rhos)) if rhos else float("nan"),
        "nan_avg_h_old": nan_old,
        "nan_avg_h_new": nan_new,
        "mean_abs_prop_delta": float(gen["prop_delta"].abs().mean()),
        "max_abs_prop_delta": float(gen["prop_delta"].abs().max()),
        "mean_abs_adj_delta": float(gen["adj_delta"].abs().mean()),
        "max_adj_boost": float(1 + STOCKOUT_MAX_BOOST),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for store_cd, (name, store_id) in STORE_MAP.items():
        cmp = compare_store(store_cd, store_id)
        cmp.to_csv(OUT_DIR / f"popularity_stockout_{store_id}.csv", index=False)
        s = summarize(cmp, store_id)
        s["name"] = name
        summaries.append(s)
        print(f"\n=== {name}({store_id}) — bread/pastry ===")
        print(f"  품목수={s['n_general_items']}  "
              f"Spearman(rank_pct old vs new)={s['spearman_rank_pct']:+.3f}")
        print(f"  avg_stockout_h NaN: old={s['nan_avg_h_old']} → new={s['nan_avg_h_new']} "
              f"(new=매진 안 하는 품목 → boost 0)")
        print(f"  proportion |Δ|: mean={s['mean_abs_prop_delta']*100:.3f}%p  "
              f"max={s['max_abs_prop_delta']*100:.3f}%p")
        print(f"  adj_stockout |Δ|: mean={s['mean_abs_adj_delta']:.4f} (boost 범위 1.0~{s['max_adj_boost']:.2f})")

    pd.DataFrame(summaries).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nwrote {OUT_DIR}/summary.csv + per-store csv")


if __name__ == "__main__":
    main()

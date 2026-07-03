"""4매장 카테고리 총량 수요이전 흡수 검증 (W0 게이트 다매장 확장).

store_daily.build_store_daily 로 4매장(광교/삼성타운/메세나폴리스/광화문) item-level
daily 를 만들어 하나로 합친 뒤, src.bakery.analysis.demand_absorption.run_absorption
(광교 단독에서 검증·머지된 동일 로직)에 태운다. placebo(미래 d+7 품절강도)도 함께.

핵심: run_absorption 은 store 루프라 4매장×카테고리를 자동 처리. 광교는 bonavi_daily
경로(PR#18)와 store_daily 경로 두 개가 되므로 부호/판정 일관성 sanity check 가능.

실행: PYTHONPATH=scripts uv run python scripts/absorption_4stores.py
산출: reports/demand_absorption/results_4stores.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bakery.analysis.demand_absorption import (
    build_absorption_panel,
    fit_absorption,
    run_absorption,
)
from store_daily import STORE_MAP, build_store_daily

OUT_DIR = Path("reports/demand_absorption")
GENERAL_CATEGORIES = ("bread", "pastry")   # 게이트 판정 대상 (단일품목/시즌 제외)


def build_all_stores_daily() -> pd.DataFrame:
    """4매장 daily 를 하나로 합친다. bulk(대량예약)는 흡수와 무관하므로 제외."""
    frames = []
    for store_cd, (name, store_id) in STORE_MAP.items():
        daily = build_store_daily(store_cd, store_id, exclude_bulk=True)
        frames.append(daily)
        print(f"  {name}({store_id}): {len(daily):,} rows, "
              f"{daily['category_id'].nunique()} cats, "
              f"{daily['item_id'].nunique()} items")
    return pd.concat(frames, ignore_index=True)


def placebo_results(daily: pd.DataFrame) -> list:
    """미래(d+7) 품절강도로 회귀 — 허위상관/잔차 confound 크기 하한."""
    panel = build_absorption_panel(daily).sort_values("date")
    panel["stockout_hours"] = (panel.groupby(["store_id", "category_id"])["stockout_hours"]
                               .shift(-7))
    panel = panel.dropna(subset=["stockout_hours"])
    out = []
    for store_id, category_id in panel[["store_id", "category_id"]].drop_duplicates().itertuples(index=False):
        res = fit_absorption(panel, store_id, category_id)
        if res is not None:
            out.append(res)
    return out


def _print_table(title: str, results: list) -> None:
    print(f"\n{title}")
    for r in sorted(results, key=lambda x: (x.store_id, x.category_id)):
        flag = "★" if r.category_id in GENERAL_CATEGORIES else " "
        print(f" {flag} {r.store_id}/{r.category_id:9s} β={r.beta:+.3f} "
              f"CI90[{r.ci_low:+.3f},{r.ci_high:+.3f}] δ={r.delta:.3f} "
              f"{r.verdict:12s} (n={r.n})")


def main() -> None:
    print("4매장 daily 빌드 (exclude_bulk=True):")
    daily = build_all_stores_daily()

    results = run_absorption(daily)
    _print_table("=== 실제 (매장×카테고리) — ★=게이트 대상(bread/pastry) ===", results)

    placebo = placebo_results(daily)
    _print_table("=== placebo (미래 d+7 품절강도) ===", placebo)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([{**r.__dict__, "arm": "real"} for r in results]
                        + [{**r.__dict__, "arm": "placebo"} for r in placebo])
    rows.to_csv(OUT_DIR / "results_4stores.csv", index=False)
    print(f"\nwrote {OUT_DIR}/results_4stores.csv ({len(rows)} rows)")

    # 게이트 요약: 일반 카테고리 walk-away 유무
    general = [r for r in results if r.category_id in GENERAL_CATEGORIES]
    walkaways = [r for r in general if r.verdict == "walkaway"]
    print(f"\n게이트 요약: 일반 카테고리 {len(general)}건 중 "
          f"walk-away {len(walkaways)}건, absorb {sum(r.verdict=='absorb' for r in general)}건")
    if walkaways:
        print("  ⚠️ walk-away 발견:", [(r.store_id, r.category_id) for r in walkaways])


if __name__ == "__main__":
    main()

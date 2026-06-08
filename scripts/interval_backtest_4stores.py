"""[DEPRECATED 2026-06-08] v5 conformal 구간(범위)예측 — 폐기. 점추정+품절/매진 위험수치로 전환 (docs/poc_scope_v6.md §8 D1). 데이터 검증 단계 산출물로만 보존.

4매장 v5 대칭 conformal 95% 구간예측 + 실제 일생산량(QT_MADE) overlay.

광교 단독 plot_interval_symmetric95.py 를 4매장으로 일반화한다.
- 발주구간/수요 target: bulk 제외 + α=0.7 보정 + bread/pastry/sandwich 합 (canonical).
- 파란 선: inventory.parquet QT_MADE (동일 카테고리 총 생산, bulk/α 미적용).
  → 생산은 수요 target 과 정의가 달라(과잉생산 포함) 보통 구간 위에 위치한다.

빌드 경로는 v4_new_data_backtest 의 광교 canonical 을 store_cd 로 매개화해 재사용.
conformal backtest 는 interval_backtest.run 을 그대로 호출.

caveat: filter_seasonal 의 광교 시즌 품목 제외는 store_id=='store_gw01' 에서만 적용.
타 3매장은 시즌 제외 없이(no-op) 진행 — 매장간 비대칭이지만 광교 canonical 을 보존.

산출물(reports/interval_4stores/):
  interval_4stores.png            — 2x2 매장별 구간+실측+생산 overlay
  interval_4stores_predictions.csv — 매장별 symmetric@95 test 행
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from bakery.features.category_aggregate import (
    TARGET_CATEGORIES,
    build_category_daily,
    build_features,
)
from interval_backtest import run
from store_daily import (
    STORE_MAP,
    build_store_closing_rows,
    build_store_daily,
    item_category_map,
)
from v4_new_data_backtest import ALPHA, TARGET_COL, V2

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path("reports/interval_4stores")


def _item_category_map() -> dict[str, str]:
    return item_category_map()


def build_store_features(store_cd: str, store_id: str) -> pd.DataFrame:
    daily_raw = build_store_daily(store_cd, store_id, exclude_bulk=True)
    closing = build_store_closing_rows(store_cd)
    cd = build_category_daily(daily_raw=daily_raw, discount_rows=closing, alpha=ALPHA)
    return build_features(cd, target_col=TARGET_COL)


def build_production_daily(store_cd: str) -> pd.DataFrame:
    """inventory.parquet → 해당 매장 TARGET_CATEGORIES 일별 총 생산량(QT_MADE)."""
    inv = pd.read_parquet(V2 / "inventory.parquet")
    inv = inv[inv["CD_PARTNER"].astype(str) == store_cd].copy()
    inv["date"] = pd.to_datetime(inv["DT_SALE"].astype(str))
    inv["item_id"] = inv["CD_ITEM"].astype(str)
    inv["QT_MADE"] = pd.to_numeric(inv["QT_MADE"], errors="coerce").fillna(0)
    cat_map = _item_category_map()
    inv["category_id"] = inv["item_id"].map(cat_map).fillna("etc")
    inv = inv[inv["category_id"].isin(TARGET_CATEGORIES)]
    prod = inv.groupby("date")["QT_MADE"].sum().reset_index()
    prod = prod.rename(columns={"QT_MADE": "production"})
    return prod


def backtest_store(store_cd: str) -> pd.DataFrame:
    name, store_id = STORE_MAP[store_cd]
    print(f"\n===== [{name}] features build =====")
    feats = build_store_features(store_cd, store_id)
    preds = run(
        feats,
        n_folds=4,
        min_train_days=365,
        calibration_days=210,
        horizon_days=30,
    )
    sub = preds[(preds["variant"] == "symmetric") & (preds["coverage_level"] == 0.95)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)
    prod = build_production_daily(store_cd)
    sub = sub.merge(prod, on="date", how="left")
    sub["store"] = name
    return sub


def plot_store(ax, sub: pd.DataFrame, name: str) -> None:
    within = (sub["actual"] >= sub["lower"]) & (sub["actual"] <= sub["upper"])
    coverage = within.mean()
    mean_width = (sub["upper"] - sub["lower"]).mean()
    over_upper = (sub["actual"] > sub["upper"]).mean()

    ax.fill_between(
        sub["date"], sub["lower"], sub["upper"],
        color="#4C72B0", alpha=0.18, label="95% 예측구간 (대칭)",
    )
    ax.plot(sub["date"], sub["upper"], color="#4C72B0", lw=0.7, alpha=0.5)
    ax.plot(sub["date"], sub["lower"], color="#4C72B0", lw=0.7, alpha=0.5)
    ax.plot(
        sub["date"], sub["anchor"], color="#C44E52", lw=1.6,
        label="예측값 (q0.90 앵커 = 권장 발주점)",
    )
    if sub["production"].notna().any():
        ax.plot(
            sub["date"], sub["production"], color="#1f77b4", lw=1.4, alpha=0.85,
            label="실제 생산량 (QT_MADE, 동일 카테고리)",
        )
    ax.scatter(
        sub.loc[within, "date"], sub.loc[within, "actual"],
        color="#2A2A2A", s=18, zorder=5, label="실측 (보정수요, 구간 내)",
    )
    ax.scatter(
        sub.loc[~within, "date"], sub.loc[~within, "actual"],
        color="#D62728", s=34, marker="X", zorder=6, label="실측 (구간 벗어남)",
    )

    fold_change = sub.index[sub["fold"].ne(sub["fold"].shift())].tolist()[1:]
    for idx in fold_change:
        ax.axvline(sub["date"].iloc[idx], color="gray", ls=":", lw=0.7, alpha=0.5)

    ax.set_title(
        f"{name} — 적중률 {coverage:.1%} · 평균폭 {mean_width:.0f}개 · 상한초과 {over_upper:.1%}",
        fontsize=11,
    )
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, alpha=0.25)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sub = []
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), sharex=False)
    for ax, store_cd in zip(axes.flat, STORE_MAP):
        name, _ = STORE_MAP[store_cd]
        sub = backtest_store(store_cd)
        all_sub.append(sub)
        plot_store(ax, sub, name)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9, framealpha=0.9)
    fig.suptitle(
        "4매장 카테고리 합 수요 — v5 대칭 conformal 95% 구간예측 + 실제 생산량(파란선)",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT_DIR / "interval_4stores.png"
    fig.savefig(out_png, dpi=140)

    out_csv = OUT_DIR / "interval_4stores_predictions.csv"
    pd.concat(all_sub, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nsaved {out_png}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()

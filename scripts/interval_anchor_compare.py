"""[DEPRECATED 2026-06-08] v5 conformal 구간(범위)예측 — 폐기. 점추정+품절/매진 위험수치로 전환 (docs/poc_scope_v6.md §8 D1). 데이터 검증 단계 산출물로만 보존.

앵커 분위수 q0.90 vs q0.50 구간예측 비교 — 4매장.

interval_backtest_4stores 의 q0.90(권장 발주점) 그래프와 동일 파이프라인을
q0.50(중앙값) 앵커로도 돌려 나란히 비교한다. 목적: "실측이 구간 하단에 깔리는"
현상이 앵커를 높은 분위수로 잡았기 때문임을 시각적으로 보이기 위함.

- q0.90: 실측이 구간 하단(pos≈0.33) — 보수적 발주점.
- q0.50: 실측이 구간 중앙(pos≈0.5) — 중앙값 중심.

매장당 feature 는 1회만 빌드, run() 을 production_q 만 바꿔 2회 호출.
conformal 보정으로 두 경우 모두 coverage≈95% 는 유지된다(폭만 달라짐).

산출물(reports/interval_4stores/):
  interval_anchor_compare.png   — 4매장 × (q0.90 | q0.50) 구간 비교
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from interval_backtest import run
from interval_backtest_4stores import build_production_daily, build_store_features
from store_daily import STORE_MAP

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path("reports/interval_4stores")
ANCHORS = (0.90, 0.50)


def backtest_at_q(feats: pd.DataFrame, prod: pd.DataFrame, production_q: float) -> pd.DataFrame:
    preds = run(
        feats,
        n_folds=4,
        min_train_days=365,
        calibration_days=210,
        horizon_days=30,
        production_q=production_q,
    )
    sub = preds[(preds["variant"] == "symmetric") & (preds["coverage_level"] == 0.95)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)
    return sub.merge(prod, on="date", how="left")


def plot_panel(ax, sub: pd.DataFrame, name: str, q: float) -> None:
    within = (sub["actual"] >= sub["lower"]) & (sub["actual"] <= sub["upper"])
    coverage = within.mean()
    mean_width = (sub["upper"] - sub["lower"]).mean()
    pos = ((sub["actual"] - sub["lower"]) / (sub["upper"] - sub["lower"])).mean()

    ax.fill_between(sub["date"], sub["lower"], sub["upper"],
                    color="#4C72B0", alpha=0.18, label="95% 예측구간 (대칭)")
    ax.plot(sub["date"], sub["upper"], color="#4C72B0", lw=0.7, alpha=0.5)
    ax.plot(sub["date"], sub["lower"], color="#4C72B0", lw=0.7, alpha=0.5)
    ax.plot(sub["date"], sub["anchor"], color="#C44E52", lw=1.6,
            label=f"예측값 (q{q:.2f} 앵커)")
    if sub["production"].notna().any():
        ax.plot(sub["date"], sub["production"], color="#1f77b4", lw=1.4, alpha=0.85,
                label="실제 생산량 (QT_MADE)")
    ax.scatter(sub.loc[within, "date"], sub.loc[within, "actual"],
               color="#2A2A2A", s=16, zorder=5, label="실측 (구간 내)")
    ax.scatter(sub.loc[~within, "date"], sub.loc[~within, "actual"],
               color="#D62728", s=30, marker="X", zorder=6, label="실측 (구간 밖)")

    ax.set_title(f"{name} · q{q:.2f} — 적중 {coverage:.0%} · 폭 {mean_width:.0f} · "
                 f"실측위치 {pos:.2f}", fontsize=10)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, alpha=0.25)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(STORE_MAP)
    fig, axes = plt.subplots(n, 2, figsize=(20, 5 * n), sharex=False)
    for row, store_cd in enumerate(STORE_MAP):
        name, store_id = STORE_MAP[store_cd]
        print(f"\n===== [{name}] features build =====")
        feats = build_store_features(store_cd, store_id)
        prod = build_production_daily(store_cd)
        for col, q in enumerate(ANCHORS):
            print(f"  -- q{q:.2f} backtest --")
            sub = backtest_at_q(feats, prod, q)
            plot_panel(axes[row, col], sub, name, q)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9, framealpha=0.9)
    fig.suptitle("앵커 분위수 비교 — 좌: q0.90(권장 발주점, 실측 하단) · 우: q0.50(중앙값, 실측 중앙)",
                 fontsize=15, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = OUT_DIR / "interval_anchor_compare.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()

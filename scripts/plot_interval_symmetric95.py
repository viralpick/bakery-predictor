"""[DEPRECATED 2026-06-08] v5 conformal 구간(범위)예측 — 폐기. 점추정+품절/매진 위험수치로 전환 (docs/poc_scope_v6.md §8 D1). 데이터 검증 단계 산출물로만 보존.

광교 v5 대칭 conformal 95% 구간 시각화 — 예측값·예측구간·실측값 한 그래프.

입력: reports/interval_predictions.csv (interval_backtest.py 산출)
출력: reports/interval_symmetric95.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SRC = Path("reports/interval_predictions.csv")
OUT = Path("reports/interval_symmetric95.png")


def main() -> None:
    df = pd.read_csv(SRC)
    sub = df[(df["variant"] == "symmetric") & (df["coverage_level"] == 0.95)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)

    within = (sub["actual"] >= sub["lower"]) & (sub["actual"] <= sub["upper"])
    coverage = within.mean()
    mean_width = (sub["upper"] - sub["lower"]).mean()
    over_upper = (sub["actual"] > sub["upper"]).mean()  # 매진(상한 초과)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(
        sub["date"], sub["lower"], sub["upper"],
        color="#4C72B0", alpha=0.18, label="95% 예측구간 (대칭)",
    )
    ax.plot(sub["date"], sub["upper"], color="#4C72B0", lw=0.8, alpha=0.5)
    ax.plot(sub["date"], sub["lower"], color="#4C72B0", lw=0.8, alpha=0.5)
    ax.plot(
        sub["date"], sub["anchor"], color="#C44E52", lw=1.8,
        label="예측값 (q0.90 앵커 = 권장 발주점)",
    )
    ax.scatter(
        sub.loc[within, "date"], sub.loc[within, "actual"],
        color="#2A2A2A", s=26, zorder=5, label="실측 (보정수요, 구간 내)",
    )
    ax.scatter(
        sub.loc[~within, "date"], sub.loc[~within, "actual"],
        color="#D62728", s=46, marker="X", zorder=6,
        label="실측 (구간 벗어남)",
    )

    # fold 경계 (재캘리브레이션 지점) 표시
    fold_change = sub.index[sub["fold"].ne(sub["fold"].shift())].tolist()[1:]
    for idx in fold_change:
        ax.axvline(sub["date"].iloc[idx], color="gray", ls=":", lw=0.8, alpha=0.6)

    ax.set_title(
        "광교 카테고리 합 수요 — v5 대칭 conformal 95% 구간예측\n"
        f"실측 적중률 {coverage:.1%} · 평균 폭 {mean_width:.0f}개 · 상한 초과(매진) {over_upper:.1%}",
        fontsize=13,
    )
    ax.set_xlabel("날짜 (2025 backtest, 3-fold expanding)")
    ax.set_ylabel("일 수요 (보정수요, 단위: 개)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=45)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT} | coverage={coverage:.3f} width={mean_width:.1f} over_upper={over_upper:.3f}")


if __name__ == "__main__":
    main()

"""요일별 강수량 효과 그래프 (4매장).

기존 그래프(x=강수량 범위, 요일 비교)는 "요일 간 비교"라 강수 효과가 안 보였음.
이번엔 뒤집어서: x축=요일, 각 요일 안에서 강수량 범위별 평균 판매량(qty)을 묶음 막대로.
=> 한 요일 안에서 비가 올수록 수요가 어떻게 변하는지 직접 비교 가능.

매장→관측소: 광교=119(수원), 서울 3매장=108(서울).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

V2 = Path("data/internal/v2")
WEATHER = Path("data/external/weather_observed.parquet")
OUT = Path("reports/rain_by_dow.png")

STORE_STATION = {"광교": 119, "삼성타운": 108, "메세나폴리스": 108, "광화문": 108}
DOW_LABELS = ["월", "화", "수", "목", "금", "토", "일"]
RAIN_BINS = [-0.1, 0.1, 5, 20, 1e9]
RAIN_LABELS = ["0mm(맑음)", "0-5mm", "5-20mm", "20mm+"]
BIN_COLORS = ["#f4c542", "#7fb3d5", "#2e86c1", "#1a5276"]


def load_weather() -> pd.DataFrame:
    w = pd.read_parquet(WEATHER)
    w["date"] = pd.to_datetime(w["date"])
    w["sumRn"] = pd.to_numeric(w["sumRn"], errors="coerce").fillna(0)
    return w[["date", "station_id", "sumRn"]]


def store_panel(daily: pd.DataFrame, weather: pd.DataFrame,
                store: str, station: int) -> pd.DataFrame:
    d = daily[daily["store"] == store].merge(
        weather[weather["station_id"] == station], on="date", how="inner"
    ).copy()
    d["dow"] = d["date"].dt.dayofweek
    d["rain_bin"] = pd.cut(d["sumRn"], bins=RAIN_BINS, labels=RAIN_LABELS)
    return d


def plot_store(ax, d: pd.DataFrame, store: str) -> None:
    n_bins = len(RAIN_LABELS)
    width = 0.8 / n_bins
    x = np.arange(7)
    for i, label in enumerate(RAIN_LABELS):
        sub = d[d["rain_bin"] == label]
        means = sub.groupby("dow")["qty"].mean().reindex(range(7))
        counts = sub.groupby("dow")["qty"].size().reindex(range(7)).fillna(0)
        offset = (i - (n_bins - 1) / 2) * width
        bars = ax.bar(x + offset, means.values, width, label=label,
                      color=BIN_COLORS[i], edgecolor="white", linewidth=0.3)
        for xi, (b, c) in enumerate(zip(bars, counts.values)):
            if c > 0 and not np.isnan(b.get_height()):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                        f"{int(c)}", ha="center", va="bottom", fontsize=6,
                        color="#555")
    ax.set_title(f"{store}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DOW_LABELS)
    ax.set_ylabel("평균 판매량(qty)")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    daily = pd.read_parquet(V2 / "daily_4stores.parquet").rename(
        columns={"DT_SALE": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    weather = load_weather()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, (store, station) in zip(axes.flat, STORE_STATION.items()):
        d = store_panel(daily, weather, store, station)
        plot_store(ax, d, store)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               title="일강수량 구간", fontsize=10, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("요일별 × 강수량 구간 평균 판매량 — 막대 위 숫자=일수(n)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved: {OUT}")

    # 콘솔 요약: 요일별 맑음 대비 비온날(>=5mm) 차이
    print("\n=== 요일별 '비(>=5mm) vs 맑음(0mm)' 판매량 차이 (4매장 합산 관점) ===")
    for store, station in STORE_STATION.items():
        d = store_panel(daily, weather, store, station)
        dry = d[d["rain_bin"] == "0mm(맑음)"].groupby("dow")["qty"].mean()
        wet = d[d["sumRn"] >= 5].groupby("dow")["qty"].mean()
        print(f"\n[{store}]")
        for dw in range(7):
            if dw in dry.index and dw in wet.index and dry[dw] > 0:
                diff = (wet[dw] - dry[dw]) / dry[dw] * 100
                print(f"  {DOW_LABELS[dw]}: 맑음 {dry[dw]:6.1f} → 비 {wet[dw]:6.1f}  ({diff:+5.1f}%)")


if __name__ == "__main__":
    main()

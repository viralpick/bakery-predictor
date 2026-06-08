"""가정 2·3 다매장 재검증 — 리포트용 요약만 출력.

가정2: 매장별 강수(비 오는 날 vs 안 오는 날) 매출 차이 + 온도-계절성 흡수 점검.
가정3: 매장별 마감 회수율 = closing / (closing + waste) — 실수요 단서 (cost-min α 대체).

매장→관측소: 광교=119(수원), 나머지 서울 3매장=108(서울).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

V2 = Path("data/internal/v2")
WEATHER = Path("data/external/weather_observed.parquet")

STORE_STATION = {"광교": 119, "삼성타운": 108, "메세나폴리스": 108, "광화문": 108}


def load_weather() -> pd.DataFrame:
    w = pd.read_parquet(WEATHER)
    w["date"] = pd.to_datetime(w["date"])
    for c in ["avgTa", "sumRn"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w["sumRn"] = w["sumRn"].fillna(0)
    w["is_rain"] = w["sumRn"] >= 1.0  # 1mm 이상 = 비 온 날
    w["month"] = w["date"].dt.month
    return w[["date", "station_id", "avgTa", "sumRn", "is_rain", "month"]]


def main() -> None:
    daily = pd.read_parquet(V2 / "daily_4stores.parquet").rename(columns={"DT_SALE": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    weather = load_weather()

    print("=" * 70)
    print("가정2-A. 매장별 비 오는 날 vs 안 오는 날 매출(qty) 차이")
    print("=" * 70)
    for store, station in STORE_STATION.items():
        d = daily[daily["store"] == store].merge(
            weather[weather["station_id"] == station], on="date", how="inner"
        )
        rain = d.loc[d["is_rain"], "qty"]
        dry = d.loc[~d["is_rain"], "qty"]
        t, p = stats.ttest_ind(rain, dry, equal_var=False)
        diff_pct = (rain.mean() - dry.mean()) / dry.mean() * 100
        print(
            f"  {store:6s}(관측소{station}): 비온날 {rain.mean():6.1f}(n={len(rain)}) vs "
            f"맑은날 {dry.mean():6.1f}(n={len(dry)}) | 차이 {diff_pct:+5.1f}% | "
            f"Welch t={t:+.2f} p={p:.3f}"
        )

    print()
    print("=" * 70)
    print("가정2-B. 온도(avgTa)가 계절성(month)에 흡수되는지 — 상관")
    print("=" * 70)
    for store, station in STORE_STATION.items():
        d = daily[daily["store"] == store].merge(
            weather[weather["station_id"] == station], on="date", how="inner"
        )
        # avgTa vs month-cyclic (계절 위치)
        msin = np.sin(2 * np.pi * d["month"] / 12)
        mcos = np.cos(2 * np.pi * d["month"] / 12)
        r_ta_seasonal = max(abs(d["avgTa"].corr(msin)), abs(d["avgTa"].corr(mcos)))
        r_ta_qty = d["avgTa"].corr(d["qty"])
        print(
            f"  {store:6s}: |corr(avgTa, 월계절)|={r_ta_seasonal:.3f}  "
            f"corr(avgTa, qty)={r_ta_qty:+.3f}"
        )

    print()
    print("=" * 70)
    print("가정3. 매장별 마감 회수율 = closing / (closing + waste)  [실수요 단서]")
    print("=" * 70)
    wa = pd.read_parquet(V2 / "waste_alpha_4stores.parquet")
    agg = wa.groupby("store").agg(
        closing_qty=("closing_qty", "sum"),
        waste_qty=("out", "sum"),
        made=("made", "sum"),
        sold_total=("sold_total", "sum"),
    )
    agg["recovery_rate"] = agg["closing_qty"] / (agg["closing_qty"] + agg["waste_qty"])
    agg["waste_rate"] = agg["waste_qty"] / agg["made"]
    for store in ["광교", "광화문", "메세나폴리스", "삼성타운"]:
        if store not in agg.index:
            continue
        r = agg.loc[store]
        print(
            f"  {store:6s}: 회수율 {r['recovery_rate']*100:5.1f}%  "
            f"(마감판매 {int(r['closing_qty']):>7,} / 폐기 {int(r['waste_qty']):>7,})  "
            f"폐기율 {r['waste_rate']*100:4.1f}%"
        )


if __name__ == "__main__":
    main()

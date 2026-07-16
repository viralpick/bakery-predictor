"""연도별 모델 미커버 이상치 — 각 연도 예측량6 + 발주부족4 + 주2 + 월1.

anomaly_customer_lists.py와 동일 방법(detrend 잔차 / de-seasonalize 발주부족 / 비휴일 WPE
주·월 / 수원 날씨 크로스체크 / 공휴일 제외)을 **연도별로 분리 랭킹**한다.
추세(−26% 다년 하락)는 detrend로 제거되므로 각 연도의 로컬 이상치가 공정 비교된다.

실행: uv run python scripts/anomaly_by_year.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reclassify_anomalies import _holiday_lookup, _nearest_holiday
from anomaly_customer_lists import _weather_by_date, _enrich, _fmt, FULL_PREDS

YEARS = [2023, 2024, 2025]
N_PRED, N_ORDER, N_WEEK, N_MONTH = 6, 4, 2, 1


def _prep() -> pd.DataFrame:
    p = pd.read_csv(FULL_PREDS)
    p["date"] = pd.to_datetime(p["date"])
    holidays = _holiday_lookup()
    p = _enrich(p, holidays, _weather_by_date()).sort_values("date").reset_index(drop=True)
    p["resid_dt"] = p["resid"] - p["resid"].rolling(56, center=True, min_periods=21).median()
    p["month"] = p["date"].dt.month
    p["dow_i"] = p["date"].dt.dayofweek
    p["shortfall"] = p["actual"] - p["production"]
    p["short_dt"] = p["shortfall"] - p.groupby(["dow_i", "month"])["shortfall"].transform("median")
    p["_hol"] = p["date"].map(lambda d: _nearest_holiday(d, holidays)[0])
    return p


def _period_year(py: pd.DataFrame, freq: str, top: int) -> pd.DataFrame:
    nh = py[~py["_hol"]].copy()
    nh["key"] = nh["date"].dt.to_period("W" if freq == "W" else "M")
    g = nh.groupby("key").agg(start=("date", "min"), end=("date", "max"), n=("date", "size"),
                              actual=("actual", "sum"), expected=("expected", "sum"),
                              resid=("resid", "sum")).reset_index(drop=True)
    g = g[g["n"] >= (4 if freq == "W" else 18)]
    g["wpe_pct"] = (g["resid"] / g["actual"].clip(lower=1) * 100).round(1)
    g["direction"] = np.where(g["resid"] < 0, "저조", "급등")
    return g.reindex(g["wpe_pct"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def main() -> None:
    p = _prep()
    for y in YEARS:
        py = p[p["date"].dt.year == y]
        pred = py[~py["is_holiday"]].reindex(
            py[~py["is_holiday"]]["resid_dt"].abs().sort_values(ascending=False).index).head(N_PRED)
        so = py[(py["shortfall"] > 0) & (~py["is_holiday"])]
        order = so.reindex(so["short_dt"].sort_values(ascending=False).index).head(N_ORDER)
        weeks = _period_year(py, "W", N_WEEK)
        months = _period_year(py, "M", N_MONTH)

        pred.to_csv(f"reports/anomaly_year_{y}_pred.csv", index=False)
        order.to_csv(f"reports/anomaly_year_{y}_order.csv", index=False)

        print("=" * 66)
        print(f"【 {y}년 】")
        print(f"■ 예측량(잔차, 추세제거) {len(pred)}일")
        print(_fmt(pred, dt_col="resid_dt"))
        print(f"■ 발주부족(매진, 계절제거) {len(order)}일")
        oc = order.assign(pct=order["shortfall"] / order["production"] * 100, expected=order["production"])
        print(_fmt(oc, dt_col="short_dt"))
        print(f"■ 주 {len(weeks)}")
        print(weeks[["start", "end", "n", "actual", "expected", "wpe_pct", "direction"]].to_string(index=False))
        print(f"■ 월 {len(months)}")
        print(months[["start", "end", "n", "actual", "expected", "wpe_pct", "direction"]].to_string(index=False))
    print("\nwrote reports/anomaly_year_*.csv")


if __name__ == "__main__":
    main()

"""고객 문의용 이상치 상위 리스트 + 내부 개선 리스트 + 날씨 크로스체크.

full_preds.csv(anomaly_detect가 덤프한 3년 OOS 예측) 위에서 재분석 — 백테스트 재실행 없음.
사용자 스펙:
- 일 ~20 = 예측량(잔차) 상위 12 + 발주량(발주부족=매진) 상위 8.
- 주 ~5, 월 ~3 = 예측량 기준.
- 공휴일/특수일/대체공휴일로 설명되는 건 **삭제 아니라 별도 리스트**(내부 개선=prior 확장).
- 각 날짜에 극한 날씨(폭우·폭설·폭염·한파·강풍) 크로스체크(우리 모델이 날씨 feature를
  쓰므로, 극한날씨+큰잔차 = 모델이 덜 반영 → weather feature 개선 후보).

랭크: 예측량=|잔차|(단위) desc · 발주량=발주부족량(actual−production) desc · 주/월=|WPE|(비휴일일만).
실행: uv run python scripts/anomaly_customer_lists.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reclassify_anomalies import _holiday_lookup, _nearest_holiday

FULL_PREDS = "reports/anomaly_full_preds.csv"
WEATHER = "data/external/weather_observed.parquet"
STATION = 119                    # 수원 = 광교 관할 ASOS
MASK = 2                         # 연휴 인접일 ±일 (공휴일 판정)
N_PRED_DAYS, N_ORDER_DAYS, N_WEEKS, N_MONTHS = 12, 8, 5, 3
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]
# 상업 특수일(비공휴일) month-day → 설명 가능 후보로 annotate
COMMERCIAL = {(2, 14): "발렌타인", (3, 14): "화이트데이", (11, 11): "빼빼로데이",
              (12, 24): "크리스마스이브", (12, 31): "연말"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _weather_by_date() -> pd.DataFrame:
    w = pd.read_parquet(WEATHER)
    w = w[w["station_id"] == STATION].copy()
    w["date"] = pd.to_datetime(w["date"])
    for c in ["sumRn", "maxTa", "minTa", "ddMefs", "sumDpthFhsc", "maxWs"]:
        w[c] = _num(w[c])
    w["snow"] = w[["ddMefs", "sumDpthFhsc"]].max(axis=1)
    return w[["date", "sumRn", "maxTa", "minTa", "snow", "maxWs"]]


def _weather_flag(r) -> str:
    tags = []
    if pd.notna(r["sumRn"]) and r["sumRn"] >= 30:
        tags.append(f"{'폭우' if r['sumRn'] >= 50 else '호우'}({r['sumRn']:.0f}mm)")
    if pd.notna(r["snow"]) and r["snow"] >= 1:
        tags.append(f"{'폭설' if r['snow'] >= 5 else '눈'}({r['snow']:.0f}cm)")
    if pd.notna(r["maxTa"]) and r["maxTa"] >= 33:
        tags.append(f"폭염({r['maxTa']:.0f}도)")
    if pd.notna(r["minTa"]) and r["minTa"] <= -12:
        tags.append(f"한파({r['minTa']:.0f}도)")
    if pd.notna(r["maxWs"]) and r["maxWs"] >= 14:
        tags.append(f"강풍({r['maxWs']:.0f}m/s)")
    return " ".join(tags)


def _enrich(df: pd.DataFrame, holidays: dict, weather: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(weather, on="date", how="left")
    df["dow"] = df["date"].dt.dayofweek.map(lambda i: DOW_KR[i])
    hol = df["date"].map(lambda d: _nearest_holiday(d, holidays))
    df["holiday"] = [h[1] for h in hol]
    df["is_holiday"] = [h[0] for h in hol]
    df["commercial"] = df["date"].map(lambda d: COMMERCIAL.get((d.month, d.day), ""))
    df["weather"] = df.apply(_weather_flag, axis=1)
    df["direction"] = np.where(df["resid"] < 0, "저조", "급등")
    return df


def _fmt(df: pd.DataFrame, dt_col: str = "") -> str:
    cols = ["date", "dow", "actual", "expected", "pct"]
    d = df.copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    d["actual"] = d["actual"].round(0).astype(int)
    d["expected"] = d["expected"].round(0).astype(int)
    d["pct"] = d["pct"].round(1)
    if dt_col:
        d["국소편차"] = d[dt_col].round(0).astype(int)   # 추세·계절 제거 후 로컬 대비 편차
        cols.append("국소편차")
    cols += ["direction", "weather", "commercial"]
    return d[cols].to_string(index=False)


def _period(p: pd.DataFrame, freq: str, holidays: dict, top: int) -> pd.DataFrame:
    """비휴일일만으로 주/월 WPE 집계 → |WPE| 상위(휴일이 몰지 않은 지속 이상)."""
    p = p.copy()
    p["is_hol"] = p["date"].map(lambda d: _nearest_holiday(d, holidays)[0])
    nh = p[~p["is_hol"]].copy()
    nh["key"] = nh["date"].dt.to_period("W" if freq == "W" else "M")
    g = nh.groupby("key").agg(start=("date", "min"), end=("date", "max"), n=("date", "size"),
                              actual=("actual", "sum"), expected=("expected", "sum"),
                              resid=("resid", "sum")).reset_index(drop=True)
    g = g[g["n"] >= (4 if freq == "W" else 18)]
    g["wpe_pct"] = (g["resid"] / g["actual"].clip(lower=1) * 100).round(1)
    g["direction"] = np.where(g["resid"] < 0, "저조", "급등")
    return g.reindex(g["wpe_pct"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def main() -> None:
    p = pd.read_csv(FULL_PREDS)
    p["date"] = pd.to_datetime(p["date"])
    holidays = _holiday_lookup()
    weather = _weather_by_date()
    p = _enrich(p, holidays, weather).sort_values("date").reset_index(drop=True)

    # detrend/de-seasonalize (advisor): 다년 하락 추세·주말/여름 계절성은 체계적 편향(내부
    # recalibration)이라 day 리스트에서 뺀다 → 로컬 대비 튀는 point 이상치만 남긴다.
    # 지속 시프트는 주/월 렌즈가 이미 잡으므로 정보 손실 아님.
    p["resid_dt"] = p["resid"] - p["resid"].rolling(56, center=True, min_periods=21).median()
    p["month"] = p["date"].dt.month
    p["dow_i"] = p["date"].dt.dayofweek
    p["shortfall"] = p["actual"] - p["production"]
    p["short_dt"] = p["shortfall"] - p.groupby(["dow_i", "month"])["shortfall"].transform("median")

    # 예측량: 공휴일 제외 → |detrend 잔차| 상위 12. 공휴일분은 내부 리스트.
    pred = p.reindex(p["resid_dt"].abs().sort_values(ascending=False).index)
    pred_cust = pred[~pred["is_holiday"]].head(N_PRED_DAYS)
    pred_internal = p.reindex(p["resid"].abs().sort_values(ascending=False).index)
    pred_internal = pred_internal[pred_internal["is_holiday"]].head(15)

    # 발주부족(매진): actual>production, 공휴일 제외 → de-seasonalize 부족량 상위 8.
    so = p[p["shortfall"] > 0].copy()
    so = so.reindex(so["short_dt"].sort_values(ascending=False).index)
    order_cust = so[~so["is_holiday"]].head(N_ORDER_DAYS)
    order_internal = so.reindex(so["shortfall"].sort_values(ascending=False).index)
    order_internal = order_internal[order_internal["is_holiday"]].head(12)

    weeks = _period(p, "W", holidays, N_WEEKS)
    months = _period(p, "M", holidays, N_MONTHS)

    for df, path in [(pred_cust, "pred_days"), (order_cust, "order_days"),
                     (pred_internal, "internal_pred"), (order_internal, "internal_order")]:
        df.to_csv(f"reports/anomaly_cust_{path}.csv", index=False)
    weeks.to_csv("reports/anomaly_cust_weeks.csv", index=False)
    months.to_csv("reports/anomaly_cust_months.csv", index=False)

    print("=" * 70)
    print(f"■ 고객 문의 — 예측량(잔차, 추세제거) 상위 {len(pred_cust)}일 (공휴일 제외)")
    print(_fmt(pred_cust, dt_col="resid_dt"))
    print(f"\n■ 고객 문의 — 발주부족(매진, 계절제거) 상위 {len(order_cust)}일 (공휴일 제외)")
    oc = order_cust.assign(pct=(order_cust["shortfall"] / order_cust["production"] * 100))
    print(_fmt(oc.assign(expected=oc["production"]), dt_col="short_dt"))  # expected 칸=버퍼발주
    print(f"\n■ 고객 문의 — 주 상위 {len(weeks)} (비휴일일 WPE)")
    print(weeks[["start", "end", "n", "actual", "expected", "wpe_pct", "direction"]].to_string(index=False))
    print(f"\n■ 고객 문의 — 월 상위 {len(months)} (비휴일일 WPE)")
    print(months[["start", "end", "n", "actual", "expected", "wpe_pct", "direction"]].to_string(index=False))
    print("\n" + "=" * 70)
    print(f"□ 내부 개선(설명가능=공휴일) — 예측 {len(pred_internal)} / 발주부족 {len(order_internal)}. holiday별 prior 확장 후보:")
    print(pred_internal[["date", "dow", "pct", "direction", "holiday", "weather"]].assign(
        date=pred_internal["date"].dt.strftime("%Y-%m-%d")).to_string(index=False))
    print("\nwrote reports/anomaly_cust_*.csv")
    print("※ 날씨=수원(119). 극한날씨+큰잔차 = 모델 weather feature 개선 후보(고객문의 전 우리 판단).")
    print("※ commercial=비공휴일 상업특수일(화이트데이 등) — 설명 가능 후보(참고).")


if __name__ == "__main__":
    main()

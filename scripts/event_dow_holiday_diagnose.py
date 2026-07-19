"""특수일 lift를 요일·대체공휴일·연휴길이로 분해 — EventLevelPrior 일반화 안전성 진단.

사용자 우려(2026-07-18):
  (1) 크리스마스는 케이크 때문에 빵 생산량 고정 → 레벨 안정이 수요 아닌 공급 아티팩트.
      → "xmas에서 통했으니 일반화"라는 논리 무효. 다른 특수일은 진짜 수요 효과.
  (2) 대체공휴일 고려 필요.
  (3) 연속 공휴일(연휴) → 여행 → 주거지 상권 매출 하락 가능.

절대 median 앵커(현 EventLevelPrior)가 이 요인들을 뭉개는지 확인. 로컬 동일요일 baseline
대비 lift = actual / (같은 요일 ±N주 중앙값, 공휴일 제외) — 추세·요일 동시 통제.

값싼 진단: reports/raw_adjusted_series.csv(광교 2021-25) + calendar, store 재빌드 없음.
실행: PYTHONPATH=scripts uv run --with matplotlib python scripts/event_dow_holiday_diagnose.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.data.calendar import build_calendar_daily

SERIES = Path("reports/raw_adjusted_series.csv")
DOW_HALFWIN_WEEKS = 6  # 동일요일 baseline 윈도우 (±6주)
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]

# 진단 대상 이벤트 (양력 고정만; 음력은 별도 날짜)
SOLAR_EVENTS = {
    "xmas": (12, 25),
    "childrens": (5, 5),
    "sinjeong": (1, 1),
    "samiljeol": (3, 1),
    "gwangbokjeol": (8, 15),
}


def local_dow_baseline(df: pd.DataFrame) -> pd.Series:
    """각 날짜에 대해 같은 요일 ±DOW_HALFWIN_WEEKS 중앙값(공휴일·주말 제외 '평상' 레벨).

    추세(−26% 하락)와 요일을 동시에 통제. 이벤트일 자신은 baseline 계산에서 제외.
    """
    s = df.set_index("date")["adjusted_demand_unit"]
    normal = df[(df["is_public_holiday"] == 0)].set_index("date")["adjusted_demand_unit"]
    out = {}
    for d in df["date"]:
        lo, hi = d - pd.Timedelta(weeks=DOW_HALFWIN_WEEKS), d + pd.Timedelta(weeks=DOW_HALFWIN_WEEKS)
        same_dow = normal[(normal.index >= lo) & (normal.index <= hi)
                          & (normal.index.dayofweek == d.dayofweek) & (normal.index != d)]
        out[d] = float(same_dow.median()) if len(same_dow) >= 3 else np.nan
    return pd.Series(out)


def run() -> None:
    series = pd.read_csv(SERIES, parse_dates=["date"])[["date", "adjusted_demand_unit"]]
    cal = build_calendar_daily(series["date"].min(), series["date"].max())
    df = series.merge(cal, on="date", how="left")

    base = local_dow_baseline(df)
    df["dow_base"] = df["date"].map(base)
    df["lift"] = df["adjusted_demand_unit"] / df["dow_base"]

    rows = []
    for name, (m, day) in SOLAR_EVENTS.items():
        ev = df[(df["date"].dt.month == m) & (df["date"].dt.day == day)]
        for _, r in ev.iterrows():
            rows.append(dict(
                event=name, date=r["date"].date(), dow=DOW_KR[r["date"].dayofweek],
                actual=r["adjusted_demand_unit"], dow_base=r["dow_base"],
                lift=r["lift"], sub=int(r["is_substitute_holiday"]),
                streak=int(r["off_streak_length"]),
            ))
    res = pd.DataFrame(rows)

    print("=== 특수일 lift 분해 (광교 2021-25, 로컬 동일요일 baseline 대비) ===")
    print("lift>1=평상 동일요일보다 높음. streak=연속 off일 수. sub=대체공휴일.\n")
    for name in SOLAR_EVENTS:
        g = res[res["event"] == name]
        print(f"--- {name} ---")
        for _, r in g.iterrows():
            base_str = f"{r['dow_base']:.0f}" if pd.notna(r["dow_base"]) else "n/a"
            lift_str = f"{r['lift']:.2f}" if pd.notna(r["lift"]) else "n/a"
            print(f"  {r['date']}({r['dow']}) actual={r['actual']:5.0f} "
                  f"base={base_str:>4} lift={lift_str:>4} streak={r['streak']} "
                  f"{'[대체공휴일]' if r['sub'] else ''}")
        lifts = g["lift"].dropna()
        if len(lifts) >= 2:
            print(f"  → lift 범위 [{lifts.min():.2f}, {lifts.max():.2f}] "
                  f"std={lifts.std():.3f} (요일 통제 후에도 흔들리면 median앵커 위험)\n")
        else:
            print()

    # 요일 효과: 평일 vs 주말/일요일 lift 대비
    res["is_sun"] = pd.to_datetime(res["date"]).dt.dayofweek == 6
    print("=== 요일 민감도 (이벤트별 평일 vs 일요일 lift) ===")
    for name in SOLAR_EVENTS:
        g = res[(res["event"] == name)].dropna(subset=["lift"])
        wk = g[~g["is_sun"]]["lift"]
        sun = g[g["is_sun"]]["lift"]
        if len(sun) and len(wk):
            print(f"  {name:14s}: 평일 median {wk.median():.2f} vs 일요일 {sun.median():.2f}")
    res.to_csv("reports/event_dow_holiday_diagnose.csv", index=False)
    print("\n저장: reports/event_dow_holiday_diagnose.csv")


if __name__ == "__main__":
    run()

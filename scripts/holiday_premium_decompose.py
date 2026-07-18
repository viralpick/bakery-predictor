"""전 공휴일 프리미엄 분해 — 요일·주말·연휴·대체공휴일 축 (광교 2021-25).

사용자(architect) 질문(2026-07-18):
  (1) 평일 어린이날이 다른 평일 공휴일보다 눈에 띄게 많이 팔리는지? (이벤트 고유 vs 일반 공휴일 프리미엄)
  (2) 주말 어린이날 + 대체공휴일 연계 → 주말 수요가 연휴에 넓게 흡수되는지?
  (3) 다른 공휴일도 동일 분석 시 일관성:
      - 평일 공휴일 vs 그냥 평일 격차?
      - 주말 공휴일은 '주말 + 특수일'이라 격차?
      - 연속 휴일 형성(금/월 휴일·대체·인접 합산)과 그로 인한 영향(여행)?

방법: lift = actual / 로컬 동일요일 baseline(±6주, 공휴일 제외). 추세·요일 동시 통제.
값싼 진단: reports/raw_adjusted_series.csv + calendar, store 재빌드 없음.
실행: PYTHONPATH=scripts uv run --with matplotlib python scripts/holiday_premium_decompose.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.data.calendar import build_calendar_daily

SERIES = Path("reports/raw_adjusted_series.csv")
HALFWIN = 6  # 동일요일 baseline ±주
DOW = ["월", "화", "수", "목", "금", "토", "일"]


def local_dow_baseline(df: pd.DataFrame) -> dict:
    normal = df[df["is_public_holiday"] == 0].set_index("date")["adjusted_demand_unit"]
    out = {}
    for d in df["date"]:
        lo, hi = d - pd.Timedelta(weeks=HALFWIN), d + pd.Timedelta(weeks=HALFWIN)
        idx = normal.index
        same = normal[(idx >= lo) & (idx <= hi) & (idx.dayofweek == d.dayofweek) & (idx != d)]
        out[d] = float(same.median()) if len(same) >= 3 else np.nan
    return out


def load() -> pd.DataFrame:
    s = pd.read_csv(SERIES, parse_dates=["date"])[["date", "adjusted_demand_unit"]]
    cal = build_calendar_daily(s["date"].min(), s["date"].max())
    df = s.merge(cal, on="date", how="left")
    base = local_dow_baseline(df)
    df["dow_base"] = df["date"].map(base)
    df["lift"] = df["adjusted_demand_unit"] / df["dow_base"]
    df["dow"] = df["date"].dt.dayofweek
    return df


def _norm_name(name: str) -> str:
    """대체공휴일을 원 명절로 통합, 영문 명칭 정리."""
    if not isinstance(name, str):
        return ""
    n = name.replace("Alternative holiday for ", "").replace(" (observed)", "")
    return n


def part_a_by_holiday(hol: pd.DataFrame) -> None:
    print("\n========== A. 명절별 occurrence (요일·연휴·대체 맥락) ==========")
    print("lift>1 = 평상 동일요일 대비 높음. streak=연속 off일. pos=연휴 내 위치.\n")
    hol = hol.copy()
    hol["base_name"] = hol["holiday_name"].map(_norm_name)
    for name, g in hol.groupby("base_name"):
        if g["lift"].notna().sum() == 0:
            continue
        print(f"--- {name} ---")
        for _, r in g.sort_values("date").iterrows():
            wk = "주말" if r["dow"] >= 5 else "평일"
            sub = "[대체]" if r["is_substitute_holiday"] else ""
            lift = f"{r['lift']:.2f}" if pd.notna(r["lift"]) else "n/a"
            print(f"  {r['date'].date()}({DOW[r['dow']]},{wk}) lift={lift:>4} "
                  f"streak={int(r['off_streak_length'])} pos={int(r['off_position_in_streak'])} {sub}")
        lifts = g["lift"].dropna()
        wkday = g[g["dow"] < 5]["lift"].dropna()
        print(f"  → 전체 median lift={lifts.median():.2f} | 평일만 median={wkday.median():.2f}"
              f" (n_평일={len(wkday)})\n" if len(wkday) else f"  → median lift={lifts.median():.2f}\n")


def part_b_dow_class(hol: pd.DataFrame) -> None:
    print("========== B. 평일공휴일 vs 주말공휴일 프리미엄 (일반 vs 격차) ==========")
    wk = hol[hol["dow"] < 5]["lift"].dropna()
    we = hol[hol["dow"] >= 5]["lift"].dropna()
    print(f"  평일 공휴일 (n={len(wk)}): median lift={wk.median():.2f}  "
          f"[{wk.quantile(.25):.2f}, {wk.quantile(.75):.2f}]  → 평상 평일 대비 프리미엄")
    print(f"  주말 공휴일 (n={len(we)}): median lift={we.median():.2f}  "
          f"[{we.quantile(.25):.2f}, {we.quantile(.75):.2f}]  → 평상 주말 대비")
    print("  (평일 lift>>1 = 공휴일에 몰림 / 주말 lift≤1 = 주말공휴일은 평상주말보다 안 높거나 낮음)\n")


def part_c_event_vs_generic(hol: pd.DataFrame) -> None:
    print("========== C. 이벤트 고유성 — 평일공휴일 lift 랭킹 ==========")
    print("각 명절 '평일 occurrence만' median lift. 어린이날이 일반 공휴일보다 튀는지.\n")
    hol = hol.copy()
    hol["base_name"] = hol["holiday_name"].map(_norm_name)
    wk = hol[hol["dow"] < 5]
    rows = []
    for name, g in wk.groupby("base_name"):
        lifts = g["lift"].dropna()
        if len(lifts) >= 1:
            rows.append((name, lifts.median(), len(lifts)))
    rows.sort(key=lambda x: -x[1])
    for name, med, n in rows:
        print(f"  {name:28s} median lift={med:.2f} (n_평일={n})")
    all_wk = wk["lift"].dropna()
    print(f"\n  [전체 평일공휴일 median={all_wk.median():.2f}] — 이 선 위/아래로 이벤트 고유성 판정\n")


def part_d_streak(hol: pd.DataFrame) -> None:
    print("========== D. 연휴 길이 효과 (여행/주거지 하락) ==========")
    print("평일 공휴일만, off_streak_length 버킷별 lift. 길수록 낮으면 여행효과.\n")
    wk = hol[hol["dow"] < 5].copy()
    wk["streak_bucket"] = pd.cut(wk["off_streak_length"], [0, 1, 2, 10],
                                 labels=["1(고립)", "2", "3+(연휴)"])
    for b, g in wk.groupby("streak_bucket", observed=True):
        lifts = g["lift"].dropna()
        if len(lifts):
            print(f"  streak {b:>8}: median lift={lifts.median():.2f} (n={len(lifts)})")
    print("\n  --- 연휴 블록(연속 off≥3, 공휴일 포함) 수요 궤적: 여행 dip 확인 ---")
    df_full = hol.attrs["full"]
    df_full = df_full.sort_values("date").reset_index(drop=True)
    is_off = df_full["is_off_day"].astype(bool)
    block_id = (is_off != is_off.shift()).cumsum()
    for bid, blk in df_full.groupby(block_id):
        if not blk["is_off_day"].iloc[0] or len(blk) < 3:
            continue
        if blk["is_public_holiday"].sum() == 0:
            continue
        span = f"{blk['date'].iloc[0].date()}~{blk['date'].iloc[-1].date()}"
        vals = "→".join(f"{v:.0f}" for v in blk["adjusted_demand_unit"])
        names = [_norm_name(n) for n in blk["holiday_name"] if isinstance(n, str) and n]
        print(f"  {span} ({len(blk)}일, {'·'.join(dict.fromkeys(names))}): {vals}")


def run() -> None:
    df = load()
    hol = df[df["is_public_holiday"] == 1].copy()
    hol.attrs["full"] = df
    print(f"=== 광교 공휴일 프리미엄 분해 (2021-25, n_공휴일={len(hol)}) ===")
    part_a_by_holiday(hol)
    part_b_dow_class(hol)
    part_c_event_vs_generic(hol)
    part_d_streak(hol)
    df.to_csv("reports/holiday_premium_decompose.csv", index=False)
    print("\n저장: reports/holiday_premium_decompose.csv")


if __name__ == "__main__":
    run()

"""탐지된 이상치를 2버킷으로 재분류 — 휴일 소스 교체(holidays.KR) + 연휴 인접일 마스크.

calendar_raw는 2023 휴일 행이 없어 오분류. bakery.data.calendar.build_calendar_daily
(holidays.KR 기반, 모델이 실제 쓴 캘린더)로 교체하고 명절 스필오버(±MASK일)까지 마스킹.

버킷 A (모델 갭, 내부 액션): 휴일/연휴 인접 → EventLevelPrior 확장 신호(고객 문의 아님).
버킷 B (진짜 미설명, 고객 문의): 어떤 휴일과도 무관 → "그날 무슨 일이 있었나".

실행: uv run python scripts/reclassify_anomalies.py
"""
from __future__ import annotations

import pandas as pd

from bakery.data.calendar import build_calendar_daily

MASK = 2  # 연휴 인접일(명절 전날/스필오버) ±일


def _holiday_lookup() -> dict:
    cal = build_calendar_daily("2022-12-01", "2026-02-28")
    cal["date"] = pd.to_datetime(cal["date"])
    hol = cal[cal["is_public_holiday"] == 1][["date", "holiday_name"]]
    return dict(zip(hol["date"], hol["holiday_name"]))


def _nearest_holiday(d: pd.Timestamp, holidays: dict) -> tuple[bool, str]:
    """±MASK일 내 휴일이 있으면 (True, '이름(+offset)'). 없으면 (False, '')."""
    best = None
    for off in range(-MASK, MASK + 1):
        name = holidays.get(d + pd.Timedelta(days=off))
        if name:
            tag = name if off == 0 else f"{name}({off:+d}일)"
            if best is None or abs(off) < best[0]:
                best = (abs(off), tag)
    return (True, best[1]) if best else (False, "")


def _classify(df: pd.DataFrame, date_col: str, holidays: dict) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    flags = df[date_col].map(lambda d: _nearest_holiday(d, holidays))
    df["holiday_related"] = [f[0] for f in flags]
    df["holiday_tag"] = [f[1] for f in flags]
    df["bucket"] = df["holiday_related"].map(lambda x: "A_모델갭(휴일)" if x else "B_미설명(고객문의)")
    return df


def main() -> None:
    holidays = _holiday_lookup()

    days = _classify(pd.read_csv("reports/anomaly_days.csv"), "date", holidays)
    weeks = _classify(pd.read_csv("reports/anomaly_weeks.csv"), "start", holidays)

    days.to_csv("reports/anomaly_days_classified.csv", index=False)
    weeks.to_csv("reports/anomaly_weeks_classified.csv", index=False)

    cols = ["date", "dow", "actual", "expected", "pct", "z", "direction", "tier", "bucket", "holiday_tag"]
    a = days[days["bucket"].str.startswith("A")]
    b = days[days["bucket"].str.startswith("B")]
    print(f"=== 버킷 A — 모델 갭(휴일 미스, 내부 prior 확장): {len(a)}일 ===")
    print(a[cols].to_string(index=False))
    print(f"\n=== 버킷 B — 진짜 미설명(고객 문의): {len(b)}일 ===")
    print(b[cols].to_string(index=False))
    print(f"\n=== 주별 이상치 ({len(weeks)}주) ===")
    print(weeks[["start", "end", "n_days", "actual", "expected", "wpe_pct", "z", "direction", "bucket", "holiday_tag"]].to_string(index=False))
    print("\nwrote reports/anomaly_{days,weeks}_classified.csv")


if __name__ == "__main__":
    main()

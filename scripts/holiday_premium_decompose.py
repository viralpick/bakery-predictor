"""전 공휴일 프리미엄 분해 — 요일·주말·연휴·대체공휴일 축 (광교).

계산은 `bakery.analysis.holiday_premium.decompose_holiday_premium`로 옮겼다(Phase 6).
이 스크립트는 동결 시리즈(tests/fixtures/frozen/raw_adjusted_series.csv, 2026-07-16
생성 원본을 추적 fixture로 이동한 것 — Fix round 1)를 읽어 표를 print하는 wrapper다.
이 경로를 tests/test_holiday_premium.py의 FROZEN_SERIES와 동일하게 맞춰
wrapper와 golden 테스트가 서로 다른 입력을 보는 drift를 원천 차단한다.
현 vintage 실행은 `uv run bakery analysis-run experiments/analysis_gwangyo.yaml`을 쓴다.

실행: PYTHONPATH=scripts uv run python scripts/holiday_premium_decompose.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from bakery.analysis.holiday_premium import decompose_holiday_premium as decompose  # noqa: E402
from bakery.data.calendar import build_calendar_daily  # noqa: E402

SERIES = Path("tests/fixtures/frozen/raw_adjusted_series.csv")
OUT = Path("reports/holiday_premium_decompose.csv")


def run() -> None:
    series = pd.read_csv(SERIES, parse_dates=["date"])[["date", "adjusted_demand_unit"]]
    calendar = build_calendar_daily(series["date"].min(), series["date"].max())
    tables = decompose(series, calendar)
    print(f"=== 광교 공휴일 프리미엄 분해 (n_공휴일={len(tables['by_holiday'])}) ===")
    for label in ("dow_class", "event_ranking", "streak_buckets"):
        print(f"\n--- {label} ---")
        print(tables[label].to_string(index=False))
    tables["full"].to_csv(OUT, index=False)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    run()

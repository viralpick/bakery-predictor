"""holiday_premium 프리미티브 — 동결 입력 golden 대조.

golden은 2026-07-28에 `scripts/holiday_premium_decompose.py`를 실제 실행해 캡처한 값이다
(원본은 reports/raw_adjusted_series.csv, 2026-07-16 생성, 동결). docs 기록 수치를 쓰지 않는
이유는 Phase 7 신규데이터 편입으로 값이 이동했기 때문이다(측정 헌장/회귀 게이트 규칙).

동결 CSV는 tests/fixtures/frozen/에 추적본으로 두었다(reports/는 gitignore 대상이라
다른 머신/CI에서는 조용히 skip되어 이 게이트가 무력화됐었다 — Fix round 1 교정).
"""
from pathlib import Path

import pandas as pd
import pytest

from bakery.analysis.holiday_premium import (
    HALFWIN,
    MIN_BASELINE_SAMPLES,
    decompose_holiday_premium,
    local_dow_baseline,
)
from bakery.data.calendar import build_calendar_daily

FROZEN_SERIES = Path("tests/fixtures/frozen/raw_adjusted_series.csv")


@pytest.fixture(scope="module")
def frozen_tables():
    # 추적 fixture다 — 없으면 체크아웃이 깨진 것이므로 skip이 아니라 실패여야 한다
    # (gitignore된 reports/ 를 읽던 옛 경로에서는 CI가 조용히 skip하고 green으로 보였다).
    assert FROZEN_SERIES.exists(), f"{FROZEN_SERIES} 없음 — 추적 fixture가 누락됐다"
    series = pd.read_csv(FROZEN_SERIES, parse_dates=["date"])[["date", "adjusted_demand_unit"]]
    calendar = build_calendar_daily(series["date"].min(), series["date"].max())
    return decompose_holiday_premium(series, calendar)


def test_constants():
    assert HALFWIN == 6
    assert MIN_BASELINE_SAMPLES == 3


def test_dow_class_golden(frozen_tables):
    """golden(2026-07-28 캡처): 평일 n=71 median 1.25 [1.10,1.38] / 주말 n=22 0.89 [0.78,1.00]."""
    dow_class = frozen_tables["dow_class"].set_index("dow_class")
    weekday = dow_class.loc["평일"]
    weekend = dow_class.loc["주말"]
    assert weekday["n"] == 71
    assert round(weekday["median_lift"], 2) == 1.25
    assert round(weekday["q25"], 2) == 1.10
    assert round(weekday["q75"], 2) == 1.38
    assert weekend["n"] == 22
    assert round(weekend["median_lift"], 2) == 0.89
    assert round(weekend["q25"], 2) == 0.78
    assert round(weekend["q75"], 2) == 1.00


def test_event_ranking_golden_top_entries(frozen_tables):
    """golden: Christmas Day 1.52(n=3) 1위, New Year's Day 1.42(n=3) 2위."""
    ranking = frozen_tables["event_ranking"]
    top = ranking.iloc[0]
    assert top["base_name"] == "Christmas Day"
    assert round(top["median_lift"], 2) == 1.52
    assert top["n_weekday"] == 3
    second = ranking.iloc[1]
    assert second["base_name"] == "New Year's Day"
    assert round(second["median_lift"], 2) == 1.42
    assert second["n_weekday"] == 3


def test_event_ranking_is_sorted_descending(frozen_tables):
    lifts = frozen_tables["event_ranking"]["median_lift"].tolist()
    assert len(lifts) == 21        # 빈 리스트로 무증상 통과 방지
    assert lifts == sorted(lifts, reverse=True)


def test_by_holiday_row_count_matches_dow_class_total(frozen_tables):
    by_holiday = frozen_tables["by_holiday"]
    assert by_holiday["lift"].notna().sum() == 71 + 22


def test_streak_buckets_golden(frozen_tables):
    """golden(2026-07-28): 고립 n=20 median 1.345475 / streak2 n=0(NaN) / 연휴 n=51 median 1.190351."""
    buckets = frozen_tables["streak_buckets"].set_index("streak_bucket")
    assert buckets.index.tolist() == ["1(고립)", "2", "3+(연휴)"]
    assert buckets.loc["1(고립)", "n"] == 20
    assert round(buckets.loc["1(고립)", "median_lift"], 6) == 1.345475
    assert buckets.loc["2", "n"] == 0
    assert pd.isna(buckets.loc["2", "median_lift"])       # 관측 0건 → median 정의 불가
    assert buckets.loc["3+(연휴)", "n"] == 51
    assert round(buckets.loc["3+(연휴)", "median_lift"], 6) == 1.190351


def test_local_dow_baseline_uses_same_dow_only():
    """동일요일 ±HALFWIN 주 median. 공휴일은 baseline에서 제외된다."""
    dates = pd.date_range("2025-01-06", periods=28, freq="D")   # 월요일 시작
    series = pd.DataFrame({"date": dates, "adjusted_demand_unit": range(100, 128)})
    calendar = pd.DataFrame({"date": dates, "is_public_holiday": 0})
    baseline = local_dow_baseline(series, calendar)
    # 2025-01-13(월)의 동일요일 이웃 = 01-06(100), 01-20(114), 01-27(121) → median 114
    assert baseline[pd.Timestamp("2025-01-13")] == 114.0


def test_local_dow_baseline_nan_when_too_few_samples():
    dates = pd.date_range("2025-01-06", periods=8, freq="D")
    series = pd.DataFrame({"date": dates, "adjusted_demand_unit": range(8)})
    calendar = pd.DataFrame({"date": dates, "is_public_holiday": 0})
    baseline = local_dow_baseline(series, calendar)
    # 월요일 이웃이 1개(01-13)뿐 → MIN_BASELINE_SAMPLES=3 미달 → NaN
    assert pd.isna(baseline[pd.Timestamp("2025-01-06")])


def test_script_delegates_to_primitive():
    import sys
    sys.path.insert(0, "scripts")
    import holiday_premium_decompose

    assert holiday_premium_decompose.decompose is decompose_holiday_premium

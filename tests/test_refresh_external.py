"""bakery.ingest.refresh 순수 헬퍼(append_new_dates/gap_days) + SourceSpec 배선 테스트.

Task 9: 외부 8종 소스 통합 갱신 CLI. API 호출 없이 확인 가능한 부분만 검증한다
(실제 fetch는 .env 필요 — 이 테스트 스위트는 dry-run/순수 로직만 다룬다).
"""
from __future__ import annotations

import pandas as pd
import pytest

from bakery.ingest import refresh


def test_observed_appends_only_new_dates():
    existing = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "v": [1, 2]})
    fetched = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-03"]), "v": [2, 3]})
    merged = refresh.append_new_dates(existing, fetched, date_col="date")
    assert list(merged["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert merged["v"].tolist() == [1, 2, 3]  # 기존 유지, 신규만 추가


def test_append_is_idempotent_on_repeat():
    existing = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "v": [1]})
    once = refresh.append_new_dates(existing, existing, date_col="date")
    twice = refresh.append_new_dates(once, existing, date_col="date")
    assert once.equals(twice)


def test_append_new_dates_works_on_non_timestamp_key():
    """append_new_dates는 date뿐 아니라 ym/quarter 같은 문자열 키에도 동작해야 한다
    (population="ym", consumption="quarter" 소스가 이 경로를 그대로 재사용)."""
    existing = pd.DataFrame({"quarter": ["2025Q3", "2025Q4"], "v": [1, 2]})
    fetched = pd.DataFrame({"quarter": ["2025Q4", "2026Q1"], "v": [99, 3]})
    merged = refresh.append_new_dates(existing, fetched, date_col="quarter")
    assert merged["quarter"].tolist() == ["2025Q3", "2025Q4", "2026Q1"]
    assert merged["v"].tolist() == [1, 2, 3]  # 2025Q4는 기존값(2) 유지, 신규(99) 무시


def test_freshness_gap_days():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-07-20"])})
    gap = refresh.gap_days(df, today=pd.Timestamp("2026-07-25"), date_col="date")
    assert gap == 5


def test_gap_days_empty_df_returns_sentinel():
    df = pd.DataFrame({"date": pd.to_datetime([])})
    gap = refresh.gap_days(df, today=pd.Timestamp("2026-07-25"), date_col="date")
    assert gap == -1


def test_external_sources_registers_eight_specs():
    names = {spec.name for spec in refresh.EXTERNAL_SOURCES.values()}
    assert len(refresh.EXTERNAL_SOURCES) == 8
    assert {"calendar", "weather", "living_population", "population", "consumption", "competitor",
            "forecast_short_term_daily", "forecast_mid_term_daily"} == names


def test_select_sources_all_returns_all_specs():
    specs = refresh.select_sources("all")
    assert len(specs) == 8


def test_select_sources_single_name():
    specs = refresh.select_sources("weather")
    assert [s.name for s in specs] == ["weather"]


def test_select_sources_unknown_name_raises():
    with pytest.raises(KeyError):
        refresh.select_sources("not-a-real-source")


def test_refresh_source_dry_run_never_calls_refresh_fn():
    """dry_run=True는 refresh_fn(실제 API 호출)을 절대 호출하지 않는다 — freshness만 보고."""
    called = {"count": 0}

    def _boom():
        called["count"] += 1
        raise AssertionError("dry_run must not invoke refresh_fn")

    spec = refresh.SourceSpec(
        name="weather", dataset_key="weather_observed", kind="observed",
        date_col="date", refresh_fn=_boom,
    )
    result = refresh.refresh_source(spec, today=pd.Timestamp("2026-07-25"), dry_run=True)
    assert called["count"] == 0
    assert result.name == "weather"
    assert result.added_rows == 0

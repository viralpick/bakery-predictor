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


# --- coverage guard (리뷰 후속: data-loss 방지) ---------------------------
#
# 모든 어댑터가 자기 fetch 창만으로 parquet을 통째 덮어쓰므로, fetch 창이
# on-disk 이력보다 좁으면(예: weather 기본 시작일 2024가 2021년부터 있는
# 파일을 덮어씀) 과거 이력이 조용히 삭제될 수 있다. 아래는 실제 API 없이
# 가짜 refresh_fn으로 그 축소 시나리오를 재현해 가드를 검증한다.

def _write_parquet(path, dates: list[str], values: list[int]) -> None:
    pd.DataFrame({"date": pd.to_datetime(dates), "v": values}).to_parquet(path, index=False)


def test_coverage_guard_restores_snapshot_on_shrink(tmp_path, monkeypatch):
    target = tmp_path / "fake_obs.parquet"
    _write_parquet(target, ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
                    [0, 1, 2, 3, 4])
    monkeypatch.setattr(refresh.paths, "dataset", lambda name: target)

    def _shrink_fetch():
        _write_parquet(target, ["2024-01-01", "2025-01-01"], [30, 40])
        return target

    spec = refresh.SourceSpec(name="fake_obs", dataset_key="fake_obs", kind="observed",
                               date_col="date", refresh_fn=_shrink_fetch)
    result = refresh.refresh_source(spec, today=pd.Timestamp("2026-07-25"), dry_run=False, force=False)

    restored = pd.read_parquet(target)
    assert len(restored) == 5  # 원본 5행으로 복원됨
    assert restored["v"].tolist() == [0, 1, 2, 3, 4]
    assert result.applied is False
    assert result.added_rows == 0
    assert result.message is not None and "shrink" in result.message


def test_coverage_guard_force_keeps_shrunk_result(tmp_path, monkeypatch):
    target = tmp_path / "fake_obs.parquet"
    _write_parquet(target, ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
                    [0, 1, 2, 3, 4])
    monkeypatch.setattr(refresh.paths, "dataset", lambda name: target)

    def _shrink_fetch():
        _write_parquet(target, ["2024-01-01", "2025-01-01"], [30, 40])
        return target

    spec = refresh.SourceSpec(name="fake_obs", dataset_key="fake_obs", kind="observed",
                               date_col="date", refresh_fn=_shrink_fetch)
    result = refresh.refresh_source(spec, today=pd.Timestamp("2026-07-25"), dry_run=False, force=True)

    kept = pd.read_parquet(target)
    assert len(kept) == 2  # force=True → 축소된 결과 그대로 유지
    assert kept["v"].tolist() == [30, 40]
    assert result.applied is True
    assert result.added_rows == 2 - 5


def test_coverage_guard_keeps_superset_without_restore(tmp_path, monkeypatch):
    target = tmp_path / "fake_obs.parquet"
    _write_parquet(target, ["2023-01-01", "2024-01-01", "2025-01-01"], [2, 3, 4])
    monkeypatch.setattr(refresh.paths, "dataset", lambda name: target)

    def _superset_fetch():
        _write_parquet(
            target,
            ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
             "2023-01-01", "2024-01-01", "2025-01-01"],
            [-4, -3, -2, -1, 2, 3, 4],
        )
        return target

    spec = refresh.SourceSpec(name="fake_obs", dataset_key="fake_obs", kind="observed",
                               date_col="date", refresh_fn=_superset_fetch)
    result = refresh.refresh_source(spec, today=pd.Timestamp("2026-07-25"), dry_run=False, force=False)

    kept = pd.read_parquet(target)
    assert len(kept) == 7  # superset 그대로 유지, 원복 안 됨
    assert result.applied is True
    assert result.added_rows == 7 - 3


def test_coverage_guard_skipped_for_forecast_kind(tmp_path, monkeypatch):
    """forecast는 창이 매 호출마다 미래로 슬라이드하는 게 정상이라 가드 대상에서 제외."""
    target = tmp_path / "fake_forecast.parquet"
    _write_parquet(target, ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"],
                    [0, 1, 2, 3, 4])
    monkeypatch.setattr(refresh.paths, "dataset", lambda name: target)

    def _sliding_fetch():
        _write_parquet(target, ["2026-07-26", "2026-07-27"], [6, 7])
        return target

    spec = refresh.SourceSpec(name="fake_forecast", dataset_key="fake_forecast", kind="forecast",
                               date_col="date", refresh_fn=_sliding_fetch)
    result = refresh.refresh_source(spec, today=pd.Timestamp("2026-07-25"), dry_run=False, force=False)

    kept = pd.read_parquet(target)
    assert len(kept) == 2  # 가드 미적용 — 어댑터 출력 그대로
    assert result.applied is True

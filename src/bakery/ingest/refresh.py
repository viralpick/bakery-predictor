"""외부 8종 소스 통합 갱신 (`bakery refresh-external`).

★어댑터 실측(Task 9 조사) — 브리프의 append 모델과 다름★
기존 `ingest/*_api.py`의 backfill 함수는 전부:
  - 자기 자신이 parquet을 직접 쓴다 (dry-run 아닌 이상 side-effect).
  - 반환값은 raw rows DataFrame이 아니라 `Path`(또는 forecast는 `dict[str, Path]`).
  - fetch한 "이번 창(window)"만으로 파일을 통째로 덮어쓴다 — 기존 on-disk 데이터와
    병합하지 않는다 (예: weather_api.backfill(start, end)는 [start, end] 구간만
    combined해서 weather_observed.parquet을 overwrite).

따라서 이 모듈은 브리프가 가정한 "refresh_fn이 신규 rows를 반환 → append_new_dates로
병합" 파이프라인을 강제하지 않는다. 대신:
  - `refresh_fn`은 기존 CLI(`ingest-*`)가 쓰던 것과 동일한 기본 인자로 실제
    backfill을 호출하는 얇은 콜러블(재구현 금지, 그대로 호출만).
  - `refresh_source`는 fetch 전/후의 on-disk parquet을 diff해서 added_rows를
    계산한다 — 병합이 아니라 "어댑터가 덮어쓴 결과를 관측"하는 방식.
  - `append_new_dates`/`gap_days`는 그 자체로 테스트된 순수 헬퍼로 남긴다. 실제
    소스 중 이 append 모델이 "그대로" 맞는 곳은 없다(모든 adapter가 자체
    overwrite) — 대신 순수 함수로서 정확성이 검증되어 있고, 향후 어댑터가
    raw rows를 반환하는 형태로 바뀌면 바로 재사용 가능하다.

⚠️ 알려진 한계 (그대로 두고 리포트에 기록):
  - `living_population_api.backfill`은 서울 열린데이터광장 rolling ~2개월
    윈도우만 반환한다. on-disk `living_population.parquet`은 2017년까지의
    CSV backfill 이력을 포함하므로, refresh_fn 호출 시 어댑터가 파일을
    "최근 2개월분"으로 통째 덮어써 과거 이력이 사라진다. 이 위험 때문에
    실제 fetch 경로는 이 태스크에서 실행하지 않았고(.env 미가정), dry_run만
    검증했다. 실사용 전 living_population_api.backfill 자체에 병합 로직을
    추가하거나 별도 CSV 재적재 스텝을 거쳐야 한다.
  - forecast(short/mid)는 둘 다 동일한 `forecast_api.backfill_forecast()` 한
    호출로 채워진다. `--source all`로 갱신하면 이 함수가 두 번 불릴 수 있어
    API 호출이 중복되지만(허용 budget 내), 별도 dedup은 하지 않았다(KISS).
  - freshness 기준 컬럼은 소스마다 다르다: 실제 날짜(`date`/`fcst_date`) 4종,
    월 스냅샷(`ym`) 1종, 분기(`quarter`) 1종, 날짜 개념이 아예 없는 경우
    (`competitor_raw`, business_id 단위 스냅샷) 1종 — 이 마지막 경우는 parquet
    파일의 mtime을 freshness 근사치로 쓴다.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from ..data import paths
from . import (
    calendar_api,
    competitor_api,
    consumption_api,
    forecast_api,
    living_population_api,
    population_api,
    weather_api,
)

_SENTINEL_GAP_DAYS = -1  # 데이터 없음(빈 df) — 관례적 sentinel


def append_new_dates(existing: pd.DataFrame, fetched: pd.DataFrame,
                      date_col: str) -> pd.DataFrame:
    """기존 키(date_col 값)는 보존, fetched의 신규 키만 추가.

    date_col은 실제 날짜(Timestamp)일 필요 없다 — 정렬/비교 가능한 값이면
    ym("2026-04")·quarter("2025Q4") 같은 문자열 키로도 동작한다.
    """
    have = set(existing[date_col])
    new = fetched[~fetched[date_col].isin(have)]
    if new.empty:
        return existing.reset_index(drop=True)
    return (pd.concat([existing, new], ignore_index=True)
            .sort_values(date_col).reset_index(drop=True))


def gap_days(df: pd.DataFrame, today: pd.Timestamp, date_col: str) -> int:
    """오늘과 df의 최신 date_col 값 사이 일수. 빈 df는 -1(sentinel)."""
    if df.empty:
        return _SENTINEL_GAP_DAYS
    return int((today.normalize() - df[date_col].max().normalize()).days)


@dataclass
class SourceSpec:
    name: str
    dataset_key: str          # paths.dataset(...) 키
    kind: str                 # "observed" | "forecast"
    refresh_fn: Callable[[], object]  # 기존 ingest_*.backfill을 그대로 감싼 콜러블
    date_col: str | None = "date"     # None=날짜 개념 없음(mtime fallback)


@dataclass
class RefreshResult:
    name: str
    added_rows: int
    last_date: pd.Timestamp | None
    gap_days: int


def _read_dataset(dataset_key: str) -> pd.DataFrame:
    path = paths.dataset(dataset_key)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _period_to_timestamp(df: pd.DataFrame, date_col: str) -> pd.Series:
    """ym("2026-04")/quarter("2025Q4") 같은 기간 문자열을 대표 Timestamp로 변환."""
    sample = str(df[date_col].iloc[0])
    if "Q" in sample:
        return df[date_col].map(lambda q: pd.Period(q, freq="Q").end_time.normalize())
    return pd.to_datetime(df[date_col], format="%Y-%m")


def _freshness(df: pd.DataFrame, spec: SourceSpec) -> tuple[pd.Timestamp | None, int]:
    """(last_date, gap_days). competitor처럼 date_col=None이면 parquet mtime을 근사치로 쓴다."""
    if spec.date_col is None:
        path = paths.dataset(spec.dataset_key)
        if not path.exists():
            return None, _SENTINEL_GAP_DAYS
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s").normalize()
        return mtime, int((pd.Timestamp.today().normalize() - mtime).days)
    if df.empty:
        return None, _SENTINEL_GAP_DAYS
    if spec.date_col in ("ym", "quarter"):
        ts_col = _period_to_timestamp(df, spec.date_col)
        wrapped = pd.DataFrame({"_ts": ts_col})
        last_date = wrapped["_ts"].max()
        gap = gap_days(wrapped, today=pd.Timestamp.today(), date_col="_ts")
        return last_date, gap
    last_date = pd.to_datetime(df[spec.date_col]).max()
    gap = gap_days(df.assign(**{spec.date_col: pd.to_datetime(df[spec.date_col])}),
                    today=pd.Timestamp.today(), date_col=spec.date_col)
    return last_date, gap


def refresh_source(spec: SourceSpec, today: pd.Timestamp, dry_run: bool = False) -> RefreshResult:
    """소스 하나 갱신. dry_run=True면 refresh_fn(실제 API 호출)을 절대 부르지 않고
    현재 on-disk freshness만 보고한다.

    observed/forecast 구분은 어댑터 자체의 overwrite 동작에 내재돼 있다(둘 다
    refresh_fn 호출 후 파일 전체를 다시 읽어 diff하는 동일 코드 경로) — 여기서
    별도 분기하지 않는다.
    """
    before = _read_dataset(spec.dataset_key)
    if dry_run:
        last_date, gap = _freshness(before, spec)
        return RefreshResult(name=spec.name, added_rows=0, last_date=last_date, gap_days=gap)

    spec.refresh_fn()
    after = _read_dataset(spec.dataset_key)
    added_rows = len(after) - len(before)
    last_date, gap = _freshness(after, spec)
    return RefreshResult(name=spec.name, added_rows=added_rows, last_date=last_date, gap_days=gap)


def freshness_summary(specs: list[SourceSpec]) -> pd.DataFrame:
    """각 소스의 현재 on-disk freshness 요약 (fetch 없음 — dry_run과 동일 관측)."""
    rows = []
    for spec in specs:
        df = _read_dataset(spec.dataset_key)
        last_date, gap = _freshness(df, spec)
        rows.append({"source": spec.name, "last_date": last_date, "gap_days": gap})
    return pd.DataFrame(rows)


def select_sources(source: str) -> list[SourceSpec]:
    """source="all"이면 8종 전부, 아니면 이름 하나. 미등록 이름은 KeyError."""
    if source == "all":
        return list(EXTERNAL_SOURCES.values())
    if source not in EXTERNAL_SOURCES:
        raise KeyError(f"unknown source '{source}'. known: {sorted(EXTERNAL_SOURCES)}")
    return [EXTERNAL_SOURCES[source]]


def _default_calendar_backfill() -> object:
    return calendar_api.backfill(2024, Date.today().year)


def _default_weather_backfill() -> object:
    return weather_api.backfill(Date(2024, 1, 1), Date.today())


def _default_living_population_backfill() -> object:
    start = Date.today() - timedelta(days=30)
    return living_population_api.backfill(start, Date.today())


EXTERNAL_SOURCES: dict[str, SourceSpec] = {
    spec.name: spec
    for spec in [
        SourceSpec("calendar", "calendar_raw", "observed",
                    refresh_fn=_default_calendar_backfill, date_col="date"),
        SourceSpec("weather", "weather_observed", "observed",
                    refresh_fn=_default_weather_backfill, date_col="date"),
        SourceSpec("living_population", "living_population", "observed",
                    refresh_fn=_default_living_population_backfill, date_col="date"),
        SourceSpec("population", "population", "observed",
                    refresh_fn=population_api.backfill, date_col="ym"),
        SourceSpec("consumption", "consumption", "observed",
                    refresh_fn=consumption_api.backfill, date_col="quarter"),
        SourceSpec("competitor", "competitor_raw", "observed",
                    refresh_fn=competitor_api.backfill, date_col=None),
        SourceSpec("forecast_short_term_daily", "forecast_short_term_daily", "forecast",
                    refresh_fn=forecast_api.backfill_forecast, date_col="date"),
        SourceSpec("forecast_mid_term_daily", "forecast_mid_term_daily", "forecast",
                    refresh_fn=forecast_api.backfill_forecast, date_col="fcst_date"),
    ]
}

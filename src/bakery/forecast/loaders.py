"""forward 예측이 의존하는 실데이터 로더.

cli.py의 item·decision 경로와 공유되므로 cli는 이 모듈을 alias로 import back한다
(`_load_forecast_weather`, `_load_real_daily`).
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from ..data import paths
from ..data.weather import load_weather_forecast_from_local
from ..features.category_aggregate import TARGET_CATEGORIES
from ..ingest.store_mapping import load_store_mapping

console = Console()

# 실데이터 진입점 — 재고정보 시트가 있는 파일만 생산량/폐기량/품목단가를 갖는다.
REAL_DAILY_PARQUET_PATH = str(paths.dataset("bonavi_daily"))


def load_forecast_weather(horizon: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Long-form horizon weather frame keyed by (store_id, date), one row per
    (store, day) — each store's nx/ny/mid_reg from the store mapping is
    matched against the latest forecast parquet, falling back to recent
    observed averages when the forecast is missing.
    """
    short_p = paths.dataset("forecast_short_term_daily")
    mid_p = paths.dataset("forecast_mid_term_daily")
    observed_p = paths.dataset("weather_observed")
    if not short_p.exists() and not mid_p.exists():
        console.print(
            "[yellow]forecast[/] parquet 없음 — `bakery ingest-forecast` 먼저 실행. "
            "이번엔 fallback (최근 28일 평균)으로 horizon 채움."
        )
    mapping = load_store_mapping()
    return load_weather_forecast_from_local(
        short_daily_path=short_p,
        mid_daily_path=mid_p,
        observed_parquet_path=observed_p,
        mapping=mapping,
        horizon_start=horizon[0],
        horizon_end=horizon[-1],
    )


def load_real_daily(store_id: str) -> pd.DataFrame:
    """bonavi_daily.parquet을 store_id + TARGET_CATEGORIES로 필터."""
    daily = pd.read_parquet(REAL_DAILY_PARQUET_PATH)
    n_stores = daily["store_id"].nunique()
    if n_stores != 1:
        raise ValueError(f"real path assumes single-store data; found {n_stores} stores. Multi-store needs store-qualified receipts/merge wiring.")
    daily["item_id"] = daily["item_id"].astype(str)
    daily = daily[daily["store_id"] == store_id]
    daily = daily[daily["category_id"].isin(TARGET_CATEGORIES)]
    return daily.reset_index(drop=True)

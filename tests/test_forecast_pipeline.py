"""Forecast pipeline: precipitation text parsing, daily aggregation, mid-term
melt, and the short→mid→fallback selection logic in load_weather_forecast_from_local.

External API calls themselves are not exercised here — they're covered by the
manual smoke runs that produced data/external/forecast_*.parquet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bakery.data.weather import (
    WEATHER_DAILY_COLUMNS,
    load_weather_forecast_from_local,
)
from bakery.ingest.forecast_api import (
    aggregate_short_term_to_daily,
    latest_mid_term_tmfc,
    latest_short_term_base,
    parse_precipitation,
)


def test_parse_precipitation_handles_kma_strings():
    assert parse_precipitation("강수없음") == 0.0
    assert parse_precipitation("") == 0.0
    assert parse_precipitation(None) == 0.0
    assert parse_precipitation("1.0mm") == 1.0
    assert parse_precipitation("3.5 mm") == 3.5
    assert parse_precipitation("30~50mm") == 40.0
    assert parse_precipitation("1.0mm 미만") == 0.5
    assert parse_precipitation("50mm 이상") == 60.0


def test_latest_short_term_base_picks_recent_publish():
    # 14:35 → published 14:00 already available (14:00 + 30min lag)
    base_date, base_time = latest_short_term_base(now=pd.Timestamp("2026-05-18 14:35"))
    assert base_date == "20260518"
    assert base_time == "1400"
    # 02:10 → too early for 02:00 (only 10min after publish); falls back to yesterday 23:00
    bd2, bt2 = latest_short_term_base(now=pd.Timestamp("2026-05-18 02:10"))
    assert bd2 == "20260517"
    assert bt2 == "2300"


def test_latest_mid_term_tmfc_picks_recent_publish():
    tm = latest_mid_term_tmfc(now=pd.Timestamp("2026-05-18 19:30"))
    assert tm == "202605181800"
    tm2 = latest_mid_term_tmfc(now=pd.Timestamp("2026-05-18 03:00"))
    assert tm2 == "202605171800"


def test_aggregate_short_term_to_daily_basic():
    """Synthetic short-term frame: TMP, TMX/TMN, REH, PCP, PTY → daily aggregates."""
    rows = []
    base = pd.Timestamp("2026-06-01 05:00")
    for hour in (6, 9, 12, 15, 18, 21):
        fdt = pd.Timestamp("2026-06-01") + pd.Timedelta(hours=hour)
        for cat, val in [("TMP", 18 + hour / 2), ("REH", 60), ("PTY", 0), ("SKY", 1)]:
            rows.append({"base_dt": base, "fcst_dt": fdt, "nx": 60, "ny": 127, "category": cat, "fcstValue": str(val)})
    rows.append({"base_dt": base, "fcst_dt": pd.Timestamp("2026-06-01 15:00"), "nx": 60, "ny": 127, "category": "TMX", "fcstValue": "28"})
    rows.append({"base_dt": base, "fcst_dt": pd.Timestamp("2026-06-01 06:00"), "nx": 60, "ny": 127, "category": "TMN", "fcstValue": "17"})
    rows.append({"base_dt": base, "fcst_dt": pd.Timestamp("2026-06-01 12:00"), "nx": 60, "ny": 127, "category": "PCP", "fcstValue": "강수없음"})
    raw = pd.DataFrame(rows)
    daily = aggregate_short_term_to_daily(raw)
    assert len(daily) == 1
    r = daily.iloc[0]
    assert r["max_temp"] == 28.0  # from TMX, not from TMP max
    assert r["min_temp"] == 17.0  # from TMN
    assert r["humidity"] == 60.0
    assert r["precipitation_mm"] == 0.0


def test_load_weather_forecast_uses_short_when_available(tmp_path):
    """Short-term covers D+0~D+3; the loader should pick those rows first."""
    short_p = tmp_path / "short.parquet"
    mid_p = tmp_path / "mid.parquet"
    obs_p = tmp_path / "obs.parquet"
    horizon_start = pd.Timestamp("2026-06-01")
    horizon_end = pd.Timestamp("2026-06-07")
    short_df = pd.DataFrame(
        {
            "nx": [60, 60, 60],
            "ny": [127, 127, 127],
            "date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "avg_temp": [22.0, 23.0, 24.0],
            "max_temp": [27.0, 28.0, 29.0],
            "min_temp": [17.0, 18.0, 19.0],
            "humidity": [55.0, 60.0, 65.0],
            "precipitation_mm": [0.0, 5.0, 0.0],
            "snow_depth_cm": [0.0, 0.0, 0.0],
            "max_pop": [10.0, 70.0, 20.0],
            "any_rain_pty": [0, 1, 0],
            "any_snow_pty": [0, 0, 0],
            "sky_modal": [1, 3, 1],
        }
    )
    short_df.to_parquet(short_p, index=False)
    mid_df = pd.DataFrame(
        {
            "tm_fc": [pd.Timestamp("2026-05-29 18:00")] * 4,
            "ta_reg_id": ["11B10101"] * 4,
            "day_offset": [4, 5, 6, 7],
            "fcst_date": pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07"]),
            "taMin": [18.0, 19.0, 17.0, 20.0],
            "taMax": [28.0, 29.0, 27.0, 30.0],
            "reg_id": ["11B00000"] * 4,
            "rnSt_am": [20, 30, 60, 20],
            "rnSt_pm": [20, 40, 70, 30],
            "wf_am": ["맑음", "구름", "비", "맑음"],
            "wf_pm": ["맑음", "구름", "비", "맑음"],
            "mid_land_reg_id": ["11B00000"] * 4,
            "mid_ta_reg_id": ["11B10101"] * 4,
        }
    )
    mid_df.to_parquet(mid_p, index=False)
    # Empty observed → fallback uses hardcoded defaults
    out = load_weather_forecast_from_local(
        short_p, mid_p, obs_p,
        station_id=108, nx=60, ny=127,
        mid_land_reg_id="11B00000", mid_ta_reg_id="11B10101",
        horizon_start=horizon_start, horizon_end=horizon_end,
    )
    assert len(out) == 7
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == [
        "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
        "2026-06-05", "2026-06-06", "2026-06-07",
    ]
    # D+0~D+2 from short
    assert out.iloc[0]["max_temp"] == 27.0
    # D+5 (2026-06-06) is the rainy day per mid forecast (rnSt 60/70 → is_rain=1)
    rainy = out[out["date"] == pd.Timestamp("2026-06-06")].iloc[0]
    assert rainy["is_rain"] == 1
    # All schema columns present
    for col in WEATHER_DAILY_COLUMNS:
        assert col in out.columns


def test_load_weather_forecast_falls_back_when_files_missing(tmp_path):
    out = load_weather_forecast_from_local(
        tmp_path / "missing_short.parquet",
        tmp_path / "missing_mid.parquet",
        tmp_path / "missing_obs.parquet",
        station_id=108, nx=60, ny=127,
        mid_land_reg_id="11B00000", mid_ta_reg_id="11B10101",
        horizon_start=pd.Timestamp("2026-06-01"),
        horizon_end=pd.Timestamp("2026-06-07"),
    )
    assert len(out) == 7
    # All rows should use the hard-coded fallback defaults (humidity=60)
    assert (out["humidity"] == 60.0).all()


def test_load_weather_forecast_horizon_validation(tmp_path):
    with pytest.raises(ValueError, match="empty horizon"):
        load_weather_forecast_from_local(
            tmp_path / "s.parquet", tmp_path / "m.parquet", tmp_path / "o.parquet",
            station_id=108, nx=60, ny=127,
            mid_land_reg_id="11B00000", mid_ta_reg_id="11B10101",
            horizon_start=pd.Timestamp("2026-06-10"),
            horizon_end=pd.Timestamp("2026-06-01"),
        )

"""c-1 미래-horizon 카테고리 예측의 leakage-safe 성질 회귀 테스트.

핵심 불변식: 미래 행 target=NaN → lag가 미래-reaching 구간에서 NaN(미래 실측 원천 차단),
fit은 미래 행 제외. 실데이터 IO 없이 합성 history로 검증한다(결정론·고속).
"""
import numpy as np
import pandas as pd

from bakery.cli import _extend_category_features
from bakery.features.category_aggregate import fill_forecast_weather

TARGET = "adjusted_demand_unit"


def _synth_hist(n_days: int = 400) -> pd.DataFrame:
    """합성 카테고리 daily history: date + target만(build_features가 나머지 파생)."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.arange(n_days, dtype=float)
    return pd.DataFrame({"date": dates, TARGET: 100.0 + 10.0 * np.sin(rng / 7.0) + rng * 0.05})


def test_future_rows_target_is_nan():
    feats, horizon = _extend_category_features(
        _synth_hist(), horizon_days=7, alpha=0.8, target_col=TARGET
    )
    fut = feats[feats["date"].isin(horizon)]
    assert len(fut) == 7
    assert fut[TARGET].isna().all()


def test_short_lag_is_leakage_free_on_horizon():
    """lag1: 첫 미래일(day+1)만 마지막 관측서 채워지고 day+2부터 NaN(미래값 미참조).
    lag7: 7일 horizon 전체가 history 도달 → 전부 non-NaN."""
    feats, horizon = _extend_category_features(
        _synth_hist(), horizon_days=7, alpha=0.8, target_col=TARGET
    )
    fut = feats[feats["date"].isin(horizon)].sort_values("date").reset_index(drop=True)
    lag1 = fut[f"{TARGET}_lag1"]
    assert not np.isnan(lag1.iloc[0])       # day+1 = 마지막 관측일 값
    assert lag1.iloc[1:].isna().all()       # day+2.. = NaN(미래 실측 없음)
    assert fut[f"{TARGET}_lag7"].notna().all()


def test_fit_set_excludes_future_dates():
    feats, horizon = _extend_category_features(
        _synth_hist(), horizon_days=7, alpha=0.8, target_col=TARGET
    )
    is_future = feats["date"].isin(horizon)
    train = feats[~is_future].dropna(subset=[TARGET])
    assert not train["date"].isin(horizon).any()
    assert len(train) == 400


def test_fill_forecast_weather_observed_priority():
    """observed 값은 forecast로 덮지 않고, 결측(미래)만 채운다."""
    d = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2026-01-01"]),
        "avgTa": [5.0, np.nan],
    })
    fw = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2026-01-01"]),
        "avgTa": [99.0, 12.0],
    })
    out = fill_forecast_weather(d, fw).set_index("date")["avgTa"]
    assert out[pd.Timestamp("2023-01-01")] == 5.0    # observed 우선
    assert out[pd.Timestamp("2026-01-01")] == 12.0   # 미래 forecast 보충


def test_fill_forecast_weather_leaves_unmapped_cols_nan():
    """forecast에 없는 컬럼(구름/풍속)은 건드리지 않아 NaN 유지."""
    d = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]),
        "avgTa": [np.nan], "avgWs": [np.nan],
    })
    fw = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "avgTa": [10.0]})
    out = fill_forecast_weather(d, fw).set_index("date")
    assert out.loc[pd.Timestamp("2026-01-01"), "avgTa"] == 10.0
    assert np.isnan(out.loc[pd.Timestamp("2026-01-01"), "avgWs"])

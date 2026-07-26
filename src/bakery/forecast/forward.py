"""카테고리 총량 forward 합성 헬퍼 — future 예측과 category backtest가 공유.

`_category_base_predict`·`_blend_event_prior`는 cli.py의 category backtest 경로
(`_category_total_fold_predictions`)와도 공유되므로 cli는 이 모듈을 import back한다.
"""

from __future__ import annotations

import pandas as pd

from ..features.category_aggregate import EVENTS, LUNAR_EVENTS, CategoryDaily, build_features
from ..models.category_total import fit_category_total
from ..models.distributional_total import fit_distributional_total
from ..models.event_prior import EventLevelPrior


def _category_base_predict(
    train: pd.DataFrame, test: pd.DataFrame, *,
    target_col: str, total_model: str, production_quantile: float,
) -> tuple:
    """train으로 카테고리 총량 모델 fit → test의 (base_median, base_prod) 반환(clip≥0).

    total_model 분기: lightgbm(production_q fit 고정) | distributional(production_q predict 시).
    fold backtest·future 예측 공용(중복 제거)."""
    import numpy as np

    if total_model == "distributional":
        model = fit_distributional_total(train, target_col=target_col)
        base_prod = np.clip(model.predict_production(test, production_q=production_quantile), 0.0, None)
    elif total_model == "lightgbm":
        model = fit_category_total(train, target_col=target_col, production_q=production_quantile)
        base_prod = np.clip(model.predict_production(test), 0.0, None)
    else:
        raise ValueError(f"unknown total_model: {total_model!r} (expected 'lightgbm' or 'distributional')")
    base_median = np.clip(model.predict_expected(test), 0.0, None)
    return base_median, base_prod


def _blend_event_prior(
    train: pd.DataFrame, dates, base_median, base_prod, *, target_col: str,
) -> tuple:
    """EventLevelPrior를 train(pre-test history)으로 fit 후 이벤트일만 레벨-앵커 블렌드.

    leakage-safe: prior는 예측창 이전 데이터로만 fit, level_for는 ed<date 엄격 필터."""
    import numpy as np

    prior = EventLevelPrior(events=EVENTS, lunar_events=LUNAR_EVENTS).fit(train, target_col=target_col)
    base_median, base_prod = prior.blend(dates, base_median, base_prod)
    return np.clip(base_median, 0.0, None), np.clip(base_prod, 0.0, None)


# item-schema 예보 → category weather 스키마 부분 매핑(기온/강수/습도만; 구름/풍속 미대응).
_FORECAST_TO_CATEGORY_WEATHER = {
    "avg_temp": "avgTa", "max_temp": "maxTa", "min_temp": "minTa",
    "precipitation_mm": "sumRn", "humidity": "avgRhm",
}


def _forecast_to_category_weather(forecast_weather: pd.DataFrame, store_id: str) -> pd.DataFrame | None:
    """_load_forecast_weather의 (store_id,date) 프레임 → category weather 스키마(date-keyed).

    기온/강수/습도만 매핑하고 rain_level은 sumRn에서 재계산. 구름(avgTca)·풍속(avgWs)·
    apparent_temp는 예보에 없어 미제공(fill_forecast_weather가 NaN 유지). store 미매칭이면 None."""
    fw = forecast_weather[forecast_weather["store_id"] == store_id]
    if fw.empty:
        return None
    out = pd.DataFrame({"date": pd.to_datetime(fw["date"].to_numpy())})
    for src_col, dst_col in _FORECAST_TO_CATEGORY_WEATHER.items():
        if src_col in fw.columns:
            out[dst_col] = fw[src_col].to_numpy()
    if "sumRn" in out.columns:
        out["rain_level"] = pd.cut(
            out["sumRn"].fillna(0), bins=[-1, 0, 5, 20, 1e9], labels=[0, 1, 2, 3]
        ).astype(int)
    return out


def _extend_category_features(
    hist: pd.DataFrame, *, horizon_days: int, alpha: float, target_col: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """history 카테고리 daily에 미래 horizon_days 행(target=NaN)을 append 후 build_features.

    미래 행의 target이 NaN이라 lag=shift가 미래-reaching 구간에서 NaN이 된다(leakage 차단).
    (feats, horizon) 반환. build_features와 분리해 leakage 회귀 테스트가 붙는 지점."""
    hist = hist.sort_values("date").reset_index(drop=True)
    last = hist["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    ext = pd.concat([hist, pd.DataFrame({"date": horizon})], ignore_index=True)
    feats = build_features(CategoryDaily(df=ext, alpha=alpha), target_col=target_col)
    return feats.sort_values("date").reset_index(drop=True), horizon

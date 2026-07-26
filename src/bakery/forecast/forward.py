"""카테고리 총량 forward 합성 헬퍼 — future 예측과 category backtest가 공유.

`_category_base_predict`·`_blend_event_prior`는 cli.py의 category backtest 경로
(`_category_total_fold_predictions`)와도 공유되므로 cli는 이 모듈을 import back한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.category_aggregate import (
    DEFAULT_ALPHA,
    EVENTS,
    LUNAR_EVENTS,
    CategoryDaily,
    build_category_daily,
    build_features,
    fill_forecast_weather,
)
from ..models.category_total import fit_category_total
from ..models.distributional_total import fit_distributional_total
from ..models.event_prior import EventLevelPrior
from ..models.item_proportion import distribute_total
from .loaders import load_forecast_weather, load_real_daily


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


@dataclass(frozen=True)
class ForwardForecast:
    """forward 2층 예측 + 중간값. 5b 설명 함수가 재계산 없이 소비.

    category_totals: [date, base_median, base_prod, prior_median, prior_prod]
        base_* = Stage1 예측(event_prior blend 전), prior_* = blend 후.
    proportions: compute_proportions 출력(target_date→date), factor 컬럼 포함.
    item_quantities: [store_id, item_id, category_id, date, demand_point, our_order].
    """
    category_totals: pd.DataFrame
    proportions: pd.DataFrame
    item_quantities: pd.DataFrame


def forecast_forward(
    store_id: str, *, daily: pd.DataFrame | None = None, horizon_days: int = 7,
    total_model: str = "lightgbm", event_prior: bool = True,
    production_quantile: float = 0.85, alpha: float = DEFAULT_ALPHA,
    use_forecast: bool = True,
) -> ForwardForecast:
    """마지막 관측일 다음 horizon_days일의 카테고리 총량 예측 → 품목 배분.

    forward-only, leakage-safe(fit은 관측 history만). daily=None이면 real 소스
    로드(cli byte-equal 경로), 아니면 주입 프레임으로 build_category_daily(테스트).
    중간값(base/prior 총량, 비중 factor)을 ForwardForecast로 노출한다.
    """
    target_col = "adjusted_demand_unit"
    if daily is None:
        daily = load_real_daily(store_id)
        hist = build_category_daily(alpha=alpha).df
    else:
        hist = build_category_daily(daily_raw=daily, alpha=alpha).df
    feats, horizon = _extend_category_features(
        hist, horizon_days=horizon_days, alpha=alpha, target_col=target_col,
    )
    if use_forecast:
        fw = load_forecast_weather(horizon)
        cat_fw = _forecast_to_category_weather(fw, store_id) if fw is not None else None
        if cat_fw is not None:
            feats = fill_forecast_weather(feats, cat_fw)
    feats = feats.sort_values("date").reset_index(drop=True)
    is_future = feats["date"].isin(horizon)
    train = feats[~is_future].dropna(subset=[target_col])
    test = feats[is_future]
    base_median, base_prod = _category_base_predict(
        train, test, target_col=target_col,
        total_model=total_model, production_quantile=production_quantile,
    )
    pre_median = np.asarray(base_median, dtype=float)   # blend 이전 스냅샷
    pre_prod = np.asarray(base_prod, dtype=float)
    if event_prior:
        base_median, base_prod = _blend_event_prior(
            train, test["date"], base_median, base_prod, target_col=target_col,
        )
    dates = test["date"].to_numpy()
    prop_result = distribute_total(daily, pd.Series(np.asarray(base_prod, dtype=float), index=dates))
    order = prop_result.quantities.rename(columns={"qty": "our_order"})
    point = distribute_total(daily, pd.Series(np.asarray(base_median, dtype=float), index=dates)) \
        .quantities.rename(columns={"qty": "demand_point"})
    preds = order.merge(point, on=["item_id", "date"], how="left")
    preds["item_id"] = preds["item_id"].astype(str)
    cat_src = daily.drop_duplicates("item_id").assign(item_id=lambda d: d["item_id"].astype(str))
    cat_map = cat_src.set_index("item_id")["category_id"]
    preds["store_id"] = store_id
    preds["category_id"] = preds["item_id"].map(cat_map)
    item_quantities = preds[
        ["store_id", "item_id", "category_id", "date", "demand_point", "our_order"]
    ].reset_index(drop=True)
    category_totals = pd.DataFrame({
        "date": dates,
        "base_median": pre_median, "base_prod": pre_prod,
        "prior_median": np.asarray(base_median, dtype=float),
        "prior_prod": np.asarray(base_prod, dtype=float),
    })
    proportions = prop_result.proportions.rename(columns={"target_date": "date"})
    return ForwardForecast(
        category_totals=category_totals, proportions=proportions,
        item_quantities=item_quantities,
    )

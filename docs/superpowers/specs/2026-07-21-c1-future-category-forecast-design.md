# c-1: 미래-horizon 카테고리 예측 (분포회귀 발주 스택)

**날짜**: 2026-07-21
**로드맵**: 발주 안전마진 표준 스택 step (c)의 공통 하드파트 c-1. (a)(b)는 PR#49로 머지 완료.

## 배경 / 문제

`prospective-eval`의 카테고리 발주 경로(`_category_total_fold_predictions`)는 **history 내부 backtest**만 한다 — test window가 과거라 feature(lag/rolling 포함)가 이미 존재한다. 운영 산출물(`predict-next-week`, `v6-predict`)은 **미래 7일**을 예측해야 하는데, 카테고리 집계 프레임에 그 날짜 행이 아예 없다. c-1은 이 **미래-horizon 카테고리 예측 경로**를 만든다. c-2a(predict-next-week 소비)·c-2b(decision σ(x)) 는 별도.

### leakage는 재귀가 아니라 "미래 y=NaN"으로 해결된다 (검증됨)

item 경로(`GlobalLGBM._join_history`)는 미래 target 행의 `y_col = np.nan`으로 두고 history와 concat한다. lag = `y.shift(k)`라 미래로 뻗는 lag(`lag1`·`rmean7`·`ewma7`)는 자동으로 NaN이 된다(미래 실측이 원천적으로 없음 = leakage 차단). `lag7/14/28`은 관측된 history에서 채워진다. LightGBM은 NaN을 native 처리하고, **NGBoost도 NaN feature를 predict에서 그대로 처리한다(실측 확인, sklearn 1.8 트리 결측 지원)**. → 재귀 예측 불필요. item 패턴을 카테고리 집계에 미러링하면 된다.

## 산출물

```python
def _category_future_order_predictions(
    store_id: str, *, horizon_days: int = 7,
    production_quantile: float = 0.85, total_model: str = "lightgbm",
    event_prior: bool = True, alpha: float = DEFAULT_ALPHA,
    use_forecast: bool = True,
) -> pd.DataFrame:  # [item_id, date, our_order]  (미래 horizon_days개 날짜)
```

c-1의 범위 = 이 함수 + leakage 회귀 테스트 + smoke run. **predict-next-week/v6-predict 배선은 하지 않는다**(c-2).

## 흐름

1. `cat_daily = build_category_daily(alpha)` — 광교 단일매장 parquet(기존 경로와 동일하게 store-agnostic 읽되 단일매장 가정).
2. 미래 horizon 날짜(마지막 관측일 +1 ~ +horizon_days) 행을 `cat_daily.df`에 append. `adjusted_demand_unit` 및 집계 컬럼 = NaN.
3. `build_features(extended)` 실행:
   - 캘린더/공휴일/이벤트: 날짜 결정론 → 미래에도 정상.
   - **날씨: 부분 예보주입.** `add_weather_features`를 forecast 프레임 override를 받도록 수정(아래). 기온(avgTa/maxTa/minTa)·강수(sumRn→rain_level)·습도(avgRhm)는 예보에서 채우고, 구름(avgTca)·풍속(avgWs)·apparent_temp는 예보에 없어 미래 행 NaN(두 모델 관용).
   - competitor: 미래 date 미매칭 시 NaN(minor, 관용).
   - lag/rolling: seam 넘어 계산 → 미래-reaching은 NaN(leakage-safe).
4. history 행(dropna)으로 fit → 미래 행 predict: median(`predict_expected`) / q0.85(`predict_production`). `total_model`로 lightgbm|distributional 선택(PR#49 배선 재사용).
5. `event_prior=True`면 EventLevelPrior 블렌드(이벤트일만; 7일 horizon엔 대개 inert).
6. `distribute_total`(history 비율)로 카테고리 총량 → item `our_order`.

## 날씨 함수 수정 (shared, 동작 보존)

`category_aggregate.add_weather_features(df, weather_path=..., station_id=...)`에 optional `forecast_weather: pd.DataFrame | None = None` 인자 추가.
- `forecast_weather=None`(기본) → **현 동작 그대로**(관측 parquet left-join). 기존 backtest·테스트 무영향.
- 프레임이 주어지면: 관측(history)으로 채운 뒤, 관측이 없는(미래) 날짜의 기온/강수/습도 컬럼을 forecast 프레임(item 스키마 `avg_temp`/`precipitation_mm`/`humidity`...)에서 매핑해 채우고 rain_level 재계산. 매핑 없는 컬럼(구름/풍속/apparent_temp)은 NaN 유지.
- 매핑: `avg_temp→avgTa, max_temp→maxTa, min_temp→minTa, precipitation_mm→sumRn, humidity→avgRhm`. store별 forecast 프레임에서 해당 store_id·date 선택.

## leakage 보장 / 테스트

- **불변식**: 미래 예측 output이 미래 actual에 무의존. `_join_history`와 동일 원리(미래 y=NaN)라 구조적으로 성립.
- 신규 테스트 `test_category_future_forecast.py`:
  1. 미래 행 target을 임의값으로 바꿔도 미래 예측 불변(leakage 없음) — item leakage 테스트 패턴.
  2. output shape: horizon_days개 날짜 × item, `our_order ≥ 0`.
  3. `add_weather_features(forecast_weather=None)`가 기존과 동일 컬럼 산출(동작 보존).
- `test_features_leakage.py`·`test_split_leakage.py` 무회귀.

## 비목표 (c-1에서 안 함)

- predict-next-week/v6-predict 배선(c-2a/c-2b).
- category-σ(x) → item-risk 매핑(c-2b).
- 예보 없는 구름/풍속 대체(부분주입으로 확정 — 아키텍트 결정).
- 재귀 multi-step(NaN-lag로 충분; 정확도 개선은 선택지, 별도).

# bakery-predictor

베이커리 매장 판매량/품절 수요예측 PoC. 자세한 배경은 `@spec.md` 참조.

## 단계

- v0: 내부 데이터 baseline (Seasonal Naive, MA, Global LightGBM)
- v1: + 캘린더/특일 + 날씨
- v1.5 (unified): + potential_demand target + cannibalization + quantile production model
- v2: + 지하철 / 상권 / 연령대 / 소비정보

현재 단계: **v1.5 통합 수요 모델** (`lightgbm_v2`). 외부 캘린더/날씨는 실 API 백필. spec.md §6의 censored demand 처리 + 매장×카테고리 캐니벌라이제이션 + quantile 권장 생산량을 단일 회귀로 통합.

## 절대 규칙

1. **Time leakage 금지** — lag/rolling feature는 split 이후, 또는 명시적 cutoff 이전 데이터로만 계산한다. 예측 시점 이후의 sales, weather 관측값, 지하철 실측값을 feature로 쓰지 않는다. 작성 시 `test_split_leakage.py` / `test_features_leakage.py`가 반드시 통과해야 한다.
2. **품절 데이터는 censored** — 품절일 판매량은 실수요가 아니다. 학습/평가에서 품절 flag(`is_stockout`, `stockout_time`)를 보존하고, 무리하게 결측 처리하지 않는다. 판매량 모델과 품절 위험 모델은 분리한다.
3. **Random split 금지** — train/validation/test는 반드시 시간순. backtest는 rolling 또는 expanding window. 단일 holdout만 쓰지 않는다.
4. **Synthetic ↔ Real 경계 명시** — `data/synthetic.py`는 PoC 용도 한정. `data/loader.py`가 실데이터 진입점이며 동일한 schema(`data/schema.py`)를 반환해야 한다. 실데이터 수령 시 loader만 교체.
5. **MAPE 단독 지표 금지** — 판매량 0/희소 품목에서 폭발한다. 메인 지표는 **WAPE**. MAE/RMSE는 보조. item-level/category-level WAPE도 함께 본다.

## 실행

```bash
uv sync                                                       # 의존성 설치
uv run bakery generate-data                                   # synthetic hourly/daily/weather/calendar parquet
uv run bakery ingest-calendar / ingest-weather                # 실 API 백필 (.env 필요)
uv run bakery ingest-forecast                                 # 단기+중기 예보 (운영 시 사용)
uv run bakery backtest --source real --variants v0,v1,v2      # 3종 LGBM + baselines 동시 비교
uv run bakery predict-next-week --source real --model lightgbm_v2 --use-forecast --production-quantile 0.85
uv run pytest                                                 # 테스트
```

## 디렉토리

- `src/bakery/data/` — schema / synthetic / calendar / weather / loader (실데이터 교체 지점)
- `src/bakery/features/` — date / lag / rolling / calendar / weather / potential_demand / cannibalization / stockout_history
- `src/bakery/models/` — base + seasonal_naive / moving_average / lightgbm_regressor (`feature_set="v0"|"v1"|"v2"`, quantile 옵션) + stockout_classifier (운영 watchlist)
- `src/bakery/ingest/` — data.go.kr API 어댑터 (calendar / weather)
- `src/bakery/evaluation/` — split, metrics, backtest
- `tests/` — leakage 회귀 테스트가 핵심
- `reports/` — 산출물 (gitignored): fold_results.csv, predictions.csv, feature_importance_*.csv, next_week_predictions.csv

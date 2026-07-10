# bakery-predictor

베이커리 매장 판매량/품절 수요예측 PoC. 자세한 배경은 `@spec.md` 참조.

## 단계

지금까지(v0~v5)는 **PoC 착수 전 데이터 검증/검토 단계**였다. 이제부터가 실측 PoC.

- v0~v2: 내부 baseline → 캘린더/날씨 → 외부데이터(지하철/상권/연령/소비)
- v1.5 (unified): potential_demand target + cannibalization + quantile production
- v4: 카테고리 합 + 품목 비율 + 신제품 tracker (3-stage)
- v5: conformal 구간예측 — ⚠️ **DEPRECATED** (점추정+위험수치로 전환)

현재 단계: **진짜 PoC 착수 (260605 아티제 실무자 미팅)**. 상세 범위 = `@docs/poc_scope_v6.md`.
- **산출물**: 점추정 + 품절/매진 위험 수치 (구간/범위 예측 폐기)
- **검증 대상**: 광교 단독. 타매장은 광교 예측 보조 데이터로만.
- **검증 방식**: 4주 구축 + 4주 전향적(prospective) 실측 — 기존 발주 시스템과 성능 비교
- **metric**: 폐기비용↓ / 매진 time median↑ / 매진률↓ (운영 KPI, 아티제와 최종 합의)
- bulk→품목비율 예측은 수요 이전 통계검증 통과 시에만. 카테고리 정의는 아티제 제공.

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
uv run bakery backtest --source real --variants v0,v1,v2      # 3종 LGBM + baselines 동시 비교 (real: v2/v3는 yhat_adjusted_demand 출력, --closing-alpha 기본 0.5)
uv run bakery predict-next-week --source real --model lightgbm_v2 --use-forecast --production-quantile 0.85  # real: yhat_adjusted_demand 출력, --closing-alpha 기본 0.5
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

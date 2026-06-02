# v5 확률 모델 — 요일별 구간 예측 설계

작성일: 2026-06-02
브랜치: `feat/convex-cost-metric` (또는 신규 분기)
상태: 설계 확정, 구현 대기

## 1. 배경 / 목표

고객사 요구: **요일별로 판매량 구간(범위)을 제공하고, 실제 판매량이 그 구간 안에 들어오는 것을 타깃**으로 한다.

현재 v4 모델은 카테고리 합 단위로 **q0.90 단일 점예측(production)**을 발주 추천으로 낸다. 이를 **구간 예측(prediction interval)**으로 확장하되, 다음 두 가지를 동시에 만족시킨다:

1. **발주 의사결정 유지**: q0.90은 매진을 줄이기 위한 의도적 안전장치다. 이를 **구간의 중심 앵커 + 발주 추천점**으로 그대로 유지한다.
2. **Coverage 보장**: "실판매가 구간 안에 들어온다"는 타깃은 통계적으로 **calibrated coverage** 문제다. 단순 quantile 학습만으로는 명목 coverage(예: 90%)와 실측이 어긋난다(메모리 기준 명목 q0.90인데 operational backtest 실측 매진 13.6% — under-coverage). 이를 **conformal calibration**으로 명목값에 수렴시킨다.

### 비목표 (YAGNI)

- 분포 전체 추정(NGBoost/LightGBMLSS) — PoC엔 과함, 도입 안 함
- 점예측 정확도(WAPE) 자체 개선 — 본 작업 범위 밖. 기존 `expected`/`production` 모델 유지
- 발주점을 q0.90에서 변경 — 앵커 고정

## 2. 핵심 설계 결정

| 결정 | 내용 | 근거 |
|---|---|---|
| 중심 앵커 | q0.90 production (현행) | 매진 회피용 의도적 편향점. 발주 추천 = 앵커 |
| 구간 생성 | q0.90 앵커 ± conformalized 잔차 마진 | "q0.90 중심" 제약 하에서 CQR이 수렴하는 형태. calibration 보장은 유지 |
| 하한 모델 | LGBM quantile `q_lo` 모델 추가 | feature-adaptive 하한(휴일/날씨 반영). 진짜 CQR에 근접 |
| 마진 대칭성 | **대칭·비대칭 둘 다 구현** | 사후 분석(coverage/width/매진율)으로 적절성 결정 |
| 그룹화 | 요일별(Mondrian conformal) | 고객사가 "요일별 구간" 요구. 토요일 넓고 평일 좁음 |
| coverage 수준 | 80% + 95% 둘 다 산출 | PM 평가 framework 기획안(Coverage@80/@95) |
| split | train → calibration → test 시간순 | leakage 금지(CLAUDE.md 절대규칙 1·3) |

## 3. 아키텍처

```
[기존 유지] CategoryTotalModel
   ├─ expected   (regression_l1, 평균 예측 — WAPE 보조)
   ├─ production (quantile α=0.90 — 발주 앵커, 중심)   ← 안 건드림
   └─ production_lo (quantile α=q_lo — 신규, adaptive 하한)

                      │
        [신규] ConformalInterval layer
                      │
  시간순 calibration set에서 요일별 잔차 수집:
    상방 잔차  r_hi = actual − production_pred        (앵커 기준)
    하방 잔차  r_lo = production_lo_pred − actual     (하한 모델 기준)
                      │
  요일별(dow) 경험분위수로 마진 산출 (coverage 수준별):
    비대칭:  하한 = production_lo_pred − Q_dow(r_lo, 1−α)
            상한 = production_pred   + Q_dow(r_hi, 1−α)
    대칭:    δ_dow = Q_dow(|actual − production_pred|, 1−α)
            구간 = [production_pred − δ_dow, production_pred + δ_dow]
                      │
        발주 추천 = production_pred (q0.90, 중심 앵커 — 변동 없음)
        구간 = [하한, 상한] (요일별, coverage 수준별)
```

## 4. 컴포넌트 명세

### 4.1 신규 — `src/bakery/models/conformal_interval.py`

분포 가정 없는 split conformal + Mondrian(요일) 그룹화. 대칭/비대칭 두 모드.

```python
@dataclass
class ConformalInterval:
    mode: str                 # "symmetric" | "asymmetric"
    coverage: float           # 0.80, 0.95
    margins_by_dow: dict      # dow -> (m_lo, m_hi)
    pooled_margin: tuple      # fallback (표본 적은 dow용)
    min_group_n: int = 30     # 그룹 표본 미달 시 pooled 사용

    def calibrate(self, cal_df, center_pred, lo_pred=None) -> "ConformalInterval": ...
    def predict_interval(self, center_pred, dow, lo_pred=None) -> (lower, upper): ...
```

- `calibrate`: calibration set의 요일별 잔차 → 경험분위수. 유한표본 보정 `(1−α)(1+1/n)` 분위수.
- 표본 미달 요일은 `pooled_margin`으로 fallback (광교 요일당 ~260일이라 대개 충분, 신매장 대비 안전장치).
- 새 모델 학습 아님 — 이미 적합된 예측값의 잔차만 사용.

### 4.2 수정 — `src/bakery/models/category_total.py`

- `fit_category_total`에 `q_lo` 파라미터 추가 → `production_lo` 모델(quantile) 학습. `CategoryTotalModel`에 `production_lo` 필드 + `predict_production_lo` 메서드.
  - `q_lo` 기본 시작점: coverage 80% → `q_lo=0.10`, 95% → `q_lo=0.05` (conformal 보정 전 초기 하한). 보정 후 실측 coverage에 맞춰짐.
  - **`production_lo`는 비대칭 모드에서만 사용**(feature-adaptive 하한). 대칭 모드는 `production_pred` 앵커 ± δ만 쓰므로 `production_lo` 불필요 — 두 variant 비교 시 이 차이도 함께 평가.
- `expanding_window_backtest`에 **calibration fold 분리**: 각 fold에서 train → calibration → test 3분할(시간순). production 모델 자체 로직은 불변.
- conformal layer는 calibration 구간에서 적합 후 test 구간 구간예측 생성.

### 4.3 확장 — `src/bakery/evaluation/metrics.py`

현재 wape/mae/rmse/grouped_wape만 존재. 신규 추가:

```python
def coverage(actual, lower, upper) -> float          # 구간 적중률
def coverage_by_group(actual, lower, upper, group)    # 요일별 coverage
def interval_width(lower, upper) -> float             # 평균 폭 (폐기 proxy)
def pinball_loss(actual, pred, q) -> float            # quantile loss
def mase(actual, pred, train_actual, season=7) -> float
```

### 4.4 Stage 2 품목 전파

카테고리 구간 폭을 품목 비율(기존 Stage 2 proportion)로 비례 전파. 새 모델 아님 — 비율 곱셈만. 품목 단위 직접 conformal은 표본 부족으로 비목표(설계 §1 granularity 결정: 카테고리 구간 메인 + 품목 비율 분배).

## 5. 두 variant 비교 프로토콜

대칭/비대칭을 동일 fold·동일 calibration으로 A/B. `scripts/interval_backtest.py` 신규:

| 비교 지표 | 판단 |
|---|---|
| Coverage@80/@95 (전체·요일별) | 명목값에 가까운 쪽 우위 |
| Interval width 평균 | 좁은 쪽 우위 (같은 coverage 전제) |
| 매진율(상한/앵커 초과) | conformal 전후 비교 — 명목 수준 수렴 확인 |
| Pinball | 보조 |

핵심 검증: **명목 q0.90 대비 실측 매진율이 conformal calibration으로 목표 수준(예: 10% 또는 5%)에 수렴하는가** — 사장님 핵심 우려("q0.90 실운영 감당 가능?")의 정량 답.

## 6. Time leakage 방어 (CLAUDE.md 절대규칙)

- calibration set은 **train 이후 · test 이전** 시간 구간. 미래 잔차로 마진 계산 금지.
- expanding window 유지, random split 금지.
- `tests/test_split_leakage.py`에 conformal 3분할 시간순 회귀 테스트 추가.

## 7. 테스트 (`tests/test_conformal_interval.py` 신규)

- coverage 단조성: 95% 구간 ⊇ 80% 구간 (폭 ≥)
- calibration 정확도: 합성 데이터에서 명목 coverage ≈ 실측 (±tolerance)
- 대칭 모드: 구간이 중심 대칭
- 비대칭 모드: 하/상한 마진 독립
- 표본 미달 요일 → pooled fallback 동작
- 시간순 split leakage 회귀

## 8. 산출물

- 코드: `conformal_interval.py`(신규), `category_total.py`(수정), `metrics.py`(확장)
- 스크립트: `scripts/interval_backtest.py`(대칭/비대칭 A/B + coverage 리포트)
- 리포트: `reports/interval_coverage.csv`(요일별·coverage수준별), `reports/interval_predictions.csv`
- 테스트: `test_conformal_interval.py`, `test_split_leakage.py` 보강

## 9. 단계 (구현 순서)

1. `metrics.py` coverage/width/pinball/mase 추가 + 단위 테스트 (평가 인프라 먼저)
2. `conformal_interval.py` + `test_conformal_interval.py` (TDD)
3. `category_total.py`에 `q_lo` 모델 + calibration fold 분리
4. `scripts/interval_backtest.py` — 광교 대칭/비대칭 A/B
5. 사후 분석 → 대칭 vs 비대칭 결정, 매진율 수렴 검증
6. Stage 2 품목 전파

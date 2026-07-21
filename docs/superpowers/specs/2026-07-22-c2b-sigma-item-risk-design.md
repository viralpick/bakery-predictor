# c-2b: category σ(x) → item-risk 매핑 (decision 코어 + predict-next-week)

**날짜**: 2026-07-22
**로드맵**: 발주 안전마진 표준 스택 step (c)의 c-2b. c-1(미래 카테고리 예측)·c-2a(predict-next-week 배선)는 PR#50 머지 완료.
**관련 스펙**: `2026-07-21-c1-future-category-forecast-design.md`, `2026-07-21-distributional-total-src-promotion-design.md`
**관련 메모리**: `project_distributional_forecasting_stack`, `project_kpi_priority_framing`

## 배경 / 문제

분포모델(`DistributionalTotalModel`, NGBoost LogNormal)은 category-total 수요의 **μ(x)·σ(x)를 동시 추정**한다. σ(x)는 지금까지 발주 base(q0.85)를 만드는 데만 쓰였고, item-level 위험 수치로는 한 번도 내려가지 않았다.

두 소비처가 σ를 놓치고 있다:
1. `decision/risk.py`의 Monte-Carlo 셸은 demand 분산을 **하드코딩 `DEFAULT_DEMAND_CV=0.30`** placeholder로 만든다 (docs §8.4 "parametric placeholder until a proper predictive distribution feeds in"). 분포모델 σ가 바로 그 "proper predictive distribution"이다.
2. c-2a가 배선한 predict-next-week category 출력은 `stockout_prob`을 **NaN**으로 남겨뒀다 ("σ(x)→item-risk 매핑은 c-2b").

c-2b는 이 매핑을 만든다.

## granularity (검증됨)

`build_category_daily`는 `groupby("date")`로 **store-total 1개/날짜**를 만든다(모든 TARGET_CATEGORIES 합). 따라서 `predict_sigma(test)`는 **날짜당 σ 1개**를 준다. 이 σ를 그날 모든 item에 broadcast한다.

**broadcast는 heuristic이 아니라 결정론적 비율 가정 하에서 수학적으로 정확하다**: store-total ~ LogNormal(μ, σ)이고 item demand = 고정비율 p × total이면, p·LogNormal(μ, σ) = LogNormal(μ+log p, σ) — **σ 불변**. 즉 item demand ~ LogNormal(log(demand_point_i), σ_tot). item median(=demand_point_i)만 다르고 log-space σ는 store-total과 동일하다.

## 아키텍트 결정 (확정)

- **범위**: `decision/risk.py` 코어 확장 + predict-next-week 소비. v6-predict/pipeline은 코어 확장으로 자동 수혜.
- **σ 분배**: 그대로 broadcast (공유-μ). item σ_log = category σ_log, 모든 item 동일. item-level σ가 미검증이므로 독립노이즈 인플레이션(overclaim)은 채택하지 않는다.
- **분포족**: risk 셸에 LogNormal 경로 추가. σ 미제공 시 기존 Normal cv 경로 유지(backward-compat).

## 산출물

### 1. `decision/risk.py` — LogNormal 경로 (backward-compat)

```python
def simulate_item_risk(
    demand_point: float,
    order_qty: float,
    params: RiskParams = RiskParams(),
    rng: np.random.Generator | None = None,
    *,
    demand_sigma_log: float | None = None,
) -> RiskResult:
```

- `demand_sigma_log`가 유효(not None, >0, `demand_point>0`)하면:
  `demand = exp(rng.normal(log(demand_point), demand_sigma_log, n))` — LogNormal, median=demand_point.
- 그 외(기본 None, 또는 `demand_point<=0`) → 기존 `Normal(point, cv·point)` 경로 그대로.
  **기존 v6 호출·테스트 무회귀**(인자 미전달 시 동작 불변).
- σ는 날짜/item마다 다르므로 `RiskParams`(전 item 공유 dataclass) 필드가 **아니라** per-call 인자.
- `_sample_demand`를 분기(신규 `_sample_demand_lognormal` 또는 내부 if). 나머지(short/leftover/cost 집계)는 분포족 무관하게 재사용 — 셸 구조 보존.
- **LogNormal 정의역 가드**: `demand_point<=0`이면 LogNormal 불가 → Normal fallback(docstring 명시). category-total 배분 결과는 구조적 양수라 정상 경로엔 미발생.

### 2. `decision/pipeline.py` — σ 컬럼 스루

- `items` 프레임에 optional `demand_sigma_log` 컬럼 허용. 있으면 per-row로 `simulate_item_risk(..., demand_sigma_log=...)`에 전달, 없으면 None.
- `_recommend_one` 시그니처에 `demand_sigma_log: float | None = None` 추가, `build_recommendation`이 컬럼 존재 시 행별로 읽어 넘긴다. 컬럼 없으면 기존 동작(무회귀).
- `METRIC_COLS`·출력 스키마 불변(σ는 입력 전용, RiskResult는 이미 p_stockout/p_waste/expected_* 노출).

### 3. `cli.py` `_category_base_predict` — σ 반환 확장

- 현재 `(base_median, base_prod)` 반환. `(base_median, base_prod, base_sigma)`로 확장.
  - `distributional` → `base_sigma = model.predict_sigma(test)` (날짜별 σ_log 배열).
  - `lightgbm` → `base_sigma = None` (σ 개념 없음).
- **재fit 회피**: σ를 얻으려 모델을 다시 적합하지 않는다 — 이미 fit된 `model`에서 뽑는다.
- 호출부 2곳(future 경로 + fold backtest 경로) 시그니처 갱신. fold backtest는 σ를 안 쓰면 `_`로 무시.

### 4. `cli.py` `_category_future_order_predictions` — σ broadcast

- `_category_base_predict`에서 받은 `base_sigma`(날짜별)를 `dates`에 정렬해, item 배분 후 프레임에 `demand_sigma_log` 컬럼으로 broadcast(그날 모든 item 동일값). `base_sigma=None`이면 컬럼 NaN.
- event_prior 블렌드는 median/prod만 조정하고 σ는 건드리지 않는다(σ는 분포모델 원값 유지 — 블렌드는 point-anchor라 분산 재추정 아님).
- 반환 컬럼: `[store_id, item_id, category_id, date, demand_point, our_order, demand_sigma_log]` (기존 6 + σ 1).

### 5. `cli.py` `_predict_next_week_category` — risk 채우기

- 현재 `stockout_prob=float("nan")` 대신, 각 행의 `(demand_point, our_order, demand_sigma_log)`로 `simulate_item_risk`를 호출해 `p_stockout`을 `stockout_prob`에 채운다.
  - `demand_sigma_log`가 NaN/None인 행(lightgbm) → `stockout_prob=NaN` 유지(현 동작).
- **8-col 출력 스키마 불변**(c-2a 핀): `[store_id, item_id, category_id, date, demand_col, stockout_prob, recommended_production, model]`. `p_waste`/`expected_cost`는 CSV에 추가하지 않는다 — 풍부한 RiskResult는 v6-predict/pipeline 경로에서만 노출.
- 재현성: 행별 독립 RNG(seed 고정). 배치 성능(33 item×7일=231행 MC)은 n_samples 기본 5000에서 문제없음(수 초).

## 정직한 한계 (spec·docstring 명시)

- **결정론적 비율 가정** 하에서 item order = p × category order, item point = p × category median이면 order/point 비율이 그날 모든 item 동일 → item p_stockout ≈ category exceedance(≈1−q0.85=0.15)로 수렴. **round(0)·배수생산(3/6/9) 정책·event_prior 블렌드**가 order를 분위수에서 밀어낼 때만 item별로 유의미하게 갈린다.
- 이는 공유-μ 커플링의 정직한 귀결이다 — 풀링은 item-level 위험 해상도를 만들지 않는다. **item soldout은 부차 KPI**(`project_kpi_priority_framing`: 폐기율=1차, item 매진=고객경험 서브KPI)이므로 수용 범위. p_stockout의 1차 가치는 "magic 상수 0.30 대신 모델-grounded σ"로 decision 파이프라인 비용추정을 정직하게 만드는 것.
- item-level σ 미검증 → broadcast는 overclaim 회피 선택(`feedback_verified_vs_inferred`).

## 테스트

### `tests/test_decision_layer.py` (기존 확장 — risk·pipeline 테스트가 여기 모여 있음)
1. **LogNormal median**: `demand_sigma_log` 제공 시 대량 샘플 median ≈ `demand_point` (rtol 완화, MC라 근사).
2. **σ 단조성**: `order > demand_point` 고정, σ↑ → `p_stockout`↑ (정확 부등호, 두 σ값 비교).
3. **backward-compat**: `demand_sigma_log=None` → 기존 Normal 경로와 **정확 동일** RiskResult(같은 seed·params, allclose).
4. **양수 가드**: `demand_point<=0` + σ 제공 → Normal fallback(예외 없이 정상 반환).

### `tests/test_predict_next_week.py` (기존 확장)
5. **distributional risk**: category+distributional 경로에서 `stockout_prob` 유한·`0 < p < 1`(monkeypatch로 c-1 스텁 — demand_point/our_order/σ 고정 프레임 주입, 결정론 검증).
6. **lightgbm NaN 유지**: category+lightgbm 시 `stockout_prob` NaN.
7. **8-col 스키마 핀**(c-2a 테스트 유지 — 컬럼 집합 정확 일치).

### `tests/test_decision_layer.py` (파이프라인 부분)
8. **파이프라인 σ 스루**: `items`에 `demand_sigma_log` 컬럼 있으면 risk가 그 σ를 쓴다(σ 다른 두 프레임 → p_stockout 다름). 컬럼 없으면 기존 동작.

### 회귀
- `test_features_leakage.py`·`test_split_leakage.py` 무회귀(risk는 post-prediction, 구조적 무관).
- 전체 `uv run pytest` — 신규 인자가 기존 호출부 안 깨는지.

## 비목표 (c-2b에서 안 함)

- p_waste/expected_cost를 predict-next-week CSV에 추가(8-col 핀 유지).
- item-level 분포모델(independent σ 학습) — broadcast로 확정.
- cannibalization 상관 샘플링(risk 셸의 기존 독립가정 유지, docs §8.4).
- conformal 재보정(분포모델 near-calibrated, raw 분위수 사용 방향 유지).
- 4매장 일반화·NegBin 등 타분포족.

## 검증 절차

1. `uv run pytest tests/test_decision_layer.py tests/test_predict_next_week.py`
2. `uv run pytest tests/test_split_leakage.py tests/test_features_leakage.py` (회귀 무손상)
3. `uv run pytest` (전체 — backward-compat)
4. 광교 실측 smoke: `uv run bakery predict-next-week --source real --order-level category --total-model distributional` → stockout_prob 유한 확인.
5. 3축 리뷰(재사용성·품질·효율).

## 리스크 / 열린 문제

- σ broadcast + 비율 배분 → p_stockout 저해상도(위 한계 명시). 배수생산 정책이 predict-next-week 경로엔 아직 미적용(v6 policy에만) → predict-next-week의 p_stockout은 round(0)만으로 갈림, 거의 균일할 수 있음. **honest 노출**(NaN보다 나음, magic 상수 제거가 코어 가치).
- NGBoost σ의 절대 calibration은 category-total OOS서만 검증됨 — item broadcast의 절대 정확도는 미검증(상대 신호로 사용).

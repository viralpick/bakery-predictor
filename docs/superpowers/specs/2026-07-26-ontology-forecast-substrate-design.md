# Spec 5a — 온톨로지 Forecast Substrate (예측 배선 정합)

- 날짜: 2026-07-26
- 로드맵: Harness Backbone 5단계(온톨로지 + 액션레이어)의 **하위 스펙 5a**
- 상위 스펙: `docs/superpowers/specs/2026-07-24-harness-backbone-design.md` §10-(B)
- 후속: **Spec 5b — 2층 설명 레이어**(별도 스펙; 본 스펙이 노출하는 중간값을 소비)
- 관련 메모리: `project_harness_backbone`, `project_poc_v7_aos_demonstrator`, `project_distributional_forecasting_stack`, `project_modeling_v4`

---

## 1. 배경 / 문제

원 스펙 B(§10-B)는 "`scenario.py`가 `GlobalLGBM(feature_set="v2")`를 직접 fit하는 네 번째 예측 배선처를 canonical 진입점으로 재배선"이었다. 그러나 architect가 지정한 실제 목적은 **"왜 이 수량?"의 2층 설명**(총량 수준 + 품목 수준)이며, 그 병목은 scenario가 아니라 온톨로지가 **예측 자체를 갖고 있지 않다는 점**이다.

현재 온톨로지 함수(`functions.py`)의 "수요"는 예측이 아니라 **`adjusted_demand` 컬럼의 기간 평균**이다:

```python
# functions.py:61
def _item_demand_points(period, demand_col):
    grouped = period.groupby("item_id")[demand_col].mean()   # ← 과거 평균, 예측 아님
    return grouped.reset_index(name="demand_point")
```

그래서 "다음주 목요일 팥빵 왜 K개 생산?" 같은 **forward 질문에 근본적으로 답할 수 없다**. 한편 canonical forward 2층 예측은 이미 존재하지만 `cli.py`에 갇혀 있어 온톨로지가 공유하지 못한다.

### 배선 지도 (검증 완료)

| 조각 | 위치 | 상태 |
|---|---|---|
| 총량 Stage 1 | `models/category_total.py`, `models/distributional_total.py` | ✅ src |
| event_prior 블렌드 | `models/event_prior.py` (`EventLevelPrior.blend`) | ✅ src |
| 품목 비중 Stage 2 | `models/item_proportion.py` (`distribute_total`, `ItemProportionResult`가 factor 컬럼 노출) | ✅ src |
| **forward 2층 합성** | `cli.py::_category_future_order_predictions` (private) | ⚠️ **cli에 갇힘** |
| 온톨로지 "수요" | `functions.py::_item_demand_points` = **컬럼 평균(예측 아님)** | ❌ 결함(병목) |
| scenario 예측 | `scenario.py::_fit_demand_model` = **직접 GlobalLGBM v2 fit** | ❌ 부차 배선처 |

`_category_future_order_predictions`의 실제 합성 흐름:
1. `build_category_daily(alpha)` → history
2. `_extend_category_features` → 미래 행 target=NaN append(leakage-safe lag)
3. (opt) 예보 날씨 주입 `fill_forecast_weather`
4. `_category_base_predict(train, test, total_model, production_quantile)` → `base_median`, `base_prod` (Stage 1)
5. (opt) `_blend_event_prior` → `base_median`, `base_prod` 보정 (event_prior)
6. `distribute_total(daily, base_prod)` → `our_order`; `distribute_total(daily, base_median)` → `demand_point` (Stage 2)
7. 반환 `[store_id, item_id, category_id, date, demand_point, our_order]` — **중간값(Stage1 base/prior 총량, 비중 factor)은 폐기**

---

## 2. 목표 / 비목표

### 목표
1. forward 2층 예측(총량 → event_prior → 품목 배분)을 **cli 밖 공유 seam**으로 추출하고, **중간값을 구조화 반환**한다(5b 설명이 재계산 없이 소비).
2. `functions.py`의 demand_point를 **컬럼 평균 → canonical forward 예측**으로 재배선한다.
3. cli의 forward CSV 산출은 seam을 소비하는 얇은 래퍼로 전환하되 **byte-equal** 유지(단일 출처).

### 비목표 (명시적 제외)
- **`scenario.what_if_driver` 재배선** — 부차·리스크(item-level GlobalLGBM v2 → 총량×비중 이전은 v7 테스트/데모 의미를 바꾼다). 본 스펙 제외, 후속 판단 대상. functions.py가 진짜 병목이라는 점에 architect 동의(2026-07-26).
- **설명 함수 일체**(`explain_category_total`, `explain_item_order`), grounded 도구, Q셋 — **Spec 5b**.
- **synthetic 소스 forward 예측** — 본 seam은 real 소스(canonical PoC = 광교) 지향(`build_category_daily`/`_load_real_daily` 의존). synthetic 온톨로지 데모는 기존 컬럼-proxy fallback 유지(라벨).
- **다중 카테고리 배분** — 현 canonical은 단일 빵 총량 1개(`distribute_total`이 date당 단일 total을 전 품목에 배분). 다중 카테고리는 범위 밖.
- **임의 과거 period replay를 온톨로지에 개방** — seam은 forward-only(마지막 관측일 다음 horizon). 온톨로지 함수는 다가오는 horizon을 예측(architect 예시 = "다음주"). 과거 period를 예측대상으로 주는 historical replay는 범위 밖(windowed_backtest fold 재발명).

---

## 3. 설계 — Seam 추출

### 3.1 위치 / 진입점
신규 패키지 **`src/bakery/forecast/`**. `_category_future_order_predictions`의 transitive private 의존을 cli에서 이 패키지로 **이동**하고 cli는 다시 import한다(단일 출처, 순환의존 없음 — `models/`·`features/`에만 의존):

- `forecast/loaders.py` ← `_load_real_daily`, `_load_forecast_weather` (⚠️ **cli의 item·decision 경로와 공유** — 이동 시 그 경로들도 `forecast.loaders`에서 import)
- `forecast/forward.py` ← `_extend_category_features`, `_forecast_to_category_weather`(forward 전용) + `_category_base_predict`, `_blend_event_prior`(⚠️ **cli의 category 마진 스택과 공유** — 이동 시 그 경로도 import) + 신규 `ForwardForecast`·`forecast_forward`

### 3.2 시그니처 (forward-only, horizon 기반)

```python
def forecast_forward(
    store_id: str,
    *,
    horizon_days: int = 7,                # 마지막 관측 history 다음 N일 예측
    total_model: str = "lightgbm",        # lightgbm | distributional
    event_prior: bool = True,
    production_quantile: float = 0.85,
    alpha: float = DEFAULT_ALPHA,
    use_forecast: bool = True,
) -> ForwardForecast: ...
```

현 `_category_future_order_predictions`와 **동일 forward 의미**(마지막 관측일 다음 `horizon_days`일 예측, fit은 관측 history만 = `date < 첫 미래일`, leakage-safe: 미래 행 target=NaN append로 lag가 seam 너머 계산). 임의 과거 cutoff replay(historical backtest를 온톨로지에 개방)는 windowed_backtest fold 재발명이라 **범위 밖**(§2). date-recency 비의존(마지막 history일 기준)이라 재현적.

### 3.3 반환 — 중간값 노출

```python
@dataclass(frozen=True)
class ForwardForecast:
    # Stage 1 총량 + event_prior 중간값 (date당 1행; event_prior off면 prior_*==base_*)
    category_totals: pd.DataFrame   # [date, base_median, base_prod, prior_median, prior_prod]
    # Stage 2 비중 factor 분해 (item_proportion.ItemProportionResult.proportions 그대로, target_date→date 정규화)
    proportions: pd.DataFrame       # [date, item_id, category_id, proportion, base_sold,
                                    #  trend_pct, avg_stockout_h, closing_rate, days_since_first,
                                    #  adj_trend, adj_stockout, adj_closing, adj_new]
    # 최종 품목 수량 (현 cli 반환과 동일 스키마)
    item_quantities: pd.DataFrame   # [store_id, item_id, category_id, date, demand_point, our_order]
```

- `base_*` = Stage 1 예측(prior 보정 전), `prior_*` = event_prior 블렌드 후. → 5b `explain_category_total`이 "특수일 앵커가 base를 얼마나 끌어올렸나"를 **실제 수치**로 서술.
- ⚠️ 구현 주의: 현 cli는 `_blend_event_prior`가 `base_median/base_prod`를 **제자리 덮어써** prior-이전 값을 잃는다. seam은 블렌드 **전** `base_*`를 스냅샷하고 블렌드 **후** 값을 `prior_*`로 담아 둘 다 보존해야 한다.
- `proportions`의 factor 컬럼(`adj_trend·adj_stockout·adj_closing·adj_new`, base=`base_sold`) → 5b `explain_item_order`가 `raw_weight = base_sold × adj_trend × adj_stockout × adj_closing × adj_new` (정규화 전)을 그대로 분해. `distribute_total`은 base_prod·base_median 두 번 호출하나 비중은 date당 동일(`compute_proportions` 결정론) → `proportions`는 1회만 노출.

### 3.4 event_prior 서술의 충실성 (advisor #1)
`prior_*`는 **pre-test 히스토리로 fit한 레벨-앵커 블렌드값**이지 "크리스마스=고정 N개 룰"이 아니다. 본 seam은 `base_*`/`prior_*`를 **엔진이 실제로 낸 값 그대로** 노출하며, "룰"이라는 이상화된 라벨을 만들지 않는다. (5b 설명 함수도 이 값을 그대로 서술 — grounding이 없애려는 fabrication 방지.)

---

## 4. 설계 — cli 래퍼화

`_category_future_order_predictions` / `_predict_next_week_category`는 `forecast_forward`를 호출하고 `ForwardForecast.item_quantities`에서 flat CSV(`next_week_predictions.csv`)를 구성한다. 컬럼·반올림·순서·프리뷰 로직 불변.

**Acceptance(게이트)**: 추출 전후 `next_week_predictions.csv` **byte-equal**(동일 인자·시드). 추출이 cli 출력을 바꾸지 않음을 증명.

---

## 5. 설계 — functions.py 재배선

### 5.1 변경
`_item_demand_points`(컬럼 평균)를 대체: 온톨로지 함수는 `forecast_forward(store_id, horizon_days=...).item_quantities`를 요청 (item_id, date)로 슬라이스해 `demand_point`를 얻는다. `period`는 다가오는 horizon 내 forward 대상으로 해석(horizon_days는 요청 period end가 마지막 관측일에서 며칠 뒤인지로 도출). `_resolve_demand_proxy`(컬럼 자동선택)는 forward 경로에선 불필요해지나, synthetic fallback 경로용으로 라벨과 함께 잔존.

### 5.2 계약 변경 — 영향 소비처 (grep 열거)
| 함수 | demand_point 소비 | 조치 |
|---|---|---|
| `rank_stockout_risk` (functions.py:79) | `_item_demand_points` | forward 예측으로 교체 |
| `explain_order` (functions.py:131) | `_item_demand_points` | forward 예측으로 교체 (5b가 이 위에 얹힘) |
| `what_if` / `WhatIfResult` (functions.py:153) | demand_point를 **인자로 수령** | 시그니처 무변경(호출부가 forward 값 주입) |
| `rank_stockout_earliness` (functions.py:85) | observed `stockout_time` 사용 | **무영향** |

- **의도적 변경(라벨)**: 온톨로지 demand_point = 과거 평균 → forward 예측. 게이트 대상 아님, docstring·리뷰에 명시.
- 테스트: functions 테스트의 fixture가 forward 경로(다가오는 horizon 예측)를 태우도록 마이그레이션. 과거 period 슬라이스 fixture → forward 대상으로 교체. real-source fixture 필요.

### 5.3 guarded fallback — 무조건 교체가 아니라 period-타입 분기 (architect 승인 2026-07-26)
⚠️ **구현 중 발견**: 무조건 forward 교체는 `grounding/questions.py::_ctx`를 깨뜨린다 — 그건 rank_stockout_risk/explain_order를 **항상 과거(historical) period**(eval-gold, `dd.min()~dd.max()`)로 호출하는데, forward 예측은 미래 horizon만 산출해 과거 period와 겹치지 않아 ValueError가 난다(미래는 정답이 없으므로 gold를 과거로 만드는 게 옳다).
→ **guarded fallback** 채택: `_resolve_demand_points`가 `_is_forward_period`(period 시작 > 마지막 관측일)로 분기 — **forward면 forecast_forward, historical면 기존 컬럼평균**. architect의 "왜 K개 생산"(미래) 질문은 예측 경로로 정상 답하고, grounding eval-gold(과거 관측 요약)는 무변경으로 유지된다. scope=functions.py만, grounding 미변경. (advisor·리뷰어 지지, architect 확정.)
- 알려진 caveat(docstring 라벨): `demand_col`은 historical 경로에만 적용(forward는 forecast_forward 산출), period가 관측경계를 걸치면 historical로 분류돼 과거 부분만 요약. per-call LightGBM refit(캐시 없음)=5b 최적화 후보.

---

## 6. Acceptance 기준

1. **불변(hard 게이트)**
   - `next_week_predictions.csv` byte-equal (seam 추출이 cli 산출 불변).
   - canonical category_total == harness `windowed_backtest` 회귀(rtol=1e-9) 유지 — 추출이 총량 엔진을 안 건드림.
   - `item_proportion`(`distribute_total`) 산출 불변.
2. **coherence invariant** (5b 두 설명이 같은 이야기임을 보증)
   - date당 `Σ item_quantities.our_order == category_totals.prior_prod` (배분 정규화 Σ비중=1).
   - `demand_point` 경로도 동일: `Σ demand_point == prior_median`.
3. **의도적 변경(라벨)**: 온톨로지 demand_point = 컬럼평균 → forward 예측 (테스트로 새 동작 고정).
4. **faithfulness**: `ForwardForecast`의 `base_*`/`prior_*`/`proportions` factor값 == 엔진 중간 산출(reconcile 테스트). 이상화된 "룰" 라벨 없음.
5. 전체 pytest 통과 + leakage 테스트(`test_split_leakage`, `test_features_leakage`) 통과(seam이 `date < cutoff` fit 유지).

---

## 7. 리스크

- **cli 헬퍼 이동 폭발반경** — `_extend_category_features`/`_category_base_predict`/`_blend_event_prior`가 cli의 다른 경로(item v2·backtest)와 상수(`EVENTS`, `LUNAR_EVENTS`, `DEFAULT_ALPHA`)를 공유. 이동 시 import 재배선 필요 → 구현 착수 전 소비처 grep으로 열거(no-yolo 규칙 3).
- **real-source 결합** — seam이 `build_category_daily`/`_load_real_daily`에 결합(real 전용). synthetic 온톨로지 데모는 이 경로를 못 탄다 → fallback 유지·라벨. (PoC = 광교 real이라 실무상 문제 없음.)
- **functions.py 계약 변경으로 fixture 붕괴** — 과거 period fixture가 forward 예측을 못 태움 → real-source·미래 period fixture로 마이그레이션 스텝을 플랜에 명시.
- **byte-equal 실패** — 헬퍼 이동 중 로직 미세 변경 → 진행 중단·원인 규명(추출 원칙).
- **coherence invariant 미성립** — `distribute_total`이 date당 단일 total 전제. 다중 카테고리 유입 시 깨짐 → 단일 총량 범위 고정(§2 비목표).

---

## 8. 5b 연결 (후속 스펙)

본 seam이 노출하는 `ForwardForecast.category_totals`(base/prior)와 `proportions`(factor)를 Spec 5b가 소비:
- `explain_category_total` ← `category_totals`(base_median → prior_median 보정 서술)
- `explain_item_order` ← `proportions`(비중 factor 분해) + `item_quantities`(qty=총량×비중) + 기존 decision lineage(safety/floor/rounding)

5a가 forward 예측과 중간값 노출을 확보하므로, 5b는 **새 모델링 없이** grounded 설명 함수·도구·Q셋만 얹는다.

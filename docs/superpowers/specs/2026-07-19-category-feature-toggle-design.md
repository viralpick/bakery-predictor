# 카테고리 스택 feature 그룹 toggle 설계

날짜: 2026-07-19
브랜치: feat/holiday-feature-fix-track12
상태: 승인됨 (architect 확인, "이렇게 진행")

## 배경 / 문제

광교 모델 재측정은 `scripts/unified_policy_kpi.py`가 `_category_order_predictions`를
호출하는 카테고리 스택 경로로 이뤄진다. 작은 수정마다 전체를 다 돌리는 불편과 별개로,
**부차적 feature를 실험적으로 넣었다 뺐다** 할 방법이 없다. 현재 `build_features`는
6개 add_* 스텝을 무조건 조립하고, 모델은 `select_feature_cols`로 존재하는 컬럼을 전부
feature로 쓴다 → 그룹을 안 만들면 자동으로 그 feature가 빠진다.

## 호출 사슬 (조립 지점)

```
scripts/unified_policy_kpi.py  →  _category_order_predictions(STORE, **kw)   [cli.py:2030]
   →  build_features(build_category_daily(α), target_col)                     [category_aggregate.py:404]
   →  select_feature_cols() = 산출 컬럼 전부 (date/target/LEAK_COLS 제외)      [category_total.py:44]
```

`build_features`가 유일한 조립 지점이므로 여기에 toggle을 건다.

## 설계 (그룹 단위 · 카테고리 스택 · 최소형)

### ① 그룹 레지스트리 — `category_aggregate.py`

```python
FEATURE_GROUPS = {
    "cyclic_calendar": add_cyclic_calendar,
    "holiday":         add_holiday_features,
    "event":           add_event_features,
    "weather":         add_weather_features,
    "competitor":      add_competitor_features,
}
```

`add_lag_rolling_ewma`는 레지스트리 **제외 = 항상 on**. 이유: (a) target 기반
autoregressive 코어 신호 — 빼면 모델이 망가짐, (b) 유일하게 `target_col` 인자가 필요해
시그니처가 다름. 항상 마지막에 실행.

### ② `build_features`에 `drop_groups` 파라미터

```python
def build_features(cd, target_col="adjusted_demand_unit", *, drop_groups=frozenset()):
    unknown = drop_groups - FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}. "
                         f"choose from {sorted(FEATURE_GROUPS)}")
    df = cd.df.copy()
    for name, fn in FEATURE_GROUPS.items():
        if name not in drop_groups:
            df = fn(df)
    return add_lag_rolling_ewma(df, target_col)
```

기본값 `frozenset()` → 현 동작과 100% 동일(회귀 안전). 기존 add_* 호출 순서 보존.

### ③ `_category_order_predictions`에 `drop_features` 인자

`drop_features: frozenset[str] = frozenset()` 추가 →
`build_features(..., drop_groups=drop_features)`로 전달. 기본값 빈 set.

### ④ 노출 — `scripts/unified_policy_kpi.py`

`argparse --drop weather,competitor` 추가 → 파싱한 `frozenset`을 각
`_category_order_predictions(..., drop_features=DROP)` 호출에 전달. 소스 수정 없이
`uv run python scripts/unified_policy_kpi.py --drop weather` 로 실험. 숨은 env I/O 대신
명시적 arg(code-quality: 암묵적 의존/숨은 I/O 금지).

### ⑤ 테스트 — `tests/`

- `drop_groups=frozenset()` 산출 컬럼셋 == 현재(회귀, 정확 집합 비교).
- `drop_groups={"weather"}` → weather 그룹 컬럼만 정확히 빠지고 나머지 그대로.
- unknown 그룹 → `ValueError`.
- 기존 leakage 테스트 통과 유지.

## 추가: item-level LGBM(A) 경로 toggle (2026-07-20, 승인 후 추가)

당초 non-goal이었으나 architect 요청으로 대칭 구현. A 경로 = `GlobalLGBM(feature_set=v0~v3)`
(item별 개별 모델이 아니라 전 품목 global LGBM, WAPE backtest 트랙).

**핵심 안전성**: enrichment(`_enrich_if_needed`)는 그대로 두고 **모델 입력 컬럼 선택만**
끈다. 드롭한 그룹 컬럼도 프레임엔 존재 → `_check_feature_set_columns`(컬럼 존재 요구)와
충돌 없음(순수 ablation). B 경로처럼 조립 지점 한 곳(`build_numeric_columns`)만 건드림.

- `FEATURE_GROUPS` 레지스트리(`lightgbm_regressor.py`): name → `*_FEATURE_COLUMNS` 상수.
  toggle 대상 = calendar / weather / cannibalization / competitor / living_pop /
  population / consumption. base(date/lag/rolling)는 코어 = 항상 on.
- `build_numeric_columns(feature_set, *, drop_groups=())` — 기존 로직 후 그룹 컬럼 필터,
  순서 보존. feature_set에 없는 그룹 드롭은 no-op(그룹명은 유효해야). unknown → ValueError.
- `GlobalLGBM(..., drop_groups=())` — 전달 + 모델명에 `_drop-<groups>` 표기.
- `_build_forecasters(..., drop_groups=())` → `cmd_backtest(..., drop="")` CLI.
  노출: **backtest만**(WAPE ablation 용도). `bakery backtest --variants v2 --drop weather`.
- 테스트 `tests/test_lgbm_feature_toggle.py` 7개.

주의: A와 B의 `--drop` 그룹 어휘는 다르다(각자 자기 경로의 feature 상수 기준). A는 calendar
하나, B는 cyclic_calendar/holiday/event로 분리 — 구현이 실제로 다르기 때문.

## Non-goals (이번 스코프 밖)

- item-level(A) toggle의 predict-next-week / v6 노출 (GlobalLGBM 인자로는 가능, CLI 미노출).
- 개별 컬럼 단위 on/off.
- 매장 선택 필터 — 실데이터가 `store_gw01` 단일매장이라 no-op, 다매장 재적재 시점까지 defer.
- cli.py 분할 / preprocessing 캐싱 — 별도 트랙(원인 측정 후 결정).

## 회귀 안전 근거

`build_features` 기존 호출부(cli.py:2042, event_prior, ontology/grounding/run, calendar)는
모두 `drop_groups` 기본값으로 진입 → 동작 불변. ⑤(a) 회귀 테스트가 이를 보증.

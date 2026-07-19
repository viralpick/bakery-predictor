"""build_features 그룹 단위 feature toggle (drop_groups).

카테고리 스택의 유일한 feature 조립 지점(build_features)에서 부차적 feature 그룹을
실험적으로 빼는 기능. 기본값(빈 set)은 현 동작과 동일해야 한다(회귀 안전).
"""

import numpy as np
import pandas as pd
import pytest

from bakery.features.category_aggregate import (
    CategoryDaily,
    FEATURE_GROUPS,
    build_features,
)

# add_cyclic_calendar가 산출하는 컬럼 — 순수 date 기반(파일 I/O 없음)이라 결정적.
CYCLIC_COLUMNS = {
    "dow", "month", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "dom", "dom_sin", "dom_cos",
}
# add_lag_rolling_ewma 산출 — 항상 on이어야 하는 autoregressive 코어.
LAG_COLUMNS = {
    "adjusted_demand_unit_lag1", "adjusted_demand_unit_lag7",
    "adjusted_demand_unit_lag14", "adjusted_demand_unit_lag28",
    "adjusted_demand_unit_rmean7", "adjusted_demand_unit_rmean28",
    "adjusted_demand_unit_rstd7", "adjusted_demand_unit_rstd28",
    "adjusted_demand_unit_ewma7", "adjusted_demand_unit_ewma28",
}


def _toy_cd(n: int = 120) -> CategoryDaily:
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {"date": dates, "adjusted_demand_unit": np.arange(n, dtype=float) + 50.0}
    )
    return CategoryDaily(df=df, alpha=0.8)


def test_drop_empty_is_noop():
    """drop_groups 기본값(빈 set)은 인자 생략과 동일한 컬럼셋 — 현 동작 보존."""
    cd = _toy_cd()
    default_cols = set(build_features(cd).columns)
    explicit_empty = set(build_features(cd, drop_groups=frozenset()).columns)
    assert default_cols == explicit_empty


def test_drop_cyclic_removes_exactly_its_columns():
    """단일 그룹 드롭 시 그 그룹 컬럼만 정확히 빠지고 나머지는 그대로."""
    cd = _toy_cd()
    full = set(build_features(cd).columns)
    dropped = set(build_features(cd, drop_groups={"cyclic_calendar"}).columns)
    assert full - dropped == CYCLIC_COLUMNS
    assert dropped == full - CYCLIC_COLUMNS


def test_lag_rolling_always_present_even_when_all_groups_dropped():
    """레지스트리 전체를 드롭해도 lag/rolling 코어는 남는다."""
    cd = _toy_cd()
    cols = set(build_features(cd, drop_groups=frozenset(FEATURE_GROUPS)).columns)
    assert LAG_COLUMNS <= cols
    assert CYCLIC_COLUMNS.isdisjoint(cols)


def test_unknown_group_raises():
    cd = _toy_cd()
    with pytest.raises(ValueError, match="unknown feature groups"):
        build_features(cd, drop_groups={"bogus"})

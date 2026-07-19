"""item-level global LGBM(A 경로) feature 그룹 toggle.

build_numeric_columns / GlobalLGBM에서 부차적 feature 그룹을 실험적으로 빼는 기능.
기본값(빈 set)은 현 동작과 동일해야 한다(회귀 안전). enrichment는 그대로 두고 모델
입력 선택만 끄므로 _check_feature_set_columns와 충돌하지 않는다.
"""

import pytest

from bakery.features.weather_features import WEATHER_FEATURE_COLUMNS
from bakery.features.cannibalization import CANNIBALIZATION_FEATURE_COLUMNS
from bakery.models.lightgbm_regressor import (
    FEATURE_GROUPS,
    GlobalLGBM,
    build_numeric_columns,
)


def test_drop_empty_is_noop():
    """drop_groups 기본값(빈 set)은 인자 생략과 동일 — 현 동작 보존."""
    default = build_numeric_columns("v2")
    explicit_empty = build_numeric_columns("v2", drop_groups=frozenset())
    assert default == explicit_empty


def test_drop_weather_removes_exactly_weather_columns():
    """단일 그룹 드롭 시 그 그룹 컬럼만 정확히 빠지고 순서·나머지는 그대로."""
    full = build_numeric_columns("v2")
    dropped = build_numeric_columns("v2", drop_groups={"weather"})
    assert set(full) - set(dropped) == set(WEATHER_FEATURE_COLUMNS)
    # 남은 컬럼은 원래 순서를 보존한다.
    assert dropped == [c for c in full if c not in set(WEATHER_FEATURE_COLUMNS)]


def test_drop_multiple_groups():
    full = build_numeric_columns("v2")
    dropped = build_numeric_columns("v2", drop_groups={"weather", "cannibalization"})
    removed = set(WEATHER_FEATURE_COLUMNS) | set(CANNIBALIZATION_FEATURE_COLUMNS)
    assert set(full) - set(dropped) == removed


def test_drop_group_absent_in_feature_set_is_noop():
    """v0엔 weather 그룹이 없으므로 드롭해도 변화 없음(그룹명은 유효)."""
    assert build_numeric_columns("v0", drop_groups={"weather"}) == build_numeric_columns("v0")


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="unknown feature groups"):
        build_numeric_columns("v2", drop_groups={"bogus"})


def test_all_registry_groups_are_valid_for_v3():
    """레지스트리 전 그룹을 v3에서 드롭하면 base(date/lag/rolling)만 남는다."""
    full = build_numeric_columns("v3")
    stripped = build_numeric_columns("v3", drop_groups=frozenset(FEATURE_GROUPS))
    all_group_cols = set().union(*(set(cols) for cols in FEATURE_GROUPS.values()))
    assert set(stripped) == set(full) - all_group_cols
    assert stripped  # base 코어는 남는다


def test_globallgbm_reflects_drop_in_feature_columns():
    """GlobalLGBM(drop_groups=...)가 numeric/feature_columns에 반영."""
    model = GlobalLGBM(feature_set="v2", drop_groups={"weather"})
    assert set(WEATHER_FEATURE_COLUMNS).isdisjoint(model.numeric_columns)
    assert set(WEATHER_FEATURE_COLUMNS).isdisjoint(model.feature_columns)

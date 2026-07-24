from __future__ import annotations
from enum import Enum


class ForecasterKind(str, Enum):
    CATEGORY_TOTAL = "category_total"
    DISTRIBUTIONAL = "distributional"
    POINT = "point_forecaster"
    COMPOSITE = "composite_pipeline"


_KIND: dict[str, ForecasterKind] = {
    "category_total": ForecasterKind.CATEGORY_TOTAL,
    "distributional_total": ForecasterKind.DISTRIBUTIONAL,
    "seasonal_naive": ForecasterKind.POINT,
    "moving_average": ForecasterKind.POINT,
    "lightgbm": ForecasterKind.POINT,
    "lightgbm_v1": ForecasterKind.POINT,
    "lightgbm_v2": ForecasterKind.POINT,
    "lightgbm_v3": ForecasterKind.POINT,
    "category_v4": ForecasterKind.COMPOSITE,
}

LAYER_NAMES: frozenset[str] = frozenset({"event_prior", "decision", "conformal_order"})


def kind_of(name: str) -> ForecasterKind:
    return _KIND[name]


_RUNNABLE_KINDS: frozenset[ForecasterKind] = frozenset(
    {ForecasterKind.CATEGORY_TOTAL, ForecasterKind.DISTRIBUTIONAL}
)


def is_runnable(name: str) -> bool:
    """실행 가능한 forecaster(category_total/distributional_total)면 True. 미등록/미지원=False."""
    try:
        return kind_of(name) in _RUNNABLE_KINDS
    except KeyError:
        return False


def build_forecaster(name: str):
    """forecaster 이름 → 어댑터 인스턴스. 미등록 KeyError."""
    from bakery.harness.forecasters import (
        CategoryTotalForecaster, DistributionalTotalForecaster,
    )
    factories = {
        "category_total": CategoryTotalForecaster,
        "distributional_total": DistributionalTotalForecaster,
    }
    return factories[name]()

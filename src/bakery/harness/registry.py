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


def is_supported_phase1(name: str) -> bool:
    """Phase 1은 category_total 경로만 실행(나머지는 taxonomy 등록만)."""
    return kind_of(name) is ForecasterKind.CATEGORY_TOTAL

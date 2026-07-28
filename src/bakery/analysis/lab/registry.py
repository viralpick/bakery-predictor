"""이름 → 핸들러 registry. harness registry.py의 kind/is_runnable 패턴 미러링."""
from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_DATA, KIND_HYPOTHESIS, AnalysisResult

HandlerFn = Callable[[AnalysisInputs], AnalysisResult]


@dataclass(frozen=True)
class Handler:
    name: str
    kind: str
    title: str
    fn: HandlerFn
    needs_predictions: bool = False     # harness-run predictions.csv 필요
    needs_multistore: bool = False      # 4매장 비교 전용
    needs_single_store: bool = False    # 광교 전용 소스(category_daily) 사용 → multistore 금지


DATA_ANALYSES: dict[str, Handler] = {}
HYPOTHESES: dict[str, Handler] = {}


def _register(
    target: dict[str, Handler], name: str, kind: str, title: str, flags: dict[str, Any]
) -> Callable[[HandlerFn], HandlerFn]:
    def deco(fn: HandlerFn) -> HandlerFn:
        if name in DATA_ANALYSES or name in HYPOTHESES:
            raise ValueError(f"핸들러 이름 중복 등록: {name}")
        target[name] = Handler(name=name, kind=kind, title=title, fn=fn, **flags)
        return fn
    return deco


def register_data(name: str, title: str, **flags: bool) -> Callable[[HandlerFn], HandlerFn]:
    """입력 데이터 분석 핸들러 등록 데코레이터."""
    return _register(DATA_ANALYSES, name, KIND_DATA, title, flags)


def register_hypothesis(name: str, title: str, **flags: bool) -> Callable[[HandlerFn], HandlerFn]:
    """가설 검증 핸들러 등록 데코레이터."""
    return _register(HYPOTHESES, name, KIND_HYPOTHESIS, title, flags)


def load_handlers() -> None:
    """핸들러 모듈을 import해 registry를 채운다(멱등)."""
    from bakery.analysis.lab.handlers import HANDLER_MODULES

    for module in HANDLER_MODULES:
        importlib.import_module(f"bakery.analysis.lab.handlers.{module}")


def all_names() -> frozenset[str]:
    load_handlers()
    return frozenset(DATA_ANALYSES) | frozenset(HYPOTHESES)


def resolve(name: str) -> Handler:
    load_handlers()
    if name in DATA_ANALYSES:
        return DATA_ANALYSES[name]
    if name in HYPOTHESES:
        return HYPOTHESES[name]
    raise KeyError(f"미등록 분석/가설: {name}")

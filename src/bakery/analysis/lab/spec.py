"""analysis-run YAML spec — pydantic 검증 + 폐기/오타 이름 강제.

harness `config.py`의 SpecError/DEPRECATED 패턴 미러링. registry를 import하지
않고 known_names를 주입받아 순환 import를 피한다(runner가 주입한다).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

MULTISTORE = "multistore"
DEFAULT_ALPHA = 0.8            # 측정 헌장: adjusted_demand = 정상 + 0.8×마감

# v5 conformal 구간예측 계열 — 점추정+위험수치 전환으로 폐기. 이식 금지.
DEPRECATED_ANALYSES: frozenset[str] = frozenset(
    {"diag_anchor_gh", "diag_chuseok_gh", "diagnose_conformal_residual"}
)


class AnalysisSpecError(ValueError):
    """spec 형식/이름 규칙 위반."""


class AnalysisDataSpec(BaseModel):
    model_config = {"extra": "forbid"}

    source: Literal["real"] = "real"
    store: str = "store_gw01"          # store_gw01 | multistore


class AnalysisSpec(BaseModel):
    model_config = {"extra": "forbid"}   # target 등 미지원 키는 즉시 거부

    name: str
    data: AnalysisDataSpec
    predictions: Path | None = None
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    data_analyses: dict[str, bool] = Field(default_factory=dict)
    hypotheses: dict[str, bool] = Field(default_factory=dict)
    params: dict[str, dict] = Field(default_factory=dict)

    def enabled(self, kind: str) -> list[str]:
        """kind('data_analyses'|'hypotheses')에서 True인 이름들(YAML 순서 보존)."""
        section: dict[str, bool] = getattr(self, kind)
        return [name for name, is_on in section.items() if is_on]

    def all_requested(self) -> list[str]:
        return list(self.data_analyses) + list(self.hypotheses)


def load_analysis_spec(
    path: str | Path, *, known_names: frozenset[str] | None = None
) -> AnalysisSpec:
    """YAML → AnalysisSpec. 폐기 이름은 항상 거부, 미등록 이름은 known_names 주면 거부."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        spec = AnalysisSpec(**raw)
    except Exception as exc:
        raise AnalysisSpecError(str(exc)) from exc
    _enforce_names(spec, known_names)
    return spec


def _enforce_names(spec: AnalysisSpec, known_names: frozenset[str] | None) -> None:
    requested = spec.all_requested()
    deprecated = [n for n in requested if n in DEPRECATED_ANALYSES]
    if deprecated:
        raise AnalysisSpecError(
            f"{deprecated}는 DEPRECATED(v5 conformal 구간예측 폐기) — 이식 대상 아님. "
            "spec에서 키를 삭제하라(off로도 남기지 말 것)."
        )
    if known_names is None:
        return
    unknown = [n for n in requested if n not in known_names]
    if unknown:
        raise AnalysisSpecError(f"미등록 분석/가설 이름: {unknown}. registry 등록명을 확인하라.")

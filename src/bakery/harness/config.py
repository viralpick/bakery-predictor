from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_FORECASTERS: list[str] = ["category_total", "distributional_total"]
DEFAULT_LAYERS: list[str] = ["event_prior"]
DEFAULT_METRICS: list[str] = [
    "wape", "wpe", "waste_rate", "soldout_median",
    "stockout_item_rate", "shortfall_day_rate",
]
DEPRECATED_FORECASTERS = {"conformal_interval"}


class SpecError(ValueError):
    """canonical 강제 규칙 위반."""


class DataSpec(BaseModel):
    source: Literal["real", "synthetic", "parquet"]
    store: str = "store_gw01"


class WindowSpec(BaseModel):
    scheme: Literal["expanding", "rolling"] = "expanding"
    n_folds: int = 52
    window_days: int = 730
    horizon_days: int = 7


class ExperimentSpec(BaseModel):
    name: str
    data: DataSpec
    target: str = "adjusted_demand_unit"
    forecaster: list[str] = Field(default_factory=lambda: list(DEFAULT_FORECASTERS))
    layers: list[str] = Field(default_factory=lambda: list(DEFAULT_LAYERS))
    event_priors: str | None = "gwangyo"
    window: WindowSpec = Field(default_factory=WindowSpec)
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    alpha: float = 0.8
    production_q: float = 0.85
    allow_deprecated: bool = False

    @field_validator("forecaster", "layers", mode="before")
    @classmethod
    def _wrap_single(cls, v):
        return [v] if isinstance(v, str) else v


def load_spec(path: str | Path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        spec = ExperimentSpec(**raw)
    except Exception as exc:
        raise SpecError(str(exc)) from exc
    _enforce(spec)
    return spec


def _enforce(spec: ExperimentSpec) -> None:
    if spec.target == "potential_demand" and not spec.allow_deprecated:
        raise SpecError("target=potential_demand는 오염 소스라 금지. allow_deprecated: true 필요.")
    if spec.metrics == ["mape"]:
        warnings.warn("MAPE 단독은 희소 품목에서 폭발한다. WAPE 병기 권장.", UserWarning)
    if "event_prior" in spec.layers and spec.event_priors is None:
        warnings.warn("event_prior layer가 있으나 event_priors 프리셋 키 미지정.", UserWarning)
    for name in spec.forecaster:
        if name in DEPRECATED_FORECASTERS:
            warnings.warn(f"{name}는 DEPRECATED forecaster.", UserWarning)

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_FORECASTERS: list[str] = ["category_total", "distributional_total"]
DEFAULT_LAYERS: list[str] = ["event_prior"]
DEFAULT_METRICS: list[str] = ["wape", "wpe", "stockout_risk", "surplus_mean_units", "surplus_rate"]
DEPRECATED_FORECASTERS = {"conformal_interval"}

MONDAY = 0
SUNDAY = 6


class SpecError(ValueError):
    """canonical 강제 규칙 위반."""


class DataSpec(BaseModel):
    source: Literal["real", "synthetic", "parquet"]
    store: str = "store_gw01"


class WindowSpec(BaseModel):
    """백테스트 fold 창 사양.

    lead_days / anchor_dow 는 운영 리드타임 정렬용 opt-in 파라미터다. 기본값
    (0 / None)은 현 헤드라인 동작(리드타임 0 + 인덱스 기반 연속 블록)과 동일하며,
    엔진 동등성 게이트(rtol=1e-9)가 이를 보증한다.
    """

    scheme: Literal["expanding", "rolling"] = "expanding"
    n_folds: int = 52
    window_days: int = 730
    horizon_days: int = 7
    lead_days: int = 0            # train/prior cutoff = test_start − lead_days. 0 = 현 동작
    anchor_dow: int | None = None  # None = 현 동작(인덱스 기반). 0=월요일 시작 블록
    # True면 원점 이후를 보는 자기회귀 feature(lag/rolling/ewma)를 가린다.
    # lead_days>0에서만 의미가 있다(원점=test_start). 기본 False = 현 동작.
    align_features: bool = False

    @field_validator("lead_days")
    @classmethod
    def _check_lead_days(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"lead_days는 0 이상이어야 한다(0=리드타임 없음): {v}")
        return v

    @field_validator("anchor_dow")
    @classmethod
    def _check_anchor_dow(cls, v: int | None) -> int | None:
        if v is not None and not MONDAY <= v <= SUNDAY:
            raise ValueError(f"anchor_dow는 {MONDAY}(월)~{SUNDAY}(일) 범위여야 한다: {v}")
        return v


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
    # "windowed" = 대상일 블록 fold(헤드라인). "panel" = 원점 기준 fold + 원점기준
    # AR feature(운영 정렬). 기본값은 헤드라인 경로다.
    engine: Literal["windowed", "panel"] = "windowed"
    # "category" = 카테고리 총량 발주까지(헤드라인, 기본). "item" = 총량을 품목별로
    # 배분해 `item_orders` 까지 낸다. 배분은 날짜마다 compute_proportions를 돌려
    # **느리므로** KPI가 필요한 실험에서만 켠다. 총량 지표는 배분 여부와 무관하게 동일.
    order_level: Literal["category", "item"] = "category"
    # KPI(비용·매진·아띠제 대비 절감률) 산출 여부. order_level="item" 필수 — 발주는
    # 품목별이고 폐기·매진시각은 품목 층위가 아니면 정의되지 않는다.
    # 매진시각 시뮬(도착 프로필 역산)이 무거워 기본 off.
    kpi: bool = False

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
    if spec.kpi and spec.order_level != "item":
        raise SpecError(
            "kpi: true는 order_level: item이 필요하다 — 폐기·매진시각은 품목 층위가 "
            "아니면 정의되지 않는다."
        )
    if spec.target == "potential_demand" and not spec.allow_deprecated:
        raise SpecError("target=potential_demand는 오염 소스라 금지. allow_deprecated: true 필요.")
    if spec.metrics == ["mape"]:
        warnings.warn("MAPE 단독은 희소 품목에서 폭발한다. WAPE 병기 권장.", UserWarning)
    if "event_prior" in spec.layers and spec.event_priors is None:
        warnings.warn("event_prior layer가 있으나 event_priors 프리셋 키 미지정.", UserWarning)
    for name in spec.forecaster:
        if name in DEPRECATED_FORECASTERS:
            warnings.warn(f"{name}는 DEPRECATED forecaster.", UserWarning)

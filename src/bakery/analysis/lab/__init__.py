"""데이터분석 + 가설검증 레이어(Phase 6) — analysis-run 오케스트레이션.

harness backbone(예측 평면)의 형제 표면. 이 레이어는 모델을 실행하지 않고,
canonical 입력 데이터와 (선택적으로) harness-run이 남긴 예측 artifact만 읽는다.
"""
from bakery.analysis.lab.result import (
    KIND_DATA,
    KIND_HYPOTHESIS,
    REASON_MULTISTORE_REQUIRED,
    REASON_OFF,
    REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED,
    AnalysisReport,
    AnalysisResult,
    SkippedResult,
)
from bakery.analysis.lab.spec import (
    DEFAULT_ALPHA,
    DEPRECATED_ANALYSES,
    MULTISTORE,
    AnalysisDataSpec,
    AnalysisSpec,
    AnalysisSpecError,
    load_analysis_spec,
)

__all__ = [
    "KIND_DATA", "KIND_HYPOTHESIS", "REASON_OFF", "REASON_PREDS_REQUIRED",
    "REASON_MULTISTORE_REQUIRED", "REASON_SINGLE_STORE_REQUIRED",
    "AnalysisResult", "SkippedResult", "AnalysisReport",
    "AnalysisDataSpec", "AnalysisSpec", "AnalysisSpecError", "load_analysis_spec",
    "DEPRECATED_ANALYSES", "MULTISTORE", "DEFAULT_ALPHA",
]

"""analysis-run 실행 엔진 — 켜진 항목만 돌리고, 끈/못 돌린 항목은 사유와 함께 남긴다.

harness runner.py와 달리 모델을 실행하지 않는다(예측 artifact는 읽기 전용).
핸들러 예외는 전체 실행을 죽이지 않고 그 항목만 error 스킵으로 강등한다 —
14개 항목 중 하나가 데이터 부족으로 실패해도 나머지 리포트는 나와야 한다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from bakery.analysis.lab import registry
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import (
    REASON_MULTISTORE_REQUIRED,
    REASON_OFF,
    REASON_PREDS_REQUIRED,
    REASON_SINGLE_STORE_REQUIRED,
    AnalysisReport,
    AnalysisResult,
    SkippedResult,
)
from bakery.analysis.lab.spec import AnalysisSpec


def _gate_reason(handler: registry.Handler, inputs: AnalysisInputs) -> str | None:
    """실행 전 게이트. 통과면 None, 아니면 스킵 사유."""
    if handler.needs_predictions and not inputs.has_predictions:
        return REASON_PREDS_REQUIRED
    if handler.needs_multistore and not inputs.is_multistore:
        return REASON_MULTISTORE_REQUIRED
    if handler.needs_single_store and inputs.is_multistore:
        return REASON_SINGLE_STORE_REQUIRED
    return None


def _handlers_in_order(spec: AnalysisSpec) -> list[tuple[registry.Handler, bool]]:
    """(핸들러, 켜짐여부) — registry 전체를 돌아 spec 미명시는 off로 취급한다."""
    registry.load_handlers()
    requested = {**spec.data_analyses, **spec.hypotheses}
    sections = (registry.DATA_ANALYSES, registry.HYPOTHESES)
    return [(handler, requested.get(name, False))
            for section in sections for name, handler in section.items()]


def _write_tables(result: AnalysisResult, out: Path) -> None:
    for label, table in result.tables:
        table.to_csv(out / f"{result.name}__{label}.csv", index=False)


def run_analysis(spec: AnalysisSpec, *, out_dir: Path) -> AnalysisReport:
    """spec에서 켜진 분석/가설을 실행해 AnalysisReport를 만든다."""
    inputs = AnalysisInputs.from_spec(spec)
    out = out_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump(mode="json")
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")

    results: list[AnalysisResult] = []
    skipped: list[SkippedResult] = []
    for handler, is_on in _handlers_in_order(spec):
        reason = REASON_OFF if not is_on else _gate_reason(handler, inputs)
        if reason is None:
            reason = _run_one(handler, inputs, out, results)
        if reason is not None:
            skipped.append(SkippedResult(name=handler.name, kind=handler.kind,
                                         title=handler.title, reason=reason))
    return AnalysisReport(name=spec.name, spec_resolved=resolved,
                          results=results, skipped=skipped)


def _run_one(handler: registry.Handler, inputs: AnalysisInputs,
             out: Path, results: list[AnalysisResult]) -> str | None:
    """핸들러 1개 실행. 성공하면 results에 추가하고 None, 실패하면 사유 문자열."""
    try:
        result = handler.fn(inputs)
    except Exception as exc:                     # noqa: BLE001 — 항목 단위 격리가 목적
        return f"error: {exc}"
    _write_tables(result, out)
    results.append(result)
    return None

"""분석/가설 실행 결과 컨테이너 — 핸들러 반환 계약."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

KIND_DATA = "data"
KIND_HYPOTHESIS = "hypothesis"

REASON_OFF = "off"                                  # YAML에서 꺼짐
REASON_PREDS_REQUIRED = "preds_required"            # preds artifact 미지정/부재
REASON_MULTISTORE_REQUIRED = "multistore_required"  # 4매장 전용 항목인데 단매장 spec
# 광교 전용 소스(category_daily=bonavi_daily)를 쓰는 항목인데 multistore spec.
# 게이트 없이 실행하면 광교 수치가 4매장 분석으로 라벨링되는 조용한 오데이터가 된다.
REASON_SINGLE_STORE_REQUIRED = "single_store_required"


@dataclass
class AnalysisResult:
    """핸들러 1개의 산출물. tables/verdict가 회귀 대조 대상, figures는 리포트 전용."""

    name: str
    kind: str
    title: str
    tables: list[tuple[str, pd.DataFrame]]
    figures: list[Any]
    verdict: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkippedResult:
    """실행하지 않은 항목 — 은폐 방지를 위해 리포트에 사유와 함께 남는다."""

    name: str
    kind: str
    title: str
    reason: str


@dataclass
class AnalysisReport:
    name: str
    spec_resolved: dict
    results: list[AnalysisResult]
    skipped: list[SkippedResult]

    def table_of(self, name: str, table_name: str) -> pd.DataFrame:
        """실행된 항목 `name`의 `table_name` 테이블. 없으면 KeyError."""
        for result in self.results:
            if result.name != name:
                continue
            for label, table in result.tables:
                if label == table_name:
                    return table
            raise KeyError(f"{name}에 테이블 '{table_name}' 없음")
        raise KeyError(f"실행된 항목에 '{name}' 없음")

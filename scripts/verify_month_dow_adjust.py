"""검증: 광교 월 × 요일 12×7 매트릭스 — 마감 조정 전후 비교.

계산은 `bakery.analysis.month_dow`로 옮겼다(Phase 6). 현 vintage 실행은
`bakery analysis-run`(hypotheses.month_dow_adjust)을 쓴다 — 이 wrapper는
canonical category_daily로 표를 print한다(레거시 α=0.5 직독 경로는 폐기).

실행: uv run python scripts/verify_month_dow_adjust.py
"""
from __future__ import annotations

from bakery.analysis.month_dow import (
    ADJUSTED_COLUMN,
    RAW_COLUMN,
    adjust_effect_table,
    month_dow_matrix,
)
from bakery.features.category_aggregate import build_category_daily


def main() -> None:
    series = build_category_daily(alpha=0.8).df
    print("=== raw (sold_total_unit) — 월 × 요일 일평균 ===")
    print(month_dow_matrix(series, RAW_COLUMN).round(1).to_string())
    print("\n=== adjusted_demand_unit — 월 × 요일 일평균 ===")
    print(month_dow_matrix(series, ADJUSTED_COLUMN).round(1).to_string())
    print("\n=== 조정 효과(delta_pct 하위 10칸) ===")
    table = adjust_effect_table(series).sort_values("delta_pct")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

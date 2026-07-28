"""월 × 요일 12×7 매트릭스 — 마감 조정(adjust) 전후 비교.

출처: scripts/verify_month_dow_adjust.py. 스크립트는 레거시 sales.parquet 직독 +
α=0.5였고, 이 모듈은 (date, 값) 시리즈를 인자로 받는 순수함수라 소스/α에 무관하다.
"""
from __future__ import annotations

import pandas as pd

DOW_LABELS: tuple[str, ...] = ("월", "화", "수", "목", "금", "토", "일")
MONTH_DOW_VALUE_COLUMNS: tuple[str, ...] = ("sold_total_unit", "adjusted_demand_unit",
                                            "sold_closing")
RAW_COLUMN = "sold_total_unit"
ADJUSTED_COLUMN = "adjusted_demand_unit"
CLOSING_COLUMN = "sold_closing"


def _with_axes(series: pd.DataFrame) -> pd.DataFrame:
    frame = series.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month
    frame["dow"] = frame["date"].dt.dayofweek
    return frame


def month_dow_matrix(series: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """월(행) × 요일(열) 일평균 매트릭스. 관측 없는 칸은 NaN."""
    frame = _with_axes(series)
    matrix = frame.groupby(["month", "dow"])[value_column].mean().unstack("dow")
    matrix = matrix.reindex(columns=range(len(DOW_LABELS)))
    matrix.columns = list(DOW_LABELS)
    return matrix


def adjust_effect_table(series: pd.DataFrame) -> pd.DataFrame:
    """월×요일 칸별 raw vs adjusted 차이 — 마감 조정이 어느 칸을 얼마나 낮추는가."""
    frame = _with_axes(series)
    table = (frame.groupby(["month", "dow"])
             .agg(raw_mean=(RAW_COLUMN, "mean"),
                  adjusted_mean=(ADJUSTED_COLUMN, "mean"),
                  closing_mean=(CLOSING_COLUMN, "mean"))
             .reset_index())
    table["delta"] = table["adjusted_mean"] - table["raw_mean"]
    table["delta_pct"] = (table["delta"] / table["raw_mean"] * 100).where(
        table["raw_mean"] > 0, 0.0)
    return table

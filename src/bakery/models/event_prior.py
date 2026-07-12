"""Post-model level-anchor prior for sharp rare calendar events (xmas 등).

트리가 학습창당 2~5샘플의 sharp 이벤트를 못 잡는 문제를, 예측 이후
leakage-safe 레벨-앵커 블렌드로 보정한다. 자세한 배경은
docs/superpowers/specs/2026-07-12-event-level-prior-design.md 참조.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EVENTS: dict[str, tuple[int, int]] = {"xmas": (12, 25)}
DEFAULT_K = 1.5


class EventLevelPrior:
    def __init__(self, events: dict[str, tuple[int, int]] | None = None, k: float = DEFAULT_K):
        self.events = dict(events) if events is not None else dict(DEFAULT_EVENTS)
        self.k = k
        self._event_actuals: list[tuple[pd.Timestamp, float]] = []  # (date, actual)

    def fit(self, history: pd.DataFrame, date_col: str = "date",
            target_col: str = "adjusted_demand_unit") -> "EventLevelPrior":
        d = history[[date_col, target_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col])
        mask = d[date_col].apply(self.is_event_day)
        ev = d[mask].dropna(subset=[target_col])
        self._event_actuals = sorted(
            (row[date_col], float(row[target_col])) for _, row in ev.iterrows()
        )
        return self

    def is_event_day(self, date: pd.Timestamp) -> bool:
        date = pd.Timestamp(date)
        return any((date.month, date.day) == (m, day) for m, day in self.events.values())

    def level_for(self, date: pd.Timestamp) -> tuple[float | None, int]:
        date = pd.Timestamp(date)
        past = [a for (ed, a) in self._event_actuals if ed < date]
        if not past:
            return None, 0
        return float(np.mean(past)), len(past)

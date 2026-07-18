"""Post-model level-anchor prior for sharp rare calendar events (xmas/설/추석 등).

트리가 학습창당 2~5샘플의 sharp 이벤트를 못 잡는 문제를, 예측 이후
leakage-safe 레벨-앵커 블렌드로 보정한다. 자세한 배경은
docs/superpowers/specs/2026-07-12-event-level-prior-design.md 참조.

이벤트별로 과거 실측을 분리 저장한다(per-event): 한 매장에 여러 이벤트가
등록돼도(예: 광교 xmas+추석) 서로의 레벨이 섞이지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EVENTS: dict[str, tuple[int, int]] = {"xmas": (12, 25)}
DEFAULT_K = 1.5


class EventLevelPrior:
    def __init__(self, events: dict[str, tuple[int, int]] | None = None,
                 k: float = DEFAULT_K, min_events: int = 2,
                 lunar_events: dict[str, dict[int, str]] | None = None,
                 recency: int | None = None):
        self.events = dict(events) if events is not None else dict(DEFAULT_EVENTS)
        self.k = k
        self.min_events = min_events
        # recency=N: anchor를 최근 N회 이벤트 median으로 제한(레벨 추종). None=전체 history(기본).
        # 수요 추세(예: -26% 하락)가 있을 때 absolute median 앵커가 과대예측하는 것을 완화.
        self.recency = recency
        self.lunar_events = dict(lunar_events) if lunar_events else {}
        # 음력 날짜(normalized) → 이벤트명. per-event 분리 및 날짜→이벤트 판별용.
        self._lunar_date_to_name: dict[pd.Timestamp, str] = {}
        for name, datemap in self.lunar_events.items():
            for date_str in datemap.values():
                self._lunar_date_to_name[pd.Timestamp(date_str).normalize()] = name
        # 이벤트명 → 정렬된 [(date, actual)]. 이벤트별 분리 저장.
        self._event_actuals: dict[str, list[tuple[pd.Timestamp, float]]] = {}

    def _event_name_for(self, date: pd.Timestamp) -> str | None:
        """이 날짜가 속한 등록 이벤트명. 어느 이벤트도 아니면 None.

        solar 우선: 양력/음력 날짜가 겹치면 양력 이벤트명을 반환(현 이벤트셋은 충돌 없음).
        """
        date = pd.Timestamp(date)
        for name, (m, day) in self.events.items():
            if (date.month, date.day) == (m, day):
                return name
        return self._lunar_date_to_name.get(date.normalize())

    def fit(self, history: pd.DataFrame, date_col: str = "date",
            target_col: str = "adjusted_demand_unit") -> "EventLevelPrior":
        d = history[[date_col, target_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col])
        d = d.dropna(subset=[target_col])
        d = d[d[date_col].apply(self.is_event_day)]   # 이벤트일만(작은 집합)
        actuals: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        for _, row in d.iterrows():
            name = self._event_name_for(row[date_col])
            actuals.setdefault(name, []).append((row[date_col], float(row[target_col])))
        for name in actuals:
            actuals[name].sort()
        self._event_actuals = actuals
        return self

    def is_event_day(self, date: pd.Timestamp) -> bool:
        return self._event_name_for(date) is not None

    def level_for(self, date: pd.Timestamp) -> tuple[float | None, int]:
        date = pd.Timestamp(date)
        name = self._event_name_for(date)
        if name is None:
            return None, 0
        # per-event: 같은 이벤트의 과거 실측만(엄격 ed<date). median: anomaly-robust (§8).
        past = [a for (ed, a) in self._event_actuals.get(name, []) if ed < date]
        if not past:
            return None, 0
        n_past = len(past)  # shrink용 신뢰도(min_events 판정)는 전체 표본 기준
        anchor = past[-self.recency:] if self.recency is not None else past
        return float(np.median(anchor)), n_past

    def blend(self, dates, base_expected, base_production):
        dates = [pd.Timestamp(d) for d in dates]
        exp = np.asarray(base_expected, dtype=float).copy()
        prod = np.asarray(base_production, dtype=float).copy()
        for i, d in enumerate(dates):
            if not self.is_event_day(d):
                continue
            prior, n_past = self.level_for(d)
            if prior is None or n_past < self.min_events or exp[i] <= 0:
                continue
            shrink = n_past / (n_past + self.k)
            blended_exp = shrink * prior + (1 - shrink) * exp[i]
            correction = blended_exp / exp[i]
            exp[i] = blended_exp
            prod[i] = prod[i] * correction
        return exp, prod

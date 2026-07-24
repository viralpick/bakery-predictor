"""특수일 EventLevelPrior 프리셋 (scripts/store_predictive_power.py에서 승격, 단일 출처).

XMAS/CHILDRENS는 양력 고정, SEOLLAL/CHUSEOK는 음력 이벤트라 bakery.data.calendar의
LUNAR_EVENT_DATES를 그대로 import (원본 스크립트와 동일 소스 → 값 드리프트 방지).
매장×이벤트 opt-in은 OOS 순개선 확인된 것만 등록 (docs/holiday_prior_scan_result.md).
"""
from __future__ import annotations

from bakery.data.calendar import LUNAR_EVENT_DATES

XMAS = {"xmas": (12, 25)}
CHILDRENS = {"childrens": (5, 5)}   # 어린이날 (양력 고정)
SEOLLAL = {"seollal": LUNAR_EVENT_DATES["days_to_seollal"]}
CHUSEOK = {"chuseok": LUNAR_EVENT_DATES["days_to_chuseok"]}

STORE_EVENT_PRIORS: dict[str, dict] = {
    "gwangyo":      {"events": {**XMAS, **CHILDRENS}, "lunar_events": dict(CHUSEOK)},
    "samsung":      {"events": dict(XMAS), "lunar_events": {}},
    "mecenatpolis": {"events": dict(XMAS), "lunar_events": dict(SEOLLAL)},
    "gwanghwamun":  {"events": dict(XMAS), "lunar_events": {}},
}


def resolve_event_priors(key: str | None) -> tuple[dict | None, dict | None]:
    if key is None:
        return None, None
    cfg = STORE_EVENT_PRIORS[key]
    return cfg.get("events"), cfg.get("lunar_events")

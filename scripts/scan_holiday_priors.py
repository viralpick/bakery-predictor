"""트랙2 후보 명절 A/B 스캔 — 광교 EventLevelPrior 확장 대상 선별.

이상치 분석(트랙1 doc)에서 "크게 튄 날 12일 = 명절"이나 광교엔 xmas+추석만 등록됨.
후보 명절 각각을 임시 등록해 verify_event_prior 패턴(leave-past-out, per-event)으로
base(LightGBM) vs blend(EventLevelPrior) WAPE·bias를 측정 → **순개선(WAPE↓)만 채택**.

기존 원칙(store_predictive_power 주석): OOS 순개선 확인된 매장×이벤트만 등록.

실행: PYTHONPATH=scripts uv run --with matplotlib python scripts/scan_holiday_priors.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from bakery.data.calendar import LUNAR_EVENT_DATES
from verify_event_prior import (
    build_feat,
    evaluate_event,
    event_days_by_name,
    print_event_result,
)
from bakery.models.event_prior import EventLevelPrior

CD_CODE, STORE_ID, LABEL = "1000000047", "store_gw01", "광교"

# 양력 고정 명절 (월, 일)
CANDIDATE_SOLAR = {
    "sinjeong": (1, 1),       # 신정
    "samiljeol": (3, 1),      # 삼일절
    "childrens": (5, 5),      # 어린이날
    "gwangbokjeol": (8, 15),  # 광복절
}

# 부처님오신날 (음력 4/8 → 양력). 2025-05-05는 어린이날과 겹쳐 solar 우선 규칙상 childrens로 분류됨.
BUDDHA_DATES = {
    2021: "2021-05-19", 2022: "2022-05-08", 2023: "2023-05-27",
    2024: "2024-05-15", 2025: "2025-05-05",
}

CANDIDATE_LUNAR = {
    "seollal": LUNAR_EVENT_DATES["days_to_seollal"],  # 설날 (기존 datemap 재사용)
    "buddha": BUDDHA_DATES,
}


def _scan(feat, base_cfg: dict, recency: int | None) -> list[tuple]:
    """base_cfg에 recency 적용해 후보별 base vs blend WAPE·2025-fold err 산출."""
    prior_cfg = dict(base_cfg, recency=recency)
    prior_template = EventLevelPrior(**prior_cfg)
    groups = event_days_by_name(feat, prior_template)
    rows = []
    for event_name, days in sorted(groups.items()):
        result = evaluate_event(feat, days, prior_cfg)
        if result is None:
            rows.append((event_name, None, None, None))
            continue
        bw, blw = result["base_wape"], result["blend_wape"]
        # 2025 fold(=2026 prospective proxy) blend err%
        r25 = next((r for r in result["rows"] if str(r["date"]).startswith("2025")), None)
        err25 = ((r25["blend"] - r25["actual"]) / r25["actual"] * 100) if r25 else None
        rows.append((event_name, bw, blw, err25))
    return rows


def _tag(bw, blw) -> str:
    if bw is None:
        return "평가불가"
    if abs(blw - bw) < 1e-6:
        return "무효(prior 미작동)"
    return f"채택({bw:.3f}→{blw:.3f})" if blw < bw else f"기각({bw:.3f}→{blw:.3f})"


def run() -> None:
    print(f"=== 트랙2 후보 명절 A/B 스캔 ({LABEL}) — recency None vs 2 ===")
    print("leave-past-out · train window=730d · per-event. err25=2025 fold blend err%(=2026 proxy)\n")
    base_cfg = dict(events=CANDIDATE_SOLAR, lunar_events=CANDIDATE_LUNAR)
    feat = build_feat(CD_CODE, STORE_ID)

    full = {r[0]: r for r in _scan(feat, base_cfg, recency=None)}
    rec2 = {r[0]: r for r in _scan(feat, base_cfg, recency=2)}

    print(f"{'명절':<14} {'full: WAPE→blend (err25)':<34} {'recency2: WAPE→blend (err25)':<34}")
    for name in sorted(full):
        _, bw_f, blw_f, e_f = full[name]
        _, bw_r, blw_r, e_r = rec2[name]
        def fmt(bw, blw, e):
            if bw is None:
                return "평가불가"
            es = f"{e:+.0f}%" if e is not None else "n/a"
            return f"{_tag(bw, blw):<20} err25={es}"
        print(f"{name:<14} {fmt(bw_f, blw_f, e_f):<34} {fmt(bw_r, blw_r, e_r):<34}")


if __name__ == "__main__":
    run()

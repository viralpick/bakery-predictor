"""예약(대량) 주문 검출 `flag_bulk_lines` 단위 테스트.

규칙 (line 단위 mask 반환):
  Tier-1 단일품목 집중: (receipt,item) qty ≥ 10 AND ≥ 3×median_daily(item) AND active_days(item) ≥ 14
  Tier-2 다품목 event : receipt total ≥ 30 AND receipt maxit ≥ 5 → 그 영수증의 qty ≥ 5 라인만 flag
"""
from __future__ import annotations

import pandas as pd

from bakery.data.bulk import flag_bulk_lines


def _lines(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """(receipt_id, item_id, date, qty) 행들 → per-line frame."""
    return pd.DataFrame(rows, columns=["receipt_id", "item_id", "date", "qty"])


def _daily_history(item_id: str, per_day_qty: int, n_days: int) -> list[tuple[str, str, str, int]]:
    """한 품목이 n_days 동안 매일 per_day_qty개 팔린 정상 이력(각 1 영수증)."""
    return [
        (f"r_hist_{item_id}_{d}", item_id, f"2024-01-{d + 1:02d}", per_day_qty)
        for d in range(n_days)
    ]


def test_tier1_single_item_spike_flagged():
    # A는 20일간 매일 3개(median=3, active=20). 한 영수증에 15개 → 예약.
    rows = _daily_history("A", 3, 20) + [("r_big", "A", "2024-02-01", 15)]
    df = _lines(rows)
    mask = flag_bulk_lines(df)
    assert mask.sum() == 1
    assert mask[df["receipt_id"] == "r_big"].iloc[0]


def test_tier1_popular_item_not_flagged():
    # B는 median=10. 한 영수증 12개 → 12<3×10=30 이라 예약 아님(정상 인기품목).
    rows = _daily_history("B", 10, 20) + [("r_pop", "B", "2024-02-01", 12)]
    df = _lines(rows)
    mask = flag_bulk_lines(df)
    assert mask.sum() == 0


def test_tier1_below_floor_not_flagged():
    # C는 median=1(active=20). 한 영수증 8개: 8≥3×1이지만 floor 10 미만 → 아님.
    rows = _daily_history("C", 1, 20) + [("r_c", "C", "2024-02-01", 8)]
    df = _lines(rows)
    assert flag_bulk_lines(df).sum() == 0


def test_tier1_low_active_days_not_flagged():
    # D는 active_days=5뿐(median 신뢰 불가). qty 20이어도 보수적으로 남김.
    rows = _daily_history("D", 2, 5) + [("r_d", "D", "2024-02-01", 20)]
    assert flag_bulk_lines(_lines(rows)).sum() == 0


def test_tier2_multi_item_event_flagged():
    # 한 영수증에 5품목 × 6개 = total 30, maxit 6 → 5개 라인 모두 flag(각 6≥5).
    # (각 품목은 이력 median 낮아 Tier-1로도 걸리지 않게 active<14로 둠)
    rows = [("r_ev", f"E{i}", "2024-02-01", 6) for i in range(5)]
    mask = flag_bulk_lines(_lines(rows))
    assert mask.sum() == 5


def test_tier2_diffuse_basket_not_flagged():
    # total 30이지만 30품목 × 1개씩(maxit 1 < 5) → 분산 장바구니, 예약 아님.
    rows = [("r_basket", f"F{i}", "2024-02-01", 1) for i in range(30)]
    assert flag_bulk_lines(_lines(rows)).sum() == 0


def test_tier2_line_level_keeps_small_addon():
    # total 32 = 한 품목 30 + 소품목 1×2. maxit 30≥5, total≥30 → 30라인만 flag,
    # qty<5 애드온 2라인은 보존(line-level, footfall proxy 유지).
    rows = [
        ("r_mix", "G_big", "2024-02-01", 30),
        ("r_mix", "G_a", "2024-02-01", 1),
        ("r_mix", "G_b", "2024-02-01", 1),
    ]
    df = _lines(rows)
    mask = flag_bulk_lines(df)
    assert mask.sum() == 1
    assert mask[df["item_id"] == "G_big"].iloc[0]
    assert not mask[df["item_id"] == "G_a"].iloc[0]


def test_tier2_below_total_not_triggered():
    # total 25 (< 30), 여러 품목 5개씩 → Tier-2 미발동. Tier-1도 이력 없어 미발동.
    rows = [("r_25", f"H{i}", "2024-02-01", 5) for i in range(5)]
    assert flag_bulk_lines(_lines(rows)).sum() == 0


def test_multiple_lines_same_receipt_item_summed():
    # 같은 (영수증,품목)이 여러 SLIP_LINE으로 쪼개져 들어와도 합산해 판정.
    # J median=2(active=20). 한 영수증에 8+8 두 라인 = 16개 → Tier-1 flag(16≥10, 16≥3×2).
    rows = _daily_history("J", 2, 20) + [
        ("r_split", "J", "2024-02-01", 8),
        ("r_split", "J", "2024-02-01", 8),
    ]
    df = _lines(rows)
    mask = flag_bulk_lines(df)
    assert mask[df["receipt_id"] == "r_split"].sum() == 2


def test_returns_boolean_series_aligned_to_index():
    rows = _daily_history("A", 3, 20) + [("r_big", "A", "2024-02-01", 15)]
    df = _lines(rows).sample(frac=1.0, random_state=0)  # 인덱스 셔플
    mask = flag_bulk_lines(df)
    assert mask.index.equals(df.index)
    assert mask.dtype == bool

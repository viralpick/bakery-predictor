"""배수 제약 하 총량 보존 배분 — 규칙과 총량 보존을 정확값으로 못박는다."""
import numpy as np
import pandas as pd
import pytest

from bakery.models.batch_rounding import (
    BATCH_RULE_TO_UNIT,
    NO_BATCH_CONSTRAINT,
    distribute_with_batch,
    estimate_batch_units,
    resolve_batch_units,
    round_to_batch,
)

# ---------------------------------------------------------------------------
# ★배수 반올림 규칙 (architect 확정)
#   k=3: 0~4.5 → 3 / 4.5~7.5 → 6 / 7.5~10.5 → 9. 경계는 위로. 0 금지.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("qty", "expected"), [
    (0.0, 3),      # ★0도 최소 1배수 — 0 금지
    (0.4, 3),
    (1.0, 3),
    (4.4, 3),
    (4.5, 6),      # ★경계는 위로 (banker's rounding이면 6이 안 나온다)
    (4.6, 6),
    (7.4, 6),
    (7.5, 9),      # ★같은 규칙의 다음 경계
    (10.4, 9),
    (10.5, 12),
])
def test_round_to_batch_three(qty, expected):
    assert round_to_batch(qty, 3) == expected


@pytest.mark.parametrize(("qty", "expected"), [
    (0.0, 2), (0.9, 2), (1.0, 2), (2.9, 2), (3.0, 4), (5.0, 6),
])
def test_round_to_batch_even(qty, expected):
    """짝수 제약: 경계 (n+0.5)*2 = 1,3,5… 에서 위로."""
    assert round_to_batch(qty, 2) == expected


def test_round_to_batch_boundary_is_half_up_not_bankers():
    """★파이썬 round()는 banker's rounding(2.5→2)이라 쓰면 안 된다.

    4.5/3 = 1.5 인데 banker's면 2가 아니라 2(짝수)로 가서 우연히 맞지만,
    7.5/3 = 2.5 는 banker's면 2 → 6이 되어 규칙(9)과 어긋난다.
    """
    assert round(2.5) == 2                  # 파이썬 기본 동작(참조)
    assert round_to_batch(7.5, 3) == 9      # 우리 규칙은 위로


def test_round_to_batch_no_constraint_still_forbids_zero():
    """k=1(제약 없음)도 0을 내지 않는다 — 발주 0은 진열 없음이라 의도가 아니다."""
    assert round_to_batch(0.0, NO_BATCH_CONSTRAINT) == 1
    assert round_to_batch(0.4, NO_BATCH_CONSTRAINT) == 1
    assert round_to_batch(2.6, NO_BATCH_CONSTRAINT) == 3


def test_round_to_batch_rejects_negative():
    with pytest.raises(ValueError, match="qty"):
        round_to_batch(-1.0, 3)


def test_batch_rule_map_matches_artisee_master():
    """아띠제 마스터 표기 3종이 모두 매핑돼 있다."""
    assert BATCH_RULE_TO_UNIT == {"짝수": 2, "3의배수": 3, "8의배수": 8}


def test_resolve_batch_units_defaults_to_unconstrained():
    units = resolve_batch_units(["a", "b", "c"], {"a": 3})
    assert units.tolist() == [3, NO_BATCH_CONSTRAINT, NO_BATCH_CONSTRAINT]


# ---------------------------------------------------------------------------
# ★총량 보존 — 이 모듈의 존재 이유
# ---------------------------------------------------------------------------

def _targets(**kwargs) -> pd.Series:
    return pd.Series(kwargs, dtype=float)


def test_total_preserved_exactly_with_mixed_units():
    """★제약 품목은 배수를 지키고 비제약 품목이 잔차를 흡수해 합이 정확히 일치한다."""
    targets = _targets(a=10.4, b=7.6, c=30.2, d=25.9, e=25.9)
    out = distribute_with_batch(targets, 100, unit_map={"a": 3, "b": 8})
    assert int(out["qty"].sum()) == 100
    assert int(out.loc[out.item_id == "a", "qty"].iloc[0]) == 9    # 10.4 → 9 (3배수)
    assert int(out.loc[out.item_id == "b", "qty"].iloc[0]) == 8    # 7.6 → 8 (8배수)


def test_every_item_respects_its_unit_and_is_positive():
    targets = _targets(a=4.6, b=1.2, c=50.0, d=20.0, e=9.9)
    out = distribute_with_batch(targets, 90, unit_map={"a": 3, "b": 8, "e": 2})
    for row in out.itertuples():
        assert row.qty > 0, f"{row.item_id} 가 0"
        assert row.qty % row.unit == 0, f"{row.item_id} 가 배수 {row.unit} 위반"
    assert int(out["qty"].sum()) == 90


def test_total_preserved_when_upward_adjustment_needed():
    """목표 합이 총량보다 작을 때(잔차 +) 흡수."""
    out = distribute_with_batch(_targets(a=1.0, b=1.0, c=1.0), 30)
    assert int(out["qty"].sum()) == 30


def test_total_preserved_when_downward_adjustment_needed():
    """목표 합이 총량보다 클 때(잔차 −) 흡수. 최소 1은 지킨다."""
    out = distribute_with_batch(_targets(a=50.0, b=50.0, c=50.0), 10)
    assert int(out["qty"].sum()) == 10
    assert (out["qty"] >= 1).all()


def test_residual_absorption_prefers_largest_remainder():
    """★잔차 +1은 소수부가 가장 큰 품목에 간다(largest remainder)."""
    out = distribute_with_batch(_targets(a=1.9, b=1.1, c=1.0), 5).set_index("item_id")
    # base = floor → 1,1,1 = 3, 잔차 2 → 소수부 큰 a(0.9) 먼저, 다음 b(0.1)
    assert int(out.loc["a", "qty"]) == 2
    assert int(out.loc["b", "qty"]) == 2
    assert int(out.loc["c", "qty"]) == 1


def test_order_of_items_is_preserved():
    """반환 순서가 입력 순서와 같아야 한다 — 조인이 조용히 어긋나는 것을 막는다."""
    targets = _targets(z=5.0, a=5.0, m=5.0)
    out = distribute_with_batch(targets, 15, unit_map={"a": 3})
    assert out["item_id"].tolist() == ["z", "a", "m"]


def test_fails_loud_when_total_below_minimum():
    """★총량이 최소 필요량보다 작으면 조용히 총량을 깨지 않고 예외를 낸다."""
    with pytest.raises(ValueError, match="흡수할 수 없다"):
        distribute_with_batch(_targets(a=1.0, b=1.0, c=1.0), 2, unit_map={"a": 8})


def test_fails_loud_when_only_constrained_items_and_total_mismatch():
    """비제약 품목이 없으면 잔차를 흡수할 수단이 없다 — 불일치를 숨기지 않는다."""
    with pytest.raises(ValueError, match="총량 보존 불가"):
        distribute_with_batch(_targets(a=10.0, b=10.0), 100, unit_map={"a": 3, "b": 3})


def test_only_constrained_items_ok_when_total_happens_to_match():
    """배수만으로 총량이 우연히 맞으면 통과한다."""
    out = distribute_with_batch(_targets(a=9.0, b=6.0), 15, unit_map={"a": 3, "b": 3})
    assert int(out["qty"].sum()) == 15


def test_rejects_negative_total():
    with pytest.raises(ValueError, match="total"):
        distribute_with_batch(_targets(a=1.0), -5)


# ---------------------------------------------------------------------------
# 데이터에서 배수 추정 — 마스터 누락 보완
# ---------------------------------------------------------------------------

def _inventory(item: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"item_id": [item] * len(values), "production_qty": values})


def test_estimate_finds_clean_multiple():
    """총생산이 전부 3의 배수면 k=3으로 추정된다."""
    inv = _inventory("x", [3.0, 6.0, 9.0, 12.0] * 20)
    out = estimate_batch_units(inv, min_rows=60)
    assert out.loc[0, "item_id"] == "x"
    assert int(out.loc[0, "unit"]) == 3
    assert out.loc[0, "align_rate"] == pytest.approx(1.0, rel=1e-12)


def test_estimate_ignores_chance_alignment():
    """★우연 정렬은 채택하지 않는다 — 임의 값이면 배수 없음."""
    rng = np.random.default_rng(0)
    inv = _inventory("y", (rng.integers(1, 50, 200)).astype(float).tolist())
    assert estimate_batch_units(inv, min_rows=60).empty


def test_estimate_skips_sparse_items():
    """행이 적으면 판정하지 않는다(노이즈로 배수를 발명하는 것 방지)."""
    assert estimate_batch_units(_inventory("z", [3.0, 6.0, 9.0]), min_rows=60).empty

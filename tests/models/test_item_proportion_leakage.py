# tests/models/test_item_proportion_leakage.py
import pandas as pd
import pytest
from bakery.models.item_proportion import compute_proportions, distribute_total


def _hist(rows):
    return pd.DataFrame(rows, columns=["date", "item_id", "category_id", "sold_units", "is_stockout", "stockout_time"])


def test_compute_proportions_ignores_rows_at_or_after_cutoff():
    cutoff = pd.Timestamp("2024-02-01")
    base = [
        ["2024-01-10", "a", "bread", 10, False, pd.NaT],
        ["2024-01-10", "b", "bread", 30, False, pd.NaT],
        ["2024-01-20", "a", "bread", 10, False, pd.NaT],
        ["2024-01-20", "b", "bread", 30, False, pd.NaT],
    ]
    hist1 = _hist([[pd.Timestamp(d), i, c, s, so, t] for d, i, c, s, so, t in base])
    # cutoff 이후 + cutoff 당일(==) 둘 다 주입 — compute_proportions는 < cutoff만 써야 하므로
    # 둘 다 비율에 영향 없어야 한다 (== 케이스가 <→<= 오프바이원 회귀를 잡는다).
    future = [
        [pd.Timestamp("2024-02-05"), "a", "bread", 9999, False, pd.NaT],
        [pd.Timestamp("2024-02-01"), "a", "bread", 9999, False, pd.NaT],
    ]
    hist2 = pd.concat([hist1, _hist(future)], ignore_index=True)

    p1 = compute_proportions(hist1, cutoff).set_index("item_id")["proportion"].sort_index()
    p2 = compute_proportions(hist2, cutoff).set_index("item_id")["proportion"].sort_index()
    assert p1.round(9).equals(p2.round(9))


# ---------------------------------------------------------------------------
# ★리드타임 축 — 기존 가드가 못 잡는다
#
# 위 테스트는 "cutoff 이후를 안 쓴다"만 보장하고, cutoff 자체는 대상일로 고정돼 있다.
# 리드타임이 있으면 발주 시점엔 `대상일 - lead_days` 까지만 알 수 있으므로, 그 구간
# (cutoff, 대상일) 실적이 배분 비율에 새면 leakage다. C1(운영 horizon 불일치)에서
# 총량 경로에 있던 것과 같은 유형의 결함이다.
# ---------------------------------------------------------------------------

LEAD_DAYS = 5
TARGET = pd.Timestamp("2024-02-01")


def _two_item_history():
    """a:b = 10:30 이 안정적으로 유지되는 history (배분 비율 0.25:0.75)."""
    rows = []
    for day in pd.date_range("2024-01-01", "2024-01-26"):
        rows.append([day, "a", "bread", 10, False, pd.NaT])
        rows.append([day, "b", "bread", 30, False, pd.NaT])
    return _hist(rows)


def _spike_in_lead_window():
    """(대상일 − lead_days, 대상일) 구간 = 발주 시점엔 모르는 실적."""
    return _hist([
        [pd.Timestamp("2024-01-29"), "a", "bread", 99999, False, pd.NaT],
        [pd.Timestamp("2024-01-31"), "a", "bread", 99999, False, pd.NaT],
    ])


def _props(history, **kwargs):
    out = compute_proportions(history, TARGET, **kwargs)
    return out.set_index("item_id")["proportion"].sort_index()


def test_lead_window_leaks_without_cutoff():
    """★대조군: cutoff를 안 주면 리드타임 구간 실적이 실제로 비율을 바꾼다.

    이 단언이 깨지면 아래 회귀 테스트가 무의미해진다(주입한 spike가 애초에
    아무 영향이 없다는 뜻이므로). 결함의 존재를 먼저 고정한다.
    """
    clean = _two_item_history()
    leaky = pd.concat([clean, _spike_in_lead_window()], ignore_index=True)
    assert _props(clean)["a"] != pytest.approx(_props(leaky)["a"], rel=1e-9)


def test_cutoff_date_blocks_lead_window_leak():
    """★회귀: cutoff_date를 주면 리드타임 구간 실적이 비율에 영향을 주지 못한다."""
    clean = _two_item_history()
    leaky = pd.concat([clean, _spike_in_lead_window()], ignore_index=True)
    cutoff = TARGET - pd.Timedelta(days=LEAD_DAYS)
    before = _props(clean, cutoff_date=cutoff)
    after = _props(leaky, cutoff_date=cutoff)
    assert before["a"] == pytest.approx(after["a"], rel=1e-12)
    assert before["b"] == pytest.approx(after["b"], rel=1e-12)


def test_cutoff_none_preserves_legacy_behaviour():
    """cutoff_date=None은 기존 동작과 정확히 같다(계약 보존)."""
    hist = _two_item_history()
    assert _props(hist, cutoff_date=None)["a"] == pytest.approx(_props(hist)["a"], rel=1e-12)
    assert _props(hist, cutoff_date=TARGET)["a"] == pytest.approx(_props(hist)["a"], rel=1e-12)


def test_target_date_label_is_independent_of_cutoff():
    """★출력 라벨은 대상일이어야 한다 — cutoff로 밀리면 조인이 조용히 어긋난다."""
    hist = _two_item_history()
    out = compute_proportions(hist, TARGET, cutoff_date=TARGET - pd.Timedelta(days=LEAD_DAYS))
    assert out["target_date"].unique().tolist() == [TARGET]


def test_distribute_total_lead_days_blocks_leak():
    """distribute_total(lead_days=)가 배분 수량까지 leakage를 막는다."""
    clean = _two_item_history()
    leaky = pd.concat([clean, _spike_in_lead_window()], ignore_index=True)
    totals = pd.Series({TARGET: 100.0})

    def qty_of(history):
        res = distribute_total(history, totals, lead_days=LEAD_DAYS)
        return res.quantities.set_index("item_id")["qty"].sort_index()

    assert qty_of(clean)["a"] == pytest.approx(qty_of(leaky)["a"], rel=1e-12)
    # 총량 보존도 함께 확인 — 배분은 재분배이지 생성이 아니다
    assert qty_of(leaky).sum() == pytest.approx(100.0, rel=1e-12)


def test_distribute_total_lead_days_zero_is_legacy():
    """lead_days=0이 기존 동작(무인자)과 정확히 같다."""
    hist = _two_item_history()
    totals = pd.Series({TARGET: 100.0})
    a = distribute_total(hist, totals).quantities.set_index("item_id")["qty"].sort_index()
    b = distribute_total(hist, totals, lead_days=0).quantities.set_index("item_id")["qty"].sort_index()
    assert a["a"] == pytest.approx(b["a"], rel=1e-12)


def test_distribute_total_rejects_negative_lead_days():
    """음수 리드타임은 fails-loud — 조용히 미래를 보게 되는 것을 막는다."""
    with pytest.raises(ValueError, match="lead_days"):
        distribute_total(_two_item_history(), pd.Series({TARGET: 100.0}), lead_days=-1)

# tests/models/test_item_proportion_leakage.py
import pandas as pd
from bakery.models.item_proportion import compute_proportions


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

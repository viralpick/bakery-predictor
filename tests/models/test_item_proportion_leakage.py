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
    # 미래(>= cutoff)에 극단 판매를 넣어도 비율이 바뀌면 안 된다 (누수 검출).
    future = [[pd.Timestamp("2024-02-05"), "a", "bread", 9999, False, pd.NaT]]
    hist2 = pd.concat([hist1, _hist(future)], ignore_index=True)

    p1 = compute_proportions(hist1, cutoff).set_index("item_id")["proportion"].sort_index()
    p2 = compute_proportions(hist2, cutoff).set_index("item_id")["proportion"].sort_index()
    assert p1.round(9).equals(p2.round(9))

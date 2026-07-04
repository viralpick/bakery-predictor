"""Tests for closing-discount basket composition (다중시각 재검증 ③).

A closing-discount purchase that sits in a basket alongside full-price items is
part of a real shopping trip (leans high α); a discount-only single-item basket
is a bargain-hunter trip (leans low α). This behavioral signal is independent of
the depth-cut supply-driven confound that muted angle ①. These tests pin the
basket classifier and the composition summary on hand-built baskets.
"""

import pandas as pd

from bakery.analysis.basket_composition import (
    CLOSING_LABEL,
    FULLPRICE_LABEL,
    basket_composition_summary,
    classify_baskets,
)

CAT = "bread"


def _lines(records):
    """records: list of (basket_id, label, category_id, qty, paid)."""
    return pd.DataFrame(records, columns=["basket_id", "label", "category_id", "qty", "paid"])


def test_classify_baskets_flags_closing_and_fullprice():
    lines = _lines([
        # basket B1: closing bread + full-price cake  → mixed
        ("B1", CLOSING_LABEL, "bread", 1, 3000.0),
        ("B1", FULLPRICE_LABEL, "cake", 1, 5000.0),
        # basket B2: closing bread only  → closing-only
        ("B2", CLOSING_LABEL, "bread", 2, 6000.0),
        # basket B3: full-price only  → not a closing basket
        ("B3", FULLPRICE_LABEL, "bread", 1, 4000.0),
    ])
    baskets = classify_baskets(lines).set_index("basket_id")
    assert list(baskets.loc["B1", ["has_closing", "has_fullprice"]]) == [True, True]
    assert list(baskets.loc["B2", ["has_closing", "has_fullprice"]]) == [True, False]
    assert list(baskets.loc["B3", ["has_closing", "has_fullprice"]]) == [False, True]


def test_summary_mixed_rate_and_value_share():
    lines = _lines([
        ("B1", CLOSING_LABEL, "bread", 1, 3000.0),
        ("B1", FULLPRICE_LABEL, "cake", 1, 5000.0),   # mixed; full-price value 5000/8000
        ("B2", CLOSING_LABEL, "bread", 2, 6000.0),    # closing-only; full-price value 0
        ("B3", FULLPRICE_LABEL, "bread", 1, 4000.0),  # no closing → excluded from closing metrics
    ])
    s = basket_composition_summary(lines)
    assert s["n_closing_baskets"] == 2
    assert s["mixed_rate"] == 0.5                     # 1 of 2 closing baskets is mixed
    # full-price value share across closing baskets: 5000 / (3000+5000+6000)
    assert s["fullprice_value_share"] == 5000.0 / 14000.0


def test_summary_basket_sizes_closing_vs_noclosing():
    lines = _lines([
        ("B1", CLOSING_LABEL, "bread", 1, 3000.0),
        ("B1", FULLPRICE_LABEL, "cake", 1, 5000.0),   # closing basket, size 2
        ("B2", CLOSING_LABEL, "bread", 1, 3000.0),    # closing basket, size 1
        ("B3", FULLPRICE_LABEL, "bread", 1, 4000.0),  # non-closing basket, size 1
        ("B4", FULLPRICE_LABEL, "bread", 1, 4000.0),
        ("B4", FULLPRICE_LABEL, "cake", 1, 5000.0),   # non-closing basket, size 2
    ])
    s = basket_composition_summary(lines)
    assert s["mean_size_closing"] == 1.5              # (2+1)/2
    assert s["mean_size_noclosing"] == 1.5            # (1+2)/2


def test_category_filter_restricts_closing_definition():
    """With category='bread', only bread-closing lines define a closing basket."""
    lines = _lines([
        ("B1", CLOSING_LABEL, "cake", 1, 3000.0),     # cake closing — not bread
        ("B1", FULLPRICE_LABEL, "bread", 1, 4000.0),
        ("B2", CLOSING_LABEL, "bread", 1, 3000.0),    # bread closing
        ("B2", FULLPRICE_LABEL, "cake", 1, 5000.0),
    ])
    s = basket_composition_summary(lines, closing_category=CAT)
    assert s["n_closing_baskets"] == 1                # only B2 has a bread-closing line

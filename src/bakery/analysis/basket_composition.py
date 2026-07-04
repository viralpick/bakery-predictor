"""Closing-discount basket composition (다중시각 재검증, angle ③).

Angle ① (depth-cut natural experiment) is muted by the closing channel being
supply-driven: closing volume barely responds to discount depth, so a price
instrument can't identify α. This angle sidesteps that confound by reading
*behavior* directly at the receipt level.

A basket is one receipt = (판매일자, POS번호, 영수증번호) within the store. If a
customer who buys a closing-discount item ALSO buys full-price items, they are on
a real shopping trip and the discounted item is plausibly wanted demand (leans
high α). If closing purchases are lone discount-only baskets, they look like
bargain-hunter trips the discount created (leans low α). We report the mixed-rate
(share of closing baskets that also contain a full-price line), the full-price
value share within closing baskets, and closing vs non-closing basket sizes.

Interpretation caveat (important): the closing discount is evening-time-locked
(≈90% of closing lines fall in 20-21h) and full-price line availability collapses
from ~75% in the daytime to ~5% at 20-21h. So a low mixed-rate is largely
*explained by the hour* (little full-price stock is left), not by bargain-hunting
psychology. This confound means the basket metrics reproduce Phase A's evening
bait-cannibalization finding rather than cleanly identifying α — read them as
descriptive, not as an independent α discriminator. See
docs/basket_composition_result.md.
"""

CLOSING_LABEL = "closing"
FULLPRICE_LABEL = "none"


def classify_baskets(lines):
    """Per-basket flags/aggregates from receipt line-items.

    Parameters
    ----------
    lines : pd.DataFrame
        Columns: basket_id, label, category_id, qty, paid. One row per receipt line.

    Returns
    -------
    pd.DataFrame
        One row per basket_id with has_closing, has_fullprice, n_lines,
        total_paid, fullprice_paid, closing_paid.
    """
    df = lines.copy()
    df["_is_closing"] = df["label"] == CLOSING_LABEL
    df["_is_fullprice"] = df["label"] == FULLPRICE_LABEL
    grouped = df.groupby("basket_id", observed=True)
    out = grouped.agg(
        has_closing=("_is_closing", "any"),
        has_fullprice=("_is_fullprice", "any"),
        n_lines=("label", "size"),
        total_paid=("paid", "sum"),
    )
    out["fullprice_paid"] = grouped.apply(
        lambda g: g.loc[g["_is_fullprice"], "paid"].sum(), include_groups=False
    )
    out["closing_paid"] = grouped.apply(
        lambda g: g.loc[g["_is_closing"], "paid"].sum(), include_groups=False
    )
    return out.reset_index()


def _closing_basket_ids(lines, closing_category):
    """basket_ids whose closing line is in `closing_category` (or any if None)."""
    is_closing = lines["label"] == CLOSING_LABEL
    if closing_category is not None:
        is_closing = is_closing & (lines["category_id"] == closing_category)
    return set(lines.loc[is_closing, "basket_id"])


def basket_composition_summary(lines, closing_category=None):
    """Composition metrics for closing baskets vs the rest.

    `closing_category` restricts what counts as a "closing basket" to receipts
    that discounted an item of that category (e.g. only bread-closing lines).
    """
    baskets = classify_baskets(lines).set_index("basket_id")
    closing_ids = _closing_basket_ids(lines, closing_category)
    closing = baskets.loc[baskets.index.isin(closing_ids)]
    noclosing = baskets.loc[~baskets.index.isin(closing_ids)]

    n_closing = len(closing)
    total_paid = closing["total_paid"].sum()
    return {
        "n_closing_baskets": int(n_closing),
        "n_noclosing_baskets": int(len(noclosing)),
        "mixed_rate": float(closing["has_fullprice"].mean()) if n_closing else float("nan"),
        "fullprice_value_share": (
            float(closing["fullprice_paid"].sum() / total_paid)
            if total_paid > 0 else float("nan")
        ),
        "mean_size_closing": float(closing["n_lines"].mean()) if n_closing else float("nan"),
        "mean_size_noclosing": (
            float(noclosing["n_lines"].mean()) if len(noclosing) else float("nan")
        ),
    }

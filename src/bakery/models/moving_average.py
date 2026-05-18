"""Moving average / EWMA baseline. Predicts a flat value per (store,item) from
the tail of the training window. Two flavors:
- simple MA over last `window` days
- EWMA with `alpha` (overrides window if set)

This is intentionally trend-blind and weekday-blind — it's the floor against
which Seasonal Naive demonstrates that weekday structure matters.
"""

from __future__ import annotations

import pandas as pd

from .base import Forecaster


class MovingAverage(Forecaster):
    name = "moving_average"

    def __init__(self, window: int = 28, alpha: float | None = None, y_col: str = "sold_units"):
        if window < 1:
            raise ValueError("window must be >= 1")
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.window = window
        self.alpha = alpha
        self.y_col = y_col
        self._lookup: pd.DataFrame | None = None
        self._global_fallback = 0.0
        if alpha is not None:
            self.name = f"ewma_a{alpha}"

    def fit(self, train: pd.DataFrame) -> MovingAverage:
        self._check_keys(train)
        df = train.sort_values("date")
        cutoff = df["date"].max() - pd.Timedelta(days=self.window)
        recent = df[df["date"] > cutoff]
        if self.alpha is None:
            agg = recent.groupby(["store_id", "item_id"], observed=True)[self.y_col].mean()
        else:
            agg = recent.groupby(["store_id", "item_id"], observed=True)[self.y_col].apply(
                lambda s: s.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1] if len(s) else 0.0
            )
        self._lookup = agg.rename("yhat").reset_index()
        self._global_fallback = float(recent[self.y_col].mean()) if not recent.empty else 0.0
        return self

    def predict(self, target: pd.DataFrame) -> pd.Series:
        if self._lookup is None:
            raise RuntimeError("call fit() before predict()")
        self._check_keys(target)
        merged = target.merge(self._lookup, on=["store_id", "item_id"], how="left")
        merged["yhat"] = merged["yhat"].fillna(self._global_fallback)
        return pd.Series(merged["yhat"].to_numpy(), index=target.index, name="yhat")

"""Seasonal-naive baseline: predict t = mean of (store,item) sales on the same
day-of-week over the last N weeks of training data.

This is the model the LightGBM has to beat to justify its existence.
"""

from __future__ import annotations

import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def __init__(self, n_weeks: int = 4, y_col: str = "sold_units"):
        if n_weeks < 1:
            raise ValueError("n_weeks must be >= 1")
        self.n_weeks = n_weeks
        self.y_col = y_col
        self._lookup: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        self._check_keys(train)
        df = train.copy()
        df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
        cutoff = df["date"].max() - pd.Timedelta(weeks=self.n_weeks)
        recent = df[df["date"] > cutoff]
        self._lookup = (
            recent.groupby(["store_id", "item_id", "dow"], observed=True)[self.y_col]
            .mean()
            .rename("yhat")
            .reset_index()
        )
        self._global_fallback = float(recent[self.y_col].mean()) if not recent.empty else 0.0
        return self

    def predict(self, target: pd.DataFrame) -> pd.Series:
        if self._lookup is None:
            raise RuntimeError("call fit() before predict()")
        self._check_keys(target)
        out = target.copy()
        out["dow"] = pd.to_datetime(out["date"]).dt.dayofweek
        merged = out.merge(self._lookup, on=["store_id", "item_id", "dow"], how="left")
        merged["yhat"] = merged["yhat"].fillna(self._global_fallback)
        return pd.Series(merged["yhat"].to_numpy(), index=target.index, name="yhat")

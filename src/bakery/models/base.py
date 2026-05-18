"""Common Forecaster interface.

fit(train_df) — learn from history.
predict(target_df) — produce a sold_units forecast for each (store,item,date).

target_df must contain at least store_id/item_id/date; for baselines that
is enough; the LightGBM model also needs the feature columns it was trained on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_KEYS = ("store_id", "item_id", "date")


class Forecaster(ABC):
    name: str = "abstract"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Forecaster:
        ...

    @abstractmethod
    def predict(self, target: pd.DataFrame) -> pd.Series:
        """Return a Series of predicted sold_units aligned to target.index."""

    def _check_keys(self, df: pd.DataFrame) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{self.name} requires columns {missing}")

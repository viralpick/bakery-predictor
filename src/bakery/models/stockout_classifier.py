"""Per-day stockout-risk classifier (LightGBM binary).

Operations-side watchlist model — surfaces "what's most likely to stock out
next week" for production planning. The main demand model (GlobalLGBM)
handles censoring directly via potential_demand target and stockout-context
features, so this classifier is no longer the only stockout-aware signal.

Output: P(is_stockout=1 | features). Inputs: date + calendar + weather +
sales lag/rolling + stockout history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.calendar_features import CALENDAR_FEATURE_COLUMNS
from ..features.date_features import DATE_FEATURE_COLUMNS, add_date_features
from ..features.lag_features import LAG_FEATURE_COLUMNS, add_lag_features
from ..features.rolling_features import ROLLING_FEATURE_COLUMNS, add_rolling_features
from ..features.stockout_history import STOCKOUT_HISTORY_COLUMNS, add_stockout_history
from ..features.weather_features import WEATHER_FEATURE_COLUMNS

CATEGORICAL_COLUMNS: list[str] = ["store_id", "item_id", "category_id", "dow", "month"]


@dataclass
class ClassifierParams:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    objective: str = "binary"
    metric: str = "auc"
    verbose: int = -1
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        extra = d.pop("extra")
        d.update(extra)
        return d


class StockoutClassifier:
    name = "stockout_classifier"

    def __init__(self, params: ClassifierParams | None = None, feature_set: str = "v1"):
        if feature_set not in {"v0", "v1"}:
            raise ValueError(f"feature_set must be v0|v1, got {feature_set!r}")
        self.params = params or ClassifierParams()
        self.feature_set = feature_set
        self.numeric_columns = self._numeric_columns(feature_set)
        self.feature_columns = CATEGORICAL_COLUMNS + self.numeric_columns
        self.model: lgb.Booster | None = None
        self._train_history: pd.DataFrame | None = None

    @staticmethod
    def _numeric_columns(feature_set: str) -> list[str]:
        base = [
            *(c for c in DATE_FEATURE_COLUMNS if c not in {"dow", "month"}),
            *LAG_FEATURE_COLUMNS,
            *ROLLING_FEATURE_COLUMNS,
            *STOCKOUT_HISTORY_COLUMNS,
        ]
        if feature_set == "v1":
            base += CALENDAR_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS
        return base

    def fit(self, train: pd.DataFrame) -> StockoutClassifier:
        if "is_stockout" not in train.columns:
            raise ValueError("train frame must include 'is_stockout' column")
        self._train_history = train.copy()
        feats = self._build_features(train)
        feats = feats.dropna(subset=self.numeric_columns, how="any")
        if feats.empty:
            raise RuntimeError("after dropping NaN history rows, no training data left")
        x = self._encode_categoricals(feats[self.feature_columns])
        y = feats["is_stockout"].astype(int).to_numpy()
        dataset = lgb.Dataset(x, label=y, categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
        self.model = lgb.train(self.params.to_dict(), dataset, num_boost_round=self.params.n_estimators)
        return self

    def predict_proba(self, target: pd.DataFrame) -> pd.Series:
        if self.model is None or self._train_history is None:
            raise RuntimeError("call fit() before predict_proba()")
        target_with_flag = target.copy()
        if "is_stockout" not in target_with_flag.columns:
            target_with_flag["is_stockout"] = False
        joined = pd.concat([self._train_history, target_with_flag], ignore_index=True)
        joined = joined.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
        feats = self._build_features(joined)
        # restrict to target rows by inner-joining on keys
        target_keys = target[["store_id", "item_id", "date"]]
        feats = feats.merge(target_keys, on=["store_id", "item_id", "date"], how="inner")
        x = self._encode_categoricals(feats[self.feature_columns])
        proba = self.model.predict(x)
        out = feats[["store_id", "item_id", "date"]].assign(stockout_prob=proba)
        aligned = target.merge(out, on=["store_id", "item_id", "date"], how="left")
        # rows with insufficient history (e.g., brand-new item) fall back to 0.
        return pd.Series(
            np.clip(aligned["stockout_prob"].fillna(0.0).to_numpy(), 0.0, 1.0),
            index=target.index,
            name="stockout_prob",
        )

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_date_features(df)
        out = add_lag_features(out)
        out = add_rolling_features(out)
        out = add_stockout_history(out)
        return out

    def _encode_categoricals(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for col in CATEGORICAL_COLUMNS:
            x[col] = x[col].astype("category")
        return x

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit() before feature_importance()")
        imp = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

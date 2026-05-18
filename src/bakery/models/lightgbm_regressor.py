"""Global LightGBM regression model.

One model fits all (store, item) pairs together. Categorical IDs let the
tree share splits across items. Lag/rolling features carry the time signal.

`feature_set="v0"`: date + lag + rolling features only (matches synthetic
baseline scope). `feature_set="v1"`: same plus calendar + weather features
merged onto the daily frame upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.calendar_features import CALENDAR_FEATURE_COLUMNS
from ..features.cannibalization import (
    CANNIBALIZATION_FEATURE_COLUMNS,
    add_cannibalization_features,
)
from ..features.date_features import DATE_FEATURE_COLUMNS, add_date_features
from ..features.lag_features import LAG_FEATURE_COLUMNS, add_lag_features
from ..features.rolling_features import ROLLING_FEATURE_COLUMNS, add_rolling_features
from ..features.weather_features import WEATHER_FEATURE_COLUMNS
from .base import Forecaster

CATEGORICAL_COLUMNS: list[str] = ["store_id", "item_id", "category_id", "dow", "month"]

# Base numeric columns shared across feature sets.
_BASE_NUMERIC_COLUMNS: list[str] = [
    *(c for c in DATE_FEATURE_COLUMNS if c not in {"dow", "month"}),
    *LAG_FEATURE_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
]

VALID_FEATURE_SETS = ("v0", "v1", "v2")


def build_numeric_columns(feature_set: str) -> list[str]:
    if feature_set == "v0":
        return list(_BASE_NUMERIC_COLUMNS)
    if feature_set == "v1":
        return [*_BASE_NUMERIC_COLUMNS, *CALENDAR_FEATURE_COLUMNS, *WEATHER_FEATURE_COLUMNS]
    if feature_set == "v2":
        return [
            *_BASE_NUMERIC_COLUMNS,
            *CALENDAR_FEATURE_COLUMNS,
            *WEATHER_FEATURE_COLUMNS,
            *CANNIBALIZATION_FEATURE_COLUMNS,
        ]
    raise ValueError(f"unknown feature_set: {feature_set!r}. Use one of {VALID_FEATURE_SETS}")


def _default_target(feature_set: str) -> str:
    """v0/v1 trained on observed sold_units (history compatibility).
    v2 trained on censoring-corrected potential_demand."""
    return "potential_demand" if feature_set == "v2" else "sold_units"


@dataclass
class LGBMParams:
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    objective: str = "regression"
    metric: str = "mae"
    # For objective="quantile": the target quantile (0 < α < 1). 0.5=median,
    # 0.85=safety-margin "production target". Ignored otherwise.
    alpha: float = 0.5
    verbose: int = -1
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "objective": self.objective,
            "metric": self.metric,
            "verbose": self.verbose,
            "seed": self.seed,
        }
        if self.objective == "quantile":
            d["alpha"] = self.alpha
        d.update(self.extra)
        return d


class GlobalLGBM(Forecaster):
    def __init__(
        self,
        params: LGBMParams | None = None,
        y_col: str | None = None,
        feature_set: str = "v0",
    ):
        if feature_set not in VALID_FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {VALID_FEATURE_SETS}; got {feature_set!r}")
        self.params = params or LGBMParams()
        self.y_col = y_col if y_col is not None else _default_target(feature_set)
        self.feature_set = feature_set
        self.name = self._build_name(feature_set, self.params)
        self.numeric_columns = build_numeric_columns(feature_set)
        self.feature_columns = CATEGORICAL_COLUMNS + self.numeric_columns
        self.model: lgb.Booster | None = None
        self._train_history: pd.DataFrame | None = None
        self._extra_history_cols: list[str] = []
        if feature_set in {"v1", "v2"}:
            self._extra_history_cols.extend(CALENDAR_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS)
        if feature_set == "v2":
            # Cannibalization aggregates run over (sold_units, is_stockout) at the
            # store/category level. When y_col is potential_demand, sold_units is
            # no longer the trained target, so we explicitly carry it in history.
            self._extra_history_cols.append("is_stockout")
            if self.y_col != "sold_units":
                self._extra_history_cols.append("sold_units")

    @staticmethod
    def _build_name(feature_set: str, params: LGBMParams) -> str:
        base = "lightgbm" if feature_set == "v0" else f"lightgbm_{feature_set}"
        if params.objective == "quantile":
            return f"{base}_q{int(params.alpha * 100):02d}"
        return base

    def fit(self, train: pd.DataFrame) -> GlobalLGBM:
        self._check_keys(train)
        self._check_feature_set_columns(train, "train")
        history_cols = [
            "store_id",
            "item_id",
            "category_id",
            "date",
            self.y_col,
            *self._extra_history_cols,
        ]
        self._train_history = train[history_cols].copy()
        feats = self._build_features(train, fitting=True)
        feats = feats.dropna(subset=self.numeric_columns, how="any")
        if feats.empty:
            raise RuntimeError("after dropping NaN lag rows, no training data left")
        x = self._encode_categoricals(feats[self.feature_columns])
        y = feats[self.y_col].astype(float).to_numpy()
        dataset = lgb.Dataset(x, label=y, categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False)
        self.model = lgb.train(self.params.to_dict(), dataset, num_boost_round=self.params.n_estimators)
        return self

    def predict(self, target: pd.DataFrame) -> pd.Series:
        if self.model is None or self._train_history is None:
            raise RuntimeError("call fit() before predict()")
        self._check_keys(target)
        self._check_feature_set_columns(target, "target")
        joined = self._join_history(target)
        feats = self._build_features(joined, fitting=False)
        target_mask = feats["date"].isin(target["date"]) & feats["store_id"].isin(target["store_id"].unique())
        feats = feats.loc[target_mask]
        feats = feats.merge(
            target[["store_id", "item_id", "date"]], on=["store_id", "item_id", "date"], how="inner"
        )
        x = self._encode_categoricals(feats[self.feature_columns])
        yhat = self.model.predict(x)
        yhat = np.clip(yhat, a_min=0.0, a_max=None)
        out = feats[["store_id", "item_id", "date"]].assign(yhat=yhat)
        aligned = target.merge(out, on=["store_id", "item_id", "date"], how="left")
        return pd.Series(aligned["yhat"].fillna(0.0).to_numpy(), index=target.index, name="yhat")

    def _build_features(self, df: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        out = add_date_features(df)
        out = add_lag_features(out, y_col=self.y_col)
        out = add_rolling_features(out, y_col=self.y_col)
        if self.feature_set == "v2":
            out = add_cannibalization_features(out)
        return out

    def _join_history(self, target: pd.DataFrame) -> pd.DataFrame:
        """Stack train history under the target frame so lag/rolling features can
        be computed for target rows from prior history. v1+ carry calendar/weather
        through both sides; v2 also carries sold_units/is_stockout so the
        cannibalization aggregates remain computable across the seam."""
        target_slim = target[["store_id", "item_id", "date"]].copy()
        target_slim[self.y_col] = np.nan
        for col in self._extra_history_cols:
            if col in target.columns:
                target_slim[col] = target[col].to_numpy()
            elif col == "is_stockout":
                target_slim[col] = False
            else:
                target_slim[col] = np.nan
        if "category_id" in target.columns:
            target_slim["category_id"] = target["category_id"].values
        else:
            cats = self._train_history.drop_duplicates(["store_id", "item_id"])
            if "category_id" in cats.columns:
                target_slim = target_slim.merge(
                    cats[["store_id", "item_id", "category_id"]],
                    on=["store_id", "item_id"],
                    how="left",
                )
        history = self._train_history.copy()
        if "category_id" not in history.columns:
            history["category_id"] = "unknown"
        if "category_id" not in target_slim.columns:
            target_slim["category_id"] = "unknown"
        joined = pd.concat([history, target_slim], ignore_index=True)
        joined = joined.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
        return joined

    def _encode_categoricals(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for col in CATEGORICAL_COLUMNS:
            x[col] = x[col].astype("category")
        return x

    def _check_feature_set_columns(self, df: pd.DataFrame, label: str) -> None:
        if self.feature_set == "v0":
            return
        required: list[str] = []
        if self.feature_set in {"v1", "v2"}:
            required.extend(CALENDAR_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS)
        # is_stockout / sold_units are required on train (for cannibalization +
        # target). On the target frame, _join_history fills them in if absent.
        if self.feature_set == "v2" and label == "train":
            required.append("is_stockout")
            if self.y_col != "sold_units":
                required.append("sold_units")
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.feature_set} {label} frame missing required columns: {sorted(missing)}. "
                "Did you forget to enrich daily with calendar/weather (and ensure DAILY_COLUMNS for v2)?"
            )

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit() before feature_importance()")
        imp = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

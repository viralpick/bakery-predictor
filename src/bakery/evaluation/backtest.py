"""Run multiple forecasters across time-series splits and collect WAPE/MAE/RMSE.

Each forecaster gets a fresh fit per fold (no warm-start carry-over).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.base import Forecaster
from .metrics import grouped_wape, summarize_with_stockout
from .split import SplitWindow, apply_split


@dataclass
class FoldResult:
    fold: int
    model: str
    wape_all: float
    wape_no_stockout: float
    mae: float
    rmse: float
    pct_underpredict: float
    pct_overpredict: float
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def run_backtest(
    df: pd.DataFrame,
    forecasters: Iterable[Forecaster],
    windows: list[SplitWindow],
    *,
    y_col: str = "sold_units",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fold_results_df, predictions_df).

    fold_results columns include both `wape_all` (operations view, against
    observed sales) and `wape_no_stockout` (demand-model view, ignoring
    capacity-truncated rows). predictions_df preserves `is_stockout` so
    downstream analyses can compute their own slices.
    """
    fold_rows: list[FoldResult] = []
    pred_chunks: list[pd.DataFrame] = []
    for w in windows:
        train, val = apply_split(df, w)
        if train.empty or val.empty:
            continue
        for forecaster in forecasters:
            fresh = _clone(forecaster)
            fresh.fit(train)
            yhat = fresh.predict(val)
            y = val[y_col].to_numpy()
            is_stockout = val["is_stockout"].to_numpy() if "is_stockout" in val.columns else np.zeros(len(val), dtype=bool)
            summary = summarize_with_stockout(y, yhat.to_numpy(), is_stockout)
            fold_rows.append(
                FoldResult(
                    fold=w.fold_index,
                    model=fresh.name,
                    wape_all=summary["wape_all"],
                    wape_no_stockout=summary["wape_no_stockout"],
                    mae=summary["mae"],
                    rmse=summary["rmse"],
                    pct_underpredict=summary["pct_underpredict"],
                    pct_overpredict=summary["pct_overpredict"],
                    train_end=w.train_end,
                    val_start=w.val_start,
                    val_end=w.val_end,
                )
            )
            cols = ["store_id", "item_id", "category_id", "date", y_col]
            if "is_stockout" in val.columns:
                cols.append("is_stockout")
            pred_chunks.append(
                val[cols].assign(model=fresh.name, fold=w.fold_index, yhat=yhat.to_numpy())
            )
    fold_df = pd.DataFrame([r.__dict__ for r in fold_rows])
    pred_df = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()
    return fold_df, pred_df


def aggregate_by_model(fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_df.groupby("model", as_index=False)
        .agg(
            wape_all=("wape_all", "mean"),
            wape_no_stockout=("wape_no_stockout", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            pct_underpredict=("pct_underpredict", "mean"),
            pct_overpredict=("pct_overpredict", "mean"),
            folds=("fold", "count"),
        )
        .sort_values("wape_no_stockout")
        .reset_index(drop=True)
    )


def per_category_wape(pred_df: pd.DataFrame, *, y_col: str = "sold_units") -> pd.DataFrame:
    return grouped_wape(pred_df, by=["model", "category_id"], y_col=y_col, yhat_col="yhat")


def _clone(forecaster: Forecaster) -> Forecaster:
    """Re-instantiate the forecaster with its same configuration."""
    cls = forecaster.__class__
    if hasattr(forecaster, "params"):  # LightGBM
        kwargs = {"params": forecaster.params}
        if hasattr(forecaster, "feature_set"):
            kwargs["feature_set"] = forecaster.feature_set
        return cls(**kwargs)
    if hasattr(forecaster, "n_weeks"):  # SeasonalNaive
        return cls(n_weeks=forecaster.n_weeks)
    if hasattr(forecaster, "alpha") and forecaster.alpha is not None:
        return cls(alpha=forecaster.alpha)
    if hasattr(forecaster, "window"):
        return cls(window=forecaster.window)
    return cls()

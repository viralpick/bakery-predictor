"""Forecast metrics. WAPE is the primary metric; MAPE is intentionally avoided.

WAPE = sum(|y - ŷ|) / sum(y). Robust to zero/sparse targets, comparable across items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(_align(y_true, y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = _align(y_true, y_pred)
    return float(np.sqrt(np.mean(diff**2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def grouped_wape(df: pd.DataFrame, by: list[str], *, y_col: str, yhat_col: str) -> pd.DataFrame:
    """Per-group WAPE (e.g., by=['store_id','item_id'] or ['category_id'])."""
    def _wape_row(g: pd.DataFrame) -> float:
        return wape(g[y_col].to_numpy(), g[yhat_col].to_numpy())

    out = df.groupby(by, observed=True).apply(_wape_row, include_groups=False)
    return out.rename("wape").reset_index()


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"wape": wape(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def summarize_with_stockout(
    y_true: np.ndarray, y_pred: np.ndarray, is_stockout: np.ndarray
) -> dict[str, float]:
    """Two WAPEs side by side:

    - `wape_all`: against observed sales (operations view; what the store sold).
    - `wape_no_stockout`: against non-stockout rows only (demand-model view; the
      'honest' accuracy unaffected by capacity-truncated rows).
    Includes a `pct_underprediction` rate on non-stockout rows since over- vs
    under-prediction have asymmetric operational cost (spec.md §5).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.asarray(is_stockout, dtype=bool)
    return {
        "wape_all": wape(y_true, y_pred),
        "wape_no_stockout": wape(y_true[mask], y_pred[mask]) if mask.any() else float("nan"),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pct_underpredict": _pct_underpredict(y_true[mask], y_pred[mask]) if mask.any() else float("nan"),
        "pct_overpredict": _pct_overpredict(y_true[mask], y_pred[mask]) if mask.any() else float("nan"),
    }


def _pct_underpredict(y_true: np.ndarray, y_pred: np.ndarray, *, tol: float = 1.0) -> float:
    """Share of rows where prediction is materially below truth (gap > tol units)."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) > tol))


def _pct_overpredict(y_true: np.ndarray, y_pred: np.ndarray, *, tol: float = 1.0) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_pred - y_true) > tol))


def _align(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    return a - b

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


def wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Percentage Error — signed bias. Σ(pred−true)/Σtrue.

    양수=체계적 과대예측, 음수=과소예측(품절 위험 방향). WAPE가 못 잡는
    '어느 쪽으로 틀렸나'를 본다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    denom = y_true.sum()
    if denom == 0:
        return float("nan")
    return float((y_pred - y_true).sum() / denom)


def quantile_exceedance_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """실측이 예측(q_α 발주)을 초과한 비율 = P(y_true > y_pred).

    보정된 q_α 발주면 ≈ 1−α (q0.85 → ≈0.15). 이보다 크게 높으면 과소발주(매진↑),
    낮으면 과대발주(폐기↑). WPE(부호 편향)와 짝으로 분포 편중을 진단한다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true > y_pred))


def grouped_wape(df: pd.DataFrame, by: list[str], *, y_col: str, yhat_col: str) -> pd.DataFrame:
    """Per-group WAPE (e.g., by=['store_id','item_id'] or ['category_id'])."""
    def _wape_row(g: pd.DataFrame) -> float:
        return wape(g[y_col].to_numpy(), g[yhat_col].to_numpy())

    out = df.groupby(by, observed=True).apply(_wape_row, include_groups=False)
    return out.rename("wape").reset_index()


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"wape": wape(y_true, y_pred), "wpe": wpe(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of points falling inside the closed interval [lower, upper]."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError(
            f"shape mismatch: actual {actual.shape}, lower {lower.shape}, upper {upper.shape}"
        )
    if len(actual) == 0:
        return float("nan")
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def coverage_by_group(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, group: np.ndarray
) -> dict:
    """Per-group coverage (e.g., by day-of-week for Mondrian conformal)."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    group = np.asarray(group)
    if not (actual.shape == lower.shape == upper.shape == group.shape):
        raise ValueError("shape mismatch among actual/lower/upper/group")
    out: dict = {}
    for g in np.unique(group):
        mask = group == g
        out[g] = coverage(actual[mask], lower[mask], upper[mask])
    return out


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width (proxy for waste/over-stocking cost)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError(f"shape mismatch: lower {lower.shape} vs upper {upper.shape}")
    if len(lower) == 0:
        return float("nan")
    return float(np.mean(upper - lower))


def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean quantile (pinball) loss at level q.

    Under-prediction (pred < actual) is weighted by q; over-prediction by (1-q).
    """
    diff = _align(actual, pred)  # actual - pred
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def mase(
    actual: np.ndarray, pred: np.ndarray, train_actual: np.ndarray, season: int = 7
) -> float:
    """Mean Absolute Scaled Error: model MAE / seasonal-naive MAE on the train series."""
    train_actual = np.asarray(train_actual, dtype=float)
    if len(train_actual) <= season:
        return float("nan")
    naive_mae = float(np.mean(np.abs(train_actual[season:] - train_actual[:-season])))
    if naive_mae == 0:
        return float("nan")
    return mae(actual, pred) / naive_mae


def summarize_with_stockout(
    y_true: np.ndarray, y_pred: np.ndarray, is_stockout: np.ndarray
) -> dict[str, float]:
    """Two WAPEs side by side:

    - `wape_all`: against observed sales (operations view; what the store sold).
    - `wape_no_stockout`: against non-stockout rows only (demand-model view; the
      'honest' accuracy unaffected by capacity-truncated rows).
    Includes a `pct_underprediction` rate on non-stockout rows since over- vs
    under-prediction have asymmetric operational cost (spec.md §5).
    Also includes `wpe` (signed bias) for diagnostics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.asarray(is_stockout, dtype=bool)
    return {
        "wape_all": wape(y_true, y_pred),
        "wape_no_stockout": wape(y_true[mask], y_pred[mask]) if mask.any() else float("nan"),
        "wpe": wpe(y_true, y_pred),
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

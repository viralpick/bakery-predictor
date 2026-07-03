import numpy as np
import pandas as pd

MAX_CONDITION_NUMBER = 1e10


def _design_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Y and X=[const, T, other_cat_sold, cat_baseline, dow dummies, month dummies, trend]."""
    y = panel["cat_sold"].to_numpy(dtype=float)
    n = len(y)
    const = np.ones((n, 1))
    cols = [const,
            panel["stockout_hours"].to_numpy(float).reshape(-1, 1),      # T = index 1
            panel["other_cat_sold"].to_numpy(float).reshape(-1, 1),
            panel["cat_baseline"].to_numpy(float).reshape(-1, 1),
            pd.get_dummies(panel["dow"], drop_first=True).to_numpy(float),
            pd.get_dummies(panel["month"], drop_first=True).to_numpy(float),
            panel["trend"].to_numpy(float).reshape(-1, 1)]
    X = np.hstack(cols)
    keep = X.std(axis=0) > 1e-12
    keep[0] = True                       # keep constant
    keep[1] = True                       # keep treatment even if degenerate → caller guards
    return y, X[:, keep], 1              # treatment is column index 1 after keep (const stays 0)


def _ols_hc3(y: np.ndarray, X: np.ndarray, treat_idx: int) -> tuple[float, float] | None:
    """OLS β and HC3 robust SE for the treatment column. numpy only. None if ill-posed."""
    n, k = X.shape
    if n - k < 5:
        return None
    XtX = X.T @ X
    try:
        cond = np.linalg.cond(XtX)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
        return None
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverages
    denom = np.clip((1.0 - h) ** 2, 1e-8, None)
    meat = X.T @ (X * (resid ** 2 / denom)[:, None])     # HC3 sandwich meat
    cov = XtX_inv @ meat @ XtX_inv
    treat_var = cov[treat_idx, treat_idx]
    if not np.isfinite(treat_var) or treat_var < 0:
        return None
    se = float(np.sqrt(treat_var))
    if not np.isfinite(se) or se <= 0:
        return None
    return float(beta[treat_idx]), se

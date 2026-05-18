"""Binary-classifier metrics for the stockout-risk model.

ROC-AUC is the headline metric. precision@k and recall@k tell us how useful
the "watchlist of high-risk items" would be in practice. We avoid sklearn
for a single dep — the math is short.
"""

from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney style AUC. Handles ties via average ranks."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average rank for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    avg_rank = sums / counts
    ranks_corrected = avg_rank[inv]
    rank_sum_pos = ranks_corrected[y == 1].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of top-k scoring rows that are actually positive."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if k <= 0 or k > len(y):
        return float("nan")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of positives captured in the top-k scoring rows."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    total_pos = int((y == 1).sum())
    if total_pos == 0 or k <= 0:
        return float("nan")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / total_pos)


def base_rate(y_true: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    return float(y.mean()) if len(y) else float("nan")

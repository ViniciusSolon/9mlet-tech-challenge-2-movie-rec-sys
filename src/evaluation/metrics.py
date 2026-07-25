"""Ranking and rating evaluation metrics for recommendation systems.

Provides ranking metrics (Precision@K, Recall@K, NDCG@K, Hit Rate@K)
and rating metrics (RMSE, MAE) used in Bloco 5 evaluation.
"""

from __future__ import annotations

import math

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cut-off rank.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items retrieved in the top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cut-off rank.

    Returns:
        Recall@K in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cut-off rank.

    Returns:
        NDCG@K in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Binary indicator: 1 if any top-K recommendation is relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cut-off rank.

    Returns:
        1.0 if at least one hit, else 0.0.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = set(recommended[:k])
    return 1.0 if top_k & relevant else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (delegates to scikit-learn).

    Args:
        y_true: Ground-truth ratings as a 1-D array.
        y_pred: Predicted ratings as a 1-D array.

    Returns:
        RMSE as a float.
    """
    from sklearn.metrics import root_mean_squared_error  # noqa: PLC0415

    return float(root_mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (delegates to scikit-learn).

    Args:
        y_true: Ground-truth ratings as a 1-D array.
        y_pred: Predicted ratings as a 1-D array.

    Returns:
        MAE as a float.
    """
    from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

    return float(mean_absolute_error(y_true, y_pred))


def average_metrics(
    recommended_lists: list[list[int]],
    relevant_sets: list[set[int]],
    k: int,
) -> dict[str, float]:
    """Compute mean ranking metrics across multiple users.

    Args:
        recommended_lists: One ordered recommendation list per user.
        relevant_sets: One relevant-item set per user.
        k: Cut-off rank.

    Returns:
        Dict with keys ``precision``, ``recall``, ``ndcg``, ``hit_rate``.
    """
    if not recommended_lists:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}

    metrics: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "hit_rate": [],
    }
    for recs, rels in zip(recommended_lists, relevant_sets, strict=True):
        metrics["precision"].append(precision_at_k(recs, rels, k))
        metrics["recall"].append(recall_at_k(recs, rels, k))
        metrics["ndcg"].append(ndcg_at_k(recs, rels, k))
        metrics["hit_rate"].append(hit_rate_at_k(recs, rels, k))

    return {key: float(np.mean(vals)) for key, vals in metrics.items()}

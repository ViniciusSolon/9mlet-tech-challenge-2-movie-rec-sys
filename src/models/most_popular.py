"""Most-popular baseline recommender (item frequency / mean rating)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from models.base import RecommenderModel


class MostPopularRecommender(RecommenderModel):
    """Score items by historical mean rating (popularity proxy)."""

    def __init__(self, default_score: float = 3.0) -> None:
        self._default_score = default_score
        self._item_means: dict[int, float] = {}
        self._global_mean = default_score
        self._fitted = False

    @property
    def name(self) -> str:
        return "most_popular"

    def fit(
        self,
        features: Any,
        targets: Any | None = None,
    ) -> MostPopularRecommender:
        """Fit item mean ratings from ``(user, item)`` features and targets."""
        if targets is None:
            raise ValueError("targets must be provided for most_popular training")
        X = np.asarray(features)
        y = np.asarray(targets, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("features must be shape (n, 2+) with item id in column 1")

        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for item_id, rating in zip(X[:, 1].astype(int), y, strict=True):
            sums[item_id] += float(rating)
            counts[item_id] += 1

        self._item_means = {
            item_id: sums[item_id] / counts[item_id] for item_id in counts
        }
        self._global_mean = float(np.mean(y)) if len(y) else self._default_score
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[float]:
        """Predict rating as the historical mean of the requested item."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        X = np.asarray(features)
        scores: list[float] = []
        for item_id in X[:, 1].astype(int):
            scores.append(self._item_means.get(int(item_id), self._global_mean))
        return scores

"""Scikit-learn baseline stub (training in Bloco 5)."""

from __future__ import annotations

from typing import Any

from models.base import RecommenderModel


class SklearnBaselineRecommender(RecommenderModel):
    """Placeholder baseline until NMF/SVD is wired."""

    def __init__(self) -> None:
        self._fitted = False

    @property
    def name(self) -> str:
        return "sklearn_baseline"

    def fit(
        self, features: Any, targets: Any | None = None
    ) -> SklearnBaselineRecommender:
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[float]:
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        size = len(features) if hasattr(features, "__len__") else 1
        return [0.0] * size

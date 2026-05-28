"""PyTorch embedding recommender stub (Bloco 5)."""

from __future__ import annotations

from typing import Any

from models.base import RecommenderModel


class TorchEmbeddingRecommender(RecommenderModel):
    """Placeholder for user/item embedding model."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self._embedding_dim = embedding_dim
        self._fitted = False

    @property
    def name(self) -> str:
        return "torch_embedding"

    def fit(
        self, features: Any, targets: Any | None = None
    ) -> TorchEmbeddingRecommender:
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[float]:
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        size = len(features) if hasattr(features, "__len__") else 1
        return [0.0] * size

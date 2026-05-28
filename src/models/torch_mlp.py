"""PyTorch MLP recommender stub (Bloco 5)."""

from __future__ import annotations

from typing import Any

from models.base import RecommenderModel


class TorchMLPRecommender(RecommenderModel):
    """Placeholder for MLP on concatenated embeddings."""

    def __init__(self, hidden_dim: int = 128) -> None:
        self._hidden_dim = hidden_dim
        self._fitted = False

    @property
    def name(self) -> str:
        return "torch_mlp"

    def fit(
        self,
        features: Any,
        targets: Any | None = None,
    ) -> TorchMLPRecommender:
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[float]:
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        size = len(features) if hasattr(features, "__len__") else 1
        return [0.0] * size

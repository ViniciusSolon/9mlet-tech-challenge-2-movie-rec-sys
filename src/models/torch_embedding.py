"""Pure dot-product embedding recommender (matrix-factorisation style).

Models implicit or explicit feedback as the inner product of learned
user and item embedding vectors (optionally with bias terms).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import RecommenderModel


class TorchEmbeddingRecommender(nn.Module, RecommenderModel):
    """Dot-product collaborative filter with learnable user/item biases.

    The predicted rating for (u, i) is:
        r̂ = <e_u, e_i> + b_u + b_i + global_bias

    Args:
        n_users: Total number of unique users.
        n_items: Total number of unique items.
        embedding_dim: Dimension of the embedding vectors.
    """

    def __init__(
        self,
        n_users: int = 100,
        n_items: int = 100,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self._fitted = False
        self._init_weights()

    def _init_weights(self) -> None:
        """Normal initialisation for embeddings; zero for biases."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted ratings for (user, item) index pairs.

        Args:
            x: Long tensor of shape (batch, 2) — columns [user_idx, item_idx].

        Returns:
            Float tensor of shape (batch, 1) with predicted scores.
        """
        users = x[:, 0]
        items = x[:, 1]
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        dot = (user_emb * item_emb).sum(dim=-1, keepdim=True)
        bias = (
            self.user_bias(users)
            + self.item_bias(items)
            + self.global_bias
        )
        return dot + bias

    @property
    def name(self) -> str:
        """Human-readable model identifier."""
        return "torch_embedding"

    def fit(
        self, features: Any, targets: Any | None = None
    ) -> TorchEmbeddingRecommender:
        """Mark the model as fitted (training is handled externally)."""
        self._fitted = True
        return self

    def predict(self, features: torch.Tensor) -> list[float]:
        """Return predicted scores for a batch of (user, item) pairs.

        Args:
            features: Long tensor of shape (N, 2).

        Returns:
            List of N predicted rating scores.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        self.eval()
        with torch.no_grad():
            scores = self.forward(features).squeeze().tolist()
        return scores if isinstance(scores, list) else [scores]

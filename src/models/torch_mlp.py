"""PyTorch MLP recommender with user/item embeddings (Bloco 5)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import RecommenderModel


class TorchMLPRecommender(nn.Module, RecommenderModel):
    """MLP rating predictor built on concatenated user+item embeddings.

    Architecture: Embedding(user) || Embedding(item) → Linear → ReLU
    → Dropout → Linear → ReLU → Linear(1).

    Args:
        n_users: Total number of unique users (vocabulary size).
        n_items: Total number of unique items (vocabulary size).
        embedding_dim: Dimension of each embedding vector.
        hidden_dim: Width of the first hidden layer.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_users: int = 100,
        n_items: int = 100,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._fitted = False
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for linear layers."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict rating scores for (user, item) index pairs.

        Args:
            x: Long tensor of shape (batch, 2) — columns [user_idx, item_idx].

        Returns:
            Float tensor of shape (batch, 1) with predicted scores.
        """
        users = x[:, 0]
        items = x[:, 1]
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        out = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc(out)

    @property
    def name(self) -> str:
        """Human-readable model identifier."""
        return "torch_mlp"

    def fit(
        self,
        features: Any,
        targets: Any | None = None,
    ) -> TorchMLPRecommender:
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

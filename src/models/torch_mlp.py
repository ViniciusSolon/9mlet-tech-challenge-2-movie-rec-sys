"""PyTorch MLP recommender stub (Bloco 5)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import RecommenderModel


class TorchMLPRecommender(nn.Module, RecommenderModel):
    """Placeholder for MLP on concatenated embeddings."""

    def __init__(
        self,
        n_users: int = 100,
        n_items: int = 100,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        users = x[:, 0]
        items = x[:, 1]
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        out = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc(out)

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

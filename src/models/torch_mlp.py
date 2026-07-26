"""PyTorch MLP recommender with user/item embeddings."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import RecommenderModel


class TorchMLPRecommender(nn.Module, RecommenderModel):
    """MLP rating predictor built on concatenated user+item embeddings."""

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
        """Initialise weights with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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

    def predict(self, features: torch.Tensor) -> list[float]:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        device = self.user_embedding.weight.device
        tensor = (
            features.to(device)
            if isinstance(features, torch.Tensor)
            else torch.tensor(features, dtype=torch.long, device=device)
        )
        self.eval()
        with torch.no_grad():
            predictions = self.forward(tensor).squeeze().cpu().numpy()
        return predictions.tolist()

"""Abstract recommender contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RecommenderModel(ABC):
    """Common interface for PyTorch and sklearn recommenders."""

    @abstractmethod
    def fit(self, features: Any, targets: Any | None = None) -> RecommenderModel:
        """Train or warm-start the model."""
        ...

    @abstractmethod
    def predict(self, features: Any) -> Any:
        """Produce scores or rankings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier."""
        ...

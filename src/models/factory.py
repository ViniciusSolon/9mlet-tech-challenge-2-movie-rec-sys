"""Factory for recommender model implementations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from models.base import RecommenderModel
from models.sklearn_baseline import SklearnBaselineRecommender
from models.torch_embedding import TorchEmbeddingRecommender
from models.torch_mlp import TorchMLPRecommender


class ModelKind(StrEnum):
    """Supported model identifiers."""

    SKLEARN_BASELINE = "sklearn_baseline"
    TORCH_EMBEDDING = "torch_embedding"
    TORCH_MLP = "torch_mlp"


_BUILDERS: dict[ModelKind, type[RecommenderModel]] = {
    ModelKind.SKLEARN_BASELINE: SklearnBaselineRecommender,
    ModelKind.TORCH_EMBEDDING: TorchEmbeddingRecommender,
    ModelKind.TORCH_MLP: TorchMLPRecommender,
}


def create_model(kind: str | ModelKind, **kwargs: Any) -> RecommenderModel:
    """Build a recommender by kind name.

    Args:
        kind: One of ``sklearn_baseline``, ``torch_embedding``, ``torch_mlp``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        A new recommender instance.

    Raises:
        ValueError: If kind is unknown.
    """
    key = ModelKind(kind) if isinstance(kind, str) else kind
    cls = _BUILDERS.get(key)
    if cls is None:
        msg = f"unknown model kind: {kind}"
        raise ValueError(msg)
    return cls(**kwargs)

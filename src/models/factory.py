"""Factory for recommender model implementations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from models.base import RecommenderModel
from models.most_popular import MostPopularRecommender
from models.sklearn_baseline import SklearnBaselineRecommender


class ModelKind(StrEnum):
    """Supported model identifiers."""

    MOST_POPULAR = "most_popular"
    SKLEARN_BASELINE = "sklearn_baseline"
    TORCH_EMBEDDING = "torch_embedding"
    TORCH_MLP = "torch_mlp"


def create_model(kind: str | ModelKind, **kwargs: Any) -> RecommenderModel:
    """Build a recommender by kind name.

    Args:
        kind: One of ``most_popular``, ``sklearn_baseline``,
            ``torch_embedding``, ``torch_mlp``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        A new recommender instance.

    Raises:
        ValueError: If kind is unknown.
        ImportError: If torch is not installed for torch-based models.
    """
    key = ModelKind(kind) if isinstance(kind, str) else kind

    if key == ModelKind.MOST_POPULAR:
        return MostPopularRecommender(**kwargs)

    if key == ModelKind.SKLEARN_BASELINE:
        return SklearnBaselineRecommender(**kwargs)

    if key == ModelKind.TORCH_EMBEDDING:
        from models.torch_embedding import TorchEmbeddingRecommender  # noqa: PLC0415

        return TorchEmbeddingRecommender(**kwargs)

    if key == ModelKind.TORCH_MLP:
        from models.torch_mlp import TorchMLPRecommender  # noqa: PLC0415

        return TorchMLPRecommender(**kwargs)

    msg = f"unknown model kind: {kind}"
    raise ValueError(msg)


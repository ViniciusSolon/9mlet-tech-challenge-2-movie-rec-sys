"""Factory for recommender model implementations."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from models.base import RecommenderModel
from models.sklearn_baseline import SklearnBaselineRecommender

if TYPE_CHECKING:
    from models.torch_embedding import TorchEmbeddingRecommender
    from models.torch_mlp import TorchMLPRecommender


class ModelKind(StrEnum):
    """Supported model identifiers."""

    SKLEARN_BASELINE = "sklearn_baseline"
    TORCH_EMBEDDING = "torch_embedding"
    TORCH_MLP = "torch_mlp"


def create_model(kind: str | ModelKind, **kwargs: Any) -> RecommenderModel:
    """Build a recommender by kind name.

    Args:
        kind: One of ``sklearn_baseline``, ``torch_embedding``, ``torch_mlp``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        A new recommender instance.

    Raises:
        ValueError: If kind is unknown.
        ImportError: If torch is not installed for torch-based models.
    """
    key = ModelKind(kind) if isinstance(kind, str) else kind

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


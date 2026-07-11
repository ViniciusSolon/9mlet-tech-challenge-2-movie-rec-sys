"""Tests for model Factory."""

from __future__ import annotations

import pytest

from models.factory import ModelKind, create_model

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


@pytest.mark.parametrize(
    "kind,expected_name",
    [
        ("sklearn_baseline", "sklearn_baseline"),
        pytest.param(
            "torch_embedding",
            "torch_embedding",
            marks=pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed"),
        ),
        pytest.param(
            "torch_mlp",
            "torch_mlp",
            marks=pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed"),
        ),
    ],
)
def test_create_model_returns_expected_kind(kind: str, expected_name: str) -> None:
    model = create_model(kind)
    assert model.name == expected_name


def test_create_model_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        create_model("unknown_model")


@requires_torch
def test_torch_mlp_forward_shape() -> None:
    model = create_model(ModelKind.TORCH_MLP, n_users=10, n_items=10, embedding_dim=8)
    x = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    out = model(x)
    assert out.shape == (2, 1)


@requires_torch
def test_torch_embedding_forward_shape() -> None:
    model = create_model(
        ModelKind.TORCH_EMBEDDING, n_users=10, n_items=10, embedding_dim=8
    )
    x = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    out = model(x)
    assert out.shape == (2, 1)


@requires_torch
def test_torch_mlp_predict_raises_before_fit() -> None:
    model = create_model(ModelKind.TORCH_MLP, n_users=10, n_items=10)
    x = torch.tensor([[0, 1]], dtype=torch.long)
    with pytest.raises(RuntimeError):
        model.predict(x)


def test_create_model_fit_predict_sklearn() -> None:
    model = create_model(ModelKind.SKLEARN_BASELINE)
    model.fit([1, 2, 3])
    scores = model.predict([1, 2])
    assert len(scores) == 2

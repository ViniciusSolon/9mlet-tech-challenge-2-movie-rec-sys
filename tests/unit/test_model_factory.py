"""Tests for model Factory."""

from __future__ import annotations

import pytest

from models.factory import ModelKind, create_model


@pytest.mark.parametrize(
    "kind,expected_name",
    [
        ("sklearn_baseline", "sklearn_baseline"),
        ("torch_embedding", "torch_embedding"),
        ("torch_mlp", "torch_mlp"),
    ],
)
def test_create_model_returns_expected_kind(kind: str, expected_name: str) -> None:
    model = create_model(kind)
    assert model.name == expected_name


def test_create_model_fit_predict_sklearn() -> None:
    model = create_model(ModelKind.SKLEARN_BASELINE)
    model.fit([1, 2, 3])
    scores = model.predict([1, 2])
    assert len(scores) == 2


def test_create_model_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        create_model("unknown_model")

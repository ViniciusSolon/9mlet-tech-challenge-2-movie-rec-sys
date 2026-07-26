"""Tests for model factory and baseline selection."""

from __future__ import annotations

import pytest

from models.factory import create_model


@pytest.mark.parametrize(
    "kind,expected_name",
    [
        ("most_popular", "most_popular"),
        ("sklearn_baseline", "sklearn_knn"),
        ("torch_embedding", "torch_embedding"),
        ("torch_mlp", "torch_mlp"),
    ],
)
def test_create_model_returns_expected_kind(kind: str, expected_name: str) -> None:
    if kind.startswith("torch"):
        pytest.importorskip("torch")
    if kind == "sklearn_baseline":
        model = create_model(kind, baseline="knn")
    else:
        model = create_model(kind)
    assert model.name == expected_name


@pytest.mark.parametrize(
    "baseline,n_neighbors",
    [("knn", 1), ("random_forest", 2)],
)
def test_create_model_fit_predict_sklearn(baseline: str, n_neighbors: int) -> None:
    model = create_model(
        "sklearn_baseline",
        baseline=baseline,
        n_neighbors=n_neighbors,
        max_depth=2,
    )
    model.fit([[0, 1], [1, 2]], [3.0, 4.0])
    scores = model.predict([[0, 1], [1, 2]])
    assert len(scores) == 2


def test_create_model_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        create_model("unknown_model")

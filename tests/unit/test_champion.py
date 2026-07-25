"""Unit tests for champion selection."""

from __future__ import annotations

import pytest

from evaluation.champion import select_champion


def test_select_champion_prefers_torch_over_better_baseline() -> None:
    candidates = [
        {"name": "sklearn_random_forest", "metrics": {"rmse": 0.5}},
        {"name": "torch_mlp", "metrics": {"rmse": 0.9}},
        {"name": "most_popular", "metrics": {"rmse": 1.2}},
    ]
    champion = select_champion(candidates)
    assert champion["name"] == "torch_mlp"


def test_select_champion_picks_best_torch() -> None:
    candidates = [
        {"name": "torch_embedding", "metrics": {"rmse": 1.1}},
        {"name": "torch_mlp", "metrics": {"rmse": 0.8}},
    ]
    assert select_champion(candidates)["name"] == "torch_mlp"


def test_select_champion_fallback_without_torch() -> None:
    candidates = [
        {"name": "most_popular", "metrics": {"rmse": 2.0}},
        {"name": "sklearn_knn", "metrics": {"rmse": 1.5}},
    ]
    assert select_champion(candidates)["name"] == "sklearn_knn"


def test_select_champion_empty_raises() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        select_champion([])

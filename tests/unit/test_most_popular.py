"""Tests for MostPopular baseline."""

from __future__ import annotations

import pytest

from models.factory import create_model
from models.most_popular import MostPopularRecommender


def test_most_popular_predicts_item_mean() -> None:
    model = MostPopularRecommender()
    model.fit([[0, 10], [1, 10], [2, 20]], [5.0, 3.0, 4.0])
    scores = model.predict([[0, 10], [0, 20], [0, 99]])
    assert scores[0] == pytest.approx(4.0)
    assert scores[1] == pytest.approx(4.0)
    assert scores[2] == pytest.approx(4.0)  # global mean fallback


def test_factory_creates_most_popular() -> None:
    model = create_model("most_popular")
    assert model.name == "most_popular"

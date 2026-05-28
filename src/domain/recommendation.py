"""Recommendation list value objects."""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import MovieId


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One ranked item suggestion."""

    movie_id: MovieId
    score: float


@dataclass(frozen=True, slots=True)
class RecommendationList:
    """Top-K recommendations for a user."""

    user_id: int
    items: tuple[Recommendation, ...]

    def movie_ids(self) -> list[int]:
        """Return ordered movie ids."""
        return [r.movie_id.value for r in self.items]

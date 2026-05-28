"""User–movie rating interaction."""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import MovieId, UserId


@dataclass(frozen=True, slots=True)
class Rating:
    """Single explicit rating event."""

    user_id: UserId
    movie_id: MovieId
    value: float
    timestamp: int | None = None

    def __post_init__(self) -> None:
        if not 0.5 <= self.value <= 5.0:
            msg = "rating must be between 0.5 and 5.0"
            raise ValueError(msg)

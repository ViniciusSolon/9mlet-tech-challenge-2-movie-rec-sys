"""Typed identifiers for users and movies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserId:
    """Opaque user identifier from MovieLens."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            msg = "user id must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MovieId:
    """Opaque movie (item) identifier from MovieLens."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            msg = "movie id must be positive"
            raise ValueError(msg)

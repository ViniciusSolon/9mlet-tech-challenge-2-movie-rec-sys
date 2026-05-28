"""Tests for domain value objects."""

from __future__ import annotations

import pytest

from domain.ids import MovieId, UserId
from domain.rating import Rating


def test_user_id_positive() -> None:
    assert UserId(1).value == 1


def test_user_id_invalid_raises() -> None:
    with pytest.raises(ValueError):
        UserId(0)


def test_rating_valid_range() -> None:
    r = Rating(UserId(1), MovieId(2), 4.0)
    assert r.value == 4.0


def test_rating_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        Rating(UserId(1), MovieId(2), 6.0)

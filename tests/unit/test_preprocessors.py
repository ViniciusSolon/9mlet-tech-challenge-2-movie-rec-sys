"""Tests for preprocessor Strategy."""

from __future__ import annotations

import pytest

from data.preprocessors import get_preprocessor


def test_explicit_keeps_high_ratings(sample_ratings) -> None:
    prep = get_preprocessor("explicit", min_rating=3.0)
    out = prep.transform(sample_ratings)
    assert (out["rating"] >= 3.0).all()
    assert (out["feedback"] == "explicit").all()


def test_implicit_filters_at_threshold(sample_ratings) -> None:
    prep = get_preprocessor("implicit", threshold=4.0)
    out = prep.transform(sample_ratings)
    assert len(out) == 2
    assert (out["interaction"] == 1).all()


def test_unknown_preprocessor_raises() -> None:
    with pytest.raises(ValueError):
        get_preprocessor("invalid")

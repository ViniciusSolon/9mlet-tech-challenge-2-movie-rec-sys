"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_ratings() -> pd.DataFrame:
    """Tiny ratings frame matching MovieLens schema."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [5.0, 2.0, 4.5, 3.0],
            "timestamp": [100, 200, 300, 400],
        }
    )

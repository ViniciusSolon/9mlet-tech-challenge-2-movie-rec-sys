"""Keep explicit star ratings with optional minimum filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from data.preprocessors.base import PreprocessorStrategy

if TYPE_CHECKING:
    import pandas as pd


class ExplicitFeedbackPreprocessor(PreprocessorStrategy):
    """Preserve ratings; optionally drop rows below min_rating."""

    def __init__(self, min_rating: float = 0.5) -> None:
        self._min_rating = min_rating

    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Filter by minimum rating and mark feedback type."""
        out = ratings.loc[ratings["rating"] >= self._min_rating].copy()
        out["feedback"] = "explicit"
        return out.reset_index(drop=True)

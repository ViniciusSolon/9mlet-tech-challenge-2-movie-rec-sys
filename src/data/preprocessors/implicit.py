"""Convert high ratings into implicit positive interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from data.preprocessors.base import PreprocessorStrategy

if TYPE_CHECKING:
    import pandas as pd


class ImplicitFeedbackPreprocessor(PreprocessorStrategy):
    """Binarize ratings at or above threshold."""

    def __init__(self, threshold: float = 4.0) -> None:
        self._threshold = threshold

    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Keep positives only with interaction flag."""
        mask = ratings["rating"] >= self._threshold
        out = ratings.loc[mask].copy()
        out["interaction"] = 1
        out["feedback"] = "implicit"
        return out.reset_index(drop=True)

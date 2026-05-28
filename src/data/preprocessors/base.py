"""Strategy interface for rating preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class PreprocessorStrategy(ABC):
    """Transform raw ratings into model-ready interactions."""

    @abstractmethod
    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ratings with strategy-specific columns."""
        ...

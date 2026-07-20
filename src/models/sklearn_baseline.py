"""Scikit-Learn baselines for recommendation rating prediction."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from models.base import RecommenderModel


class BaselineKind(StrEnum):
    """Supported baseline algorithm identifiers."""

    KNN = "knn"
    RANDOM_FOREST = "random_forest"


class SklearnBaselineRecommender(RecommenderModel):
    """Wrapper around Scikit-Learn regressors for recommendation."""

    def __init__(
        self,
        baseline: str = BaselineKind.KNN,
        n_neighbors: int = 20,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        self._baseline = (
            BaselineKind(baseline) if isinstance(baseline, str) else baseline
        )
        self._model = self._build_model(n_neighbors, max_depth, random_state)
        self._fitted = False

    @property
    def name(self) -> str:
        return f"sklearn_{self._baseline.value}"

    @property
    def estimator(self) -> BaseEstimator:
        return self._model

    def _build_model(
        self,
        n_neighbors: int,
        max_depth: int,
        random_state: int,
    ) -> RegressorMixin:
        if self._baseline == BaselineKind.KNN:
            return KNeighborsRegressor(n_neighbors=n_neighbors)
        return RandomForestRegressor(max_depth=max_depth, random_state=random_state)

    def fit(
        self,
        features: Any,
        targets: Any | None = None,
    ) -> SklearnBaselineRecommender:
        if targets is None:
            raise ValueError("targets must be provided for baseline training")
        self._model.fit(features, targets)
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[float]:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        return self._model.predict(features).tolist()

"""Metric strategy objects for rating evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class MetricStrategy(ABC):
    """Abstract strategy for a rating metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""

    @abstractmethod
    def compute(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Compute the metric for prediction arrays."""


class MSEStrategy(MetricStrategy):
    @property
    def name(self) -> str:
        return "mse"

    def compute(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        return float(mean_squared_error(y_true, y_pred))


class RMSEStrategy(MetricStrategy):
    @property
    def name(self) -> str:
        return "rmse"

    def compute(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        return float(root_mean_squared_error(y_true, y_pred))


class MAEStrategy(MetricStrategy):
    @property
    def name(self) -> str:
        return "mae"

    def compute(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        return float(mean_absolute_error(y_true, y_pred))


class R2Strategy(MetricStrategy):
    @property
    def name(self) -> str:
        return "r2"

    def compute(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        return float(r2_score(y_true, y_pred))


METRIC_STRATEGIES: list[MetricStrategy] = [
    RMSEStrategy(),
    MAEStrategy(),
    MSEStrategy(),
    R2Strategy(),
]


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Compute all configured rating metrics in a consistent order."""
    return {
        strategy.name: strategy.compute(y_true, y_pred)
        for strategy in METRIC_STRATEGIES
    }

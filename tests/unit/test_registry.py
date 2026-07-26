"""Unit tests for MLflow registry metric validation helpers."""

from __future__ import annotations

from evaluation.registry import MLflowRegistryManager


def test_validate_metrics_accepts_finite_non_negative() -> None:
    manager = MLflowRegistryManager.__new__(MLflowRegistryManager)
    assert manager.validate_metrics({"rmse": 0.9, "mae": 0.7, "r2": float("nan")})


def test_validate_metrics_rejects_negative() -> None:
    manager = MLflowRegistryManager.__new__(MLflowRegistryManager)
    assert not manager.validate_metrics({"rmse": -0.1, "mae": 0.7})

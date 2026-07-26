"""MLflow Registry automation helpers."""

from __future__ import annotations

import math

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


class MLflowRegistryManager:
    """Manage MLflow registered model lifecycle."""

    def __init__(self, tracking_uri: str, model_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def _ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.model_name)
        except MlflowException:
            self.client.create_registered_model(self.model_name)

    def archive_production(self) -> None:
        for version in self.client.get_latest_versions(
            self.model_name, stages=["Production"]
        ):
            self.client.transition_model_version_stage(
                self.model_name,
                version.version,
                "Archived",
                archive_existing_versions=False,
            )

    def register_model_version(self, run_id: str, artifact_path: str) -> int:
        self._ensure_registered_model()
        source = f"runs:/{run_id}/{artifact_path}"
        model_version = self.client.create_model_version(
            name=self.model_name,
            source=source,
            run_id=run_id,
        )
        return int(model_version.version)

    def stage_model_version(self, version: int, stage: str) -> None:
        self.client.transition_model_version_stage(
            self.model_name,
            version,
            stage,
            archive_existing_versions=False,
        )

    def validate_metrics(self, metrics: dict[str, float]) -> bool:
        for name, value in metrics.items():
            if name == "r2":
                continue
            if not math.isfinite(value) or value < 0.0:
                return False
        return True

    def promote_champion(self, version: int, metrics: dict[str, float]) -> str:
        self.stage_model_version(version, "Staging")
        if not self.validate_metrics(metrics):
            return "staging"
        self.archive_production()
        self.stage_model_version(version, "Production")
        return "production"

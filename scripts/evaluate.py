"""Evaluate PyTorch and baseline recommenders with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs"))

from settings import load_settings  # noqa: E402

from data.splits import assert_temporal_order, temporal_train_test_split  # noqa: E402
from evaluation.champion import select_champion  # noqa: E402
from evaluation.registry import MLflowRegistryManager  # noqa: E402
from evaluation.runner import (  # noqa: E402
    comparison_table,
    run_baselines,
    run_torch_eval,
    xy_from_frame,
)
from training.seeds import set_global_seeds  # noqa: E402
from utils.paths import get_processed_dir, get_project_root  # noqa: E402

_TEST_RATIO = 0.2


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluate stage."""
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="torch_mlp")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=settings.mlflow_experiment_name,
    )
    parser.add_argument(
        "--registry-model-name",
        type=str,
        default="movie-rec-sys",
    )
    return parser.parse_args()


def _json_safe(value: object) -> object:
    """Replace NaN/Inf floats with None for strict JSON serialization."""
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_metrics(path: Path, payload: dict[str, object]) -> None:
    """Persist the evaluation payload to disk as JSON."""
    path.write_text(
        json.dumps(_json_safe(payload), indent=4, allow_nan=False),
        encoding="utf-8",
    )


def _promote_champion(
    settings: object,
    registry_model_name: str,
    champion: dict[str, object],
) -> tuple[int, str]:
    """Register and promote the selected champion in MLflow Registry."""
    registry = MLflowRegistryManager(settings.mlflow_tracking_uri, registry_model_name)
    version = registry.register_model_version(
        str(champion["run_id"]),
        str(champion["artifact_path"]),
    )
    stage = registry.promote_champion(version, champion["metrics"])  # type: ignore[arg-type]
    return version, stage


def _build_result(
    temporal_ok: bool,
    candidates: list[dict[str, object]],
    champion: dict[str, object],
    version: int,
    stage: str,
) -> dict[str, object]:
    """Assemble the metrics.json payload."""
    return {
        "split": {
            "strategy": "temporal_preferred",
            "temporal_order_ok": temporal_ok,
            "test_ratio": _TEST_RATIO,
        },
        "comparison": comparison_table(candidates),
        "candidates": candidates,
        "champion": {
            "name": champion["name"],
            "run_id": champion["run_id"],
            "version": version,
            "stage": stage,
        },
    }


def main() -> int:
    """DVC evaluate stage entrypoint."""
    args = parse_args()
    settings = load_settings()
    set_global_seeds(42)
    ratings_path = get_processed_dir() / "features_ratings.parquet"
    model_path = get_project_root() / "models" / "model.pth"
    if not ratings_path.exists() or not model_path.exists():
        print("Missing inputs for evaluation.")
        return 1

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    df = pd.read_parquet(ratings_path)
    train_df, test_df = temporal_train_test_split(df, test_ratio=_TEST_RATIO, seed=42)
    temporal_ok = assert_temporal_order(train_df, test_df)
    print(f"Temporal split respected: {temporal_ok}")

    x_train, y_train = xy_from_frame(train_df)
    x_test, y_test = xy_from_frame(test_df)
    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates: list[dict[str, object]] = []
    run_baselines(x_train, y_train, x_test, y_test, args.experiment_name, candidates)
    run_torch_eval(
        args.model_type,
        args.experiment_name,
        train_df,
        test_df,
        x_test,
        y_test,
        n_users,
        n_movies,
        model_path,
        device,
        candidates,
        _TEST_RATIO,
    )

    champion = select_champion(candidates)
    version, stage = _promote_champion(settings, args.registry_model_name, champion)
    result = _build_result(temporal_ok, candidates, champion, version, stage)
    save_metrics(Path("metrics.json"), result)
    print("Comparison (by RMSE):")
    for row in result["comparison"]:
        print(f"  {row}")
    print(f"Champion: {champion['name']} version {version} in {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

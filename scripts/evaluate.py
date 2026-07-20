"""Evaluate PyTorch and sklearn recommender models with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import average_metrics, mae, rmse
from evaluation.metric_strategy import compute_metrics
from evaluation.registry import MLflowRegistryManager
from models.factory import create_model
from training.seeds import set_global_seeds
from utils.paths import get_processed_dir, get_project_root

_K = 10
_RELEVANT_THRESHOLD = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="torch_mlp")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="movie-rec-sys-training",
    )
    parser.add_argument(
        "--registry-model-name",
        type=str,
        default="movie-rec-sys",
    )
    return parser.parse_args()


def _build_test_sets(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test sets for ranking evaluation."""
    test_n = max(1, int(len(df) * 0.10))
    test_df = df.sample(n=test_n, random_state=seed)
    train_df = df.drop(test_df.index)
    return train_df, test_df


def _split_ratings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform the DataFrame into train/test arrays for regression metrics."""
    X = df[["user_idx", "movie_idx"]].to_numpy(dtype=int)
    y = df["rating"].to_numpy(dtype=float)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_torch_model(
    model_type: str,
    n_users: int,
    n_items: int,
    model_path: Path,
    embedding_dim: int = 32,
) -> torch.nn.Module:
    """Load a torch model from disk with the expected constructor arguments."""
    model = create_model(
        model_type,
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def log_mlflow_model(
    model: object,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, object],
) -> str:
    """Log metrics and model artifact for a candidate run."""
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model_name.startswith("torch"):
            mlflow.pytorch.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model.estimator, "model")
        return run.info.run_id


def evaluate_candidate(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute the standard rating metrics for a candidate model."""
    predictions = model.predict(X_test)
    return compute_metrics(y_test.tolist(), predictions)


def _compute_ranking_metrics(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute ranking metrics for a torch-based recommender."""
    seen: dict[int, set[int]] = (
        train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    )
    relevant: dict[int, set[int]] = (
        test_df[test_df["rating"] >= _RELEVANT_THRESHOLD]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    all_items = np.arange(n_items, dtype=np.int64)
    recommended_lists: list[list[int]] = []
    relevant_sets: list[set[int]] = []

    model.eval()
    with torch.no_grad():
        for user_idx, rel_items in relevant.items():
            unseen = np.array(
                [i for i in all_items if i not in seen.get(user_idx, set())],
                dtype=np.int64,
            )
            if len(unseen) == 0:
                continue
            user_col = np.full(len(unseen), user_idx, dtype=np.int64)
            X = torch.tensor(
                np.stack([user_col, unseen], axis=1), dtype=torch.long
            ).to(device)
            scores = model(X).squeeze().cpu().numpy()
            top_k_idx = np.argsort(-scores)[:k]
            recommended = unseen[top_k_idx].tolist()
            recommended_lists.append(recommended)
            relevant_sets.append(rel_items)

    return average_metrics(recommended_lists, relevant_sets, k)


def save_metrics(path: Path, payload: dict[str, object]) -> None:
    """Persist the evaluation payload to disk."""
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seeds(42)
    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"

    ratings_path = processed_dir / "features_ratings.parquet"
    model_path = models_dir / "model.pth"
    if not ratings_path.exists() or not model_path.exists():
        print("Missing inputs for evaluation.")
        return 1

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_parquet(ratings_path)
    train_df, test_df = _build_test_sets(df)
    X_train, X_test, y_train, y_test = _split_ratings(df)

    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates: list[dict[str, object]] = []
    for baseline in ["knn", "random_forest"]:
        model = create_model(
            "sklearn_baseline",
            baseline=baseline,
            n_neighbors=1 if baseline == "knn" else 2,
            max_depth=3,
        )
        model.fit(X_train, y_train)
        metrics = evaluate_candidate(model, X_test, y_test)
        run_id = log_mlflow_model(
            model,
            model.name,
            metrics,
            {
                "baseline": baseline,
                "model_type": model.name,
                "experiment": args.experiment_name,
            },
        )
        candidates.append(
            {
                "name": model.name,
                "run_id": run_id,
                "artifact_path": "model",
                "metrics": metrics,
            }
        )

    torch_model = load_torch_model(
        args.model_type,
        n_users,
        n_movies,
        model_path,
        embedding_dim=32,
    )
    torch_model.fit([], None)
    torch_model.to(device)

    metrics = evaluate_candidate(torch_model, X_test, y_test)
    ranking_metrics = _compute_ranking_metrics(
        torch_model,
        train_df,
        test_df,
        n_movies,
        _K,
        device,
    )
    metrics = {**metrics, **{f"{key}@{_K}": value for key, value in ranking_metrics.items()}}

    run_id = log_mlflow_model(
        torch_model,
        torch_model.name,
        metrics,
        {
            "model_type": args.model_type,
            "experiment": args.experiment_name,
        },
    )
    candidates.append(
        {
            "name": torch_model.name,
            "run_id": run_id,
            "artifact_path": "model",
            "metrics": metrics,
        }
    )

    champion = min(candidates, key=lambda item: item["metrics"]["rmse"])
    registry = MLflowRegistryManager(tracking_uri, args.registry_model_name)
    model_version = registry.register_model_version(
        champion["run_id"],
        champion["artifact_path"],
    )
    stage = registry.promote_champion(model_version, champion["metrics"])

    result = {
        "candidates": candidates,
        "champion": {
            "name": champion["name"],
            "run_id": champion["run_id"],
            "version": model_version,
            "stage": stage,
        },
    }
    save_metrics(Path("metrics.json"), result)
    print(f"Champion: {champion['name']} version {model_version} in {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

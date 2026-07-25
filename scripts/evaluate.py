"""Evaluate PyTorch and baseline recommenders with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs"))

from settings import load_settings  # noqa: E402
from data.splits import assert_temporal_order, temporal_train_test_split  # noqa: E402
from evaluation.metric_strategy import compute_metrics  # noqa: E402
from evaluation.metrics import average_metrics  # noqa: E402
from evaluation.registry import MLflowRegistryManager  # noqa: E402
from models.factory import create_model  # noqa: E402
from training.seeds import set_global_seeds  # noqa: E402
from utils.paths import get_processed_dir, get_project_root  # noqa: E402

_K = 10
_RELEVANT_THRESHOLD = 4.0
_TEST_RATIO = 0.2
_BASELINE_TRAIN_MAX = 200_000
_BASELINE_TEST_MAX = 50_000


def parse_args() -> argparse.Namespace:
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


def _xy_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and rating targets from a ratings frame."""
    X = frame[["user_idx", "movie_idx"]].to_numpy(dtype=int)
    y = frame["rating"].to_numpy(dtype=float)
    return X, y


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
    safe_metrics = {key.replace("@", "_at_"): value for key, value in metrics.items()}
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(safe_metrics)
        if model_name.startswith("torch"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact_path = Path(tmp_dir) / "model.pth"
                device = next(model.parameters()).device
                torch.save(model.cpu().state_dict(), artifact_path)
                model.to(device)
                mlflow.log_artifact(str(artifact_path), artifact_path="model")
        elif hasattr(model, "estimator"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                mlflow.sklearn.save_model(model.estimator, tmp_dir)
                mlflow.log_artifacts(tmp_dir, artifact_path="model")
        return run.info.run_id


def evaluate_candidate(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute the standard rating metrics for a candidate model."""
    # Evita OOM em hold-out de milhões de linhas no GPU.
    if len(X_test) > _BASELINE_TEST_MAX and not hasattr(model, "estimator"):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=_BASELINE_TEST_MAX, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
    predictions = np.atleast_1d(model.predict(X_test))
    return compute_metrics(y_test.tolist(), predictions)


def _compute_ranking_metrics(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k: int,
    device: torch.device,
    max_users: int = 1000,
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

    user_ids = list(relevant.keys())
    if len(user_ids) > max_users:
        rng = np.random.default_rng(42)
        user_ids = rng.choice(user_ids, size=max_users, replace=False).tolist()

    all_items = np.arange(n_items, dtype=np.int64)
    recommended_lists: list[list[int]] = []
    relevant_sets: list[set[int]] = []

    model.eval()
    with torch.no_grad():
        for user_idx in user_ids:
            rel_items = relevant[user_idx]
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


def _append_candidate(
    candidates: list[dict[str, object]],
    name: str,
    run_id: str,
    metrics: dict[str, float],
) -> None:
    """Append a candidate payload for champion selection."""
    candidates.append(
        {
            "name": name,
            "run_id": run_id,
            "artifact_path": "model",
            "metrics": metrics,
        }
    )


def _run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    candidates: list[dict[str, object]],
) -> None:
    """Train and log MostPopular + sklearn baselines on sampled rows."""
    if len(X_train) > _BASELINE_TRAIN_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=_BASELINE_TRAIN_MAX, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    if len(X_test) > _BASELINE_TEST_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=_BASELINE_TEST_MAX, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    popular = create_model("most_popular")
    popular.fit(X_train, y_train)
    popular_metrics = evaluate_candidate(popular, X_test, y_test)
    run_id = log_mlflow_model(
        popular,
        popular.name,
        popular_metrics,
        {
            "model_type": popular.name,
            "experiment": experiment_name,
            "train_sample": len(X_train),
        },
    )
    _append_candidate(candidates, popular.name, run_id, popular_metrics)

    for baseline in ["knn", "random_forest"]:
        n_neighbors = max(1, min(20, len(X_train)))
        model = create_model(
            "sklearn_baseline",
            baseline=baseline,
            n_neighbors=n_neighbors if baseline == "knn" else 2,
            max_depth=8,
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
                "experiment": experiment_name,
                "train_sample": len(X_train),
            },
        )
        _append_candidate(candidates, model.name, run_id, metrics)


def _run_torch_eval(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_users: int,
    n_movies: int,
    model_path: Path,
    device: torch.device,
    candidates: list[dict[str, object]],
) -> None:
    """Evaluate the trained torch model on rating and ranking metrics."""
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
    metrics = {
        **metrics,
        **{f"{key}@{_K}": value for key, value in ranking_metrics.items()},
    }
    run_id = log_mlflow_model(
        torch_model,
        torch_model.name,
        metrics,
        {
            "model_type": args.model_type,
            "experiment": args.experiment_name,
            "split": "temporal_or_random_fallback",
            "test_ratio": _TEST_RATIO,
        },
    )
    _append_candidate(candidates, torch_model.name, run_id, metrics)


def _comparison_table(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a compact comparison table sorted by RMSE."""
    rows: list[dict[str, object]] = []
    for item in candidates:
        metrics = item["metrics"]
        rows.append(
            {
                "model": item["name"],
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "precision@10": metrics.get("precision@10"),
                "recall@10": metrics.get("recall@10"),
                "ndcg@10": metrics.get("ndcg@10"),
                "hit_rate@10": metrics.get("hit_rate@10"),
            }
        )
    return sorted(rows, key=lambda row: float(row["rmse"] or 1e9))


def main() -> int:
    args = parse_args()
    settings = load_settings()
    set_global_seeds(42)
    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"

    ratings_path = processed_dir / "features_ratings.parquet"
    model_path = models_dir / "model.pth"
    if not ratings_path.exists() or not model_path.exists():
        print("Missing inputs for evaluation.")
        return 1

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_parquet(ratings_path)
    train_df, test_df = temporal_train_test_split(df, test_ratio=_TEST_RATIO, seed=42)
    temporal_ok = assert_temporal_order(train_df, test_df)
    print(f"Temporal split respected: {temporal_ok}")

    X_train, y_train = _xy_from_frame(train_df)
    X_test, y_test = _xy_from_frame(test_df)
    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates: list[dict[str, object]] = []
    _run_baselines(
        X_train, y_train, X_test, y_test, args.experiment_name, candidates
    )
    _run_torch_eval(
        args,
        train_df,
        test_df,
        X_test,
        y_test,
        n_users,
        n_movies,
        model_path,
        device,
        candidates,
    )

    champion = min(candidates, key=lambda item: item["metrics"]["rmse"])
    registry = MLflowRegistryManager(
        settings.mlflow_tracking_uri,
        args.registry_model_name,
    )
    model_version = registry.register_model_version(
        champion["run_id"],
        champion["artifact_path"],
    )
    stage = registry.promote_champion(model_version, champion["metrics"])

    comparison = _comparison_table(candidates)
    result = {
        "split": {
            "strategy": "temporal_preferred",
            "temporal_order_ok": temporal_ok,
            "test_ratio": _TEST_RATIO,
        },
        "comparison": comparison,
        "candidates": candidates,
        "champion": {
            "name": champion["name"],
            "run_id": champion["run_id"],
            "version": model_version,
            "stage": stage,
        },
    }
    save_metrics(Path("metrics.json"), result)
    print("Comparison (by RMSE):")
    for row in comparison:
        print(f"  {row}")
    print(f"Champion: {champion['name']} version {model_version} in {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

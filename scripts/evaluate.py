"""DVC Stage: Evaluate model performance with ≥ 4 metrics (Bloco 5)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import average_metrics, mae, rmse
from models.factory import create_model
from utils.paths import get_processed_dir, get_project_root

_K = 10
_RELEVANT_THRESHOLD = 4.0


def _build_test_sets(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last 10 % of interactions as the test set.

    Args:
        df: Feature-ready ratings DataFrame.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (train_df, test_df).
    """
    test_n = max(1, int(len(df) * 0.10))
    test_df = df.sample(n=test_n, random_state=seed)
    train_df = df.drop(test_df.index)
    return train_df, test_df


def _compute_ranking_metrics(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute mean Precision, Recall, NDCG, Hit Rate at K.

    For each user in the test set the model scores all items not seen
    during training, ranks them, and the top-K list is compared against
    the ground-truth relevant items (rating ≥ threshold).

    Args:
        model: Trained PyTorch recommender.
        train_df: Training interactions used to build seen-item sets.
        test_df: Held-out interactions used as ground truth.
        n_items: Total number of items in the catalogue.
        k: Ranking cut-off.
        device: Compute device.

    Returns:
        Dict with mean Precision@K, Recall@K, NDCG@K, Hit Rate@K.
    """
    seen: dict[int, set[int]] = (
        train_df.groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
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


def main() -> int:
    """Load the trained model and emit evaluation metrics."""
    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"

    ratings_path = processed_dir / "features_ratings.parquet"
    model_path = models_dir / "model.pth"

    if not ratings_path.exists() or not model_path.exists():
        print("Missing inputs for evaluation.")
        return 1

    df = pd.read_parquet(ratings_path)
    train_df, test_df = _build_test_sets(df)

    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model("torch_mlp", n_users=n_users, n_items=n_movies)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X_test = torch.tensor(
        test_df[["user_idx", "movie_idx"]].values, dtype=torch.long
    ).to(device)
    y_true = test_df["rating"].values

    with torch.no_grad():
        y_pred = model(X_test).squeeze().cpu().numpy()

    rating_metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": float(((y_true - y_pred) ** 2).mean()),
    }

    ranking = _compute_ranking_metrics(
        model, train_df, test_df, n_movies, _K, device
    )
    ranking_metrics = {f"{key}@{_K}": val for key, val in ranking.items()}

    metrics = {**rating_metrics, **ranking_metrics}
    print("Evaluation metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    with open("metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=4)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("movie-rec-sys-training")

    try:
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(metrics)
    except Exception as exc:
        print(f"Warning: MLflow tracking failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

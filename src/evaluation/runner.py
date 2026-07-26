"""Evaluation helpers: baselines, ranking, and MLflow logging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch

from evaluation.metric_strategy import compute_metrics
from evaluation.metrics import average_metrics
from models.factory import create_model

_K = 10
_RELEVANT_THRESHOLD = 4.0
_BASELINE_TRAIN_MAX = 200_000
_BASELINE_TEST_MAX = 50_000


def xy_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and rating targets from a ratings frame."""
    features = frame[["user_idx", "movie_idx"]].to_numpy(dtype=int)
    targets = frame["rating"].to_numpy(dtype=float)
    return features, targets


def sample_rows(
    features: np.ndarray,
    targets: np.ndarray,
    max_rows: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample feature/target arrays when they exceed ``max_rows``."""
    if len(features) <= max_rows:
        return features, targets
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), size=max_rows, replace=False)
    return features[idx], targets[idx]


def load_torch_model(
    model_type: str,
    n_users: int,
    n_items: int,
    model_path: Path,
    embedding_dim: int = 32,
) -> torch.nn.Module:
    """Load a torch model checkpoint with the expected constructor args."""
    model = create_model(
        model_type,
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_candidate(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute the standard rating metrics for a candidate model."""
    if len(x_test) > _BASELINE_TEST_MAX and not hasattr(model, "estimator"):
        x_test, y_test = sample_rows(x_test, y_test, _BASELINE_TEST_MAX)
    predictions = np.atleast_1d(model.predict(x_test))
    return compute_metrics(y_test.tolist(), predictions)


def log_mlflow_model(
    model: object,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, object],
) -> str:
    """Log metrics and model artifact for a candidate run."""
    safe = {key.replace("@", "_at_"): value for key, value in metrics.items()}
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(safe)
        _log_artifact(model, model_name)
        return run.info.run_id


def _log_artifact(model: object, model_name: str) -> None:
    """Persist torch state_dict or sklearn estimator under artifact path model/."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        if model_name.startswith("torch"):
            path = Path(tmp_dir) / "model.pth"
            device = next(model.parameters()).device
            torch.save(model.cpu().state_dict(), path)
            model.to(device)
            mlflow.log_artifact(str(path), artifact_path="model")
        elif hasattr(model, "estimator"):
            # Prefer pickle: skops rejects some sklearn neighbor types by default.
            mlflow.sklearn.save_model(
                model.estimator,
                tmp_dir,
                serialization_format="pickle",
            )
            mlflow.log_artifacts(tmp_dir, artifact_path="model")


def append_candidate(
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


def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    candidates: list[dict[str, object]],
) -> None:
    """Train and log MostPopular + sklearn baselines on sampled rows."""
    x_train, y_train = sample_rows(x_train, y_train, _BASELINE_TRAIN_MAX)
    x_test, y_test = sample_rows(x_test, y_test, _BASELINE_TEST_MAX)
    _eval_popular(x_train, y_train, x_test, y_test, experiment_name, candidates)
    for baseline in ("knn", "random_forest"):
        _eval_sklearn(
            baseline, x_train, y_train, x_test, y_test, experiment_name, candidates
        )


def _eval_popular(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    candidates: list[dict[str, object]],
) -> None:
    """Fit and log the MostPopular baseline."""
    model = create_model("most_popular")
    model.fit(x_train, y_train)
    metrics = evaluate_candidate(model, x_test, y_test)
    run_id = log_mlflow_model(
        model,
        model.name,
        metrics,
        {
            "model_type": model.name,
            "experiment": experiment_name,
            "train_sample": len(x_train),
        },
    )
    append_candidate(candidates, model.name, run_id, metrics)


def _eval_sklearn(
    baseline: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    candidates: list[dict[str, object]],
) -> None:
    """Fit and log one sklearn baseline variant."""
    n_neighbors = max(1, min(20, len(x_train)))
    model = create_model(
        "sklearn_baseline",
        baseline=baseline,
        n_neighbors=n_neighbors if baseline == "knn" else 2,
        max_depth=8,
    )
    model.fit(x_train, y_train)
    metrics = evaluate_candidate(model, x_test, y_test)
    run_id = log_mlflow_model(
        model,
        model.name,
        metrics,
        {
            "baseline": baseline,
            "model_type": model.name,
            "experiment": experiment_name,
            "train_sample": len(x_train),
        },
    )
    append_candidate(candidates, model.name, run_id, metrics)


def compute_ranking_metrics(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    device: torch.device,
    k: int = _K,
    max_users: int = 200,
) -> dict[str, float]:
    """Compute ranking metrics for a torch-based recommender."""
    seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    relevant = (
        test_df[test_df["rating"] >= _RELEVANT_THRESHOLD]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )
    user_ids = _sample_user_ids(list(relevant.keys()), max_users)
    recommended_lists: list[list[int]] = []
    relevant_sets: list[set[int]] = []
    model.eval()
    with torch.no_grad():
        for user_idx in user_ids:
            recs = _rank_user(model, user_idx, seen, n_items, k, device)
            if recs is None:
                continue
            recommended_lists.append(recs)
            relevant_sets.append(relevant[user_idx])
    return average_metrics(recommended_lists, relevant_sets, k)


def _sample_user_ids(user_ids: list[int], max_users: int) -> list[int]:
    """Optionally downsample the list of evaluation users."""
    if len(user_ids) <= max_users:
        return user_ids
    rng = np.random.default_rng(42)
    return rng.choice(user_ids, size=max_users, replace=False).tolist()


def _rank_user(
    model: torch.nn.Module,
    user_idx: int,
    seen: dict[int, set[int]],
    n_items: int,
    k: int,
    device: torch.device,
    score_batch: int = 4096,
) -> list[int] | None:
    """Return top-K unseen item ids for one user, or None if none remain."""
    all_items = np.arange(n_items, dtype=np.int64)
    unseen = np.array(
        [i for i in all_items if i not in seen.get(user_idx, set())],
        dtype=np.int64,
    )
    if len(unseen) == 0:
        return None
    scores = _score_pairs(model, user_idx, unseen, device, score_batch)
    return unseen[np.argsort(-scores)[:k]].tolist()


def _score_pairs(
    model: torch.nn.Module,
    user_idx: int,
    items: np.ndarray,
    device: torch.device,
    score_batch: int,
) -> np.ndarray:
    """Score (user, item) pairs in batches to limit peak memory."""
    chunks: list[np.ndarray] = []
    for start in range(0, len(items), score_batch):
        batch_items = items[start : start + score_batch]
        user_col = np.full(len(batch_items), user_idx, dtype=np.int64)
        batch = torch.tensor(
            np.stack([user_col, batch_items], axis=1),
            dtype=torch.long,
            device=device,
        )
        chunk = model(batch).squeeze().detach().cpu().numpy()
        chunks.append(np.atleast_1d(chunk))
    return np.concatenate(chunks)


def comparison_table(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a compact comparison table sorted by RMSE."""
    rows: list[dict[str, object]] = []
    for item in candidates:
        metrics = item["metrics"]  # type: ignore[index]
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


def run_torch_eval(
    model_type: str,
    experiment_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_users: int,
    n_movies: int,
    model_path: Path,
    device: torch.device,
    candidates: list[dict[str, object]],
    test_ratio: float,
) -> None:
    """Evaluate the trained torch model on rating and ranking metrics."""
    torch_model = load_torch_model(model_type, n_users, n_movies, model_path)
    torch_model.fit([], None)
    torch_model.to(device)
    metrics = evaluate_candidate(torch_model, x_test, y_test)
    ranking = compute_ranking_metrics(torch_model, train_df, test_df, n_movies, device)
    metrics = {**metrics, **{f"{key}@{_K}": value for key, value in ranking.items()}}
    run_id = log_mlflow_model(
        torch_model,
        torch_model.name,
        metrics,
        {
            "model_type": model_type,
            "experiment": experiment_name,
            "split": "temporal_or_random_fallback",
            "test_ratio": test_ratio,
        },
    )
    append_candidate(candidates, torch_model.name, run_id, metrics)

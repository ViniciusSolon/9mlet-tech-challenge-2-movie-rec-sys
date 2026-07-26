"""Evaluate PyTorch and baseline recommenders with MLflow tracking."""

from __future__ import annotations

import argparse
import json
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


def _save_model_artifact(model: object, model_name: str, tmp_dir: str) -> None:
    """Save pytorch or sklearn model artifact inside temp directory."""
    if model_name.startswith("torch"):
        artifact_path = Path(tmp_dir) / "model.pth"
        device = next(model.parameters()).device  # type: ignore
        torch.save(model.cpu().state_dict(), artifact_path)  # type: ignore
        model.to(device)  # type: ignore
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
    elif hasattr(model, "estimator"):
        mlflow.sklearn.save_model(model.estimator, tmp_dir)
        mlflow.log_artifacts(tmp_dir, artifact_path="model")


def log_mlflow_model(
    model: object, model_name: str, metrics: dict[str, float], params: dict[str, object]
) -> str:
    """Log metrics and model artifact for a candidate run."""
    safe_metrics = {key.replace("@", "_at_"): value for key, value in metrics.items()}
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(safe_metrics)
        with tempfile.TemporaryDirectory() as tmp_dir:
            _save_model_artifact(model, model_name, tmp_dir)
        return run.info.run_id


def evaluate_candidate(
    model: object, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Compute the standard rating metrics for a candidate model."""
    if len(X_test) > _BASELINE_TEST_MAX and not hasattr(model, "estimator"):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=_BASELINE_TEST_MAX, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
    preds = np.atleast_1d(model.predict(X_test))  # type: ignore
    return compute_metrics(y_test.tolist(), preds)


def _predict_user_ranking(
    model: torch.nn.Module, user_idx: int, unseen: np.ndarray, device: torch.device
) -> np.ndarray:
    """Predict scores for unseen items for a single user."""
    user_col = np.full(len(unseen), user_idx, dtype=np.int64)
    X = torch.tensor(np.stack([user_col, unseen], axis=1), dtype=torch.long).to(device)
    return model(X).squeeze().cpu().numpy()


def _get_top_k_for_user(
    model: torch.nn.Module,
    user_idx: int,
    seen_items: set[int],
    n_items: int,
    k: int,
    device: torch.device,
) -> list[int] | None:
    """Get top K recommended items for a user."""
    all_items = np.arange(n_items, dtype=np.int64)
    unseen = np.array([i for i in all_items if i not in seen_items], dtype=np.int64)
    if len(unseen) == 0:
        return None
    scores = _predict_user_ranking(model, user_idx, unseen, device)
    top_k_idx = np.argsort(-scores)[:k]
    return unseen[top_k_idx].tolist()


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
    seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    rel = (
        test_df[test_df["rating"] >= _RELEVANT_THRESHOLD]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    user_ids = list(rel.keys())
    if len(user_ids) > max_users:
        user_ids = (
            np.random.default_rng(42)
            .choice(user_ids, size=max_users, replace=False)
            .tolist()
        )

    recs, rels = [], []
    model.eval()
    with torch.no_grad():
        for u in user_ids:
            top_k = _get_top_k_for_user(
                model, u, seen.get(u, set()), n_items, k, device
            )
            if top_k is not None:
                recs.append(top_k)
                rels.append(rel[u])

    return average_metrics(recs, rels, k)


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
        {"name": name, "run_id": run_id, "artifact_path": "model", "metrics": metrics}
    )


def _sample_dataset(
    X: np.ndarray, y: np.ndarray, max_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample dataset if rows exceed max_rows."""
    if len(X) <= max_rows:
        return X, y
    idx = np.random.default_rng(42).choice(len(X), size=max_rows, replace=False)
    return X[idx], y[idx]


def _eval_and_log(
    model: object,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: dict[str, object],
    candidates: list[dict[str, object]],
) -> None:
    """Fit, evaluate, log and append candidate model."""
    model.fit(X_tr, y_tr)  # type: ignore
    metrics = evaluate_candidate(model, X_te, y_te)
    run_id = log_mlflow_model(model, model.name, metrics, params)  # type: ignore
    _append_candidate(candidates, model.name, run_id, metrics)  # type: ignore


def _run_baselines(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    exp_name: str,
    candidates: list[dict[str, object]],
) -> None:
    """Train and log MostPopular + sklearn baselines on sampled rows."""
    X_tr_s, y_tr_s = _sample_dataset(X_tr, y_tr, _BASELINE_TRAIN_MAX)
    X_te_s, y_te_s = _sample_dataset(X_te, y_te, _BASELINE_TEST_MAX)

    pop = create_model("most_popular")
    _eval_and_log(
        pop,
        X_tr_s,
        y_tr_s,
        X_te_s,
        y_te_s,
        {"model_type": pop.name, "experiment": exp_name},
        candidates,
    )

    for baseline in ["knn", "random_forest"]:
        n_neighbors = max(1, min(20, len(X_tr_s)))
        sk = create_model(
            "sklearn_baseline",
            baseline=baseline,
            n_neighbors=n_neighbors if baseline == "knn" else 2,
            max_depth=8,
        )
        _eval_and_log(
            sk,
            X_tr_s,
            y_tr_s,
            X_te_s,
            y_te_s,
            {"baseline": baseline, "model_type": sk.name, "experiment": exp_name},
            candidates,
        )


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
        args.model_type, n_users, n_movies, model_path, embedding_dim=32
    )
    torch_model.fit([], None)
    torch_model.to(device)

    metrics = evaluate_candidate(torch_model, X_test, y_test)
    ranking = _compute_ranking_metrics(
        torch_model, train_df, test_df, n_movies, _K, device
    )
    metrics = {**metrics, **{f"{k}@{_K}": v for k, v in ranking.items()}}

    params = {
        "model_type": args.model_type,
        "experiment": args.experiment_name,
        "split": "temporal_or_random_fallback",
        "test_ratio": _TEST_RATIO,
    }
    run_id = log_mlflow_model(torch_model, torch_model.name, metrics, params)
    _append_candidate(candidates, torch_model.name, run_id, metrics)


def _comparison_table(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a compact comparison table sorted by RMSE."""
    rows = [
        {
            "model": c["name"],
            "rmse": c["metrics"].get("rmse"),
            "mae": c["metrics"].get("mae"),
            "precision@10": c["metrics"].get("precision@10"),
            "recall@10": c["metrics"].get("recall@10"),
            "ndcg@10": c["metrics"].get("ndcg@10"),
            "hit_rate@10": c["metrics"].get("hit_rate@10"),
        }
        for c in candidates
    ]
    return sorted(rows, key=lambda r: float(r["rmse"] or 1e9))


def _register_and_promote(
    champion: dict[str, object],
    metrics: dict[str, float],
    registry_name: str,
    tracking_uri: str,
) -> tuple[int, str]:
    """Register champion in MLflow Model Registry and promote stage."""
    reg = MLflowRegistryManager(tracking_uri, registry_name)
    ver = reg.register_model_version(
        str(champion["run_id"]), str(champion["artifact_path"])
    )
    stage = reg.promote_champion(ver, metrics)
    return ver, stage


def main() -> int:
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

    X_tr, y_tr = _xy_from_frame(train_df)
    X_te, y_te = _xy_from_frame(test_df)
    n_users, n_movies = int(df["user_idx"].max() + 1), int(df["movie_idx"].max() + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates: list[dict[str, object]] = []
    _run_baselines(X_tr, y_tr, X_te, y_te, args.experiment_name, candidates)
    _run_torch_eval(
        args,
        train_df,
        test_df,
        X_te,
        y_te,
        n_users,
        n_movies,
        model_path,
        device,
        candidates,
    )

    champ = min(candidates, key=lambda c: c["metrics"]["rmse"])  # type: ignore
    ver, stage = _register_and_promote(
        champ, champ["metrics"], args.registry_model_name, settings.mlflow_tracking_uri
    )  # type: ignore

    comp = _comparison_table(candidates)
    res = {
        "split": {
            "strategy": "temporal_preferred",
            "temporal_order_ok": temporal_ok,
            "test_ratio": _TEST_RATIO,
        },
        "comparison": comp,
        "candidates": candidates,
        "champion": {
            "name": champ["name"],
            "run_id": champ["run_id"],
            "version": ver,
            "stage": stage,
        },
    }
    save_metrics(Path("metrics.json"), res)
    for r in comp:
        print(f"  {r}")
    print(f"Champion: {champ['name']} version {ver} in {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

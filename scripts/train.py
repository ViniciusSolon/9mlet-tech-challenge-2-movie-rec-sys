"""Train the recommender model and log artifacts to MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs"))

from settings import load_settings  # noqa: E402

from models.factory import create_model  # noqa: E402
from training.loop import (  # noqa: E402
    build_loaders,
    build_optimizer,
    run_epoch,
    update_checkpoint,
)
from training.seeds import set_global_seeds  # noqa: E402
from utils.paths import get_processed_dir, get_project_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI hyperparameters for the train stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model-type", type=str, default="torch_mlp")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _log_train_params(args: argparse.Namespace, n_users: int, n_items: int) -> None:
    """Log static training hyperparameters to the active MLflow run."""
    mlflow.log_params(
        {
            "model_type": args.model_type,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "val_split": args.val_split,
            "patience": args.patience,
            "seed": args.seed,
            "n_users": n_users,
            "n_items": n_items,
        }
    )


def _run_one_epoch(
    model: nn.Module,
    train_loader: object,
    val_loader: object,
    criterion: nn.Module,
    optimizer: object,
    scheduler: object,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train and validate one epoch; log metrics to MLflow."""
    train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = run_epoch(model, val_loader, criterion, None, device)
    scheduler.step(val_loss)
    mlflow.log_metrics({"train_mse": train_loss, "val_mse": val_loss}, step=epoch)
    print(f"Epoch {epoch + 1}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")
    return train_loss, val_loss


def _train_epochs(
    model: nn.Module,
    train_loader: object,
    val_loader: object,
    args: argparse.Namespace,
    device: torch.device,
    model_path: Path,
) -> tuple[float, int, list[dict[str, float]]]:
    """Fit the model with early stopping; return best loss, epoch, history."""
    criterion = nn.MSELoss()
    optimizer, scheduler = build_optimizer(model, args.lr)
    best_val_loss, best_epoch, patience_count = float("inf"), 0, 0
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        train_loss, val_loss = _run_one_epoch(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
        )
        history.append(
            {"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss}
        )
        prev_best = best_val_loss
        best_val_loss, patience_count, stop = update_checkpoint(
            model,
            val_loss,
            best_val_loss,
            patience_count,
            args.patience,
            str(model_path),
        )
        if best_val_loss < prev_best:
            best_epoch = epoch + 1
        if stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    return best_val_loss, best_epoch, history


def _finalize_run(
    model: nn.Module,
    model_path: Path,
    history: list[dict[str, float]],
    best_val: float,
    best_epoch: int,
    device: torch.device,
    models_dir: Path,
) -> None:
    """Log best metrics, checkpoint, history and MLflow model artifact."""
    mlflow.log_metrics({"best_val_mse": best_val, "best_epoch": float(best_epoch)})
    mlflow.log_artifact(str(model_path))
    history_path = models_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        example = torch.tensor([[0, 0]], dtype=torch.long, device=device)
        mlflow.pytorch.log_model(
            model, artifact_path="model", input_example=example.cpu().numpy()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Note: PyTorch model logging via MLflow REST API skipped ({exc})")


def main() -> int:
    """DVC train stage entrypoint."""
    args = parse_args()
    set_global_seeds(args.seed)
    settings = load_settings()
    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    ratings_path = processed_dir / "features_ratings.parquet"
    if not ratings_path.exists():
        print(f"Input {ratings_path} not found.")
        return 1

    df = pd.read_parquet(ratings_path)
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    train_loader, val_loader = build_loaders(
        df, args.val_split, args.batch_size, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        args.model_type,
        n_users=n_users,
        n_items=n_movies,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model_path = models_dir / "model.pth"
    print(f"Training {args.model_type}: {n_users} users, {n_movies} items on {device}")

    with mlflow.start_run(run_name=f"{args.model_type}_train"):
        _log_train_params(args, n_users, n_movies)
        mlflow.log_params({"device": str(device)})
        best_val, best_epoch, history = _train_epochs(
            model, train_loader, val_loader, args, device, model_path
        )
        print(f"Best val_mse={best_val:.4f} at epoch {best_epoch}")
        _finalize_run(
            model, model_path, history, best_val, best_epoch, device, models_dir
        )

    print(f"Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

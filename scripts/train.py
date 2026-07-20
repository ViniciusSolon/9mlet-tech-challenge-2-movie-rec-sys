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
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import create_model
from training.seeds import set_global_seeds
from utils.paths import get_processed_dir, get_project_root


def parse_args() -> argparse.Namespace:
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


def _build_loaders(
    df: pd.DataFrame, val_split: float, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Split data into train/validation loaders."""
    val_n = max(1, int(len(df) * val_split))
    val_df = df.sample(n=val_n, random_state=seed)
    train_df = df.drop(val_df.index)

    def _to_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        X = torch.tensor(frame[["user_idx", "movie_idx"]].values, dtype=torch.long)
        y = torch.tensor(frame["rating"].values, dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

    return _to_loader(train_df, shuffle=True), _to_loader(val_df, shuffle=False)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def main() -> int:
    args = parse_args()
    set_global_seeds(args.seed)

    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)

    ratings_path = processed_dir / "features_ratings.parquet"
    if not ratings_path.exists():
        print(f"Input {ratings_path} not found.")
        return 1

    print(f"Loading features from {ratings_path}...")
    df = pd.read_parquet(ratings_path)

    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("movie-rec-sys-training")

    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    train_loader, val_loader = _build_loaders(
        df,
        args.val_split,
        args.batch_size,
        args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(
        args.model_type,
        n_users=n_users,
        n_items=n_movies,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.5,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    model_path = models_dir / "model.pth"
    training_history: list[dict[str, float]] = []

    print(f"Training {args.model_type}: {n_users} users, {n_movies} items")

    with mlflow.start_run(run_name=f"{args.model_type}_train"):
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
                "n_items": n_movies,
            }
        )
        mlflow.log_params({"device": str(device)})

        for epoch in range(args.epochs):
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _run_epoch(model, val_loader, criterion, None, device)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1}/{args.epochs}  "
                f"train_mse={train_loss:.4f}  val_mse={val_loss:.4f}"
            )
            mlflow.log_metrics(
                {"train_mse": train_loss, "val_mse": val_loss},
                step=epoch,
            )
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_count = 0
                torch.save(model.state_dict(), str(model_path))
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Best val_mse={best_val_loss:.4f} at epoch {best_epoch}")
        mlflow.log_metrics(
            {"best_val_mse": best_val_loss, "best_epoch": float(best_epoch)}
        )
        mlflow.log_artifact(str(model_path))

        history_path = models_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(training_history, fh, indent=2)
        mlflow.log_artifact(str(history_path))

        mlflow.pytorch.log_model(model, artifact_path="model")

    print(f"Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

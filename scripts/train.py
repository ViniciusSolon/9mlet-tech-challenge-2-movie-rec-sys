"""DVC Stage: Train recommendation model with MLflow tracking."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import create_model
from utils.paths import get_processed_dir, get_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-type", type=str, default="torch_mlp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    
    ratings_path = processed_dir / "features_ratings.parquet"
    if not ratings_path.exists():
        print(f"Input {ratings_path} not found.")
        return 1
        
    print(f"Loading features from {ratings_path}...")
    df = pd.read_parquet(ratings_path)
    
    # MLflow tracking
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("movie-rec-sys-training")
    
    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    
    # Prepare data
    X = torch.tensor(df[["user_idx", "movie_idx"]].values, dtype=torch.long)
    y = torch.tensor(df["rating"].values, dtype=torch.float32)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Instantiate model via Factory
    model = create_model(
        args.model_type,
        n_users=n_users,
        n_items=n_movies,
        embedding_dim=args.embedding_dim
    )
    
    print(f"Training {args.model_type} with {len(df)} samples...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = model(batch_X).squeeze()
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_mse", avg_loss, step=epoch)
        
        # Save model
        model_path = models_dir / "model.pth"
        torch.save(model.state_dict(), str(model_path))
        mlflow.log_artifact(str(model_path))
        print(f"Model saved to {model_path}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

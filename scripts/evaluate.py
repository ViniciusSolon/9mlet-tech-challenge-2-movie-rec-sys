"""DVC Stage: Evaluate model performance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import create_model
from utils.paths import get_processed_dir, get_project_root


def main() -> int:
    processed_dir = get_processed_dir()
    models_dir = get_project_root() / "models"
    
    ratings_path = processed_dir / "features_ratings.parquet"
    model_path = models_dir / "model.pth"
    
    if not ratings_path.exists() or not model_path.exists():
        print("Missing inputs for evaluation.")
        return 1
        
    df = pd.read_parquet(ratings_path)
    # Simple evaluation on the same data for now (scaffold)
    
    n_users = int(df["user_idx"].max() + 1)
    n_movies = int(df["movie_idx"].max() + 1)
    
    # Need to know which model type was used. In a real scenario, this is in params or metadata.
    model = create_model("torch_mlp", n_users=n_users, n_items=n_movies)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    X = torch.tensor(df[["user_idx", "movie_idx"]].values, dtype=torch.long)
    y_true = df["rating"].values
    
    with torch.no_grad():
        y_pred = model(X).squeeze().numpy()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    
    metrics = {
        "rmse": float(rmse),
        "mse": float(mse)
    }
    
    # Save metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation metrics: {metrics}")
    
    # Log to MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("movie-rec-sys-training")
    
    try:
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"Warning: MLflow tracking failed: {e}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

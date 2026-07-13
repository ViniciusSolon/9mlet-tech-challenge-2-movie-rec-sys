"""Demo end-to-end: train MLP/Embedding and show top-K recommendations.

Generates synthetic MovieLens-like data in memory, trains the model and
prints a recommendation list for each user.  No external data required.

Usage:
    python scripts/demo_recommend.py [--model torch_mlp|torch_embedding]
                                     [--epochs 5] [--k 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import create_model
from training.seeds import set_global_seeds

# ---------------------------------------------------------------------------
# Synthetic catalogue
# ---------------------------------------------------------------------------
MOVIES = {
    1: "Toy Story",
    2: "Jumanji",
    3: "Grumpier Old Men",
    4: "Waiting to Exhale",
    5: "Father of the Bride II",
    6: "Heat",
    7: "Sabrina",
    8: "Tom and Huck",
    9: "Sudden Death",
    10: "GoldenEye",
}

USERS = [101, 102, 103, 104, 105]


def _make_synthetic_ratings(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic rating interactions for demo purposes.

    Args:
        seed: Random seed.

    Returns:
        DataFrame with columns userId, movieId, rating.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for user in USERS:
        # Each user rates 5-8 random movies
        n = rng.integers(5, 9)
        movies = rng.choice(list(MOVIES.keys()), size=n, replace=False)
        for movie in movies:
            rating = float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0],
                                      p=[0.05, 0.10, 0.20, 0.35, 0.30]))
            rows.append({"userId": user, "movieId": int(movie), "rating": rating})
    return pd.DataFrame(rows)


def _build_index_maps(
    df: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Build contiguous integer index maps for users and movies.

    Args:
        df: Ratings DataFrame.

    Returns:
        Tuple (user_to_idx, idx_to_user, movie_to_idx, idx_to_movie).
    """
    users = sorted(df["userId"].unique())
    movies = sorted(df["movieId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    movie_to_idx = {m: i for i, m in enumerate(movies)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}
    return user_to_idx, idx_to_user, movie_to_idx, idx_to_movie


def _train(
    model: nn.Module,
    df: pd.DataFrame,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> None:
    """Train the model on the full interaction set.

    Args:
        model: PyTorch recommender model.
        df: Feature-ready DataFrame with user_idx, movie_idx, rating.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        device: Compute device.
    """
    X = torch.tensor(df[["user_idx", "movie_idx"]].values, dtype=torch.long).to(device)
    y = torch.tensor(df["rating"].values, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(-1), by)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  train_mse={total / len(loader):.4f}")


def _recommend(
    model: nn.Module,
    user_idx: int,
    seen_movie_idxs: set[int],
    all_movie_idxs: list[int],
    k: int,
    device: torch.device,
) -> list[int]:
    """Return top-K unseen item indices for a user.

    Args:
        model: Trained PyTorch recommender.
        user_idx: Integer index of the target user.
        seen_movie_idxs: Movie indices already rated by the user.
        all_movie_idxs: All available movie indices.
        k: Number of recommendations.
        device: Compute device.

    Returns:
        Ordered list of up to K recommended movie indices.
    """
    candidates = [i for i in all_movie_idxs if i not in seen_movie_idxs]
    if not candidates:
        return []
    users_col = torch.full((len(candidates),), user_idx, dtype=torch.long).to(device)
    items_col = torch.tensor(candidates, dtype=torch.long).to(device)
    X = torch.stack([users_col, items_col], dim=1)
    model.eval()
    with torch.no_grad():
        scores = model(X).squeeze().cpu().numpy()
    top_k = np.argsort(-scores)[:k]
    return [candidates[i] for i in top_k]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="torch_mlp",
        choices=["torch_mlp", "torch_embedding"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run demo training and print recommendations for each user."""
    args = parse_args()
    set_global_seeds(args.seed)
    device = torch.device("cpu")

    print("=" * 60)
    print(f"  Demo: {args.model}  |  epochs={args.epochs}  |  top-{args.k}")
    print("=" * 60)

    # --- Data ---
    ratings = _make_synthetic_ratings(args.seed)
    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = _build_index_maps(ratings)

    df = ratings.copy()
    df["user_idx"] = df["userId"].map(user_to_idx)
    df["movie_idx"] = df["movieId"].map(movie_to_idx)

    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    print(f"\nSynthetic data: {len(df)} ratings  |  {n_users} users  |  {n_movies} movies\n")

    # --- Model ---
    model = create_model(
        args.model,
        n_users=n_users,
        n_items=n_movies,
        embedding_dim=args.embedding_dim,
    )
    model.to(device)

    print(f"Training {args.model}...")
    _train(model, df, args.epochs, lr=0.01, batch_size=32, device=device)
    model.fit(None)

    # --- Recommendations ---
    all_movie_idxs = list(range(n_movies))
    seen_by: dict[int, set[int]] = (
        df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    )

    print("\n" + "=" * 60)
    print(f"  Top-{args.k} Recommendations")
    print("=" * 60)

    for user_idx, user_id in sorted(idx_to_user.items()):
        seen = seen_by.get(user_idx, set())
        rec_idxs = _recommend(model, user_idx, seen, all_movie_idxs, args.k, device)
        rated_titles = [MOVIES[idx_to_movie[i]] for i in seen]
        rec_titles = [MOVIES[idx_to_movie[i]] for i in rec_idxs]

        print(f"\nUser {user_id}")
        print(f"  Já assistiu : {', '.join(rated_titles)}")
        print(f"  Recomendado : {', '.join(rec_titles) or '(sem candidatos)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

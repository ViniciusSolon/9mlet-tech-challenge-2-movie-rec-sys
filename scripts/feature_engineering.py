"""DVC Stage: Feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.paths import get_processed_dir  # noqa: E402


def main() -> int:
    processed_dir = get_processed_dir()

    ratings_path = processed_dir / "preprocessed_ratings.parquet"
    metadata_path = processed_dir / "enriched_metadata.parquet"

    if not ratings_path.exists() or not metadata_path.exists():
        print("Required inputs missing.")
        return 1

    print("Loading preprocessed data...")
    ratings = pd.read_parquet(ratings_path)
    metadata = pd.read_parquet(metadata_path)

    # Simple feature engineering: join ratings with movie features if needed
    # Or just prepare the tensors/matrices for the model
    print("Performing feature engineering...")

    # Example: map IDs to contiguous integers for embeddings
    user_map = {uid: i for i, uid in enumerate(ratings["userId"].unique())}
    movie_map = {mid: i for i, mid in enumerate(metadata["movieId"].unique())}

    ratings["user_idx"] = ratings["userId"].map(user_map)
    ratings["movie_idx"] = ratings["movieId"].map(movie_map)

    # Drop rows with unknown movies (if any)
    ratings = ratings.dropna(subset=["movie_idx"])
    ratings["movie_idx"] = ratings["movie_idx"].astype(int)

    out_ratings = processed_dir / "features_ratings.parquet"
    ratings.to_parquet(out_ratings)

    # Save maps as artifacts
    # (In a real scenario, use a specific artifact folder)

    print(f"Saved feature-ready ratings to {out_ratings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

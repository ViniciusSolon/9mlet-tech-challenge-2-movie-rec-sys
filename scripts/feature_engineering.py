"""DVC Stage: Feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.paths import get_processed_dir  # noqa: E402


def _build_id_maps(
    ratings: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """Map raw user/movie IDs to contiguous embedding indices."""
    user_map = {uid: i for i, uid in enumerate(ratings["userId"].unique())}
    movie_map = {mid: i for i, mid in enumerate(metadata["movieId"].unique())}
    return user_map, movie_map


def _attach_content_columns(
    ratings: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join lightweight content columns when present in enriched metadata."""
    content_cols = [
        col for col in ("year", "vote_average", "popularity") if col in metadata.columns
    ]
    if not content_cols:
        return ratings
    subset = metadata[["movieId", *content_cols]].drop_duplicates("movieId")
    return ratings.merge(subset, on="movieId", how="left")


def main() -> int:
    """Map IDs to contiguous indices and attach optional content columns."""
    processed_dir = get_processed_dir()
    ratings_path = processed_dir / "preprocessed_ratings.parquet"
    metadata_path = processed_dir / "enriched_metadata.parquet"
    if not ratings_path.exists() or not metadata_path.exists():
        print("Required inputs missing.")
        return 1

    ratings = pd.read_parquet(ratings_path)
    metadata = pd.read_parquet(metadata_path)
    user_map, movie_map = _build_id_maps(ratings, metadata)
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["userId"].map(user_map)
    ratings["movie_idx"] = ratings["movieId"].map(movie_map)
    ratings = ratings.dropna(subset=["movie_idx"])
    ratings["movie_idx"] = ratings["movie_idx"].astype(int)
    ratings = _attach_content_columns(ratings, metadata)

    out_path = processed_dir / "features_ratings.parquet"
    ratings.to_parquet(out_path)
    print(f"Saved feature-ready ratings to {out_path} ({len(ratings)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

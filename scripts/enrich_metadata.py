"""DVC Stage: Enrich MovieLens metadata with TMDB data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.external.movielens_io import build_link_table  # noqa: E402
from utils.paths import get_processed_dir, get_raw_dir  # noqa: E402


def main() -> int:
    raw_dir = get_raw_dir()
    processed_dir = get_processed_dir()

    tmdb_path = processed_dir / "movie_metadata.parquet"
    if not tmdb_path.exists():
        print(
            f"Error: {tmdb_path} not found. "
            "Run scripts/fetch_external_metadata.py first."
        )
        # Create a dummy for now if we want the pipeline to run in a demo
        return 1

    print("Building MovieLens link table (movies + links)...")
    ml_metadata = build_link_table(raw_dir)

    print(f"Loading TMDB metadata from {tmdb_path}...")
    tmdb_metadata = pd.read_parquet(tmdb_path)
    # Parquet usa snake_case (movie_id); MovieLens usa movieId.
    rename = {}
    if "movie_id" in tmdb_metadata.columns:
        rename["movie_id"] = "movieId"
    if "tmdb_id" in tmdb_metadata.columns:
        rename["tmdb_id"] = "tmdbId_tmdb"
    tmdb_metadata = tmdb_metadata.rename(columns=rename)

    print("Enriching metadata...")
    enriched = ml_metadata.merge(
        tmdb_metadata,
        on="movieId",
        how="left",
        suffixes=("", "_tmdb"),
    )

    out_path = processed_dir / "enriched_metadata.parquet"
    enriched.to_parquet(out_path)
    print(f"Saved enriched metadata to {out_path} ({len(enriched)} movies)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

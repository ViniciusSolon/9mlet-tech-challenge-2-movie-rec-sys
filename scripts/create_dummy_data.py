"""Utility to create dummy MovieLens-like data for pipeline testing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    """Write small GroupLens-named CSVs plus a tiny TMDB metadata parquet."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [1, 2, 1, 3, 2],
            "rating": [5.0, 4.0, 3.0, 5.0, 2.0],
            "timestamp": [12345678, 12345679, 12345680, 12345681, 12345682],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)"],
            "genres": ["Animation|Comedy", "Adventure|Children", "Comedy|Romance"],
        }
    )
    links = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "imdbId": ["0114709", "0113492", "0113228"],
            "tmdbId": [862, 8844, 15602],
        }
    )
    # Canonical GroupLens names (dvc.yaml) + Kaggle aliases for local flexibility.
    ratings.to_csv(raw_dir / "ratings.csv", index=False)
    ratings.to_csv(raw_dir / "rating.csv", index=False)
    movies.to_csv(raw_dir / "movies.csv", index=False)
    movies.to_csv(raw_dir / "movie.csv", index=False)
    links.to_csv(raw_dir / "links.csv", index=False)
    links.to_csv(raw_dir / "link.csv", index=False)

    pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [862, 8844, 15602],
            "overview": ["Summary 1", "Summary 2", "Summary 3"],
            "year": [1995, 1995, 1995],
            "vote_average": [8.0, 7.0, 6.5],
            "popularity": [10.0, 8.0, 5.0],
            "fetch_status": ["ok", "ok", "ok"],
        }
    ).to_parquet(processed_dir / "movie_metadata.parquet")

    print("Dummy data created under data/raw and data/processed.")


if __name__ == "__main__":
    main()

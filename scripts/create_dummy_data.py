"""Utility to create dummy MovieLens-like data for pipeline testing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [1, 2, 1, 3, 2],
            "rating": [5.0, 4.0, 3.0, 5.0, 2.0],
            "timestamp": [12345678, 12345679, 12345680, 12345681, 12345682],
        }
    ).to_csv(raw_dir / "rating.csv", index=False)

    pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Toy Story", "Jumanji", "Grumpier Old Men"],
            "genres": ["Animation|Comedy", "Adventure|Children", "Comedy|Romance"],
        }
    ).to_csv(raw_dir / "movie.csv", index=False)

    pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "imdbId": ["0114709", "0113492", "0113228"],
            "tmdbId": [862, 8844, 15602],
        }
    ).to_csv(raw_dir / "link.csv", index=False)

    pd.DataFrame(
        {
            "tmdbId": [862, 8844, 15602],
            "overview": ["Summary 1", "Summary 2", "Summary 3"],
            "fetch_status": ["ok", "ok", "ok"],
        }
    ).to_parquet(processed_dir / "movie_metadata.parquet")

    print("Dummy data created under data/raw and data/processed.")


if __name__ == "__main__":
    main()

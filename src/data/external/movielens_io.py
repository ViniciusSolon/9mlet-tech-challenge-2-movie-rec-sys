"""Load MovieLens link/movie/rating tables with flexible filenames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _resolve_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = raw_dir / name
        if path.is_file():
            return path
    msg = f"none of {candidates} found in {raw_dir}"
    raise FileNotFoundError(msg)


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """Load ``ratings.csv`` or ``rating.csv`` (GroupLens vs Kaggle naming)."""
    path = _resolve_file(raw_dir, ("ratings.csv", "rating.csv"))
    ratings = pd.read_csv(path)
    return ratings.rename(columns=str.strip)


def load_links(raw_dir: Path) -> pd.DataFrame:
    """Load ``links.csv`` or ``link.csv``."""
    path = _resolve_file(raw_dir, ("links.csv", "link.csv"))
    links = pd.read_csv(path)
    links = links.rename(columns=str.strip)
    return links


def load_movies(raw_dir: Path) -> pd.DataFrame:
    """Load ``movies.csv`` or ``movie.csv``."""
    path = _resolve_file(raw_dir, ("movies.csv", "movie.csv"))
    movies = pd.read_csv(path)
    movies = movies.rename(columns=str.strip)
    return movies


def build_link_table(raw_dir: Path) -> pd.DataFrame:
    """Join movies and links; flag invalid ``tmdbId`` values."""
    links = load_links(raw_dir)
    movies = load_movies(raw_dir)
    merged = movies.merge(links, on="movieId", how="left")
    merged["tmdbId"] = pd.to_numeric(merged["tmdbId"], errors="coerce")
    merged["imdbId"] = merged["imdbId"].astype("string")
    return merged

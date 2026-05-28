"""Structured records for external movie metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MovieMetadataRecord:
    """Normalized TMDB fields joined to MovieLens ``movieId``."""

    movie_id: int
    tmdb_id: int | None
    imdb_id: str | None
    title: str | None
    overview: str | None
    genres: str | None
    keywords: str | None
    release_year: int | None
    original_language: str | None
    vote_average: float | None
    popularity: float | None
    fetch_status: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON/parquet."""
        return asdict(self)

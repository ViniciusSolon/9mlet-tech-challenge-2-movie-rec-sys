"""Tests for metadata fetch orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from data.external.metadata_fetch import cache_path, fetch_one, records_to_parquet
from data.external.records import MovieMetadataRecord
from data.external.tmdb_client import TmdbClient


def test_records_to_parquet(tmp_path: Path) -> None:
    rec = MovieMetadataRecord(
        movie_id=1,
        tmdb_id=862,
        imdb_id="114709",
        title="Toy Story",
        overview="Plot",
        genres="Animation",
        keywords="toys",
        release_year=1995,
        original_language="en",
        vote_average=7.7,
        popularity=1.0,
        fetch_status="ok",
    )
    out = tmp_path / "movie_metadata.parquet"
    frame = records_to_parquet([rec], out)
    assert out.is_file()
    assert len(frame) == 1


@patch.object(TmdbClient, "get_movie")
def test_fetch_one_writes_cache(mock_get: MagicMock, tmp_path: Path) -> None:
    mock_get.return_value = {
        "title": "Toy Story",
        "overview": "Plot",
        "genres": [],
        "keywords": {"keywords": []},
        "release_date": "1995-01-01",
        "original_language": "en",
        "vote_average": 7.0,
        "popularity": 1.0,
    }
    client = TmdbClient("key", min_interval_sec=0.0)
    row = pd.Series({"movieId": 1, "tmdbId": 862, "imdbId": "114709", "title": "Toy"})
    rec = fetch_one(client, row, tmp_path, resume=False)
    assert rec.fetch_status == "ok"
    path = cache_path(tmp_path, 1)
    assert path.is_file()
    cached = json.loads(path.read_text(encoding="utf-8"))
    assert cached["movie_id"] == 1

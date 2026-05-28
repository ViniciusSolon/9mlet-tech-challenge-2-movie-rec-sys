"""Tests for TMDB client (HTTP mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data.external.errors import TmdbRequestError
from data.external.tmdb_client import TmdbClient


def _movie_payload() -> dict:
    return {
        "title": "Toy Story",
        "overview": "A toy story.",
        "genres": [{"name": "Animation"}],
        "keywords": {"keywords": [{"name": "toys"}]},
        "release_date": "1995-11-22",
        "original_language": "en",
        "vote_average": 7.7,
        "popularity": 100.0,
    }


@patch("data.external.tmdb_client.httpx.get")
def test_get_movie_success(mock_get: MagicMock) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _movie_payload()
    mock_get.return_value = response

    client = TmdbClient("test-key", min_interval_sec=0.0)
    data = client.get_movie(862)
    assert data is not None
    rec = client.parse_movie(1, 862, "0114709", data, status="ok")
    assert rec.overview == "A toy story."
    assert rec.release_year == 1995
    assert "Animation" in (rec.genres or "")


@patch("data.external.tmdb_client.httpx.get")
def test_get_movie_not_found(mock_get: MagicMock) -> None:
    response = MagicMock()
    response.status_code = 404
    mock_get.return_value = response

    client = TmdbClient("test-key", min_interval_sec=0.0)
    assert client.get_movie(999999) is None


@patch("data.external.tmdb_client.httpx.get")
def test_get_movie_raises_after_max_retries(mock_get: MagicMock) -> None:
    rate = MagicMock(status_code=429)
    mock_get.return_value = rate
    client = TmdbClient("test-key", min_interval_sec=0.0, max_retries=2)
    with pytest.raises(TmdbRequestError):
        client.get_movie(862)
    assert mock_get.call_count == 2


@patch("data.external.tmdb_client.httpx.get")
def test_get_movie_retries_on_429(mock_get: MagicMock) -> None:
    ok = MagicMock(status_code=200)
    ok.json.return_value = _movie_payload()
    rate = MagicMock(status_code=429)
    mock_get.side_effect = [rate, ok]

    client = TmdbClient("test-key", min_interval_sec=0.0, max_retries=3)
    data = client.get_movie(862)
    assert data is not None
    assert mock_get.call_count == 2

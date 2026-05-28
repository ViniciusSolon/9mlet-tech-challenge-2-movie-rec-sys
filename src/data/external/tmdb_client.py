"""TMDB HTTP client with rate limiting and bounded retries."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from data.external.errors import TmdbRequestError
from data.external.records import MovieMetadataRecord

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class TmdbClient:
    """Fetch movie details from TMDB API v3."""

    def __init__(
        self,
        api_key: str,
        *,
        language: str = "en-US",
        min_interval_sec: float = 0.26,
        max_retries: int = 4,
        timeout_sec: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._language = language
        self._min_interval = min_interval_sec
        self._max_retries = max(1, max_retries)
        self._timeout = timeout_sec
        self._last_call = 0.0

    def get_movie(self, tmdb_id: int) -> dict[str, Any] | None:
        """Return TMDB movie payload or ``None`` if not found (404)."""
        path = f"/movie/{tmdb_id}"
        params = {
            "api_key": self._api_key,
            "language": self._language,
            "append_to_response": "keywords",
        }
        return self._request_json(path, params)

    def parse_movie(
        self,
        movie_id: int,
        tmdb_id: int | None,
        imdb_id: str | None,
        payload: dict[str, Any] | None,
        *,
        status: str,
    ) -> MovieMetadataRecord:
        """Map API JSON to ``MovieMetadataRecord``."""
        if payload is None:
            return self._empty_record(movie_id, tmdb_id, imdb_id, status)

        genres = "|".join(g["name"] for g in payload.get("genres", []))
        kw_block = payload.get("keywords", {}).get("keywords", [])
        keywords = "|".join(k["name"] for k in kw_block)
        year = self._parse_year(payload.get("release_date"))
        return MovieMetadataRecord(
            movie_id=movie_id,
            tmdb_id=tmdb_id,
            imdb_id=imdb_id,
            title=payload.get("title"),
            overview=payload.get("overview") or None,
            genres=genres or None,
            keywords=keywords or None,
            release_year=year,
            original_language=payload.get("original_language"),
            vote_average=payload.get("vote_average"),
            popularity=payload.get("popularity"),
            fetch_status=status,
        )

    def _request_json(
        self,
        path: str,
        params: dict[str, str],
    ) -> dict[str, Any] | None:
        url = f"https://api.themoviedb.org/3{path}"
        last_detail = "unknown error"
        for attempt in range(self._max_retries):
            self._wait_rate_limit()
            try:
                response = httpx.get(url, params=params, timeout=self._timeout)
            except httpx.HTTPError as exc:
                last_detail = f"HTTP error: {exc}"
                logger.warning(
                    "tmdb http error path=%s attempt=%s/%s: %s",
                    path,
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt + 1 >= self._max_retries:
                    break
                self._backoff(attempt)
                continue
            if response.status_code == 404:
                return None
            if response.status_code in RETRY_STATUS:
                last_detail = f"status {response.status_code}"
                logger.warning(
                    "tmdb retryable status=%s path=%s attempt=%s/%s",
                    response.status_code,
                    path,
                    attempt + 1,
                    self._max_retries,
                )
                if attempt + 1 >= self._max_retries:
                    break
                self._backoff(attempt)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_detail = f"status {response.status_code}: {exc}"
                logger.warning(
                    "tmdb status error path=%s attempt=%s/%s: %s",
                    path,
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt + 1 >= self._max_retries:
                    break
                self._backoff(attempt)
                continue
            data = response.json()
            return data if isinstance(data, dict) else None

        raise TmdbRequestError(
            f"TMDB request failed after {self._max_retries} attempts ({last_detail})"
        )

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(2**attempt, 30))

    def _parse_year(self, release_date: str | None) -> int | None:
        if not release_date or len(release_date) < 4:
            return None
        try:
            return int(release_date[:4])
        except ValueError:
            return None

    def _empty_record(
        self,
        movie_id: int,
        tmdb_id: int | None,
        imdb_id: str | None,
        status: str,
    ) -> MovieMetadataRecord:
        return MovieMetadataRecord(
            movie_id=movie_id,
            tmdb_id=tmdb_id,
            imdb_id=imdb_id,
            title=None,
            overview=None,
            genres=None,
            keywords=None,
            release_year=None,
            original_language=None,
            vote_average=None,
            popularity=None,
            fetch_status=status,
        )

"""Orchestrate TMDB fetch, JSON cache, and parquet export."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from data.external.errors import TmdbRequestError
from data.external.movielens_io import build_link_table
from data.external.records import MovieMetadataRecord
from data.external.tmdb_client import TmdbClient
from utils.paths import get_external_metadata_dir, get_logs_dir

logger = logging.getLogger(__name__)
_BAR_WIDTH = 40


def setup_fetch_logging() -> None:
    """Log details to file only (terminal stays for progress bar + errors)."""
    log_dir = get_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "fetch_metadata.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _print_error(message: str) -> None:
    """Print one error line below the progress bar."""
    sys.stderr.write(f"\nERRO: {message}\n")
    sys.stderr.flush()


def _render_progress(current: int, total: int) -> None:
    """Draw an in-place ASCII progress bar."""
    if total <= 0:
        return
    ratio = min(current / total, 1.0)
    filled = int(_BAR_WIDTH * ratio)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    line = f"\r[{bar}] {current}/{total} ({ratio * 100:.1f}%)"
    sys.stderr.write(line)
    sys.stderr.flush()


def _finish_progress() -> None:
    """Move to the next line after the progress bar."""
    sys.stderr.write("\n")
    sys.stderr.flush()


def cache_path(cache_dir: Path, movie_id: int) -> Path:
    """Path for per-movie JSON cache file."""
    return cache_dir / f"{movie_id}.json"


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON cache atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_cache(path: Path) -> dict[str, Any] | None:
    """Read cached JSON if present."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_one(
    client: TmdbClient,
    row: pd.Series,
    cache_dir: Path,
    *,
    resume: bool,
) -> MovieMetadataRecord:
    """Fetch or load cache for a single movie row."""
    movie_id = int(row["movieId"])
    tmdb_raw = row.get("tmdbId")
    tmdb_id = int(tmdb_raw) if pd.notna(tmdb_raw) else None
    imdb_id = str(row["imdbId"]) if pd.notna(row.get("imdbId")) else None
    title_ml = row.get("title")

    path = cache_path(cache_dir, movie_id)
    if resume and path.is_file():
        try:
            cached = load_cache(path)
            if cached and cached.get("fetch_status") == "ok":
                return _record_from_cache(cached)
        except json.JSONDecodeError as exc:
            _print_error(f"movieId={movie_id} cache JSON inválido: {exc}")

    if tmdb_id is None or tmdb_id <= 0:
        rec = client.parse_movie(
            movie_id, None, imdb_id, None, status="missing_tmdb_id"
        )
        save_cache(path, _cache_payload(rec, title_ml, None))
        return rec

    try:
        payload = client.get_movie(tmdb_id)
    except TmdbRequestError as exc:
        _print_error(f"movieId={movie_id} tmdbId={tmdb_id} — {exc}")
        logger.exception("tmdb request failed movieId=%s tmdbId=%s", movie_id, tmdb_id)
        rec = client.parse_movie(movie_id, tmdb_id, imdb_id, None, status="api_error")
        save_cache(path, _cache_payload(rec, title_ml, None))
        return rec

    status = "ok" if payload else "not_found"
    rec = client.parse_movie(movie_id, tmdb_id, imdb_id, payload, status=status)
    save_cache(path, _cache_payload(rec, title_ml, payload))
    return rec


def run_fetch(
    raw_dir: Path,
    client: TmdbClient,
    *,
    limit: int | None,
    resume: bool,
) -> list[MovieMetadataRecord]:
    """Fetch metadata for all movies in the link table."""
    table = build_link_table(raw_dir)
    if limit is not None:
        table = table.head(limit)
    cache_dir = get_external_metadata_dir()
    records: list[MovieMetadataRecord] = []
    total = len(table)

    logger.info(
        "starting fetch total=%s resume=%s raw_dir=%s cache_dir=%s",
        total,
        resume,
        raw_dir,
        cache_dir,
    )

    for idx, (_, row) in enumerate(table.iterrows(), start=1):
        rec = fetch_one(client, row, cache_dir, resume=resume)
        records.append(rec)
        _render_progress(idx, total)

    _finish_progress()
    logger.info("finished total=%s", total)
    return records


def records_to_parquet(
    records: list[MovieMetadataRecord],
    out_path: Path,
) -> pd.DataFrame:
    """Write consolidated parquet and return DataFrame."""
    frame = pd.DataFrame([r.to_dict() for r in records])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return frame


def _cache_payload(
    rec: MovieMetadataRecord,
    title_ml: object,
    tmdb_json: dict[str, Any] | None,
) -> dict[str, Any]:
    data = rec.to_dict()
    data["title_movielens"] = title_ml
    data["tmdb_raw"] = tmdb_json
    return data


def _record_from_cache(cached: dict[str, Any]) -> MovieMetadataRecord:
    keys = MovieMetadataRecord.__dataclass_fields__
    payload = {k: cached.get(k) for k in keys}
    return MovieMetadataRecord(**payload)

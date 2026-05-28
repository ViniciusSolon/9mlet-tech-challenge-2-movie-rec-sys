#!/usr/bin/env python
"""CLI: batch-fetch TMDB metadata for MovieLens movies (pré-etapa P.4–P.7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from configs.settings import load_settings  # noqa: E402

from data.external.metadata_fetch import (  # noqa: E402
    records_to_parquet,
    run_fetch,
    setup_fetch_logging,
)
from data.external.tmdb_client import TmdbClient  # noqa: E402
from utils.paths import get_processed_dir, get_raw_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Fetch TMDB metadata for MovieLens.")
    parser.add_argument("--limit", type=int, default=None, help="Max movies (dev).")
    parser.add_argument("--resume", action="store_true", help="Skip cached JSON files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="MovieLens raw folder (default: data/raw).",
    )
    return parser.parse_args()


def main() -> int:
    """Run fetch and write ``movie_metadata.parquet``."""
    setup_fetch_logging()
    args = parse_args()
    settings = load_settings()
    api_key = settings.ensure_tmdb_key()
    raw_dir = args.raw_dir or get_raw_dir()

    client = TmdbClient(
        api_key,
        language=settings.tmdb_language,
        min_interval_sec=settings.tmdb_min_interval_sec,
        max_retries=settings.tmdb_max_retries,
    )
    records = run_fetch(raw_dir, client, limit=args.limit, resume=args.resume)
    out = get_processed_dir() / "movie_metadata.parquet"
    frame = records_to_parquet(records, out)
    ok = int((frame["fetch_status"] == "ok").sum())
    api_err = int((frame["fetch_status"] == "api_error").sum())
    not_found = int((frame["fetch_status"] == "not_found").sum())
    missing = int((frame["fetch_status"] == "missing_tmdb_id").sum())
    print(
        f"Concluído: {out}\n"
        f"  total={len(frame)} ok={ok} api_error={api_err} "
        f"not_found={not_found} missing_tmdb_id={missing}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

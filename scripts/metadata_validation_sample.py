#!/usr/bin/env python
"""Generate manual validation sample for pré-etapa P.9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.external.coverage_report import (  # noqa: E402
    build_validation_sample,
    validation_sample_to_markdown,
)
from data.external.movielens_io import load_movies  # noqa: E402
from utils.paths import get_processed_dir, get_project_root, get_raw_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="TMDB metadata validation sample.")
    parser.add_argument("--size", type=int, default=10, help="Sample size.")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path.")
    return parser.parse_args()


def main() -> int:
    """Write validation checklist markdown."""
    args = parse_args()
    parquet = get_processed_dir() / "movie_metadata.parquet"
    if not parquet.is_file():
        print(f"Arquivo não encontrado: {parquet}", file=sys.stderr)
        return 1

    metadata = pd.read_parquet(parquet)
    movies = load_movies(get_raw_dir())
    sample = build_validation_sample(metadata, movies, size=args.size)
    out = args.out or (get_project_root() / "docs" / "METADATA_VALIDATION_SAMPLE.md")
    out.write_text(validation_sample_to_markdown(sample), encoding="utf-8")
    print(f"Amostra P.9: {out} ({len(sample)} filmes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

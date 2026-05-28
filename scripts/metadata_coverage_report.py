#!/usr/bin/env python
"""Generate metadata coverage report (pré-etapa P.8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.external.coverage_report import (  # noqa: E402
    build_coverage_frame,
    coverage_to_markdown,
)
from utils.paths import get_processed_dir, get_project_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="TMDB metadata coverage report.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Path to movie_metadata.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown output (default: docs/METADATA_COVERAGE_REPORT.md)",
    )
    return parser.parse_args()


def main() -> int:
    """Write coverage markdown and print summary."""
    args = parse_args()
    parquet = args.parquet or (get_processed_dir() / "movie_metadata.parquet")
    if not parquet.is_file():
        print(f"Arquivo não encontrado: {parquet}", file=sys.stderr)
        return 1

    metadata = pd.read_parquet(parquet)
    stats = build_coverage_frame(metadata)
    out = args.out or (get_project_root() / "docs" / "METADATA_COVERAGE_REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(coverage_to_markdown(stats, parquet), encoding="utf-8")

    print(f"Relatório: {out}")
    print(
        f"ok={stats['ok']}/{stats['total_rows']} ({stats['ok_pct_total']}%) | "
        f"overview={stats['overview_pct_ok']}% | "
        f"genres={stats['genres_pct_ok']}% | "
        f"keywords={stats['keywords_pct_ok']}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

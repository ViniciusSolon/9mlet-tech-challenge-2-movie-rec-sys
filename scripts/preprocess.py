"""DVC Stage: Preprocess raw MovieLens data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.external.movielens_io import load_ratings  # noqa: E402
from data.preprocessors.registry import get_preprocessor  # noqa: E402
from utils.paths import get_processed_dir, get_raw_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="explicit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = get_raw_dir()
    processed_dir = get_processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        ratings = load_ratings(raw_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}. Download MovieLens 20M into data/raw/.")
        return 1

    print(f"Loaded {len(ratings)} ratings from {raw_dir}")
    print(f"Applying preprocessor strategy: {args.strategy}")
    preprocessor = get_preprocessor(args.strategy, min_rating=1.0)
    processed_ratings = preprocessor.transform(ratings)

    out_path = processed_dir / "preprocessed_ratings.parquet"
    processed_ratings.to_parquet(out_path)
    print(f"Saved preprocessed ratings to {out_path} ({len(processed_ratings)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

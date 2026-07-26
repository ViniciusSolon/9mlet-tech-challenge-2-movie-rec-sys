"""Script to download real MovieLens dataset into data/raw."""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DATASETS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}


def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
    print(f"Downloaded {len(content)} bytes. Extracting files...")
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for file_info in z.infolist():
            filename = Path(file_info.filename).name
            if filename in ("ratings.csv", "movies.csv", "links.csv", "tags.csv"):
                target_path = dest_dir / filename
                with z.open(file_info) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                print(f"Extracted: {target_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MovieLens dataset.")
    parser.add_argument(
        "--variant",
        choices=["small", "20m"],
        default="small",
        help=(
            "Dataset variant: 'small' (100k ratings, ~1MB) "
            "or '20m' (20M ratings, ~100MB)."
        ),
    )
    args = parser.parse_args()
    url = DATASETS[args.variant]
    download_and_extract(url, RAW_DIR)
    print("Dataset download complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

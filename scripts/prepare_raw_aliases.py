"""Ensure GroupLens CSV names exist for DVC deps (Kaggle-compatible)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.raw_aliases import ensure_raw_aliases  # noqa: E402
from utils.paths import get_raw_dir  # noqa: E402


def main() -> int:
    """Create ratings.csv / movies.csv / links.csv aliases when needed."""
    try:
        for message in ensure_raw_aliases(get_raw_dir()):
            print(message)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Minimal entrypoint to verify Docker image and project layout."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Print scaffold status and exit successfully."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    print("movie-rec-sys: hello train (scaffold)")
    print(f"project_root={root}")
    print(f"src_exists={src.is_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Normalize MovieLens raw filenames (GroupLens vs Kaggle)."""

from __future__ import annotations

from pathlib import Path

# Canonical GroupLens names expected by dvc.yaml → aliases used by Kaggle dumps.
_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ratings.csv", ("rating.csv",)),
    ("movies.csv", ("movie.csv",)),
    ("links.csv", ("link.csv",)),
)


def ensure_raw_aliases(raw_dir: Path) -> list[str]:
    """Create relative symlinks (or hardlinks) for missing canonical names.

    Args:
        raw_dir: Directory containing MovieLens CSV files.

    Returns:
        Human-readable messages describing actions taken.

    Raises:
        FileNotFoundError: If neither canonical nor alias file exists.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    for canonical, aliases in _ALIASES:
        messages.append(_ensure_one(raw_dir, canonical, aliases))
    return messages


def _ensure_one(raw_dir: Path, canonical: str, aliases: tuple[str, ...]) -> str:
    target = raw_dir / canonical
    if target.exists():
        return f"ok: {canonical}"
    source = _first_existing(raw_dir, aliases)
    if source is None:
        msg = f"missing {canonical} and aliases {aliases} in {raw_dir}"
        raise FileNotFoundError(msg)
    _link_or_copy(source, target)
    return f"linked: {canonical} -> {source.name}"


def _first_existing(raw_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = raw_dir / name
        if path.is_file():
            return path
    return None


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        target.symlink_to(source.name)
    except OSError:
        try:
            target.hardlink_to(source)
        except OSError:
            target.write_bytes(source.read_bytes())

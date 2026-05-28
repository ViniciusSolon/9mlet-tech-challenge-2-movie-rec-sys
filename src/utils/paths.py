"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return ``data/`` directory."""
    return get_project_root() / "data"


def get_raw_dir() -> Path:
    """Return ``data/raw/`` directory."""
    return get_data_dir() / "raw"


def get_processed_dir() -> Path:
    """Return ``data/processed/`` directory."""
    return get_data_dir() / "processed"


def get_external_metadata_dir() -> Path:
    """Return TMDB raw JSON cache directory."""
    return get_raw_dir() / "external_metadata"


def get_logs_dir() -> Path:
    """Return ``data/logs/`` directory."""
    return get_data_dir() / "logs"

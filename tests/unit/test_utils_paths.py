"""Tests for path utilities."""

from __future__ import annotations

from utils.paths import get_data_dir, get_processed_dir, get_raw_dir


def test_data_paths_exist_or_are_defined() -> None:
    assert get_data_dir().name == "data"
    assert get_raw_dir().name == "raw"
    assert get_processed_dir().name == "processed"

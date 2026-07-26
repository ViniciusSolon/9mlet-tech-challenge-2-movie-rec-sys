"""Unit tests for MovieLens raw filename aliases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.raw_aliases import ensure_raw_aliases


def test_ensure_raw_aliases_links_kaggle_names(tmp_path: Path) -> None:
    pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0]}).to_csv(
        tmp_path / "rating.csv", index=False
    )
    pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Comedy"]}).to_csv(
        tmp_path / "movie.csv", index=False
    )
    pd.DataFrame({"movieId": [1], "imdbId": ["1"], "tmdbId": [1]}).to_csv(
        tmp_path / "link.csv", index=False
    )

    messages = ensure_raw_aliases(tmp_path)
    assert (tmp_path / "ratings.csv").exists()
    assert (tmp_path / "movies.csv").exists()
    assert (tmp_path / "links.csv").exists()
    assert any("linked" in message for message in messages)


def test_ensure_raw_aliases_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_raw_aliases(tmp_path)

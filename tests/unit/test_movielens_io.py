"""Tests for MovieLens IO helpers."""

from __future__ import annotations

from pathlib import Path

from data.external.movielens_io import build_link_table, load_links


def test_load_links_finds_link_csv() -> None:
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not (raw / "link.csv").is_file() and not (raw / "links.csv").is_file():
        return
    links = load_links(raw)
    assert "movieId" in links.columns
    assert "tmdbId" in links.columns


def test_build_link_table_merges() -> None:
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not (raw / "movie.csv").is_file():
        return
    table = build_link_table(raw)
    assert "title" in table.columns
    assert len(table) > 0

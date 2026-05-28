"""Tests for metadata coverage report."""

from __future__ import annotations

import pandas as pd

from data.external.coverage_report import build_coverage_frame, coverage_to_markdown


def test_build_coverage_frame() -> None:
    frame = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "fetch_status": ["ok", "ok", "not_found"],
            "title": ["A", "B", None],
            "overview": ["text", None, None],
            "genres": ["G", None, None],
            "keywords": ["k", "k2", None],
            "release_year": [1999, None, None],
        }
    )
    stats = build_coverage_frame(frame)
    assert stats["total_rows"] == 3
    assert stats["ok"] == 2
    assert stats["overview_pct_ok"] == 50.0
    assert "ok" in coverage_to_markdown(stats, "data/processed/movie_metadata.parquet")

"""Tests for temporal train/test splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from data.splits import assert_temporal_order, temporal_train_test_split


def test_temporal_split_keeps_past_before_future() -> None:
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "movie_idx": [1, 2, 1, 3, 2],
            "rating": [5.0, 4.0, 3.0, 5.0, 2.0],
            "timestamp": [10, 20, 30, 40, 50],
        }
    )
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.4, seed=0)
    assert len(train_df) + len(test_df) == len(df)
    assert assert_temporal_order(train_df, test_df)


def test_temporal_split_fallback_without_timestamp() -> None:
    df = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3, 4],
            "movie_idx": [1, 2, 3, 4, 5],
            "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2, seed=42)
    assert len(test_df) == 1
    assert len(train_df) == 4


def test_temporal_split_rejects_empty() -> None:
    with pytest.raises(ValueError):
        temporal_train_test_split(pd.DataFrame(), test_ratio=0.2)

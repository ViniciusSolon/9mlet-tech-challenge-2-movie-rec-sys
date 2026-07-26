"""Tests for ranking and rating evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.metrics import (
    average_metrics,
    hit_rate_at_k,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        assert precision_at_k([1, 2, 3], {4, 5}, k=3) == pytest.approx(0.0)

    def test_partial_hit(self) -> None:
        assert precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == pytest.approx(0.5)

    def test_k_truncation(self) -> None:
        # Only top-2 are considered; item 3 (relevant) is outside cut-off
        assert precision_at_k([1, 2, 3], {3}, k=2) == pytest.approx(0.0)

    def test_zero_k_returns_zero(self) -> None:
        assert precision_at_k([1, 2], {1}, k=0) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k([1, 2], {1, 2}, k=2) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        assert recall_at_k([1, 2, 3], {1, 4}, k=3) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k([1, 2], set(), k=2) == pytest.approx(0.0)


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_no_hits(self) -> None:
        assert ndcg_at_k([1, 2, 3], {4, 5}, k=3) == pytest.approx(0.0)

    def test_single_hit_first_position(self) -> None:
        # DCG = 1/log2(2) = 1; IDCG = 1/log2(2) = 1 → NDCG = 1
        score = ndcg_at_k([5, 1, 2], {5}, k=3)
        assert score == pytest.approx(1.0)

    def test_single_hit_second_position(self) -> None:
        # DCG = 1/log2(3); IDCG = 1/log2(2) → NDCG = log2(2)/log2(3)
        expected = math.log2(2) / math.log2(3)
        assert ndcg_at_k([1, 5, 2], {5}, k=3) == pytest.approx(expected)

    def test_empty_relevant_returns_zero(self) -> None:
        assert ndcg_at_k([1, 2], set(), k=2) == pytest.approx(0.0)


class TestHitRateAtK:
    def test_hit(self) -> None:
        assert hit_rate_at_k([1, 2, 3], {3}, k=3) == pytest.approx(1.0)

    def test_no_hit(self) -> None:
        assert hit_rate_at_k([1, 2, 3], {4}, k=3) == pytest.approx(0.0)

    def test_hit_outside_k(self) -> None:
        assert hit_rate_at_k([1, 2, 3], {3}, k=2) == pytest.approx(0.0)


class TestRatingMetrics:
    def test_rmse_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        y_true = np.array([3.0, 3.0])
        y_pred = np.array([4.0, 2.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_known_value(self) -> None:
        y_true = np.array([3.0, 3.0])
        y_pred = np.array([4.0, 2.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)


class TestAverageMetrics:
    def test_returns_all_keys(self) -> None:
        recs = [[1, 2], [3, 4]]
        rels = [{1}, {3}]
        result = average_metrics(recs, rels, k=2)
        assert set(result) == {"precision", "recall", "ndcg", "hit_rate"}

    def test_empty_input(self) -> None:
        result = average_metrics([], [], k=10)
        assert all(v == pytest.approx(0.0) for v in result.values())

    def test_values_bounded(self) -> None:
        recs = [[1, 2, 3], [4, 5, 6]]
        rels = [{1, 2}, {4}]
        result = average_metrics(recs, rels, k=3)
        for val in result.values():
            assert 0.0 <= val <= 1.0

"""Retrieval metrics.

These are the numbers the project's central claim rests on, so they are worth
testing against hand-computable cases rather than trusting by inspection. A
metric that is subtly wrong produces a plausible-looking results table that
argues for the wrong conclusion.
"""

from __future__ import annotations

import math

import pytest

from graphrag.eval.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)

# --- recall -----------------------------------------------------------


def test_recall_counts_relevant_found_not_positions():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "c"}, k=3) == 0.5
    assert recall_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_recall_respects_the_cutoff():
    """An item at rank 4 must not count towards recall@3."""
    assert recall_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0
    assert recall_at_k(["x", "y", "z", "a"], {"a"}, k=4) == 1.0


def test_recall_with_no_relevant_items_is_zero_not_a_crash():
    assert recall_at_k(["a"], set(), k=3) == 0.0


# --- precision --------------------------------------------------------


def test_precision_divides_by_retrieved_not_relevant():
    assert precision_at_k(["a", "x"], {"a"}, k=2) == 0.5
    assert precision_at_k([], {"a"}, k=2) == 0.0


# --- MRR --------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved,expected",
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 0.5),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
    ],
)
def test_reciprocal_rank_uses_the_first_hit(retrieved, expected):
    assert reciprocal_rank(retrieved, {"a"}) == pytest.approx(expected)


def test_mrr_ignores_later_hits():
    """Only the first relevant item matters, by definition."""
    assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0


# --- nDCG -------------------------------------------------------------


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_found():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_penalises_a_worse_ordering():
    """Same items retrieved, different order -> strictly lower score."""
    good = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    bad = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert good > bad


def test_ndcg_matches_a_hand_computed_value():
    # One relevant item at rank 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1.
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))


# --- composition ------------------------------------------------------


def test_score_retrieval_reports_every_metric():
    scores = score_retrieval(["a", "x"], {"a"}, k=2)
    keys = scores.as_dict()
    assert set(keys) == {"recall@2", "precision@2", "mrr", "ndcg@2"}


def test_aggregate_averages_across_questions():
    a = score_retrieval(["a"], {"a"}, k=1)   # perfect
    b = score_retrieval(["x"], {"a"}, k=1)   # miss
    avg = aggregate([a, b])
    assert avg["recall@1"] == 0.5
    assert avg["mrr"] == 0.5


def test_aggregate_of_nothing_is_empty_not_a_crash():
    assert aggregate([]) == {}
